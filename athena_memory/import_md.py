"""Markdown memory importer.

Ports `runImportMarkdown` from the TS `cli.ts`. Walks one or more glob
patterns, parses Markdown bullets as memory entries, and feeds them into the
provider's existing ``_extract_and_write`` / ``_write_entries`` path.

Recognised file shapes:

* ``MEMORY.md`` (or any file whose name is ``MEMORY.md``)
* ``memory/<YYYY-MM-DD>.md`` (date in filename becomes the entry timestamp)

Section structure:

* ``# Heading`` / ``## Heading`` lines set the **category** for any bullets
  that follow until the next heading. The category is normalised to lower
  case and mapped against ``MEMORY_CATEGORIES`` (e.g. ``## Preferences``
  → ``preferences``). Unknown headings fall back to ``cases``.
* Bullet lines (``- ``, ``* ``, ``+ ``) become individual memory entries.
* Blank lines and any other content are ignored.

Use via the CLI as ``hermes-memory-lancedb import-markdown <glob>`` or
programmatically via :func:`run_import_markdown`.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Reuse the canonical category list when available, else hardcode.
try:
    from . import MEMORY_CATEGORIES as _CATS
except Exception:  # pragma: no cover - defensive
    _CATS = ["profile", "preferences", "entities", "events", "cases", "patterns"]

_BULLET_RE = re.compile(r"^[-*+]\s+(.+)$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_DATE_IN_FILENAME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DEFAULT_GLOBS = ("memory/**/*.md", "MEMORY.md")


def _normalise_category(heading: str) -> str:
    """Map a heading like '## Preferences' to a known category, else 'cases'."""
    raw = heading.strip().lower()
    # Strip leading bullet/punctuation noise.
    raw = re.sub(r"[^a-z]+", " ", raw).strip()
    # Direct match.
    if raw in _CATS:
        return raw
    # Singular/plural fallbacks.
    for cat in _CATS:
        if raw == cat or raw.rstrip("s") == cat.rstrip("s"):
            return cat
    # Heuristic mappings from common doc headings.
    aliases = {
        "user profile": "profile",
        "background": "profile",
        "facts": "entities",
        "people": "entities",
        "places": "entities",
        "decisions": "events",
        "history": "events",
        "tasks": "cases",
        "projects": "cases",
        "rules": "patterns",
        "behaviours": "patterns",
        "behaviors": "patterns",
        "habits": "patterns",
    }
    return aliases.get(raw, "cases")


def _timestamp_from_path(path: Path) -> float:
    """Extract a Unix timestamp from a date in the filename.

    Returns the current time if the filename has no ``YYYY-MM-DD`` prefix.
    """
    m = _DATE_IN_FILENAME_RE.search(path.name)
    if not m:
        return time.time()
    try:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return dt.timestamp()
    except ValueError:
        return time.time()


def parse_markdown_file(path: Path, default_category: str = "cases") -> List[Dict]:
    """Parse a single markdown file into a list of entry dicts.

    Each returned dict has keys: ``content``, ``category``, ``timestamp``,
    ``source``, ``abstract``. ``content`` is the raw bullet text (no leading
    dash); ``abstract`` is the same text truncated to 80 chars.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("import-markdown: failed to read %s: %s", path, e)
        return []

    # Strip BOM and normalise CRLF.
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = text.replace("\r\n", "\n").split("\n")

    ts = _timestamp_from_path(path)
    current_cat = default_category
    out: List[Dict] = []
    for line in lines:
        h = _HEADING_RE.match(line)
        if h:
            current_cat = _normalise_category(h.group(1))
            continue
        b = _BULLET_RE.match(line)
        if not b:
            continue
        body = b.group(1).strip()
        if len(body) < 5:
            continue
        out.append({
            "content": body,
            "category": current_cat,
            "timestamp": ts,
            "source": f"markdown:{path.name}",
            "abstract": body[:80],
        })
    return out


def _expand_globs(patterns: Iterable[str], base_dir: Path) -> List[Path]:
    seen: List[Path] = []
    seen_set = set()
    for pattern in patterns:
        # Allow either absolute or base-relative patterns.
        if os.path.isabs(pattern):
            matches = glob.glob(pattern, recursive=True)
        else:
            matches = glob.glob(str(base_dir / pattern), recursive=True)
        for m in matches:
            p = Path(m).resolve()
            if p.is_file() and p not in seen_set:
                seen.append(p)
                seen_set.add(p)
    return seen


def discover_files(
    patterns: Optional[Iterable[str]] = None,
    base_dir: Optional[Path] = None,
) -> List[Path]:
    """Return the list of markdown files matched by `patterns`.

    Defaults to ``memory/**/*.md`` + ``MEMORY.md`` rooted at ``base_dir``
    (or ``cwd`` when not given).
    """
    base = base_dir or Path.cwd()
    pats = list(patterns) if patterns else list(_DEFAULT_GLOBS)
    return _expand_globs(pats, base)


def run_import_markdown(
    write_fn: Callable[[List[Dict]], None],
    patterns: Optional[Iterable[str]] = None,
    *,
    base_dir: Optional[Path] = None,
    dry_run: bool = False,
    min_text_length: int = 5,
    importance: float = 0.7,
    on_message: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Walk the given globs, parse markdown, and forward entries to ``write_fn``.

    Parameters
    ----------
    write_fn:
        Callable invoked with a batch list of entry dicts (matches the
        provider's ``_write_entries`` signature). Skipped when ``dry_run``.
    patterns:
        Iterable of glob patterns relative to ``base_dir``. Defaults to
        ``memory/**/*.md`` + ``MEMORY.md``.
    base_dir:
        Base directory for relative globs. Defaults to ``cwd``.
    dry_run:
        If True, parse but do not write — only count.
    min_text_length:
        Bullet lines shorter than this are skipped (after stripping).
    importance:
        Default importance (0.0–1.0) for imported entries.
    on_message:
        Optional progress callback (e.g. ``click.echo``).

    Returns
    -------
    dict
        ``{"imported": int, "skipped": int, "found_files": int, "files": [str]}``
    """
    log = on_message or (lambda _msg: None)
    files = discover_files(patterns, base_dir)
    if not files:
        return {"imported": 0, "skipped": 0, "found_files": 0, "files": []}

    importance = max(0.0, min(1.0, importance))
    imported = 0
    skipped = 0
    batch: List[Dict] = []

    for f in files:
        entries = parse_markdown_file(f)
        for e in entries:
            if len(e["content"]) < min_text_length:
                skipped += 1
                continue
            e["importance"] = importance
            e["tier"] = "peripheral"
            if dry_run:
                log(
                    f"  [dry-run] would import: {e['content'][:80]}"
                    + ("..." if len(e["content"]) > 80 else "")
                )
                imported += 1
            else:
                batch.append(e)
                imported += 1

    if not dry_run and batch:
        # Send in chunks to keep the embedder happy.
        chunk = 50
        for i in range(0, len(batch), chunk):
            try:
                write_fn(batch[i : i + chunk])
            except Exception as exc:
                logger.warning("import-markdown: write failed: %s", exc)
                skipped += len(batch[i : i + chunk])
                imported -= len(batch[i : i + chunk])

    return {
        "imported": imported,
        "skipped": skipped,
        "found_files": len(files),
        "files": [str(f) for f in files],
    }


__all__ = [
    "run_import_markdown",
    "parse_markdown_file",
    "discover_files",
]
