"""Auto-capture cleanup.

Loose port of ``auto-capture-cleanup.ts`` adapted to this Python store.

Auto-captured memories are entries written by ``on_session_end``, the
extraction pipeline, or the compress hook. They tend to be lower quality
than explicit ``lancedb_remember`` writes because the LLM had to extract
them from raw turns rather than being told what to remember.

This module runs a cleanup pass at session start for the *previous* session's
auto-captures. It:

  - **Deletes** entries that are pure inbound metadata sentinels, addressing
    boilerplate, runtime wrapper notices, or other clearly-unwanted noise.
  - **Demotes** low-value working/core entries from auto-capture sources
    (composite < threshold) to ``peripheral`` so they age out fast.
  - **Cleans content** by stripping known auto-capture prefixes (untrusted
    metadata blocks, ``<relevant-memories>`` injections, addressing tokens)
    when the content is otherwise worth keeping.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel patterns
# ---------------------------------------------------------------------------

_INBOUND_META_SENTINELS = (
    "Conversation info (untrusted metadata):",
    "Sender (untrusted metadata):",
    "Thread starter (untrusted, for context):",
    "Replied message (untrusted, for context):",
    "Forwarded message context (untrusted metadata):",
    "Chat history since last reply (untrusted, for context):",
)

_SESSION_RESET_PREFIX = (
    "A new session was started via /new or /reset. Execute your Session Startup sequence now"
)

_ADDRESSING_PREFIX_RE = re.compile(r"^(?:<@!?[0-9]+>|@[A-Za-z0-9_.-]+)\s*")
_SYSTEM_EVENT_LINE_RE = re.compile(
    r"^System:\s*\[[^\n]*?\]\s*Exec\s+(?:completed|failed|started)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_RUNTIME_WRAPPER_PREFIX_RE = re.compile(
    r"^\[(?:Subagent Context|Subagent Task)\]", re.IGNORECASE
)
_RUNTIME_WRAPPER_LINE_RE = re.compile(
    r"^\[(?:Subagent Context|Subagent Task)\]\s*", re.IGNORECASE
)
_RUNTIME_WRAPPER_BOILERPLATE_RE = re.compile(
    r"(?:You are running as a subagent\b.*?(?:$|(?<=\.)\s+)|"
    r"Results auto-announce to your requester\.?\s*|"
    r"do not busy-poll for status\.?\s*|"
    r"Reply with a brief acknowledgment only\.?\s*|"
    r"Do not use any memory tools\.?\s*)",
    re.IGNORECASE,
)


def _escape(s: str) -> str:
    return re.escape(s)


_INBOUND_META_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(_escape(s) for s in _INBOUND_META_SENTINELS) + r")\s*\n```json[\s\S]*?\n```\s*",
    re.MULTILINE,
)

_RELEVANT_MEMORIES_RE = re.compile(
    r"<relevant-memories>\s*[\s\S]*?</relevant-memories>\s*", re.IGNORECASE
)
_UNTRUSTED_DATA_RE = re.compile(
    r"\[UNTRUSTED DATA[^\n]*\][\s\S]*?\[END UNTRUSTED DATA\]\s*", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _strip_session_reset_prefix(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith(_SESSION_RESET_PREFIX):
        return trimmed
    blank_idx = trimmed.find("\n\n")
    if blank_idx >= 0:
        return trimmed[blank_idx + 2 :].strip()
    lines = trimmed.split("\n")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines[2:]).strip()


def _strip_runtime_wrappers(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return trimmed
    cleaned: List[str] = []
    stripping = True
    for line in trimmed.split("\n"):
        cur = line.strip()
        if stripping and cur == "":
            continue
        if stripping and _RUNTIME_WRAPPER_PREFIX_RE.match(cur):
            remainder = _RUNTIME_WRAPPER_LINE_RE.sub("", cur).strip()
            remainder = _RUNTIME_WRAPPER_BOILERPLATE_RE.sub("", remainder)
            remainder = re.sub(r"\s{2,}", " ", remainder).strip()
            if remainder:
                cleaned.append(remainder)
                stripping = False
            continue
        stripping = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def strip_auto_capture_prefix(role: str, text: str) -> str:
    """Apply all known auto-capture cleanup transforms to ``text``."""
    if role != "user":
        return text.strip()
    normalized = text.strip()
    normalized = _RELEVANT_MEMORIES_RE.sub("", normalized)
    normalized = _UNTRUSTED_DATA_RE.sub("", normalized)
    normalized = _strip_session_reset_prefix(normalized)
    # Strip metadata blocks repeatedly until no further change.
    for _ in range(6):
        before = normalized
        normalized = _SYSTEM_EVENT_LINE_RE.sub("\n", normalized)
        normalized = _INBOUND_META_BLOCK_RE.sub("\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        if normalized == before.strip():
            break
    normalized = _ADDRESSING_PREFIX_RE.sub("", normalized).strip()
    normalized = _strip_runtime_wrappers(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def is_pure_metadata(text: str) -> bool:
    """Returns True if the text is *only* auto-capture prefix junk."""
    if not text or not text.strip():
        return True
    cleaned = strip_auto_capture_prefix("user", text)
    return len(cleaned.strip()) < 8


# ---------------------------------------------------------------------------
# Cleanup pass
# ---------------------------------------------------------------------------

@dataclass
class CleanupReport:
    scanned: int = 0
    deleted: int = 0
    demoted: int = 0
    cleaned: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "deleted": self.deleted,
            "demoted": self.demoted,
            "cleaned": self.cleaned,
            "error": self.error,
        }


# Sources that we treat as "auto-captured" — i.e. eligible for cleanup.
AUTO_CAPTURE_SOURCES = (
    "session_end",
    "session_compressed",
    "compress",
    "extraction",
    "extraction_merge",
    "turn",
    "builtin",
    "compactor",
)


def _eligible(row: Dict[str, Any]) -> bool:
    src = str(row.get("source") or "").lower()
    if not src:
        return False
    for tag in AUTO_CAPTURE_SOURCES:
        if src.startswith(tag):
            return True
    return False


def _row_value_score(row: Dict[str, Any]) -> float:
    """Heuristic 0..1 value score (higher = keep)."""
    importance = float(row.get("importance") or 0.0)
    access = int(row.get("access_count") or 0)
    content_len = len(str(row.get("content") or ""))
    abstract_len = len(str(row.get("abstract") or ""))
    # Tier weight
    tier = row.get("tier") or "peripheral"
    tier_w = {"core": 0.4, "working": 0.25, "peripheral": 0.0}.get(tier, 0.0)
    # Content weight
    cw = min(0.3, content_len / 2000.0) + min(0.1, abstract_len / 160.0)
    # Access weight (saturates fast)
    aw = min(0.2, access / 5.0)
    return min(1.0, 0.6 * importance + tier_w + cw + aw)


def cleanup_auto_captures(
    session_id: str,
    store: Any,
    *,
    user_id: str = "andrew",
    delete_threshold: float = 0.05,
    demote_threshold: float = 0.18,
    max_rows: int = 500,
) -> CleanupReport:
    """Scan the previous session's auto-captures and clean them up.

    Behaviour:
      - Rows whose content is pure metadata or below ``delete_threshold`` are deleted.
      - Rows below ``demote_threshold`` (but above the delete cutoff) are demoted
        to ``peripheral`` so they age out quickly.
      - Rows where the content can be cleaned (prefixes stripped) are updated.
    """
    report = CleanupReport()
    if not session_id:
        return report

    table = getattr(store, "_table", None)
    if table is None:
        report.error = "store has no _table"
        return report

    try:
        where = f"session_id = '{session_id}' AND user_id = '{user_id}'"
        rows = table.search().where(where, prefilter=True).limit(max_rows).to_list()
    except Exception as e:
        report.error = f"fetch failed: {e}"
        return report

    report.scanned = len(rows)

    for row in rows:
        if not _eligible(row):
            continue

        content = str(row.get("content") or "")
        mid = row.get("id") or ""
        if not mid:
            continue

        # 1) pure-metadata → delete
        if is_pure_metadata(content):
            try:
                table.delete(f"id = '{mid}'")
                report.deleted += 1
                continue
            except Exception as e:  # pragma: no cover
                logger.debug("cleanup delete failed: %s", e)
                continue

        # 2) clean prefixes if helpful
        cleaned = strip_auto_capture_prefix("user", content)
        cleaned_changed = cleaned != content.strip()

        score = _row_value_score(row)

        if score < delete_threshold:
            try:
                table.delete(f"id = '{mid}'")
                report.deleted += 1
                continue
            except Exception as e:  # pragma: no cover
                logger.debug("cleanup delete failed: %s", e)
                continue

        update_values: Dict[str, Any] = {}
        if cleaned_changed and len(cleaned) >= 8:
            update_values["content"] = cleaned[:2000]
            report.cleaned += 1

        if score < demote_threshold and (row.get("tier") or "peripheral") != "peripheral":
            update_values["tier"] = "peripheral"
            report.demoted += 1

        if update_values:
            try:
                table.update(where=f"id = '{mid}'", values=update_values)
            except Exception as e:  # pragma: no cover
                logger.debug("cleanup update failed: %s", e)

    return report


__all__ = [
    "CleanupReport",
    "AUTO_CAPTURE_SOURCES",
    "cleanup_auto_captures",
    "strip_auto_capture_prefix",
    "is_pure_metadata",
]
