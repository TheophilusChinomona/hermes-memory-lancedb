"""Management CLI for ``hermes-memory-lancedb``.

Mirrors the Commander-based ``cli.ts`` from the TS plugin (auth subcommands
intentionally omitted — OAuth is out of scope for the Python port). Built on
``click`` so each subcommand is a plain function that accepts options and
prints to stdout.

Run via either of:

  hermes-memory-lancedb <command> [options]
  python -m hermes_memory_lancedb.cli <command> [options]

Commands:

  version                Print package version.
  list                   List entries (filterable by tier/category).
  search <query>         Hybrid search via the existing pipeline.
  stats                  Counts per tier/category, storage path, table size.
  delete <id>            Delete a single entry.
  delete-bulk            Bulk delete by ids or where-clause.
  export                 Dump all entries as JSON or JSONL.
  import <file>          Load entries from JSON/JSONL.
  import-markdown [glob] Ingest MEMORY.md / memory/YYYY-MM-DD.md files.
  reembed                Re-embed all entries with the current embedder.
  migrate check|run|verify  Schema migration ops.
  reindex-fts            Drop and rebuild the FTS index.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

import click

from . import (
    LanceDBMemoryProvider,
    MEMORY_CATEGORIES,
    RetrievalTrace,
    _TABLE_NAME,
    _with_lock,
)
from .import_md import run_import_markdown

logger = logging.getLogger("hermes_memory_lancedb.cli")

# ---------------------------------------------------------------------------
# Provider plumbing
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """Read the version from pyproject metadata or fallback constant."""
    try:
        from importlib.metadata import version as _v
        return _v("hermes-memory-lancedb")
    except Exception:
        return "1.7.0"


def _make_provider(storage_path: Optional[str], user_id: Optional[str]) -> LanceDBMemoryProvider:
    """Initialise the provider against the requested storage path.

    The CLI takes ``--storage-path`` and ``--user-id`` flags at the top level.
    These set ``LANCEDB_PATH`` and the provider's ``user_id`` for the duration
    of the run.
    """
    if storage_path:
        os.environ["LANCEDB_PATH"] = storage_path
    p = LanceDBMemoryProvider()
    if not p.is_available():
        raise click.ClickException(
            "OPENAI_API_KEY is not set; the embedder cannot start. "
            "Most read-only commands still work — re-run with the env var set "
            "to enable search/import."
        )
    p.initialize("cli", user_id=user_id)
    if not p._ready:
        raise click.ClickException(
            "Provider failed to initialise. Check that lancedb is importable "
            "and that the storage path is writable."
        )
    return p


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _format_entry(row: dict, idx: Optional[int] = None) -> str:
    prefix = f"{idx + 1}. " if idx is not None else ""
    rid = row.get("id", "?")
    tier = row.get("tier", "?")
    cat = row.get("category", "?")
    text = (row.get("content") or row.get("abstract") or "")[:120]
    return f"{prefix}[{rid}] [{tier}/{cat}] {text}"


def _emit_json(obj) -> None:
    click.echo(json.dumps(obj, indent=2, default=str))


def _list_rows(
    provider: LanceDBMemoryProvider,
    *,
    tier: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
) -> List[dict]:
    """Return a list of rows from the underlying table, filtered.

    LanceDB's filter expressions are SQL-ish; we build a `WHERE` from the
    optional tier/category filters.
    """
    if provider.table is None:
        return []
    where = [f"user_id = '{provider._user_id}'"]
    if tier:
        where.append(f"tier = '{tier}'")
    if category:
        where.append(f"category = '{category}'")
    expr = " AND ".join(where)
    try:
        rows = (
            provider.table.search()
            .where(expr, prefilter=True)
            .limit(limit)
            .to_list()
        )
    except Exception as e:
        raise click.ClickException(f"list query failed: {e}")
    return rows


# ---------------------------------------------------------------------------
# Click root
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--storage-path",
    envvar="LANCEDB_PATH",
    default=None,
    help="LanceDB storage directory (default: $LANCEDB_PATH or $HERMES_HOME/lancedb).",
)
@click.option(
    "--user-id",
    default=None,
    help="Memory scoping key (default: 'andrew').",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable INFO logging from the package.",
)
@click.pass_context
def cli(ctx: click.Context, storage_path: Optional[str], user_id: Optional[str], verbose: bool):
    """Manage a LanceDB memory store from the command line."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    ctx.ensure_object(dict)
    ctx.obj["storage_path"] = storage_path
    ctx.obj["user_id"] = user_id


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@cli.command()
def version():
    """Print the package version."""
    click.echo(_get_version())


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command(name="list")
@click.option("--limit", "-n", default=20, type=int, help="Max rows to display.")
@click.option("--tier", default=None, help="Filter by tier (peripheral/working/core).")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_context
def list_cmd(ctx, limit: int, tier: Optional[str], category: Optional[str], as_json: bool):
    """List memory entries with optional tier/category filtering."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    rows = _list_rows(p, tier=tier, category=category, limit=limit)
    if as_json:
        # Strip the (huge) vector field before emitting.
        for r in rows:
            r.pop("vector", None)
        _emit_json(rows)
        return
    if not rows:
        click.echo("No memories found.")
        return
    click.echo(f"Found {len(rows)} memories:")
    for i, r in enumerate(rows):
        click.echo(_format_entry(r, i))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=6, type=int, help="Max results.")
@click.option("--trace", is_flag=True, help="Print pipeline trace.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_context
def search(ctx, query: str, limit: int, trace: bool, as_json: bool):
    """Hybrid search via the existing retrieval pipeline."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    t = RetrievalTrace() if trace else None
    if t is not None:
        # Tag the source so RetrievalStats can break down by caller.
        t._source = "cli"  # type: ignore[attr-defined]
    hits = p._hybrid_search(query, top_k=limit, trace=t)
    if as_json:
        out = {"results": [
            {k: v for k, v in h.items() if k != "vector"} for h in hits
        ]}
        if t is not None:
            out["trace"] = t.to_dict()
        _emit_json(out)
        return
    if not hits:
        click.echo("No relevant memories found.")
    else:
        click.echo(f"Found {len(hits)} memories:")
        for i, h in enumerate(hits):
            click.echo(_format_entry(h, i))
    if t is not None:
        click.echo("")
        click.echo(t.summarize())


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_context
def stats(ctx, as_json: bool):
    """Show counts per tier/category, storage path, table size."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    try:
        rows = (
            p.table.search()
            .where(f"user_id = '{p._user_id}'", prefilter=True)
            .limit(100000)
            .to_list()
        )
    except Exception as e:
        raise click.ClickException(f"stats query failed: {e}")

    tiers: dict = {}
    cats: dict = {}
    for r in rows:
        t = r.get("tier") or "peripheral"
        tiers[t] = tiers.get(t, 0) + 1
        c = r.get("category") or "unknown"
        cats[c] = cats.get(c, 0) + 1

    table_path = Path(p.storage_path) / _TABLE_NAME
    size_bytes = 0
    if table_path.exists():
        for f in table_path.rglob("*"):
            if f.is_file():
                try:
                    size_bytes += f.stat().st_size
                except OSError:
                    pass

    summary = {
        "total": len(rows),
        "tiers": tiers,
        "categories": cats,
        "storage_path": p.storage_path,
        "table_size_bytes": size_bytes,
        "retrieval": p.get_stats(),
    }
    if as_json:
        _emit_json(summary)
        return
    click.echo(f"Total memories: {summary['total']}")
    click.echo(f"Storage path:   {summary['storage_path']}")
    click.echo(f"Table size:     {size_bytes / 1024:.1f} KiB")
    click.echo("")
    click.echo("By tier:")
    for k, v in sorted(tiers.items()):
        click.echo(f"  {k}: {v}")
    click.echo("By category:")
    for k, v in sorted(cats.items()):
        click.echo(f"  {k}: {v}")
    rs = summary["retrieval"]
    if rs["total_queries"]:
        click.echo("")
        click.echo("Retrieval stats (rolling buffer):")
        click.echo(f"  queries:        {rs['total_queries']}")
        click.echo(f"  zero-result:    {rs['zero_result_queries']}")
        click.echo(f"  avg latency ms: {rs['avg_latency_ms']}")
        click.echo(f"  p95 latency ms: {rs['p95_latency_ms']}")
        click.echo(f"  avg results:    {rs['avg_result_count']}")


# ---------------------------------------------------------------------------
# delete + delete-bulk
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("entry_id")
@click.pass_context
def delete(ctx, entry_id: str):
    """Delete one entry by id."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    table_path = os.path.join(p.storage_path, _TABLE_NAME)
    with _with_lock(table_path):
        try:
            p.table.delete(f"id = '{entry_id}'")
        except Exception as e:
            raise click.ClickException(f"delete failed: {e}")
    click.echo(f"Deleted {entry_id}")


@cli.command(name="delete-bulk")
@click.option("--ids", default=None, help="Comma-separated ids to delete.")
@click.option("--filter", "filter_expr", default=None, help="LanceDB WHERE expression, e.g. tier='peripheral'.")
@click.option("--dry-run", is_flag=True, help="Count what would be deleted, don't delete.")
@click.pass_context
def delete_bulk(ctx, ids: Optional[str], filter_expr: Optional[str], dry_run: bool):
    """Bulk-delete entries by ids list or filter expression."""
    if not ids and not filter_expr:
        raise click.UsageError("provide --ids or --filter")
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])

    if ids:
        id_list = [s.strip() for s in ids.split(",") if s.strip()]
        if not id_list:
            raise click.UsageError("--ids was empty after parsing")
        # Build a WHERE expression like id IN ('a','b','c').
        joined = ",".join(f"'{i}'" for i in id_list)
        expr = f"id IN ({joined})"
    else:
        expr = filter_expr

    if dry_run:
        try:
            rows = p.table.search().where(expr, prefilter=True).limit(100000).to_list()
        except Exception as e:
            raise click.ClickException(f"dry-run query failed: {e}")
        click.echo(f"DRY RUN: would delete {len(rows)} entries matching: {expr}")
        return

    table_path = os.path.join(p.storage_path, _TABLE_NAME)
    with _with_lock(table_path):
        try:
            p.table.delete(expr)
        except Exception as e:
            raise click.ClickException(f"bulk delete failed: {e}")
    click.echo(f"Deleted entries matching: {expr}")


# ---------------------------------------------------------------------------
# export + import
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "jsonl"]))
@click.option("--out", "out_path", default=None, help="Write to file (default stdout).")
@click.pass_context
def export(ctx, fmt: str, out_path: Optional[str]):
    """Dump all entries as JSON or JSONL."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    try:
        rows = (
            p.table.search()
            .where(f"user_id = '{p._user_id}'", prefilter=True)
            .limit(100000)
            .to_list()
        )
    except Exception as e:
        raise click.ClickException(f"export query failed: {e}")
    for r in rows:
        # Drop the embedding to keep the dump compact.
        r.pop("vector", None)
        # Distance / score columns sometimes come back as numpy floats.
        for k in ("_distance", "_score"):
            if k in r:
                try:
                    r[k] = float(r[k])
                except Exception:
                    del r[k]

    if fmt == "json":
        text = json.dumps(
            {
                "version": "1.0",
                "exported_at": time.time(),
                "count": len(rows),
                "memories": rows,
            },
            indent=2,
            default=str,
        )
    else:  # jsonl
        text = "\n".join(json.dumps(r, default=str) for r in rows)

    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        click.echo(f"Wrote {len(rows)} entries to {out_path}")
    else:
        click.echo(text)


@cli.command(name="import")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--dry-run", is_flag=True, help="Parse and count without writing.")
@click.pass_context
def import_cmd(ctx, file_path: str, dry_run: bool):
    """Load entries from a JSON or JSONL file."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    text = Path(file_path).read_text(encoding="utf-8")
    entries: List[dict] = []

    # Auto-detect format: try JSON first, fall back to JSONL.
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            entries = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"failed to parse JSON: {e}")
    elif stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"failed to parse JSON: {e}")
        if isinstance(obj.get("memories"), list):
            entries = obj["memories"]
        else:
            entries = [obj]
    else:
        # JSONL.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        click.echo("No entries to import.")
        return

    if dry_run:
        click.echo(f"DRY RUN: would import {len(entries)} entries")
        return

    # Drop vector fields — we'll re-embed via _write_entries.
    for e in entries:
        e.pop("vector", None)
        e.pop("id", None)  # let _write_entries assign fresh ids

    p._write_entries(entries)
    click.echo(f"Imported {len(entries)} entries")


# ---------------------------------------------------------------------------
# import-markdown
# ---------------------------------------------------------------------------


@cli.command(name="import-markdown")
@click.argument("globs", nargs=-1)
@click.option("--dry-run", is_flag=True, help="Parse and count without writing.")
@click.option("--base-dir", default=None, type=click.Path(exists=True, file_okay=False), help="Base for relative globs.")
@click.option("--min-text-length", default=5, type=int, help="Skip bullets shorter than this.")
@click.option("--importance", default=0.7, type=float, help="Importance assigned to imports.")
@click.pass_context
def import_markdown(
    ctx,
    globs: tuple,
    dry_run: bool,
    base_dir: Optional[str],
    min_text_length: int,
    importance: float,
):
    """Ingest MEMORY.md / memory/YYYY-MM-DD.md files."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    base = Path(base_dir) if base_dir else None
    pats = list(globs) if globs else None
    result = run_import_markdown(
        write_fn=p._write_entries,
        patterns=pats,
        base_dir=base,
        dry_run=dry_run,
        min_text_length=min_text_length,
        importance=importance,
        on_message=click.echo,
    )
    click.echo(
        f"\nimport-markdown: {result['imported']} imported, "
        f"{result['skipped']} skipped (scanned {result['found_files']} files)"
        + (" [dry-run]" if dry_run else "")
    )


# ---------------------------------------------------------------------------
# reembed
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dry-run", is_flag=True, help="Count rows without re-embedding.")
@click.option("--target", default=None, type=click.Path(file_okay=False), help="Write re-embedded data to a different LanceDB path (A/B testing).")
@click.option("--batch-size", default=32, type=int, help="Rows per embedder batch.")
@click.option("--limit", default=None, type=int, help="Max rows to process.")
@click.pass_context
def reembed(ctx, dry_run: bool, target: Optional[str], batch_size: int, limit: Optional[int]):
    """Re-embed all entries with the current embedder.

    Useful after upgrading to a new embedding model, or to A/B compare
    retrieval quality on a parallel database via ``--target``.
    """
    src = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    try:
        rows = (
            src.table.search()
            .where(f"user_id = '{src._user_id}'", prefilter=True)
            .limit(limit or 100000)
            .to_list()
        )
    except Exception as e:
        raise click.ClickException(f"source query failed: {e}")

    rows = [r for r in rows if r.get("content")]
    click.echo(f"Found {len(rows)} entries to re-embed")
    if dry_run:
        click.echo("DRY RUN: no writes")
        return

    if target:
        # Refuse to clobber the source.
        if Path(target).resolve() == Path(src.storage_path).resolve():
            raise click.UsageError("--target equals source storage path; refusing")
        dst = _make_provider(target, ctx.obj["user_id"])
    else:
        dst = src

    embedder = src._embedder
    if embedder is None:
        raise click.ClickException("source embedder is not initialised")

    table_path = os.path.join(dst.storage_path, _TABLE_NAME)
    written = 0
    batch: List[dict] = []
    for r in rows:
        text = r.get("content", "")
        new_vec = embedder.embed(text)
        batch.append({
            "id": str(r.get("id") or uuid.uuid4()),
            "content": text,
            "vector": new_vec,
            "timestamp": float(r.get("timestamp") or time.time()),
            "source": r.get("source", "reembed"),
            "session_id": r.get("session_id", ""),
            "user_id": r.get("user_id", dst._user_id),
            "tags": r.get("tags", "[]") if isinstance(r.get("tags"), str) else json.dumps(r.get("tags", [])),
            "tier": r.get("tier", "peripheral"),
            "importance": float(r.get("importance") or 0.5),
            "access_count": int(r.get("access_count") or 0),
            "category": r.get("category", "general"),
            "abstract": r.get("abstract", ""),
            "overview": r.get("overview", ""),
        })
        if len(batch) >= batch_size:
            with _with_lock(table_path):
                dst.table.add(batch)
            written += len(batch)
            batch = []
            click.echo(f"  reembedded {written}/{len(rows)}")
    if batch:
        with _with_lock(table_path):
            dst.table.add(batch)
        written += len(batch)
    click.echo(f"reembed complete: {written} entries written to {dst.storage_path}")


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


@cli.group()
def migrate():
    """Schema migration ops (check/run/verify)."""


@migrate.command("check")
@click.pass_context
def migrate_check(ctx):
    """Report which schema columns are missing on the current table."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    if p.table is None:
        raise click.ClickException("table is not open")
    existing = {f.name for f in p.table.schema}
    expected = {"id", "content", "vector", "timestamp", "source", "session_id",
                "user_id", "tags", "tier", "importance", "access_count",
                "category", "abstract", "overview"}
    missing = sorted(expected - existing)
    extra = sorted(existing - expected)
    click.echo(f"Existing columns: {sorted(existing)}")
    if missing:
        click.echo(f"MISSING: {missing}  (run `migrate run` to add)")
    else:
        click.echo("Schema is up to date.")
    if extra:
        click.echo(f"Extra columns:    {extra}")


@migrate.command("run")
@click.pass_context
def migrate_run(ctx):
    """Add any missing schema columns to the current table."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    p._migrate_schema_if_needed()
    click.echo("Migration complete.")


@migrate.command("verify")
@click.pass_context
def migrate_verify(ctx):
    """Verify that all expected schema columns exist."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    existing = {f.name for f in p.table.schema}
    expected = {"tier", "importance", "access_count", "category", "abstract", "overview"}
    missing = sorted(expected - existing)
    if missing:
        raise click.ClickException(f"Schema verification failed; missing: {missing}")
    click.echo("Schema OK — all v1.1.0+ columns present.")


# ---------------------------------------------------------------------------
# reindex-fts
# ---------------------------------------------------------------------------


@cli.command(name="reindex-fts")
@click.pass_context
def reindex_fts(ctx):
    """Drop and rebuild the BM25 FTS index on the ``content`` column."""
    p = _make_provider(ctx.obj["storage_path"], ctx.obj["user_id"])
    table_path = os.path.join(p.storage_path, _TABLE_NAME)
    with _with_lock(table_path):
        try:
            p.table.create_fts_index("content", replace=True)
        except Exception as e:
            raise click.ClickException(f"FTS rebuild failed: {e}")
    click.echo("FTS index rebuilt.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None):
    """Entry point referenced by the ``[project.scripts]`` block."""
    cli(args=list(argv) if argv is not None else None, prog_name="hermes-memory-lancedb")


if __name__ == "__main__":  # pragma: no cover
    main()
