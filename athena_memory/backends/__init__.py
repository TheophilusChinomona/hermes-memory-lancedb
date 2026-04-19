"""Pluggable storage backends for hermes-memory-lancedb.

The original `_table` everywhere in the provider is a LanceDB-specific
handle. This package abstracts the persistence layer so the same
`LanceDBMemoryProvider` can run against:

  * LanceDB native (default, requires AVX2 in the host CPU)
  * Postgres + pgvector + pg_search (ParadeDB image — runs on any CPU)

The pipeline (RRF fusion, MMR, length-norm, hard min-score, cross-encoder
rerank) is backend-agnostic and operates on lists of dicts. Only the
storage I/O lives behind this interface.

Selection happens via env:

  - `HERMES_MEMORY_BACKEND=lancedb|pgvector` (explicit override)
  - Auto-detect: if `HERMES_MEMORY_DATABASE_URL` is set and no override,
    use pgvector.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import MemoryStore, StoreUnavailable

logger = logging.getLogger(__name__)


def detect_backend() -> str:
    """Return the active backend name ("lancedb" or "pgvector").

    Resolution:
      1. ``HERMES_MEMORY_BACKEND`` env var (explicit override)
      2. Auto-detect: if ``HERMES_MEMORY_DATABASE_URL`` is set → pgvector
      3. Default → lancedb
    """
    explicit = (os.environ.get("HERMES_MEMORY_BACKEND") or "").strip().lower()
    if explicit in ("lancedb", "pgvector"):
        return explicit
    if explicit:
        logger.warning(
            "Unknown HERMES_MEMORY_BACKEND=%r — falling back to lancedb",
            explicit,
        )
    if os.environ.get("HERMES_MEMORY_DATABASE_URL"):
        return "pgvector"
    return "lancedb"


def make_pgvector_store(
    database_url: Optional[str] = None,
    *,
    embedding_dim: int = 1536,
) -> "MemoryStore":
    """Construct a PgvectorStore from env / args (without initializing it)."""
    from .pgvector import PgvectorStore  # local import keeps psycopg optional

    url = database_url or os.environ.get("HERMES_MEMORY_DATABASE_URL", "")
    if not url:
        raise StoreUnavailable(
            "pgvector backend selected but HERMES_MEMORY_DATABASE_URL is not set"
        )
    return PgvectorStore(url, embedding_dim=embedding_dim)


def make_lancedb_store(
    *, db: Any, table: Any, storage_path: str, table_name: str = "memories"
) -> "MemoryStore":
    """Construct a LanceDBStore wrapping an already-opened table."""
    from .lancedb_store import LanceDBStore

    return LanceDBStore(
        db=db, table=table, storage_path=storage_path, table_name=table_name
    )


__all__ = [
    "MemoryStore",
    "StoreUnavailable",
    "detect_backend",
    "make_lancedb_store",
    "make_pgvector_store",
]
