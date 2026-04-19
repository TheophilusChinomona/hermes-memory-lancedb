"""MemoryStore ABC — persistence layer interface for memory backends.

Implementations:
  * `LanceDBStore` (in `lancedb_store.py`) — wraps the native LanceDB table
  * `PgvectorStore` (in `pgvector.py`) — Postgres + pgvector + pg_search

The provider (`LanceDBMemoryProvider`) calls these methods instead of
touching `self._table` directly. Pipeline code (rerank, MMR, etc.) is
unaffected — it operates on dict lists returned from these methods.

Method contract:
  * Search returns list of dicts with the columns enumerated below
    (`vector` is included so the pipeline can do MMR + cosine fallback).
  * `where` predicates are SQL fragments built by `ScopeManager` —
    same shape used by both backends.
"""

from __future__ import annotations

import abc
import contextlib
from typing import Any, Dict, Iterator, List, Optional, Set


class StoreUnavailable(RuntimeError):
    """Raised when a backend is selected but its driver/server isn't reachable."""


class MemoryStore(abc.ABC):
    """Persistence interface that backends implement.

    Lifecycle: ``initialize()`` once, then any sequence of read/write
    methods, then ``shutdown()``.

    Row shape (returned by ``vector_search`` / ``fts_search`` / ``get_by_id``
    / ``list_rows``)::

        {
            "id": str,
            "content": str,
            "abstract": str,
            "overview": str,
            "vector": list[float] | None,
            "timestamp": float,
            "source": str,
            "session_id": str,
            "user_id": str,
            "tags": str,         # JSON-encoded list
            "tier": str,
            "importance": float,
            "access_count": int,
            "category": str,
            "metadata": str,     # JSON-encoded dict
            "parent_id": str,
            "temporal_type": str,
            # Multi-scope columns (P1) — present only after migration
            "agent_id": str,
            "project_id": str,
            "team_id": str,
            "workspace_id": str,
            "scope": str,
        }

    Backends MAY return additional fields; the provider only reads the
    keys it needs.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self) -> None:
        """Open connections, ensure schema + indexes exist, run migrations.

        Must be idempotent — safe to call against an already-initialized
        store.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Close connections / pools. Safe to call repeatedly."""

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True iff initialize() succeeded and the store is usable."""

    # ------------------------------------------------------------------
    # Schema introspection + migration
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def existing_columns(self) -> Set[str]:
        """Return the set of column names currently present on the row table."""

    @abc.abstractmethod
    def add_columns(self, fields: Dict[str, str]) -> None:
        """Add columns with default-value SQL expressions (e.g. ``"''"``,
        ``"0"``, ``"'static'"``).  Backends translate these to their dialect.
        """

    @abc.abstractmethod
    def ensure_fts_index(self) -> None:
        """Create or refresh the BM25 / FTS index on the ``content`` column.
        Idempotent.
        """

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def vector_search(
        self,
        vec: List[float],
        *,
        where: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Cosine vector ANN over the ``vector`` column."""

    @abc.abstractmethod
    def fts_search(
        self,
        query: str,
        *,
        where: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """BM25 (or TF-IDF) full-text search over the ``content`` column."""

    @abc.abstractmethod
    def get_by_id(self, mid: str) -> Optional[Dict[str, Any]]:
        """Fetch one row by id, or ``None`` if missing."""

    @abc.abstractmethod
    def list_rows(
        self,
        *,
        where: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List rows matching ``where`` (no scoring). Used for stats / debug."""

    @abc.abstractmethod
    def vector_distance_probe(
        self,
        vec: List[float],
        *,
        where: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Single-NN probe with a max-distance cutoff — used by the dedup
        path to find a near-duplicate before writing.

        Returns the closest row if its L2 distance is below ``threshold``,
        else ``None``.
        """

    @abc.abstractmethod
    def count_by(self, column: str, *, where: Optional[str] = None) -> Dict[str, int]:
        """``GROUP BY column`` count map. Used by ``lancedb_stats``."""

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Batch INSERT. Rows already include all columns the schema expects."""

    @abc.abstractmethod
    def update_row(self, mid: str, fields: Dict[str, Any]) -> None:
        """Partial update of one row by id."""

    @abc.abstractmethod
    def delete_by_id(self, mid: str) -> None:
        """Remove one row by id (no-op if missing)."""

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def with_lock(self, name: str) -> "contextlib.AbstractContextManager[Any]":
        """Cross-process lock around a critical section.

        Default: no-op context manager (single-process backends like
        in-memory mocks). LanceDB uses a portalocker file lock; pgvector
        uses Postgres advisory locks.
        """
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Return a small dict of human-friendly state for stats output.

        Default just returns the backend name + readiness; override to
        include row counts, table size, index health, etc.
        """
        return {"backend": type(self).__name__, "ready": self.is_ready}
