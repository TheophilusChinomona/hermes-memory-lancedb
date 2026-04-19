"""LanceDBStore — adapter wrapping the native LanceDB table behind the
:class:`MemoryStore` interface.

This is a thin shim. All the actual LanceDB calls are the same ones the
provider used to make directly via ``self._table.*``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Set

from .base import MemoryStore, StoreUnavailable

logger = logging.getLogger(__name__)


class LanceDBStore(MemoryStore):
    """Adapter exposing a LanceDB table as a :class:`MemoryStore`.

    The provider creates the LanceDB connection + table and hands them to
    this adapter. Schema creation + the schema-migration default values
    are still controlled by the provider (it knows the column → default
    mapping). This adapter only wraps the runtime read/write/lock
    surface area.
    """

    def __init__(
        self,
        *,
        db: Any,
        table: Any,
        storage_path: str,
        table_name: str = "memories",
    ):
        self._db = db
        self._table = table
        self._storage_path = storage_path
        self._table_name = table_name
        self._ready = table is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        # Provider already created the table before instantiating us; nothing to do.
        self._ready = self._table is not None

    def shutdown(self) -> None:
        # LanceDB has no explicit close — drop refs.
        self._table = None
        self._db = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._table is not None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def existing_columns(self) -> Set[str]:
        if self._table is None:
            return set()
        return {f.name for f in self._table.schema}

    def add_columns(self, fields: Dict[str, str]) -> None:
        if self._table is None or not fields:
            return
        self._table.add_columns(fields)

    def ensure_fts_index(self) -> None:
        if self._table is None:
            return
        try:
            self._table.create_fts_index("content", replace=True)
        except Exception as e:
            logger.debug("LanceDB FTS index skipped: %s", e)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def vector_search(
        self,
        vec: List[float],
        *,
        where: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready or not vec:
            return []
        q = self._table.search(vec, vector_column_name="vector")
        if where:
            q = q.where(where, prefilter=True)
        return q.limit(limit).to_list()

    def fts_search(
        self,
        query: str,
        *,
        where: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready or not query:
            return []
        try:
            q = self._table.search(query, query_type="fts")
            if where:
                q = q.where(where, prefilter=True)
            return q.limit(limit).to_list()
        except Exception as e:
            logger.debug("LanceDB FTS search failed: %s", e)
            return []

    def get_by_id(self, mid: str) -> Optional[Dict[str, Any]]:
        if not self.is_ready or not mid:
            return None
        rows = (
            self._table.search()
            .where(_quote(f"id = '{mid}'"), prefilter=True)
            .limit(1)
            .to_list()
        )
        return rows[0] if rows else None

    def list_rows(
        self,
        *,
        where: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready:
            return []
        q = self._table.search()
        if where:
            q = q.where(where, prefilter=True)
        return q.limit(limit).to_list()

    def vector_distance_probe(
        self,
        vec: List[float],
        *,
        where: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_ready or not vec:
            return None
        q = self._table.search(vec, vector_column_name="vector")
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(1).to_list()
        if not rows:
            return None
        row = rows[0]
        # LanceDB stores L2 distance under "_distance" by default.
        dist = float(row.get("_distance", 9.99))
        if dist > threshold:
            return None
        return row

    def count_by(self, column: str, *, where: Optional[str] = None) -> Dict[str, int]:
        if not self.is_ready:
            return {}
        # LanceDB doesn't have GROUP BY in its query API, so we list rows
        # and aggregate in Python. Bounded by the caller's typical use
        # (stats / debug — not the hot path).
        q = self._table.search()
        if where:
            q = q.where(where, prefilter=True)
        # Cap at 5000 to avoid loading the whole table for large stores.
        rows = q.limit(5000).to_list()
        counts: Dict[str, int] = {}
        for r in rows:
            key = str(r.get(column) or "")
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self.is_ready or not rows:
            return
        # LanceDB expects ``tags`` to be a JSON string; the provider already
        # encodes it before calling. The vector field is a list[float]
        # which lance accepts directly.
        self._table.add(rows)

    def update_row(self, mid: str, fields: Dict[str, Any]) -> None:
        if not self.is_ready or not mid or not fields:
            return
        self._table.update(where=f"id = '{mid}'", values=fields)

    def delete_by_id(self, mid: str) -> None:
        if not self.is_ready or not mid:
            return
        self._table.delete(f"id = '{mid}'")

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def with_lock(self, name: str) -> Iterator[None]:
        """File-based lock matching what the provider used pre-refactor.

        The original ``_with_lock`` in ``__init__.py`` used portalocker
        keyed off the table directory path. We do the same.
        """
        try:
            import portalocker
        except ImportError:
            yield
            return
        lock_dir = os.path.join(self._storage_path or ".", ".locks")
        try:
            os.makedirs(lock_dir, exist_ok=True)
        except Exception:
            yield
            return
        lock_path = os.path.join(lock_dir, _safe_name(name) + ".lock")
        with open(lock_path, "w") as fh:
            try:
                portalocker.lock(fh, portalocker.LOCK_EX)
            except Exception as e:
                logger.debug("portalocker acquire failed (%s) — proceeding without lock", e)
                yield
                return
            try:
                yield
            finally:
                try:
                    portalocker.unlock(fh)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "backend": "LanceDBStore",
            "ready": self.is_ready,
            "storage_path": self._storage_path,
            "table": self._table_name,
        }
        if self.is_ready:
            try:
                info["row_count"] = int(self._table.count_rows())
            except Exception:
                pass
        return info


def _quote(predicate: str) -> str:
    """Pass-through; kept for symmetry should LanceDB ever change quoting rules."""
    return predicate


def _safe_name(name: str) -> str:
    """Make a name safe for use as a filename component."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:100]
