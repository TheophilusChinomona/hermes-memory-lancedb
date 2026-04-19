"""Postgres + pgvector + pg_search backend.

Designed for the ParadeDB image (`paradedb/paradedb`) which ships
both the ``vector`` and ``pg_search`` extensions out of the box. Falls
back to ``tsvector`` / ``ts_rank_cd`` if ``pg_search`` isn't available.

Connection: pass ``HERMES_MEMORY_DATABASE_URL`` (URL-encoded password).

Schema:

  ``hermes_memory.memories``
    id TEXT PRIMARY KEY
    content TEXT NOT NULL
    abstract TEXT
    overview TEXT
    vector VECTOR(<dim>)        -- pgvector
    timestamp DOUBLE PRECISION
    source TEXT
    session_id TEXT
    user_id TEXT
    tags TEXT                   -- JSON
    tier TEXT
    importance REAL
    access_count INTEGER
    category TEXT
    metadata TEXT               -- JSON
    parent_id TEXT
    temporal_type TEXT
    -- multi-scope (P1)
    agent_id TEXT
    project_id TEXT
    team_id TEXT
    workspace_id TEXT
    scope TEXT

Indexes:
    HNSW on vector  (vector_cosine_ops)
    BM25 on content (pg_search) OR GIN on to_tsvector(content)
    btree on user_id, scope, tier
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Set

from .base import MemoryStore, StoreUnavailable

logger = logging.getLogger(__name__)


_SCHEMA = "hermes_memory"
_TABLE = "memories"

# All columns the provider expects on a row dict, in order. Used for
# INSERT VALUES and SELECT ... AS row construction.
_ALL_COLUMNS = (
    "id",
    "content",
    "abstract",
    "overview",
    "vector",
    "timestamp",
    "source",
    "session_id",
    "user_id",
    "tags",
    "tier",
    "importance",
    "access_count",
    "category",
    "metadata",
    "parent_id",
    "temporal_type",
    "agent_id",
    "project_id",
    "team_id",
    "workspace_id",
    "scope",
)

# Default SQL for creating each column when migrating an existing legacy
# table (mirrors LanceDB's ``add_columns`` defaults the provider passes in).
_LEGACY_DEFAULTS = {
    "abstract": "''",
    "overview": "''",
    "tier": "'peripheral'",
    "importance": "0.5",
    "access_count": "0",
    "category": "'general'",
    "metadata": "''",
    "parent_id": "''",
    "temporal_type": "'static'",
    "agent_id": "''",
    "project_id": "''",
    "team_id": "''",
    "workspace_id": "''",
    "scope": "'global'",
}


def _vector_to_pg(vec: List[float]) -> str:
    """pgvector accepts ``[x, y, z]`` text literal."""
    return "[" + ",".join(f"{float(v):.7f}" for v in vec) + "]"


def _pg_to_vector(value: Any) -> Optional[List[float]]:
    """Parse pgvector return value (``str`` like ``[1,2,3]`` or list-like)."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [float(x) for x in inner.split(",")]
    # psycopg may return a memoryview or bytes
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _parse_legacy_default(default_sql: str) -> Any:
    """Translate the SQL-fragment defaults the provider passes
    (``"''"``, ``"0"``, ``"'static'"``) into Python values.
    """
    s = default_sql.strip()
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s in ("0", "0.0"):
        return 0
    try:
        return float(s)
    except ValueError:
        return s


class PgvectorStore(MemoryStore):
    """Postgres + pgvector backend.

    Uses ``pg_search`` for BM25 if the extension is present; otherwise
    falls back to ``tsvector`` / ``ts_rank_cd``.
    """

    def __init__(
        self,
        database_url: str,
        *,
        embedding_dim: int = 1536,
        schema: str = _SCHEMA,
        table: str = _TABLE,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
    ):
        if not database_url:
            raise StoreUnavailable("PgvectorStore: HERMES_MEMORY_DATABASE_URL is required")
        self._database_url = database_url
        self._dim = int(embedding_dim)
        self._schema = schema
        self._table = table
        self._qual = f'"{schema}"."{table}"'
        self._pool: Any = None
        self._lock = threading.Lock()
        self._ready = False
        self._has_pg_search = False
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise StoreUnavailable(
                "PgvectorStore requires psycopg[binary,pool]>=3.2 — install via "
                "`pip install \"psycopg[binary,pool]>=3.2\"`"
            ) from exc

        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    self._database_url,
                    min_size=self._pool_min,
                    max_size=self._pool_max,
                    kwargs={"autocommit": True},
                    open=False,
                )
                self._pool.open(wait=True, timeout=15.0)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
                        self._has_pg_search = True
                    except Exception as e:
                        logger.info(
                            "pg_search extension unavailable (%s) — falling back to tsvector FTS",
                            e,
                        )
                        self._has_pg_search = False

                    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
                    cur.execute(self._create_table_sql())

                    # Vector ANN index (HNSW, cosine).
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_vector_hnsw "
                        f"ON {self._qual} USING hnsw (vector vector_cosine_ops) "
                        "WITH (m = 16, ef_construction = 64);"
                    )
                    # Useful btree indexes for the WHERE clauses ScopeManager builds.
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_user_id_idx "
                        f"ON {self._qual} (user_id);"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_scope_idx "
                        f"ON {self._qual} (scope);"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_tier_idx "
                        f"ON {self._qual} (tier);"
                    )

                    self._ensure_fts_index_locked(cur)
            self._ready = True
            logger.info(
                "PgvectorStore initialized at %s (dim=%d, schema=%s, pg_search=%s)",
                _redact_url(self._database_url),
                self._dim,
                self._schema,
                self._has_pg_search,
            )

    def shutdown(self) -> None:
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Pool close failed: %s", e)
                self._pool = None
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._pool is not None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_table_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self._qual} (
            id             TEXT PRIMARY KEY,
            content        TEXT NOT NULL,
            abstract       TEXT NOT NULL DEFAULT '',
            overview       TEXT NOT NULL DEFAULT '',
            vector         vector({self._dim}),
            timestamp      DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
            source         TEXT NOT NULL DEFAULT '',
            session_id     TEXT NOT NULL DEFAULT '',
            user_id        TEXT NOT NULL DEFAULT '',
            tags           TEXT NOT NULL DEFAULT '[]',
            tier           TEXT NOT NULL DEFAULT 'peripheral',
            importance     REAL NOT NULL DEFAULT 0.5,
            access_count   INTEGER NOT NULL DEFAULT 0,
            category       TEXT NOT NULL DEFAULT 'general',
            metadata       TEXT NOT NULL DEFAULT '',
            parent_id      TEXT NOT NULL DEFAULT '',
            temporal_type  TEXT NOT NULL DEFAULT 'static',
            agent_id       TEXT NOT NULL DEFAULT '',
            project_id     TEXT NOT NULL DEFAULT '',
            team_id        TEXT NOT NULL DEFAULT '',
            workspace_id   TEXT NOT NULL DEFAULT '',
            scope          TEXT NOT NULL DEFAULT 'global'
        );
        """

    def existing_columns(self) -> Set[str]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s;",
                    (self._schema, self._table),
                )
                return {row[0] for row in cur.fetchall()}

    def add_columns(self, fields: Dict[str, str]) -> None:
        if not fields:
            return
        existing = self.existing_columns()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for name, default_sql in fields.items():
                    if name in existing:
                        continue
                    pg_type = _LEGACY_PG_TYPES.get(name, "TEXT")
                    cur.execute(
                        f'ALTER TABLE {self._qual} ADD COLUMN IF NOT EXISTS '
                        f'"{name}" {pg_type} NOT NULL DEFAULT {default_sql};'
                    )
                    logger.info("PgvectorStore: added column %s %s default %s", name, pg_type, default_sql)

    def ensure_fts_index(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_fts_index_locked(cur)

    def _ensure_fts_index_locked(self, cur: Any) -> None:
        """Create the BM25 (pg_search) or tsvector index. Idempotent."""
        if self._has_pg_search:
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_bm25 "
                    f"ON {self._qual} USING bm25 (id, content) "
                    "WITH (key_field = 'id');"
                )
                return
            except Exception as e:
                logger.warning(
                    "pg_search BM25 index creation failed (%s) — falling back to tsvector",
                    e,
                )
                self._has_pg_search = False

        # Fallback: tsvector + GIN. Uses a generated column so writes don't
        # need to compute it client-side.
        cur.execute(
            f"ALTER TABLE {self._qual} "
            f"ADD COLUMN IF NOT EXISTS content_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_content_tsv "
            f"ON {self._qual} USING GIN (content_tsv);"
        )

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
        where_sql = f"WHERE {where}" if where else ""
        sql = (
            f"SELECT {_select_columns()} "
            f"FROM {self._qual} {where_sql} "
            "ORDER BY vector <=> %s::vector "
            f"LIMIT {int(limit)};"
        )
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (_vector_to_pg(vec),))
                return [_row_to_dict(r) for r in cur.fetchall()]

    def fts_search(
        self,
        query: str,
        *,
        where: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready or not query:
            return []
        where_extra = f" AND {where}" if where else ""

        if self._has_pg_search:
            # pg_search BM25.  ``@@@`` is the BM25 match operator; ``score`` is
            # exposed for ORDER BY.
            sql = (
                f"SELECT {_select_columns()} "
                f"FROM {self._qual} "
                f"WHERE id @@@ paradedb.match('content', %s){where_extra} "
                "ORDER BY paradedb.score(id) DESC "
                f"LIMIT {int(limit)};"
            )
            params: tuple = (query,)
        else:
            # tsvector fallback — uses websearch syntax for friendly queries.
            sql = (
                f"SELECT {_select_columns()} "
                f"FROM {self._qual} "
                "WHERE content_tsv @@ websearch_to_tsquery('english', %s)"
                f"{where_extra} "
                "ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) DESC "
                f"LIMIT {int(limit)};"
            )
            params = (query, query)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, params)
                    return [_row_to_dict(r) for r in cur.fetchall()]
                except Exception as e:
                    logger.debug("PgvectorStore fts_search failed: %s", e)
                    return []

    def get_by_id(self, mid: str) -> Optional[Dict[str, Any]]:
        if not self.is_ready or not mid:
            return None
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_select_columns()} FROM {self._qual} WHERE id = %s LIMIT 1;",
                    (mid,),
                )
                row = cur.fetchone()
                return _row_to_dict(row) if row else None

    def list_rows(
        self,
        *,
        where: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready:
            return []
        where_sql = f"WHERE {where}" if where else ""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_select_columns()} FROM {self._qual} {where_sql} "
                    f"LIMIT {int(limit)};"
                )
                return [_row_to_dict(r) for r in cur.fetchall()]

    def vector_distance_probe(
        self,
        vec: List[float],
        *,
        where: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_ready or not vec:
            return None
        where_sql = f"WHERE {where}" if where else ""
        # ``<->`` is L2 distance in pgvector — matches the LanceDB behaviour
        # the existing _find_similar relies on.
        sql = (
            f"SELECT {_select_columns()}, vector <-> %s::vector AS _dist "
            f"FROM {self._qual} {where_sql} "
            "ORDER BY vector <-> %s::vector "
            "LIMIT 1;"
        )
        v = _vector_to_pg(vec)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (v, v))
                row = cur.fetchone()
                if not row:
                    return None
                dist = float(row[-1])
                if dist > threshold:
                    return None
                d = _row_to_dict(row[:-1])
                d["_distance"] = dist
                return d

    def count_by(self, column: str, *, where: Optional[str] = None) -> Dict[str, int]:
        if not self.is_ready:
            return {}
        if column not in _ALL_COLUMNS:
            raise ValueError(f"unknown column for count_by: {column}")
        where_sql = f"WHERE {where}" if where else ""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT "{column}", COUNT(*) FROM {self._qual} {where_sql} '
                    f'GROUP BY "{column}";'
                )
                return {str(k): int(c) for (k, c) in cur.fetchall()}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self.is_ready or not rows:
            return
        cols, placeholders, values = self._build_insert(rows)
        sql = (
            f"INSERT INTO {self._qual} ({cols}) VALUES {placeholders} "
            "ON CONFLICT (id) DO NOTHING;"
        )
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)

    def update_row(self, mid: str, fields: Dict[str, Any]) -> None:
        if not self.is_ready or not mid or not fields:
            return
        set_parts = []
        values: List[Any] = []
        for col, val in fields.items():
            if col not in _ALL_COLUMNS:
                continue
            if col == "vector" and isinstance(val, list):
                set_parts.append(f'"{col}" = %s::vector')
                values.append(_vector_to_pg(val))
            else:
                set_parts.append(f'"{col}" = %s')
                values.append(val)
        if not set_parts:
            return
        values.append(mid)
        sql = f"UPDATE {self._qual} SET {', '.join(set_parts)} WHERE id = %s;"
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)

    def delete_by_id(self, mid: str) -> None:
        if not self.is_ready or not mid:
            return
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._qual} WHERE id = %s;", (mid,))

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def with_lock(self, name: str) -> Iterator[None]:
        """Postgres advisory lock around a critical section.

        Hashes the lock name to a bigint key (pg_advisory_lock takes
        a single bigint). Lock is session-scoped, released on exit.
        """
        if not self.is_ready:
            yield
            return
        key = _hash_to_bigint(name)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(%s);", (key,))
                try:
                    yield
                finally:
                    try:
                        cur.execute("SELECT pg_advisory_unlock(%s);", (key,))
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("advisory unlock failed: %s", e)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "backend": "PgvectorStore",
            "ready": self.is_ready,
            "schema": self._schema,
            "table": self._table,
            "embedding_dim": self._dim,
            "fts_engine": "pg_search" if self._has_pg_search else "tsvector",
        }
        if self.is_ready:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT COUNT(*) FROM {self._qual};")
                        info["row_count"] = int(cur.fetchone()[0])
            except Exception as e:
                info["row_count_error"] = str(e)
        return info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_insert(self, rows: List[Dict[str, Any]]) -> tuple:
        """Build the (cols, placeholders, values) tuple for a batch INSERT.

        Uses %s for scalars and %s::vector cast for the vector column.
        """
        cols_quoted = ", ".join(f'"{c}"' for c in _ALL_COLUMNS)
        per_row_placeholders: List[str] = []
        values: List[Any] = []
        for row in rows:
            placeholder_parts: List[str] = []
            for col in _ALL_COLUMNS:
                v = row.get(col)
                if col == "vector":
                    if v is None:
                        placeholder_parts.append("NULL")
                        continue
                    placeholder_parts.append("%s::vector")
                    values.append(_vector_to_pg(v))
                elif col == "tags":
                    if isinstance(v, list):
                        placeholder_parts.append("%s")
                        values.append(json.dumps(v))
                    elif v is None:
                        placeholder_parts.append("'[]'")
                    else:
                        placeholder_parts.append("%s")
                        values.append(str(v))
                else:
                    placeholder_parts.append("%s")
                    # Coerce None → schema default ('' for text, 0 for int).
                    values.append(_default_for(col) if v is None else v)
            per_row_placeholders.append("(" + ", ".join(placeholder_parts) + ")")
        return cols_quoted, ", ".join(per_row_placeholders), values


_LEGACY_PG_TYPES = {
    "abstract": "TEXT",
    "overview": "TEXT",
    "tier": "TEXT",
    "importance": "REAL",
    "access_count": "INTEGER",
    "category": "TEXT",
    "metadata": "TEXT",
    "parent_id": "TEXT",
    "temporal_type": "TEXT",
    "agent_id": "TEXT",
    "project_id": "TEXT",
    "team_id": "TEXT",
    "workspace_id": "TEXT",
    "scope": "TEXT",
}

_INT_COLUMNS = {"access_count"}
_FLOAT_COLUMNS = {"timestamp", "importance"}


def _default_for(col: str) -> Any:
    if col in _INT_COLUMNS:
        return 0
    if col in _FLOAT_COLUMNS:
        return 0.0
    return ""


def _select_columns() -> str:
    """SELECT clause matching the order in _ALL_COLUMNS."""
    return ", ".join(f'"{c}"' for c in _ALL_COLUMNS)


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Map a SELECT result tuple back to a row dict the provider expects."""
    out: Dict[str, Any] = {}
    for i, col in enumerate(_ALL_COLUMNS):
        v = row[i]
        if col == "vector":
            v = _pg_to_vector(v)
        out[col] = v
    return out


def _hash_to_bigint(name: str) -> int:
    """Map an arbitrary name to a 64-bit signed int for pg_advisory_lock."""
    h = hashlib.sha256(name.encode("utf-8")).digest()
    # First 8 bytes, signed.
    val = int.from_bytes(h[:8], "big", signed=True)
    return val


def _redact_url(url: str) -> str:
    """Strip the password from a URL for safe logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return f"{scheme}://{creds}@{host}"
