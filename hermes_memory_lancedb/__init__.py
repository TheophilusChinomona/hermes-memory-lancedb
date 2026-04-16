"""hermes-memory-lancedb — LanceDB vector memory plugin for Hermes agents.

Local-first hybrid recall: vector ANN + BM25 (tantivy) merged via RRF,
Weibull decay, OpenAI text-embedding-3-small, auto-capture + auto-recall.

Ported from Theo's memory-lancedb-pro/hermes-adapt TypeScript fork.

Storage:    $LANCEDB_PATH  or  $HERMES_HOME/lancedb/
Embeddings: OpenAI text-embedding-3-small — OPENAI_API_KEY required

Install:
    pip install hermes-memory-lancedb

Activate in ~/.hermes/config.yaml:
    memory:
      provider: lancedb
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MemoryProvider base — import from Hermes if available, else use bundled ABC
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider as _MemoryProviderBase
except ImportError:
    # Running outside the Hermes venv (tests, standalone use)
    from abc import ABC, abstractmethod

    class _MemoryProviderBase(ABC):  # type: ignore[no-redef]
        @property
        @abstractmethod
        def name(self) -> str: ...
        @abstractmethod
        def is_available(self) -> bool: ...
        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...
        @abstractmethod
        def get_tool_schemas(self) -> List[Dict[str, Any]]: ...
        def system_prompt_block(self) -> str: return ""
        def prefetch(self, query: str, *, session_id: str = "") -> str: return ""
        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: pass
        def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None: pass
        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
            raise NotImplementedError
        def shutdown(self) -> None: pass
        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None: pass
        def on_session_end(self, messages: List[Dict[str, Any]]) -> None: pass
        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str: return ""
        def on_memory_write(self, action: str, target: str, content: str) -> None: pass
        def get_config_schema(self) -> List[Dict[str, Any]]: return []
        def save_config(self, values: Dict[str, Any], hermes_home: str) -> None: pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TABLE_NAME = "memories"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536
_TOP_K_VECTOR = 20
_TOP_K_BM25 = 20
_TOP_K_RETURN = 6
_RRF_K = 60

_WEIBULL_SCALE = 30.0   # characteristic decay time in days
_WEIBULL_SHAPE = 0.7    # < 1 = heavy tail (older memories fade slowly)
_DECAY_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _get_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("timestamp", pa.float64()),
        pa.field("source", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("tags", pa.string()),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weibull_weight(age_days: float) -> float:
    if age_days < 0:
        age_days = 0.0
    return math.exp(-((age_days / _WEIBULL_SCALE) ** _WEIBULL_SHAPE))


def _age_days(timestamp: float) -> float:
    return max(0.0, (time.time() - timestamp) / 86400.0)


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def _merge_rrf(vector_hits: List[Dict], bm25_hits: List[Dict], top_k: int) -> List[Dict]:
    scores: Dict[str, float] = {}
    by_id: Dict[str, Dict] = {}
    for rank, hit in enumerate(vector_hits):
        mid = hit["id"]
        scores[mid] = scores.get(mid, 0.0) + _rrf_score(rank)
        by_id[mid] = hit
    for rank, hit in enumerate(bm25_hits):
        mid = hit["id"]
        scores[mid] = scores.get(mid, 0.0) + _rrf_score(rank)
        by_id.setdefault(mid, hit)
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [by_id[mid] for mid in ranked[:top_k]]


def _msg_text(msg: Dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class _EmbedClient:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def embed(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        vec = self._client.embeddings.create(model=_EMBED_MODEL, input=text[:8000]).data[0].embedding
        with self._lock:
            if len(self._cache) > 2000:
                for k in list(self._cache.keys())[:200]:
                    del self._cache[k]
            self._cache[key] = vec
        return vec


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

def _extract_facts_from_turn(user: str, assistant: str) -> List[str]:
    combined = f"User: {user}\nAssistant: {assistant}"
    return [combined[:1200]]


def _extract_facts_from_session(messages: List[Dict]) -> List[str]:
    full = "\n".join(
        f"{m.get('role','?').capitalize()}: {_msg_text(m)}"
        for m in messages if m.get("role") in ("user", "assistant")
    )
    return [full[:4000]] if full.strip() else []


# ---------------------------------------------------------------------------
# LanceDBMemoryProvider
# ---------------------------------------------------------------------------

class LanceDBMemoryProvider(_MemoryProviderBase):
    """
    LanceDB vector memory for Hermes agents.

    Hybrid BM25 + vector recall, Weibull decay, auto-capture, auto-recall.
    Storage: $LANCEDB_PATH or $HERMES_HOME/lancedb/
    Embeddings: OpenAI text-embedding-3-small (OPENAI_API_KEY required)
    """

    def __init__(self):
        self._db = None
        self._table = None
        self._embedder: Optional[_EmbedClient] = None
        self._user_id = "andrew"
        self._session_id = ""
        self._storage_path = ""
        self._ready = False
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_queue: List[Dict] = []
        self._sync_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "lancedb"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "storage_path", "description": "Directory for LanceDB data files", "default": "~/.hermes/lancedb"},
            {"key": "user_id", "description": "User identifier for memory scoping", "default": "andrew"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "lancedb.json"
        existing: Dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def _load_config(self) -> Dict:
        try:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        except ImportError:
            from pathlib import Path
            hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

        config: Dict = {
            "storage_path": os.environ.get("LANCEDB_PATH", str(hermes_home / "lancedb")),
            "user_id": "andrew",
        }
        config_path = hermes_home / "lancedb.json"
        if config_path.exists():
            try:
                config.update({k: v for k, v in json.loads(config_path.read_text()).items() if v})
            except Exception:
                pass
        return config

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        cfg = self._load_config()
        self._storage_path = cfg["storage_path"]
        self._user_id = kwargs.get("user_id") or cfg.get("user_id", "andrew")

        try:
            import lancedb

            os.makedirs(self._storage_path, exist_ok=True)
            self._db = lancedb.connect(self._storage_path)

            if _TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(_TABLE_NAME)
            else:
                self._table = self._db.create_table(_TABLE_NAME, schema=_get_schema())

            try:
                self._table.create_fts_index("content", replace=True)
            except Exception as e:
                logger.debug("FTS index skipped: %s", e)

            self._embedder = _EmbedClient(os.environ.get("OPENAI_API_KEY", ""))
            self._ready = True
            logger.info("LanceDB memory initialized at %s", self._storage_path)

            # Warm prefetch so turn 1 has context
            self.queue_prefetch("current targets prospects contacts plans tasks decisions")

        except Exception as e:
            logger.warning("LanceDB memory init failed: %s", e, exc_info=True)
            self._ready = False

    def system_prompt_block(self) -> str:
        if not self._ready:
            return ""
        return (
            "# LanceDB Memory\n"
            "You have persistent vector memory across sessions. "
            "Call lancedb_search before most responses — any question, task, or topic "
            "may have relevant context from previous sessions. "
            "Default to searching first, then answering. Only skip if the query is "
            "clearly self-contained (e.g. a simple calculation or format request). "
            "Do not guess or fabricate from training knowledge when memory may have the answer. "
            "Use lancedb_remember to store durable facts explicitly when asked to remember something."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.5)
        with self._prefetch_lock:
            result, self._prefetch_result = self._prefetch_result, ""
        return f"## Recalled Memory\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._ready:
            return

        def _run():
            try:
                hits = self._hybrid_search(query, top_k=_TOP_K_RETURN)
                if hits:
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {h['content'][:300]}" for h in hits)
            except Exception as e:
                logger.debug("LanceDB prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="lancedb-prefetch")
        self._prefetch_thread.start()

    def _hybrid_search(self, query: str, top_k: int = _TOP_K_RETURN) -> List[Dict]:
        if not self._ready or self._table is None or self._embedder is None:
            return []
        try:
            vec = self._embedder.embed(query)
            v_results = (
                self._table.search(vec, vector_column_name="vector")
                .where(f"user_id = '{self._user_id}'", prefilter=True)
                .limit(_TOP_K_VECTOR)
                .to_list()
            )
            try:
                b_results = (
                    self._table.search(query, query_type="fts")
                    .where(f"user_id = '{self._user_id}'", prefilter=True)
                    .limit(_TOP_K_BM25)
                    .to_list()
                )
            except Exception:
                b_results = []

            def _hit(row) -> Optional[Dict]:
                ts = row.get("timestamp", 0.0) or 0.0
                w = _weibull_weight(_age_days(ts))
                if w < _DECAY_THRESHOLD:
                    return None
                return {"id": row.get("id", ""), "content": row.get("content", ""),
                        "timestamp": ts, "decay_weight": w, "source": row.get("source", "")}

            return _merge_rrf(
                [h for r in v_results if (h := _hit(r))],
                [h for r in b_results if (h := _hit(r))],
                top_k=top_k,
            )
        except Exception as e:
            logger.debug("LanceDB search failed: %s", e)
            return []

    def _write_entries(self, entries: List[Dict]) -> None:
        if not self._ready or not entries or self._table is None or self._embedder is None:
            return
        try:
            rows = []
            for entry in entries:
                text = entry.get("content", "")
                if not text.strip():
                    continue
                rows.append({
                    "id": str(uuid.uuid4()),
                    "content": text,
                    "vector": self._embedder.embed(text),
                    "timestamp": entry.get("timestamp", time.time()),
                    "source": entry.get("source", "turn"),
                    "session_id": entry.get("session_id", self._session_id),
                    "user_id": entry.get("user_id", self._user_id),
                    "tags": json.dumps(entry.get("tags", [])),
                })
            if rows:
                self._table.add(rows)
        except Exception as e:
            logger.warning("LanceDB write failed: %s", e)

    def _queue_write(self, entries: List[Dict]) -> None:
        with self._sync_lock:
            self._sync_queue.extend(entries)
        if self._sync_thread and self._sync_thread.is_alive():
            return

        def _flush():
            while True:
                with self._sync_lock:
                    batch, self._sync_queue[:] = list(self._sync_queue), []
                if not batch:
                    break
                self._write_entries(batch)

        self._sync_thread = threading.Thread(target=_flush, daemon=True, name="lancedb-sync")
        self._sync_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._ready:
            return
        self._queue_write([{
            "content": c, "source": "turn",
            "session_id": session_id or self._session_id,
            "user_id": self._user_id, "timestamp": time.time(),
        } for c in _extract_facts_from_turn(user_content, assistant_content)])

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._ready:
            return
        facts = _extract_facts_from_session(messages)
        if facts:
            self._write_entries([{
                "content": f, "source": "session_end",
                "session_id": self._session_id,
                "user_id": self._user_id, "timestamp": time.time(),
            } for f in facts])

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._ready:
            return ""
        facts = _extract_facts_from_session(messages)
        if facts:
            self._write_entries([{
                "content": f, "source": "compress",
                "session_id": self._session_id,
                "user_id": self._user_id, "timestamp": time.time(),
            } for f in facts])
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._ready or action == "remove":
            return
        self._queue_write([{
            "content": f"[{target}] {content}", "source": "builtin",
            "session_id": self._session_id, "user_id": self._user_id,
            "timestamp": time.time(), "tags": [target],
        }])

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "lancedb_search",
                "description": (
                    "Search long-term vector memory for relevant past context. "
                    "Uses hybrid BM25 + semantic search with recency weighting. "
                    "Use when you need to recall something from earlier sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        "top_k": {"type": "integer", "description": "Max results (default: 6, max: 20)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lancedb_remember",
                "description": (
                    "Explicitly store a durable fact or observation in long-term memory. "
                    "Use for key decisions, user preferences, or important outcomes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact or observation to store."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
                    },
                    "required": ["content"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "lancedb_search":
            hits = self._hybrid_search(args.get("query", ""), top_k=min(int(args.get("top_k", _TOP_K_RETURN)), 20))
            if not hits:
                return json.dumps({"results": [], "message": "No relevant memories found."})
            return json.dumps({"results": [
                {"content": h["content"][:500], "source": h.get("source", ""), "age_days": round(_age_days(h.get("timestamp", 0.0)), 1)}
                for h in hits
            ]})
        if tool_name == "lancedb_remember":
            content = args.get("content", "").strip()
            if not content:
                return json.dumps({"error": "content is required"})
            self._queue_write([{
                "content": content, "source": "explicit",
                "session_id": self._session_id, "user_id": self._user_id,
                "timestamp": time.time(), "tags": args.get("tags", []),
            }])
            return json.dumps({"stored": True, "content": content[:100]})
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        logger.info("LanceDB memory shut down.")


__all__ = ["LanceDBMemoryProvider"]
