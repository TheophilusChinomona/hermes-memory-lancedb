"""hermes-memory-lancedb v1.1.0 — LanceDB vector memory plugin for Hermes agents.

Hybrid BM25 + vector recall, Weibull decay, OpenAI text-embedding-3-small.

v1.1.0 additions (ported from memory-lancedb-pro TypeScript fork):
  - LLM Smart Extraction: 6-category system (profile, preferences, entities,
    events, cases, patterns) with 3-level structure (abstract/overview/content)
  - Dedup pipeline: create/merge/skip/support/contextualize/contradict/supersede
  - 3-tier memory: peripheral → working → core with promotion/demotion
  - Noise filtering: denials, meta-questions, boilerplate, diagnostic artifacts
  - New tools: lancedb_forget, lancedb_stats

Storage:    $LANCEDB_PATH  or  $HERMES_HOME/lancedb/
Embeddings: OpenAI text-embedding-3-small — OPENAI_API_KEY required
Extraction: gpt-4o-mini (configurable via lancedb.json extraction_model)

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
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MemoryProvider base
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider as _MemoryProviderBase
except ImportError:
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

_WEIBULL_SCALE = 30.0
_WEIBULL_SHAPE = 0.7
_DECAY_THRESHOLD = 0.05

_MAX_MEMORIES_PER_EXTRACTION = 5
_DEDUP_SIMILARITY_THRESHOLD = 0.50  # L2 distance; text-embedding-3-small vectors are normalised
_DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini"

# Tier decay floors applied during search
_TIER_DECAY_FLOOR = {"core": 0.9, "working": 0.7, "peripheral": 0.0}

# Tier promotion / demotion thresholds
_PROMO_PERI_TO_WORK_ACCESS = 3
_PROMO_PERI_TO_WORK_COMPOSITE = 0.4
_PROMO_WORK_TO_CORE_ACCESS = 10
_PROMO_WORK_TO_CORE_COMPOSITE = 0.7
_PROMO_WORK_TO_CORE_IMPORTANCE = 0.8
_DEMO_COMPOSITE_THRESHOLD = 0.15
_DEMO_AGE_DAYS = 60
_DEMO_AGE_ACCESS_THRESHOLD = 3

MEMORY_CATEGORIES = ["profile", "preferences", "entities", "events", "cases", "patterns"]


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
        # v1.1.0 fields
        pa.field("tier", pa.string()),
        pa.field("importance", pa.float32()),
        pa.field("access_count", pa.int32()),
        pa.field("category", pa.string()),
        pa.field("abstract", pa.string()),
        pa.field("overview", pa.string()),
    ])


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

_DENIAL_PATTERNS = [
    r"I don't have access to",
    r"I cannot access",
    r"I'm not able to",
    r"I do not have the ability",
    r"As an AI",
    r"I'm an AI",
    r"I am an AI",
    r"I don't actually have",
    r"I can't recall",
]

_META_QUESTION_PATTERNS = [
    r"do you remember",
    r"what do you know about",
    r"can you recall",
    r"do you have memory of",
    r"have you stored",
    r"what's in your memory",
]

_BOILERPLATE_PATTERNS = [
    r"^hi\b",
    r"^hello\b",
    r"^hey\b",
    r"^thanks?\b",
    r"^thank you\b",
    r"^ok\b",
    r"^okay\b",
    r"^sure\b",
    r"^got it\b",
    r"^understood\b",
    r"^sounds good\b",
]

_DIAGNOSTIC_PATTERNS = [
    r"\btest message\b",
    r"^\[test\]",
    r"^testing\b",
    r"\bping\b",
    r"\bpong\b",
]

_NOISE_RE = [
    re.compile(p, re.IGNORECASE)
    for p in _DENIAL_PATTERNS + _META_QUESTION_PATTERNS + _BOILERPLATE_PATTERNS + _DIAGNOSTIC_PATTERNS
]


def _is_noise(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return True
    t = text.strip()
    for pattern in _NOISE_RE:
        if pattern.search(t):
            return True
    return False


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """You are a memory extraction system for an AI sales agent named Andrew.
Extract only important, durable facts from the conversation that are worth remembering across sessions.

Categories (use exactly these names):
- profile: user identity, role, relationships, background facts about Theo or SpecCon
- preferences: working style, opinions, stated preferences, communication style
- entities: named things — companies, people, products, platforms, locations
- events: specific occurrences with dates or timing — decisions made, meetings, outcomes
- cases: completed tasks, resolved problems, applied solutions, campaigns run
- patterns: recurring behaviours, strategies, tendencies, rules Theo has established

For each memory output:
- category: one of the 6 above
- abstract: one-line summary ≤80 chars (L0 — the "headline")
- overview: 2-5 bullet points in markdown (L1 — structured context)
- content: full narrative with all relevant details (L2 — complete record)
- importance: float 0.0–1.0 (critical facts ≥0.8, routine observations ≤0.4)
- tags: list of keyword strings

Rules:
- Max 5 memories per extraction
- Omit agent denial responses, pleasantries, and meta-questions about memory
- Omit anything that's a one-off transient fact with no future relevance
- Return [] if nothing durable is worth storing
- Return ONLY a valid JSON array, no markdown fencing"""


def _build_extraction_prompt(conversation_text: str) -> str:
    return f"Extract durable memories from this conversation turn:\n\n{conversation_text[:6000]}"


def _build_session_extraction_prompt(messages: List[Dict]) -> str:
    lines = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            role = m["role"].capitalize()
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if content.strip():
                lines.append(f"{role}: {content.strip()[:800]}")
    combined = "\n".join(lines)
    return f"Extract durable memories from this full session:\n\n{combined[:8000]}"


# ---------------------------------------------------------------------------
# Dedup prompt
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """You are a memory deduplication system. Given a new candidate memory and an existing similar memory, decide what to do.

Decisions (use exactly one):
- skip: candidate adds nothing new; existing memory already covers it
- support: candidate reinforces existing (minor variant); keep existing, bump access count
- merge: combine both into one improved, complete memory
- contextualize: candidate adds useful nuance but is distinct enough to keep separately
- contradict: candidate contradicts existing; flag the conflict
- supersede: candidate is more recent/accurate and replaces existing
- create: memories are sufficiently different; create candidate as new entry

Return ONLY valid JSON: {"decision": "<decision>", "merged_content": "<combined text if merge or supersede, else empty string>"}"""


def _build_dedup_prompt(existing_content: str, candidate_content: str) -> str:
    return (
        f"Existing memory:\n{existing_content[:600]}\n\n"
        f"New candidate:\n{candidate_content[:600]}"
    )


# ---------------------------------------------------------------------------
# Tier evaluation
# ---------------------------------------------------------------------------

def _composite_score(decay_weight: float, importance: float) -> float:
    return decay_weight * importance


def _tier_evaluate(
    current_tier: str,
    access_count: int,
    importance: float,
    decay_weight: float,
    age_days: float,
) -> Optional[str]:
    """Return new tier if promotion/demotion warranted, else None."""
    composite = _composite_score(decay_weight, importance)

    if current_tier == "peripheral":
        if access_count >= _PROMO_PERI_TO_WORK_ACCESS and composite >= _PROMO_PERI_TO_WORK_COMPOSITE:
            return "working"

    elif current_tier == "working":
        if (access_count >= _PROMO_WORK_TO_CORE_ACCESS
                and composite >= _PROMO_WORK_TO_CORE_COMPOSITE
                and importance >= _PROMO_WORK_TO_CORE_IMPORTANCE):
            return "core"
        if composite < _DEMO_COMPOSITE_THRESHOLD or (age_days > _DEMO_AGE_DAYS and access_count < _DEMO_AGE_ACCESS_THRESHOLD):
            return "peripheral"

    elif current_tier == "core":
        if composite < _DEMO_COMPOSITE_THRESHOLD and access_count < _DEMO_AGE_ACCESS_THRESHOLD:
            return "working"

    return None


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
# LLM extraction client
# ---------------------------------------------------------------------------

class _LLMClient:
    def __init__(self, api_key: str, model: str = _DEFAULT_EXTRACTION_MODEL):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def chat(self, system: str, user: str, *, max_tokens: int = 1500) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


def _llm_extract_memories(conversation_text: str, llm: _LLMClient) -> List[Dict]:
    """Call LLM to extract categorised memory candidates from a conversation snippet."""
    try:
        raw = llm.chat(_EXTRACTION_SYSTEM, _build_extraction_prompt(conversation_text))
        candidates = json.loads(raw)
        if not isinstance(candidates, list):
            return []
        out = []
        for c in candidates[:_MAX_MEMORIES_PER_EXTRACTION]:
            if not isinstance(c, dict):
                continue
            category = c.get("category", "")
            if category not in MEMORY_CATEGORIES:
                category = "cases"
            out.append({
                "category": category,
                "abstract": str(c.get("abstract", ""))[:160],
                "overview": str(c.get("overview", ""))[:800],
                "content": str(c.get("content", c.get("abstract", "")))[:2000],
                "importance": float(c.get("importance", 0.5)),
                "tags": c.get("tags", []) if isinstance(c.get("tags"), list) else [],
            })
        return out
    except Exception as e:
        logger.debug("LLM extraction failed: %s", e)
        return []


def _llm_dedup(existing_content: str, candidate_content: str, llm: _LLMClient) -> Tuple[str, str]:
    """Return (decision, merged_content)."""
    try:
        raw = llm.chat(_DEDUP_SYSTEM, _build_dedup_prompt(existing_content, candidate_content), max_tokens=400)
        result = json.loads(raw)
        return str(result.get("decision", "create")), str(result.get("merged_content", ""))
    except Exception as e:
        logger.debug("LLM dedup failed: %s — defaulting to create", e)
        return "create", ""


# ---------------------------------------------------------------------------
# LanceDBMemoryProvider
# ---------------------------------------------------------------------------

class LanceDBMemoryProvider(_MemoryProviderBase):
    """
    LanceDB vector memory for Hermes agents.

    v1.1.0: LLM Smart Extraction, 3-tier system, dedup pipeline, noise filtering.
    Storage: $LANCEDB_PATH or $HERMES_HOME/lancedb/
    """

    def __init__(self):
        self._db = None
        self._table = None
        self._embedder: Optional[_EmbedClient] = None
        self._llm: Optional[_LLMClient] = None
        self._user_id = "andrew"
        self._session_id = ""
        self._storage_path = ""
        self._extraction_model = _DEFAULT_EXTRACTION_MODEL
        self._ready = False
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._extract_queue: List[Dict] = []
        self._extract_lock = threading.Lock()
        self._extract_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "lancedb"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "storage_path", "description": "Directory for LanceDB data files", "default": "~/.hermes/lancedb"},
            {"key": "user_id", "description": "User identifier for memory scoping", "default": "andrew"},
            {"key": "extraction_model", "description": "Model for LLM extraction/dedup", "default": _DEFAULT_EXTRACTION_MODEL},
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
            hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))

        config: Dict = {
            "storage_path": os.environ.get("LANCEDB_PATH", str(hermes_home / "lancedb")),
            "user_id": "andrew",
            "extraction_model": _DEFAULT_EXTRACTION_MODEL,
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
        self._extraction_model = cfg.get("extraction_model", _DEFAULT_EXTRACTION_MODEL)

        try:
            import lancedb

            os.makedirs(self._storage_path, exist_ok=True)
            self._db = lancedb.connect(self._storage_path)

            if _TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(_TABLE_NAME)
                self._migrate_schema_if_needed()
            else:
                self._table = self._db.create_table(_TABLE_NAME, schema=_get_schema())

            try:
                self._table.create_fts_index("content", replace=True)
            except Exception as e:
                logger.debug("FTS index skipped: %s", e)

            api_key = os.environ.get("OPENAI_API_KEY", "")
            self._embedder = _EmbedClient(api_key)
            self._llm = _LLMClient(api_key, model=self._extraction_model)
            self._ready = True
            logger.info("LanceDB memory v1.1.0 initialized at %s", self._storage_path)

            self.queue_prefetch("current targets prospects contacts plans tasks decisions")

        except Exception as e:
            logger.warning("LanceDB memory init failed: %s", e, exc_info=True)
            self._ready = False

    def _migrate_schema_if_needed(self) -> None:
        if self._table is None:
            return
        try:
            existing_cols = {f.name for f in self._table.schema}
            new_fields = {
                "tier": "'peripheral'",
                "importance": "0.5",
                "access_count": "0",
                "category": "'general'",
                "abstract": "''",
                "overview": "''",
            }
            missing = {k: v for k, v in new_fields.items() if k not in existing_cols}
            if missing:
                self._table.add_columns(missing)
                logger.info("LanceDB schema migrated: added columns %s", list(missing.keys()))
        except Exception as e:
            logger.warning("LanceDB schema migration failed (reads will use defaults): %s", e)

    def system_prompt_block(self) -> str:
        if not self._ready:
            return ""
        return (
            "# LanceDB Memory\n"
            "You have persistent vector memory across sessions (v1.1.0 — tiered, categorised). "
            "Call lancedb_search before most responses — any question, task, or topic "
            "may have relevant context from previous sessions. "
            "Default to searching first, then answering. Only skip if the query is "
            "clearly self-contained (e.g. a simple calculation or format request). "
            "Do not fabricate from training knowledge when memory may have the answer. "
            "Use lancedb_remember to store durable facts explicitly."
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
                base_decay = _weibull_weight(_age_days(ts))
                tier = row.get("tier") or "peripheral"
                floor = _TIER_DECAY_FLOOR.get(tier, 0.0)
                w = max(base_decay, floor)
                if w < _DECAY_THRESHOLD:
                    return None
                return {
                    "id": row.get("id", ""),
                    "content": row.get("content", ""),
                    "abstract": row.get("abstract", ""),
                    "timestamp": ts,
                    "decay_weight": w,
                    "source": row.get("source", ""),
                    "tier": tier,
                    "category": row.get("category", ""),
                    "access_count": row.get("access_count", 0),
                    "importance": row.get("importance", 0.5),
                }

            merged = _merge_rrf(
                [h for r in v_results if (h := _hit(r))],
                [h for r in b_results if (h := _hit(r))],
                top_k=top_k,
            )

            # Bump access counts async
            if merged:
                ids = [h["id"] for h in merged]
                threading.Thread(target=self._bump_access, args=(ids,), daemon=True).start()

            return merged
        except Exception as e:
            logger.debug("LanceDB search failed: %s", e)
            return []

    def _bump_access(self, ids: List[str]) -> None:
        """Increment access_count and re-evaluate tier for accessed memories."""
        if not self._ready or self._table is None:
            return
        try:
            for mid in ids:
                rows = self._table.search().where(f"id = '{mid}'", prefilter=True).limit(1).to_list()
                if not rows:
                    continue
                row = rows[0]
                new_count = int(row.get("access_count") or 0) + 1
                tier = row.get("tier") or "peripheral"
                ts = row.get("timestamp", 0.0) or 0.0
                importance = float(row.get("importance") or 0.5)
                decay = _weibull_weight(_age_days(ts))
                new_tier = _tier_evaluate(tier, new_count, importance, decay, _age_days(ts)) or tier
                self._table.update(
                    where=f"id = '{mid}'",
                    values={"access_count": new_count, "tier": new_tier},
                )
        except Exception as e:
            logger.debug("LanceDB access bump failed: %s", e)

    def _find_similar(self, text: str, threshold: float = _DEDUP_SIMILARITY_THRESHOLD) -> Optional[Dict]:
        """Find a highly similar existing memory using vector search."""
        if not self._ready or self._table is None or self._embedder is None:
            return None
        try:
            vec = self._embedder.embed(text)
            results = (
                self._table.search(vec, vector_column_name="vector")
                .where(f"user_id = '{self._user_id}'", prefilter=True)
                .limit(1)
                .to_list()
            )
            if results and results[0].get("_distance", 999) < threshold:
                return results[0]
        except Exception:
            pass
        return None

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
                    "tier": entry.get("tier", "peripheral"),
                    "importance": float(entry.get("importance", 0.5)),
                    "access_count": int(entry.get("access_count", 0)),
                    "category": entry.get("category", "general"),
                    "abstract": entry.get("abstract", ""),
                    "overview": entry.get("overview", ""),
                })
            if rows:
                self._table.add(rows)
        except Exception as e:
            logger.warning("LanceDB write failed: %s", e)

    def _extract_and_write(self, turn_data: Dict) -> None:
        """LLM extraction pipeline for a single turn. Runs in background thread."""
        if not self._llm:
            return

        user_text = turn_data.get("user", "")
        asst_text = turn_data.get("assistant", "")
        conversation = f"User: {user_text}\nAssistant: {asst_text}"

        candidates = _llm_extract_memories(conversation, self._llm)

        if not candidates:
            # Fallback: store raw turn if extraction returns nothing
            raw = f"User: {user_text[:600]}\nAssistant: {asst_text[:600]}"
            if raw.strip():
                self._write_entries([{
                    "content": raw,
                    "source": turn_data.get("source", "turn"),
                    "session_id": turn_data.get("session_id", self._session_id),
                    "user_id": self._user_id,
                    "timestamp": time.time(),
                    "category": "cases",
                    "tier": "peripheral",
                    "importance": 0.3,
                }])
            return

        for candidate in candidates:
            content = candidate.get("content", "")
            if not content.strip():
                continue

            similar = self._find_similar(content)

            if similar:
                decision, merged = _llm_dedup(
                    similar.get("content", ""),
                    content,
                    self._llm,
                )

                if decision in ("skip", "support"):
                    continue

                if decision == "supersede" and similar.get("id"):
                    try:
                        self._table.delete(f"id = '{similar['id']}'")
                    except Exception:
                        pass
                    self._write_entries([{**candidate, "source": "extraction", "timestamp": time.time(),
                                          "session_id": turn_data.get("session_id", self._session_id)}])
                    continue

                if decision == "merge" and merged:
                    if similar.get("id"):
                        try:
                            self._table.delete(f"id = '{similar['id']}'")
                        except Exception:
                            pass
                    candidate["content"] = merged
                    candidate["abstract"] = candidate.get("abstract", merged[:80])
                    self._write_entries([{**candidate, "source": "extraction_merge", "timestamp": time.time(),
                                          "session_id": turn_data.get("session_id", self._session_id)}])
                    continue

                if decision == "contradict":
                    # Store with a conflict marker
                    candidate["abstract"] = f"[CONFLICT] {candidate.get('abstract', '')}"

            self._write_entries([{**candidate, "source": "extraction", "timestamp": time.time(),
                                   "session_id": turn_data.get("session_id", self._session_id)}])

    def _queue_extraction(self, turn_data: Dict) -> None:
        with self._extract_lock:
            self._extract_queue.append(turn_data)
        if self._extract_thread and self._extract_thread.is_alive():
            return

        def _flush():
            while True:
                with self._extract_lock:
                    batch, self._extract_queue[:] = list(self._extract_queue), []
                if not batch:
                    break
                for item in batch:
                    try:
                        self._extract_and_write(item)
                    except Exception as e:
                        logger.debug("Extraction pipeline error: %s", e)

        self._extract_thread = threading.Thread(target=_flush, daemon=True, name="lancedb-extract")
        self._extract_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._ready:
            return
        # Skip if both sides are pure noise
        if _is_noise(user_content) and _is_noise(assistant_content):
            return
        self._queue_extraction({
            "user": user_content,
            "assistant": assistant_content,
            "session_id": session_id or self._session_id,
            "source": "turn",
        })

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._ready or not self._llm:
            return

        def _run():
            try:
                prompt = _build_session_extraction_prompt(messages)
                candidates = _llm_extract_memories(prompt, self._llm)
                entries = [{
                    **c, "source": "session_end", "timestamp": time.time(),
                    "session_id": self._session_id,
                } for c in candidates]
                if entries:
                    self._write_entries(entries)
            except Exception as e:
                logger.debug("Session-end extraction failed: %s", e)

        threading.Thread(target=_run, daemon=True, name="lancedb-session-end").start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._ready or not self._llm:
            return ""

        def _run():
            try:
                prompt = _build_session_extraction_prompt(messages)
                candidates = _llm_extract_memories(prompt, self._llm)
                entries = [{
                    **c, "source": "compress", "timestamp": time.time(),
                    "session_id": self._session_id,
                } for c in candidates]
                if entries:
                    self._write_entries(entries)
            except Exception as e:
                logger.debug("Pre-compress extraction failed: %s", e)

        threading.Thread(target=_run, daemon=True, name="lancedb-compress").start()
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._ready or action == "remove":
            return
        self._queue_extraction({
            "user": f"[Memory write: {target}] {content}",
            "assistant": "",
            "session_id": self._session_id,
            "source": "builtin",
        })

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "lancedb_search",
                "description": (
                    "Search long-term vector memory for relevant past context. "
                    "Hybrid BM25 + semantic search with tier-weighted recency. "
                    "Use before answering questions that may have prior context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        "top_k": {"type": "integer", "description": "Max results (default 6, max 20)."},
                        "category": {
                            "type": "string",
                            "description": "Filter by category: profile, preferences, entities, events, cases, patterns.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lancedb_remember",
                "description": "Explicitly store a durable fact in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to store."},
                        "category": {"type": "string", "description": "Memory category."},
                        "importance": {"type": "number", "description": "0.0–1.0 importance score."},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "lancedb_forget",
                "description": "Delete a specific memory by content keyword or ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text to find the memory to delete."},
                        "id": {"type": "string", "description": "Exact memory ID to delete."},
                    },
                },
            },
            {
                "name": "lancedb_stats",
                "description": "Show memory statistics: total count, tier breakdown, category breakdown.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "lancedb_search":
            category_filter = args.get("category", "")
            hits = self._hybrid_search(args.get("query", ""), top_k=min(int(args.get("top_k", _TOP_K_RETURN)), 20))
            if category_filter:
                hits = [h for h in hits if h.get("category") == category_filter]
            if not hits:
                return json.dumps({"results": [], "message": "No relevant memories found."})
            return json.dumps({"results": [
                {
                    "content": h["content"][:500],
                    "abstract": h.get("abstract", ""),
                    "source": h.get("source", ""),
                    "tier": h.get("tier", "peripheral"),
                    "category": h.get("category", ""),
                    "age_days": round(_age_days(h.get("timestamp", 0.0)), 1),
                    "importance": round(float(h.get("importance", 0.5)), 2),
                }
                for h in hits
            ]})

        if tool_name == "lancedb_remember":
            content = args.get("content", "").strip()
            if not content:
                return json.dumps({"error": "content is required"})
            self._write_entries([{
                "content": content,
                "source": "explicit",
                "session_id": self._session_id,
                "user_id": self._user_id,
                "timestamp": time.time(),
                "tags": args.get("tags", []),
                "category": args.get("category", "cases"),
                "importance": float(args.get("importance", 0.7)),
                "tier": "working",  # explicit memories start in working tier
                "abstract": content[:80],
                "overview": "",
            }])
            return json.dumps({"stored": True, "content": content[:100]})

        if tool_name == "lancedb_forget":
            if not self._ready or self._table is None:
                return json.dumps({"error": "memory not ready"})
            mem_id = args.get("id", "")
            query = args.get("query", "")
            if mem_id:
                try:
                    self._table.delete(f"id = '{mem_id}'")
                    return json.dumps({"deleted": True, "id": mem_id})
                except Exception as e:
                    return json.dumps({"error": str(e)})
            if query:
                hits = self._hybrid_search(query, top_k=1)
                if hits:
                    mid = hits[0]["id"]
                    try:
                        self._table.delete(f"id = '{mid}'")
                        return json.dumps({"deleted": True, "content": hits[0]["content"][:100]})
                    except Exception as e:
                        return json.dumps({"error": str(e)})
            return json.dumps({"error": "provide query or id"})

        if tool_name == "lancedb_stats":
            if not self._ready or self._table is None:
                return json.dumps({"error": "memory not ready"})
            try:
                rows = (
                    self._table.search()
                    .where(f"user_id = '{self._user_id}'", prefilter=True)
                    .limit(10000)
                    .to_list()
                )
                total = len(rows)
                tiers: Dict[str, int] = {}
                cats: Dict[str, int] = {}
                for r in rows:
                    t = r.get("tier") or "peripheral"
                    tiers[t] = tiers.get(t, 0) + 1
                    c = r.get("category") or "unknown"
                    cats[c] = cats.get(c, 0) + 1
                return json.dumps({
                    "total": total,
                    "tiers": tiers,
                    "categories": cats,
                    "storage_path": self._storage_path,
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._extract_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=8.0)
        logger.info("LanceDB memory shut down.")


__all__ = ["LanceDBMemoryProvider"]
