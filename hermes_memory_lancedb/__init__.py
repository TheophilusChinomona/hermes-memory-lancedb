"""hermes-memory-lancedb v1.4.0 — LanceDB vector memory plugin for Hermes agents.

Hybrid BM25 + vector recall, Weibull decay, OpenAI text-embedding-3-small.

v1.4.0 additions (P2 of memory-lancedb-pro port — write pipeline):
  - Long-context chunker: split oversized writes on sentence boundaries with
    overlap; chunks share a `parent_id` so retrieval can collapse them
  - Batch dedup: pairwise cosine within a candidate batch + a single LLM call
    over surviving candidates vs the existing pool (was 1 LLM call per pair)
  - Admission control: rolling acceptance-rate / novelty / recency / type-prior
    gate with a hard-reject cosine vs recent rejects; persists stats to
    `<storage_path>/admission_stats.json`
  - Smart metadata: per-write LLM extraction of memory_temporal_type,
    confidence, sensitivity, modality, fact_key, tags — JSON-encoded into a
    new `metadata` column
  - Noise prototype filter: ~20 bundled multilingual noise prototypes; rejects
    writes whose embedding has cosine >= 0.92 with any prototype. Combined
    with the existing regex filter (either matcher rejects)

v1.2.0 additions (P0 of memory-lancedb-pro port — retrieval quality):
  - Cross-encoder reranking via Jina (fallback: cosine of query vs doc vectors)
  - MMR diversity: defer near-duplicate hits (cosine > 0.85) to end
  - Length normalization: log2 penalty for entries longer than 500 chars
  - Hard min-score cutoff after pipeline (default 0.35)
  - Decay weight applied as score multiplier (was filter-only before)

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

# v1.4.0 (P2) modules
from .chunker import chunk_text  # noqa: E402
from .dedup import batch_dedup  # noqa: E402
from .admission import AdmissionController  # noqa: E402
from .smart_metadata import extract_smart_metadata, stringify_smart_metadata  # noqa: E402
from .noise_proto import NoisePrototypeFilter  # noqa: E402

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

# v1.2.0 retrieval pipeline tuning
_LENGTH_NORM_ANCHOR = 500
_LENGTH_NORM_FLOOR = 0.3
_MMR_SIMILARITY_THRESHOLD = 0.85
_MIN_SCORE_EARLY = 0.3
_HARD_MIN_SCORE = 0.35
_RERANK_MODEL = "jina-reranker-v3"
_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"
_RERANK_TIMEOUT_S = 5.0
_RERANK_BLEND_RERANK = 0.6
_RERANK_BLEND_ORIGINAL = 0.4
_RERANK_UNRETURNED_PENALTY = 0.8

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

# v1.4.0 P2 — write-pipeline tuning
_CHUNK_TRIGGER_CHARS = 1800   # split contents longer than this into chunks
_CHUNK_MAX_CHARS = 1500
_CHUNK_OVERLAP = 200
_BATCH_DEDUP_POOL_SIZE = 6     # how many existing memories to surface per candidate
_NOISE_PROTO_THRESHOLD = 0.92  # cosine >= this => noise (matches TS)


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
        # v1.4.0 P2 fields
        pa.field("metadata", pa.string()),   # JSON-encoded smart metadata
        pa.field("parent_id", pa.string()),  # links chunked rows to their source
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
    out = []
    for mid in ranked[:top_k]:
        h = dict(by_id[mid])
        h["score"] = scores[mid]
        out.append(h)
    return out


def _clamp01(value: float, fallback: float = 0.0) -> float:
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return max(0.0, min(1.0, fallback))
    return max(0.0, min(1.0, value))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _normalize_to_top(hits: List[Dict]) -> List[Dict]:
    """Rescale `score` so the top hit is 1.0 (no-op if max <= 0 or empty)."""
    if not hits:
        return hits
    top = max(h.get("score", 0.0) for h in hits)
    if top <= 0:
        return hits
    for h in hits:
        h["score"] = h.get("score", 0.0) / top
    return hits


def _apply_length_normalization(
    hits: List[Dict],
    anchor: int = _LENGTH_NORM_ANCHOR,
) -> List[Dict]:
    """Penalize sprawling entries that dominate via keyword density.

    factor = 1 / (1 + 0.5 * log2(max(charLen/anchor, 1)))
    Entries at or below `anchor` chars: no penalty. Floor: score * 0.3.
    """
    if anchor <= 0 or not hits:
        return hits
    out = []
    for h in hits:
        text = h.get("content") or h.get("abstract") or ""
        ratio = len(text) / anchor if anchor > 0 else 1.0
        log_ratio = math.log2(max(ratio, 1.0))
        factor = 1.0 / (1.0 + 0.5 * log_ratio)
        original = h.get("score", 0.0)
        new_h = dict(h)
        new_h["score"] = _clamp01(original * factor, original * _LENGTH_NORM_FLOOR)
        out.append(new_h)
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out


def _apply_mmr_diversity(
    hits: List[Dict],
    threshold: float = _MMR_SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """Greedy MMR — defer near-duplicates (cosine > threshold) to end.

    Operates on hits' `vector` field. Hits without vectors are always kept
    in the selected list (no similarity check possible).
    """
    if len(hits) <= 1:
        return hits
    selected: List[Dict] = []
    deferred: List[Dict] = []
    for cand in hits:
        c_vec = cand.get("vector")
        too_similar = False
        if c_vec is not None and len(c_vec) > 0:
            c_list = list(c_vec)
            for sel in selected:
                s_vec = sel.get("vector")
                if s_vec is None or len(s_vec) == 0:
                    continue
                if _cosine_similarity(c_list, list(s_vec)) > threshold:
                    too_similar = True
                    break
        if too_similar:
            deferred.append(cand)
        else:
            selected.append(cand)
    return selected + deferred


def _rerank_jina(
    query: str,
    hits: List[Dict],
    api_key: str,
    model: str = _RERANK_MODEL,
    endpoint: str = _RERANK_ENDPOINT,
    timeout_s: float = _RERANK_TIMEOUT_S,
) -> Optional[List[Dict]]:
    """Cross-encoder rerank via Jina's API.

    Returns None on any failure so callers can fall back to cosine rerank.
    Blends 60% cross-encoder score + 40% original fused score, clamped to
    [score*0.5, 1.0]. Unreturned candidates get score * 0.8 (mild penalty).
    """
    if not hits or not api_key:
        return None
    try:
        import httpx
        documents = [h.get("content", "") for h in hits]
        body = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(hits),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(endpoint, json=body, headers=headers, timeout=timeout_s)
        if resp.status_code >= 400:
            logger.debug("Jina rerank HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            return None

        returned: Dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            cross = item.get("relevance_score")
            if cross is None:
                cross = item.get("score")
            if idx is None or cross is None:
                continue
            if not isinstance(idx, int) or not 0 <= idx < len(hits):
                continue
            returned[idx] = float(cross)

        if not returned:
            return None

        out: List[Dict] = []
        for idx, cross in returned.items():
            original = dict(hits[idx])
            blended = (
                cross * _RERANK_BLEND_RERANK
                + original.get("score", 0.0) * _RERANK_BLEND_ORIGINAL
            )
            original["score"] = _clamp01(blended, original.get("score", 0.0) * 0.5)
            original["reranked_score"] = float(cross)
            out.append(original)
        for idx, h in enumerate(hits):
            if idx in returned:
                continue
            unreturned = dict(h)
            unreturned["score"] = _clamp01(
                h.get("score", 0.0) * _RERANK_UNRETURNED_PENALTY,
                h.get("score", 0.0) * 0.5,
            )
            out.append(unreturned)
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out
    except Exception as e:
        logger.debug("Jina rerank failed: %s", e)
        return None


def _rerank_cosine_fallback(
    query_vec: List[float],
    hits: List[Dict],
) -> List[Dict]:
    """Lightweight cosine rerank when no API key is configured.

    Blends 70% original fused score + 30% cosine(query, doc). Hits without
    a `vector` field pass through with their existing score unchanged.
    """
    if not hits or not query_vec:
        return hits
    try:
        out = []
        for h in hits:
            vec = h.get("vector")
            if vec is None or len(vec) == 0:
                out.append(h)
                continue
            cos = _cosine_similarity(query_vec, list(vec))
            blended = h.get("score", 0.0) * 0.7 + cos * 0.3
            new_h = dict(h)
            new_h["score"] = _clamp01(blended, h.get("score", 0.0))
            new_h["reranked_score"] = float(cos)
            out.append(new_h)
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out
    except Exception as e:
        logger.debug("Cosine rerank failed: %s", e)
        return hits


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
        # v1.4.0 P2 — admission control + noise prototype filter
        self._admission: Optional[AdmissionController] = None
        self._noise_proto: Optional[NoisePrototypeFilter] = None
        self._smart_metadata_enabled: bool = True

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

            # v1.4.0 P2: admission controller + noise prototype filter.
            # Both default ON; flip via env vars to disable for tests/CI.
            admission_enabled = os.environ.get(
                "LANCEDB_ADMISSION_ENABLED", "1"
            ).lower() not in ("0", "false", "no")
            self._admission = AdmissionController(
                self._storage_path, enabled=admission_enabled,
            )
            self._smart_metadata_enabled = os.environ.get(
                "LANCEDB_SMART_METADATA", "1"
            ).lower() not in ("0", "false", "no")
            self._noise_proto = NoisePrototypeFilter(self._storage_path)
            try:
                self._noise_proto.load_or_init(self._embedder.embed)
            except Exception as e:
                logger.debug("noise prototype init skipped: %s", e)

            self._ready = True
            logger.info("LanceDB memory v1.4.0 initialized at %s", self._storage_path)

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
                # v1.1.0
                "tier": "'peripheral'",
                "importance": "0.5",
                "access_count": "0",
                "category": "'general'",
                "abstract": "''",
                "overview": "''",
                # v1.4.0 P2
                "metadata": "''",
                "parent_id": "''",
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
                    "vector": row.get("vector"),
                }

            v_hits = [h for r in v_results if (h := _hit(r))]
            b_hits = [h for r in b_results if (h := _hit(r))]

            # Stage 1: RRF fusion — fetch a wider window than top_k so the
            # downstream stages (rerank/length-norm/MMR) have enough to work with.
            rerank_window = max(top_k * 4, 12)
            merged = _merge_rrf(v_hits, b_hits, top_k=rerank_window)
            if not merged:
                return []

            # Stage 2: Normalize RRF scores to [0, 1] so subsequent thresholds
            # (min-score, hard-min-score) operate on a meaningful scale.
            merged = _normalize_to_top(merged)

            # Stage 3: Apply tier/recency decay weight as a multiplicative boost.
            for h in merged:
                original = h.get("score", 0.0)
                h["score"] = _clamp01(
                    original * h.get("decay_weight", 1.0),
                    original * 0.3,
                )
            merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # Stage 4: Early min-score filter.
            merged = [h for h in merged if h.get("score", 0.0) >= _MIN_SCORE_EARLY]
            if not merged:
                return []

            # Stage 5: Cross-encoder rerank — Jina if API key configured,
            # else cosine fallback (free, uses already-fetched vectors).
            rerank_input = merged[: top_k * 2]
            tail = merged[top_k * 2 :]
            rerank_api_key = (
                os.environ.get("LANCEDB_RERANK_API_KEY")
                or os.environ.get("JINA_API_KEY", "")
            )
            rerank_provider = os.environ.get("LANCEDB_RERANK_PROVIDER", "auto").lower()
            reranked: Optional[List[Dict]] = None
            if rerank_provider != "none" and rerank_api_key:
                reranked = _rerank_jina(
                    query,
                    rerank_input,
                    api_key=rerank_api_key,
                    model=os.environ.get("LANCEDB_RERANK_MODEL", _RERANK_MODEL),
                    endpoint=os.environ.get("LANCEDB_RERANK_ENDPOINT", _RERANK_ENDPOINT),
                    timeout_s=float(
                        os.environ.get("LANCEDB_RERANK_TIMEOUT_S", str(_RERANK_TIMEOUT_S))
                    ),
                )
            if reranked is None and rerank_provider != "none":
                reranked = _rerank_cosine_fallback(vec, rerank_input)
            if reranked is not None:
                merged = sorted(reranked + tail, key=lambda x: x.get("score", 0.0), reverse=True)

            # Stage 6: Length normalization (penalize sprawling entries).
            merged = _apply_length_normalization(merged)

            # Stage 7: Hard min-score cutoff (drops post-rerank low-confidence hits).
            try:
                hard_min = float(os.environ.get("LANCEDB_HARD_MIN_SCORE", str(_HARD_MIN_SCORE)))
            except (TypeError, ValueError):
                hard_min = _HARD_MIN_SCORE
            merged = [h for h in merged if h.get("score", 0.0) >= hard_min]

            # Stage 8: MMR diversity — defer near-duplicates to the tail.
            merged = _apply_mmr_diversity(merged)

            # Final: take top_k.
            merged = merged[:top_k]

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

    def _is_vector_noise(self, vec: List[float]) -> bool:
        """Vector-level noise check via the prototype bank (P2)."""
        if not self._noise_proto or not self._noise_proto.initialized:
            return False
        try:
            return self._noise_proto.is_noise(vec, threshold=_NOISE_PROTO_THRESHOLD)
        except Exception:
            return False

    def _should_admit(self, text: str, vec: List[float], category: str) -> Tuple[bool, str]:
        """Combined admission check: regex noise OR vector noise OR admission gate.

        Returns (admit, reason). Reason is empty when admitted.
        """
        if _is_noise(text):
            return False, "regex noise filter"
        if vec and self._is_vector_noise(vec):
            return False, "noise prototype match"
        if self._admission and self._admission.enabled:
            decision = self._admission.evaluate(text, vec, category=category)
            if not decision.admit:
                return False, f"admission control: {decision.reason}"
        return True, ""

    def _write_entries(self, entries: List[Dict]) -> None:
        if not self._ready or not entries or self._table is None or self._embedder is None:
            return
        try:
            rows = []
            for entry in entries:
                text = entry.get("content", "")
                if not text.strip():
                    continue

                # P2: chunk long content. Each chunk shares a parent_id.
                pieces: List[Tuple[str, str]] = []
                if len(text) > _CHUNK_TRIGGER_CHARS:
                    chunks = chunk_text(text, _CHUNK_MAX_CHARS, _CHUNK_OVERLAP)
                    parent_id = str(uuid.uuid4())
                    for chunk in chunks:
                        pieces.append((chunk, parent_id))
                else:
                    pieces.append((text, entry.get("parent_id", "")))

                category = entry.get("category", "general")
                source = entry.get("source", "turn")
                base_metadata_obj = entry.get("metadata")
                if isinstance(base_metadata_obj, dict):
                    base_metadata_str = stringify_smart_metadata(base_metadata_obj)
                elif isinstance(base_metadata_obj, str):
                    base_metadata_str = base_metadata_obj
                else:
                    base_metadata_str = ""

                for chunk_text_value, parent_id in pieces:
                    vec = self._embedder.embed(chunk_text_value)

                    # P2: gate via combined noise + admission check (skipped for
                    # explicit `lancedb_remember` writes — those are user intent).
                    if source not in ("explicit",):
                        admit, reason = self._should_admit(chunk_text_value, vec, category)
                        if not admit:
                            logger.debug("LanceDB write skipped: %s", reason)
                            continue

                    # P2: smart metadata. If caller supplied one we keep it,
                    # otherwise extract on the fly when LLM is wired up.
                    metadata_str = base_metadata_str
                    if (
                        not metadata_str
                        and self._smart_metadata_enabled
                        and self._llm is not None
                    ):
                        try:
                            meta = extract_smart_metadata(
                                chunk_text_value,
                                self._llm,
                                abstract=entry.get("abstract", ""),
                                category=category,
                                source="manual" if source == "explicit" else "auto-capture",
                                timestamp=entry.get("timestamp", time.time()),
                            )
                            metadata_str = stringify_smart_metadata(meta)
                        except Exception as e:
                            logger.debug("smart metadata extraction failed: %s", e)
                            metadata_str = ""

                    rows.append({
                        "id": str(uuid.uuid4()),
                        "content": chunk_text_value,
                        "vector": vec,
                        "timestamp": entry.get("timestamp", time.time()),
                        "source": source,
                        "session_id": entry.get("session_id", self._session_id),
                        "user_id": entry.get("user_id", self._user_id),
                        "tags": json.dumps(entry.get("tags", [])),
                        "tier": entry.get("tier", "peripheral"),
                        "importance": float(entry.get("importance", 0.5)),
                        "access_count": int(entry.get("access_count", 0)),
                        "category": category,
                        "abstract": entry.get("abstract", ""),
                        "overview": entry.get("overview", ""),
                        "metadata": metadata_str,
                        "parent_id": parent_id,
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

        # P2: build an existing-pool from a single vector probe per candidate,
        # then run ONE batch dedup LLM call instead of N pairwise calls.
        existing_pool: List[Dict] = []
        candidate_to_match: Dict[int, Dict] = {}
        for idx, candidate in enumerate(candidates):
            content = candidate.get("content", "")
            if not content.strip():
                continue
            similar = self._find_similar(content)
            if similar:
                candidate_to_match[idx] = similar
                # Add to pool if not already present
                if not any(p.get("id") == similar.get("id") for p in existing_pool):
                    existing_pool.append(similar)
                if len(existing_pool) >= _BATCH_DEDUP_POOL_SIZE:
                    break

        embed_fn = self._embedder.embed if self._embedder else None
        decisions = batch_dedup(
            candidates,
            existing_pool,
            self._llm,
            embedder=embed_fn,
        )
        decisions_by_index = {d["index"]: d for d in decisions}

        for idx, candidate in enumerate(candidates):
            content = candidate.get("content", "")
            if not content.strip():
                continue

            d = decisions_by_index.get(idx, {"decision": "create", "merged_content": ""})
            decision = d.get("decision", "create")
            merged = d.get("merged_content") or ""
            similar = candidate_to_match.get(idx)

            if decision in ("skip", "support"):
                continue

            if decision == "supersede" and similar and similar.get("id"):
                try:
                    self._table.delete(f"id = '{similar['id']}'")
                except Exception:
                    pass
                self._write_entries([{**candidate, "source": "extraction", "timestamp": time.time(),
                                      "session_id": turn_data.get("session_id", self._session_id)}])
                continue

            if decision == "merge" and merged and similar:
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


__all__ = [
    "LanceDBMemoryProvider",
    "MEMORY_CATEGORIES",
    # P2 modules re-exported for convenience
    "AdmissionController",
    "NoisePrototypeFilter",
    "batch_dedup",
    "chunk_text",
    "extract_smart_metadata",
    "stringify_smart_metadata",
]
