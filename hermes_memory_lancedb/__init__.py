"""hermes-memory-lancedb v1.5.0 — LanceDB vector memory plugin for Hermes agents.

Hybrid BM25 + vector recall, Weibull decay, multi-provider embeddings.

v1.5.0 additions (P3 of memory-lancedb-pro port — reflection subsystem):
  - Reflection store: separate `reflections` LanceDB table with own schema + FTS index
  - Event store + item store + ranker + retry + slice loaders
  - Provider hooks: lazy init (LANCEDB_REFLECTION_ENABLED), session_end capture,
    optional merge into _hybrid_search results with `source: "reflection"` marker
  - New tools: lancedb_reflect, lancedb_reflections

v1.4.0 additions (P2 of memory-lancedb-pro port — write pipeline):
  - Long-content chunking with paragraph/sentence boundaries + overlap
  - Batch LLM dedup (single call vs pairwise)
  - Admission controller with rolling stats (rate limiting, similarity
    gating against recent rejects)
  - Smart metadata extraction (temporal_type, confidence, sensitivity)
  - Noise prototype filter (vector-based, complements regex _is_noise)
  - New schema columns: metadata (JSON), parent_id (chunk grouping)

v1.3.0 additions (P1 of memory-lancedb-pro port — multi-tenancy):
  - Multi-scope isolation: agent / user / project / team / workspace
    columns compose orthogonally in the search predicate
  - Multi-provider embeddings: OpenAI (default), Jina, Gemini, Ollama,
    plus any OpenAI-compatible endpoint via LANCEDB_EMBED_BASE_URL
  - Schema migration adds scope columns to existing tables
  - Embedding dimension is provider-driven (no longer hardcoded)

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

v1.5.0 additions (P3 of memory-lancedb-pro port — reflection subsystem):
  - Reflection subpackage (hermes_memory_lancedb.reflection) — 8 modules:
    store, event_store, item_store, metadata, mapped_metadata, ranking,
    retry, slices.
  - Dedicated `reflections` LanceDB table with its own FTS index +
    optional vector column.
  - Recency-weighted, importance-boosted ranking (logistic decay).
  - Lazy init via `LANCEDB_REFLECTION_ENABLED=1` (off by default for
    backwards compat).
  - on_session_end now captures a reflection extract in addition to memories.
  - _hybrid_search optionally pulls top-K reflections (env-tunable).
  - New tools: lancedb_reflect (explicit write), lancedb_reflections (search).

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
Embeddings: $LANCEDB_EMBED_PROVIDER (default openai/text-embedding-3-small)
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

from .embedders import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    EmbeddingError,
    PROVIDER_DEFAULT_MODEL,
    get_provider_from_env,
    is_provider_available,
    make_embedder,
)
from .scopes import (
    GLOBAL_SCOPE,
    SCOPE_COLUMN_DEFAULTS,
    SCOPE_COLUMNS,
    ScopeManager,
    clawteam_scopes_from_env,
    parse_agent_id_from_session_key,
)
from . import reflection as _reflection
from .reflection import (
    BuildReflectionStorePayloadsParams,
    ReflectionErrorSignalLike,
    ReflectionEventStore,
    ReflectionItemStore,
    ReflectionRanker,
    ReflectionStore,
)

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
_EMBED_MODEL = "text-embedding-3-small"  # legacy default; see embedders.make_embedder
_EMBED_DIM = 1536  # legacy default; the schema is built from the active embedder's dim
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

def _get_schema(embed_dim: int = _EMBED_DIM):
    import pyarrow as pa
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embed_dim)),
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
        # v1.3.0 P1 — multi-scope columns (all nullable strings)
        pa.field("agent_id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("team_id", pa.string()),
        pa.field("workspace_id", pa.string()),
        pa.field("scope", pa.string()),
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
# Reflection prompt (P3)
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM = """You are a reflection writer for an AI agent. Read the session and produce a structured markdown reflection in EXACTLY this format:

## Invariants
- <stable rule that should hold across all future sessions>

## Derived
- <session-specific change or delta to apply next run>

## User model deltas (about the human)
- <preference change>

## Agent model deltas (about the assistant/system)
- <new self-knowledge>

## Lessons & pitfalls (symptom / cause / fix / prevention)
- <bullet>

## Decisions (durable)
- <decision>

## Open loops / next actions
- <follow-up>

Rules:
- Each section is optional — omit a section entirely if you have nothing useful for it.
- Bullets must start with "- ".
- Keep each bullet to one short sentence.
- Do NOT include reasoning, preamble, or explanations.
- If the session is too short or contains nothing reflection-worthy, return an empty string."""


def _build_reflection_prompt(messages: List[Dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        if content.strip():
            lines.append(f"{role.capitalize()}: {content.strip()[:800]}")
    combined = "\n".join(lines)
    return f"Write a reflection for this session:\n\n{combined[:10000]}"


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
# Embedding client (legacy shim — kept for backward compatibility with tests
# that monkey-patch `hermes_memory_lancedb._EmbedClient`. New code goes
# through `hermes_memory_lancedb.embedders.make_embedder()`.)
# ---------------------------------------------------------------------------

class _EmbedClient:
    """Legacy thin OpenAI embedder. Kept for back-compat with v1.1.0/v1.2.0 tests."""

    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self.dimensions = _EMBED_DIM
        self.model = _EMBED_MODEL

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
        self._embedder: Optional[Embedder] = None
        self._llm: Optional[_LLMClient] = None
        self._user_id = "andrew"
        # P1 scope identifiers (all optional; empty string == no filter)
        self._agent_id: str = ""
        self._project_id: str = ""
        self._team_id: str = ""
        self._workspace_id: str = ""
        self._scope_manager: ScopeManager = ScopeManager()
        # Tracks whether the active table has the new scope columns. Set
        # during initialize() / _migrate_schema_if_needed() and consulted
        # by _hybrid_search to choose between legacy and composable filters.
        self._has_scope_columns: bool = True
        self._embed_dim: int = _EMBED_DIM
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

        # Reflection subsystem (P3) — disabled by default for backwards compat.
        # Enable with LANCEDB_REFLECTION_ENABLED=1.
        self._reflection_store: Optional[ReflectionStore] = None
        self._reflection_event_store: ReflectionEventStore = ReflectionEventStore()
        self._reflection_item_store: ReflectionItemStore = ReflectionItemStore()
        self._reflection_ranker: ReflectionRanker = ReflectionRanker()

    @property
    def name(self) -> str:
        return "lancedb"

    def is_available(self) -> bool:
        provider = get_provider_from_env()
        return is_provider_available(provider)

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

        # P1 scope identifiers (all optional). Read from kwargs first, then
        # env vars (LANCEDB_AGENT_ID, LANCEDB_PROJECT_ID, etc.), else "".
        self._agent_id = (
            kwargs.get("agent_id") or os.environ.get("LANCEDB_AGENT_ID", "")
            or parse_agent_id_from_session_key(session_id) or ""
        )
        self._project_id = kwargs.get("project_id") or os.environ.get("LANCEDB_PROJECT_ID", "")
        self._team_id = kwargs.get("team_id") or os.environ.get("LANCEDB_TEAM_ID", "")
        self._workspace_id = kwargs.get("workspace_id") or os.environ.get("LANCEDB_WORKSPACE_ID", "")

        # Apply CLAWTEAM_MEMORY_SCOPE env extension (no-op if unset)
        self._scope_manager = ScopeManager()
        clawteam = clawteam_scopes_from_env()
        if clawteam:
            self._scope_manager.apply_clawteam_scopes(clawteam)

        # Build the embedder via the multi-provider factory.
        try:
            self._embedder = make_embedder()
            self._embed_dim = self._embedder.dimensions
        except EmbeddingError as e:
            logger.warning("LanceDB embedder init failed: %s", e)
            self._ready = False
            return

        try:
            import lancedb

            os.makedirs(self._storage_path, exist_ok=True)
            self._db = lancedb.connect(self._storage_path)

            if _TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(_TABLE_NAME)
                self._migrate_schema_if_needed()
            else:
                self._table = self._db.create_table(
                    _TABLE_NAME, schema=_get_schema(self._embed_dim)
                )
                self._has_scope_columns = True

            try:
                self._table.create_fts_index("content", replace=True)
            except Exception as e:
                logger.debug("FTS index skipped: %s", e)

            # LLM client still uses OpenAI for extraction/dedup.
            llm_key = os.environ.get("OPENAI_API_KEY", "")
            if llm_key:
                self._llm = _LLMClient(llm_key, model=self._extraction_model)
            else:
                self._llm = None
                logger.debug(
                    "OPENAI_API_KEY not set — LLM extraction/dedup disabled (raw turn fallback only)"
                )

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
            logger.info(
                "LanceDB memory v1.5.0 initialized at %s (provider=%s, dim=%d)",
                self._storage_path,
                get_provider_from_env(),
                self._embed_dim,
            )

            # P3: lazy-init the reflection store. Default off — set
            # LANCEDB_REFLECTION_ENABLED=1 to opt in. Embedder is optional;
            # if absent, the reflection store falls back to FTS-only.
            if os.environ.get("LANCEDB_REFLECTION_ENABLED") == "1":
                self._init_reflection_store()

            self.queue_prefetch("current targets prospects contacts plans tasks decisions")

        except Exception as e:
            logger.warning("LanceDB memory init failed: %s", e, exc_info=True)
            self._ready = False

    # --- Reflection subsystem (P3) ---------------------------------------

    def _init_reflection_store(self) -> None:
        """Create and initialize the dedicated reflection store.

        Embedder is optional — if missing, the reflection store falls back to
        BM25/FTS only. Failures are logged but never block the main provider.
        """
        try:
            embed_fn = self._embedder.embed if self._embedder is not None else None
            self._reflection_store = ReflectionStore(
                storage_path=self._storage_path,
                embedder=embed_fn,
            )
            ok = self._reflection_store.initialize()
            if ok:
                logger.info("LanceDB reflection store initialized at %s", self._storage_path)
            else:
                logger.warning("LanceDB reflection store init returned False; reflections disabled.")
                self._reflection_store = None
        except Exception as e:  # noqa: BLE001
            logger.warning("Reflection store init failed: %s", e, exc_info=True)
            self._reflection_store = None

    def _capture_reflection_at_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Build a reflection-shaped markdown summary from session messages
        and persist it via the reflection store.

        Mirrors the P3 contract: an LLM call extracts lessons-learned in the
        reflection markdown schema (Invariants / Derived / Lessons / etc.)
        and the resulting payloads are written to the dedicated table.
        """
        if (
            self._reflection_store is None
            or not self._reflection_store.is_ready
            or self._llm is None
        ):
            return

        try:
            prompt = _build_reflection_prompt(messages)
            reflection_md = self._llm.chat(_REFLECTION_SYSTEM, prompt, max_tokens=1500)
            if not reflection_md or not reflection_md.strip():
                return
            params = BuildReflectionStorePayloadsParams(
                reflection_text=reflection_md,
                session_key=os.environ.get("LANCEDB_REFLECTION_SESSION_KEY", self._session_id or "session"),
                session_id=self._session_id or "session",
                agent_id=self._user_id,
                command=os.environ.get("LANCEDB_REFLECTION_COMMAND", "session_end"),
                scope=os.environ.get("LANCEDB_REFLECTION_SCOPE", "global"),
                tool_error_signals=[],
                run_at=time.time() * 1000.0,
                used_fallback=False,
                write_legacy_combined=True,
            )
            result = self._reflection_store.write_reflection(params)
            # Mirror to in-memory event/item stores for downstream lookups.
            try:
                from .reflection import build_reflection_event_payload as _build_event
                ev = _build_event(
                    scope=params.scope,
                    session_key=params.session_key,
                    session_id=params.session_id,
                    agent_id=params.agent_id,
                    command=params.command,
                    tool_error_signals=[],
                    run_at=params.run_at,
                    used_fallback=False,
                    event_id=result.get("event_id"),
                )
                self._reflection_event_store.append(ev)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.debug("Reflection capture failed: %s", e)

    def _search_reflections(self, query: str, *, top_k: int = 3) -> List[Dict]:
        """Return top-K reflections for ``query`` as result-shaped dicts.

        Each hit is annotated with ``source: "reflection"`` so callers can
        distinguish reflection-merged results from regular memory results.
        """
        store = self._reflection_store
        if store is None or not store.is_ready or not query:
            return []
        try:
            hits = store.search(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            logger.debug("Reflection search failed: %s", e)
            return []
        out: List[Dict] = []
        for h in hits:
            out.append({
                "id": h.id,
                "content": h.text,
                "abstract": h.text[:80],
                "timestamp": h.timestamp,
                "decay_weight": 1.0,
                "source": "reflection",
                "tier": "reflection",
                "category": "reflection",
                "access_count": 0,
                "importance": h.importance,
                "vector": None,
                "score": h.score,
                "kind": h.kind,
                "scope": h.scope,
                "event_id": h.event_id,
            })
        return out

    def _migrate_schema_if_needed(self) -> None:
        if self._table is None:
            return
        try:
            existing_cols = {f.name for f in self._table.schema}
            # v1.1.0 fields
            v110_fields = {
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
            # v1.3.0 P1 — multi-scope columns
            scope_fields = dict(SCOPE_COLUMN_DEFAULTS)
            new_fields = {**v110_fields, **scope_fields}
            missing = {k: v for k, v in new_fields.items() if k not in existing_cols}
            if missing:
                self._table.add_columns(missing)
                logger.info("LanceDB schema migrated: added columns %s", list(missing.keys()))
            # Determine whether scope columns are now present (either pre-existing
            # or just-added). Used to choose between legacy `user_id`-only
            # filtering and the composable scope filter.
            updated_cols = {f.name for f in self._table.schema}
            self._has_scope_columns = all(c in updated_cols for c in SCOPE_COLUMNS)
        except Exception as e:
            logger.warning("LanceDB schema migration failed (reads will use defaults): %s", e)
            # Be conservative — without confirmed migration, assume legacy schema.
            try:
                cols = {f.name for f in self._table.schema}
                self._has_scope_columns = all(c in cols for c in SCOPE_COLUMNS)
            except Exception:
                self._has_scope_columns = False

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

    def _build_scope_where(self) -> str:
        """Compose the SQL WHERE predicate for the active scope identifiers."""
        return self._scope_manager.build_where_clause(
            agent_id=self._agent_id or None,
            user_id=self._user_id or None,
            project_id=self._project_id or None,
            team_id=self._team_id or None,
            workspace_id=self._workspace_id or None,
            scope_columns_present=self._has_scope_columns,
            legacy_user_id=self._user_id or None,
        )

    def _hybrid_search(self, query: str, top_k: int = _TOP_K_RETURN) -> List[Dict]:
        if not self._ready or self._table is None or self._embedder is None:
            return []
        try:
            vec = self._embedder.embed(query)
            where_clause = self._build_scope_where()
            v_search = self._table.search(vec, vector_column_name="vector")
            if where_clause:
                v_search = v_search.where(where_clause, prefilter=True)
            v_results = v_search.limit(_TOP_K_VECTOR).to_list()
            try:
                b_search = self._table.search(query, query_type="fts")
                if where_clause:
                    b_search = b_search.where(where_clause, prefilter=True)
                b_results = b_search.limit(_TOP_K_BM25).to_list()
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

            # Stage 9 (P3): Optionally pull top-K reflections via the
            # reflection store and merge them in. Reflections carry
            # source="reflection" so consumers can highlight them.
            if self._reflection_store is not None and self._reflection_store.is_ready:
                try:
                    refl_top_k = max(1, int(os.environ.get("LANCEDB_REFLECTION_TOP_K", "3")))
                except (TypeError, ValueError):
                    refl_top_k = 3
                refl_hits = self._search_reflections(query, top_k=refl_top_k)
                if refl_hits:
                    # Reflections aren't on the same RRF score scale; preserve
                    # them by interleaving at the tail of the kept window.
                    seen_ids = {h.get("id") for h in merged}
                    refl_hits = [h for h in refl_hits if h.get("id") not in seen_ids]
                    merged = merged + refl_hits

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
            where_clause = self._build_scope_where()
            search = self._table.search(vec, vector_column_name="vector")
            if where_clause:
                search = search.where(where_clause, prefilter=True)
            results = search.limit(1).to_list()
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

                    row = {
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
                    }
                    # P1 — populate scope columns when present.
                    if self._has_scope_columns:
                        row["agent_id"] = entry.get("agent_id", self._agent_id) or ""
                        row["project_id"] = entry.get("project_id", self._project_id) or ""
                        row["team_id"] = entry.get("team_id", self._team_id) or ""
                        row["workspace_id"] = entry.get("workspace_id", self._workspace_id) or ""
                        canonical = entry.get("scope")
                        if not canonical:
                            if row["agent_id"]:
                                canonical = f"agent:{row['agent_id']}"
                            elif row["user_id"]:
                                canonical = f"user:{row['user_id']}"
                            else:
                                canonical = GLOBAL_SCOPE
                        row["scope"] = canonical
                    rows.append(row)
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

            # P3: capture a reflection alongside the regular extraction.
            try:
                self._capture_reflection_at_session_end(messages)
            except Exception as e:  # noqa: BLE001
                logger.debug("Reflection capture (session-end) failed: %s", e)

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
            {
                "name": "lancedb_reflect",
                "description": (
                    "Explicitly write a reflection (markdown with ## Invariants / ## Derived / "
                    "## Lessons sections) to the dedicated reflection store. Use when you have "
                    "a session-summary insight worth persisting separately from regular memory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection_text": {
                            "type": "string",
                            "description": "Markdown reflection in the canonical format.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "Scope tag (default 'global').",
                        },
                        "command": {
                            "type": "string",
                            "description": "Command/event name (default 'manual').",
                        },
                    },
                    "required": ["reflection_text"],
                },
            },
            {
                "name": "lancedb_reflections",
                "description": (
                    "Search ONLY the reflection store (separate from regular memories). "
                    "Returns invariants, derived deltas, lessons, and decisions tagged with the "
                    "originating session/event."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        "top_k": {"type": "integer", "description": "Max results (default 6, max 20)."},
                    },
                    "required": ["query"],
                },
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
                where_clause = self._build_scope_where()
                search = self._table.search()
                if where_clause:
                    search = search.where(where_clause, prefilter=True)
                rows = search.limit(10000).to_list()
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

        if tool_name == "lancedb_reflect":
            text = (args.get("reflection_text") or "").strip()
            if not text:
                return json.dumps({"error": "reflection_text is required"})
            if self._reflection_store is None or not self._reflection_store.is_ready:
                return json.dumps({
                    "error": "reflection store not enabled",
                    "hint": "set LANCEDB_REFLECTION_ENABLED=1 and re-initialize",
                })
            params = BuildReflectionStorePayloadsParams(
                reflection_text=text,
                session_key=self._session_id or "session",
                session_id=self._session_id or "session",
                agent_id=self._user_id,
                command=args.get("command", "manual"),
                scope=args.get("scope", "global"),
                tool_error_signals=[],
                run_at=time.time() * 1000.0,
                used_fallback=False,
            )
            try:
                result = self._reflection_store.write_reflection(params)
                return json.dumps({
                    "stored": result.get("stored", False),
                    "event_id": result.get("event_id", ""),
                    "stored_kinds": result.get("stored_kinds", []),
                    "ids": result.get("ids", []),
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        if tool_name == "lancedb_reflections":
            if self._reflection_store is None or not self._reflection_store.is_ready:
                return json.dumps({
                    "error": "reflection store not enabled",
                    "hint": "set LANCEDB_REFLECTION_ENABLED=1 and re-initialize",
                })
            query = (args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "query is required"})
            try:
                top_k = min(int(args.get("top_k", 6)), 20)
            except (TypeError, ValueError):
                top_k = 6
            try:
                hits = self._reflection_store.search(query, top_k=top_k)
            except Exception as e:
                return json.dumps({"error": str(e)})
            return json.dumps({
                "results": [
                    {
                        "id": h.id,
                        "text": h.text[:500],
                        "kind": h.kind,
                        "scope": h.scope,
                        "score": round(float(h.score), 4),
                        "importance": round(float(h.importance), 2),
                        "event_id": h.event_id,
                        "session_id": h.session_id,
                        "age_days": round(_age_days(h.timestamp / 1000.0 if h.timestamp > 1e10 else h.timestamp), 1),
                    }
                    for h in hits
                ],
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._extract_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=8.0)
        logger.info("LanceDB memory shut down.")


__all__ = [
    "LanceDBMemoryProvider",
    "MEMORY_CATEGORIES",
    # P1 re-exports for downstream consumers and tests
    "Embedder",
    "EmbeddingError",
    "ScopeManager",
    "make_embedder",
    "get_provider_from_env",
    "is_provider_available",
    "EMBEDDING_DIMENSIONS",
    "PROVIDER_DEFAULT_MODEL",
    "GLOBAL_SCOPE",
    "SCOPE_COLUMNS",
    "SCOPE_COLUMN_DEFAULTS",
    "parse_agent_id_from_session_key",
    "clawteam_scopes_from_env",
    # P2 modules re-exported for convenience
    "AdmissionController",
    "NoisePrototypeFilter",
    "batch_dedup",
    "chunk_text",
    "extract_smart_metadata",
    "stringify_smart_metadata",
    # Reflection subsystem (P3) — re-exported for convenient top-level access.
    "ReflectionStore",
    "ReflectionEventStore",
    "ReflectionItemStore",
    "ReflectionRanker",
    "BuildReflectionStorePayloadsParams",
    "ReflectionErrorSignalLike",
]
