"""Batch dedup — port of TS `src/batch-dedup.ts` plus an LLM-batch variant.

Two layers:

1. `cosine_batch_dedup(abstracts, vectors, threshold=0.85)`
     Pure vector pairwise pass that drops candidates whose embedded abstracts
     are near-duplicates of an earlier candidate in the same batch. Cheap and
     deterministic — exactly mirrors the TS implementation.

2. `batch_dedup(candidates, existing_pool, llm, embedder=None)`
     Full batch decision pass. For each candidate we compute a per-candidate
     decision against an `existing_pool` of prior memories using ONE LLM call
     (instead of one call per pair, which is what `_llm_dedup` in __init__.py
     does today). The model returns an array of decisions parallel to the
     candidate list. Falls back to per-candidate decisions on parse failure.

    Each returned dict has shape:
        {
          "index": int,           # position in input `candidates`
          "decision": str,        # one of: skip|support|merge|contextualize|
                                  # contradict|supersede|create
          "merged_content": str,  # only populated for merge / supersede
          "matched_existing_id": Optional[str],  # which existing memory matched
        }
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_THRESHOLD = 0.85

VALID_DECISIONS = {
    "skip",
    "support",
    "merge",
    "contextualize",
    "contradict",
    "supersede",
    "create",
}

_BATCH_DEDUP_SYSTEM = """You are a memory deduplication system. You will receive an array of candidate \
memories and a small pool of existing memories. For EACH candidate, decide what to do.

Decisions (use exactly one per candidate):
- skip: candidate adds nothing new; covered by an existing memory or another candidate
- support: candidate reinforces an existing memory (minor variant)
- merge: combine candidate with a matched existing memory into one improved entry
- contextualize: candidate adds nuance and should be kept separately
- contradict: candidate contradicts an existing memory; flag the conflict
- supersede: candidate replaces an existing memory (more recent / accurate)
- create: candidate is sufficiently new; create as a new entry

Return ONLY a valid JSON array of objects, one per candidate, in the same order:
  {"index": <int>, "decision": "<decision>", "merged_content": "<text or empty>", "matched_existing_id": "<id or empty>"}
"""


# ---------------------------------------------------------------------------
# Cosine sim helpers (kept in-module so this file has no other deps)
# ---------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
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


# ---------------------------------------------------------------------------
# Layer 1: pairwise cosine dedup within a batch
# ---------------------------------------------------------------------------

@dataclass
class CosineBatchDedupResult:
    surviving_indices: List[int] = field(default_factory=list)
    duplicate_indices: List[int] = field(default_factory=list)
    duplicate_of: Dict[int, int] = field(default_factory=dict)
    input_count: int = 0
    output_count: int = 0


def cosine_batch_dedup(
    abstracts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    threshold: float = DEFAULT_BATCH_THRESHOLD,
) -> CosineBatchDedupResult:
    """Mark later candidates as batch-duplicates of an earlier one when cosine > threshold."""
    n = len(abstracts)
    if n <= 1:
        return CosineBatchDedupResult(
            surviving_indices=[0] if n == 1 else [],
            duplicate_indices=[],
            duplicate_of={},
            input_count=n,
            output_count=n,
        )

    is_duplicate = [False] * n
    duplicate_of: Dict[int, int] = {}

    for i in range(n):
        if is_duplicate[i]:
            continue
        for j in range(i + 1, n):
            if is_duplicate[j]:
                continue
            vi = vectors[i] if i < len(vectors) else None
            vj = vectors[j] if j < len(vectors) else None
            if not vi or not vj:
                continue
            sim = _cosine(vi, vj)
            if sim > threshold:
                is_duplicate[j] = True
                duplicate_of[j] = i

    surviving = [i for i, dup in enumerate(is_duplicate) if not dup]
    duplicates = [i for i, dup in enumerate(is_duplicate) if dup]
    return CosineBatchDedupResult(
        surviving_indices=surviving,
        duplicate_indices=duplicates,
        duplicate_of=duplicate_of,
        input_count=n,
        output_count=len(surviving),
    )


# ---------------------------------------------------------------------------
# Layer 2: LLM batch decision pass
# ---------------------------------------------------------------------------

def _build_batch_dedup_prompt(
    candidates: Sequence[Dict[str, Any]],
    existing_pool: Sequence[Dict[str, Any]],
) -> str:
    cand_lines = []
    for i, c in enumerate(candidates):
        ab = (c.get("abstract") or "")[:160]
        ct = (c.get("content") or c.get("text") or "")[:600]
        cat = c.get("category", "?")
        cand_lines.append(f"[{i}] category={cat} abstract={ab!r}\n     content={ct!r}")

    pool_lines = []
    for ex in existing_pool[:10]:
        eid = ex.get("id", "")
        ab = (ex.get("abstract") or "")[:160]
        ct = (ex.get("content") or "")[:400]
        pool_lines.append(f"id={eid!r} abstract={ab!r}\n  content={ct!r}")

    return (
        "Existing memory pool (top matches by vector similarity):\n"
        + ("\n".join(pool_lines) or "  (empty)")
        + "\n\nCandidate batch:\n"
        + "\n".join(cand_lines)
    )


def _normalize_decision(raw: Any) -> str:
    if not isinstance(raw, str):
        return "create"
    d = raw.strip().lower()
    return d if d in VALID_DECISIONS else "create"


def batch_dedup(
    candidates: Sequence[Dict[str, Any]],
    existing_pool: Sequence[Dict[str, Any]],
    llm: Optional[Any],
    embedder: Optional[Callable[[str], Sequence[float]]] = None,
    cosine_threshold: float = DEFAULT_BATCH_THRESHOLD,
    fallback_decision: str = "create",
) -> List[Dict[str, Any]]:
    """Run the two-stage batch dedup.

    Returns a list of decision dicts parallel to `candidates`. If the LLM call
    fails or returns malformed JSON, every candidate falls back to
    `fallback_decision` (default `create`) — keeping the pipeline non-blocking.

    Layer 1 (cosine pairwise) marks within-batch duplicates as `skip`.
    Layer 2 (LLM batch) decides on the remaining candidates vs the pool.
    """
    decisions: List[Dict[str, Any]] = []
    if not candidates:
        return decisions

    # Layer 1: within-batch cosine dedup if we have an embedder.
    survivor_set = set(range(len(candidates)))
    if embedder is not None:
        try:
            abstracts = [c.get("abstract") or c.get("content") or "" for c in candidates]
            vectors = [list(embedder(a)) if a else [] for a in abstracts]
            cos_result = cosine_batch_dedup(abstracts, vectors, threshold=cosine_threshold)
            survivor_set = set(cos_result.surviving_indices)
            for dup_idx, src_idx in cos_result.duplicate_of.items():
                decisions.append({
                    "index": dup_idx,
                    "decision": "skip",
                    "merged_content": "",
                    "matched_existing_id": None,
                    "duplicate_of_candidate": src_idx,
                })
        except Exception as e:
            logger.debug("cosine_batch_dedup failed: %s", e)

    # Layer 2: LLM batch decision over survivors vs pool.
    survivors = [c for i, c in enumerate(candidates) if i in survivor_set]
    survivor_indices = [i for i in range(len(candidates)) if i in survivor_set]

    if survivors and llm is not None and existing_pool:
        try:
            user_prompt = _build_batch_dedup_prompt(survivors, existing_pool)
            raw = llm.chat(_BATCH_DEDUP_SYSTEM, user_prompt, max_tokens=800)
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("expected list")
            # Map by relative index
            llm_decisions: Dict[int, Dict[str, Any]] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                idx_raw = item.get("index")
                if not isinstance(idx_raw, int):
                    continue
                if idx_raw < 0 or idx_raw >= len(survivors):
                    continue
                llm_decisions[idx_raw] = {
                    "decision": _normalize_decision(item.get("decision")),
                    "merged_content": str(item.get("merged_content") or ""),
                    "matched_existing_id": (
                        str(item.get("matched_existing_id"))
                        if item.get("matched_existing_id")
                        else None
                    ),
                }
            for rel_idx, abs_idx in enumerate(survivor_indices):
                d = llm_decisions.get(rel_idx, {
                    "decision": fallback_decision,
                    "merged_content": "",
                    "matched_existing_id": None,
                })
                decisions.append({
                    "index": abs_idx,
                    "decision": d["decision"],
                    "merged_content": d["merged_content"],
                    "matched_existing_id": d["matched_existing_id"],
                })
        except Exception as e:
            logger.debug("batch_dedup LLM call failed: %s — defaulting", e)
            for abs_idx in survivor_indices:
                decisions.append({
                    "index": abs_idx,
                    "decision": fallback_decision,
                    "merged_content": "",
                    "matched_existing_id": None,
                })
    else:
        # No LLM or no existing pool → everyone surviving creates.
        for abs_idx in survivor_indices:
            decisions.append({
                "index": abs_idx,
                "decision": fallback_decision,
                "merged_content": "",
                "matched_existing_id": None,
            })

    decisions.sort(key=lambda d: d["index"])
    return decisions


__all__ = [
    "DEFAULT_BATCH_THRESHOLD",
    "VALID_DECISIONS",
    "CosineBatchDedupResult",
    "cosine_batch_dedup",
    "batch_dedup",
]
