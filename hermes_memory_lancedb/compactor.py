"""Memory compactor — progressive summarization of clusters.

Ports ``memory-compactor.ts``: periodically scans memories, builds clusters
of near-duplicate / topically related entries by cosine similarity, and merges
each cluster into one denser entry. Source entries are deleted, freeing rows
and improving recall quality.

Two ways to run:

- :func:`compact_memories` — pure orchestrator. Takes a duck-typed store and
  optional embedder/LLM, returns a :class:`CompactionResult`.
- ``LanceDBMemoryProvider.handle_tool_call("lancedb_compact", ...)`` — exposes
  this to agents via the ``lancedb_compact`` tool.

Heuristic auto-trigger: write paths in ``__init__.py`` increment a counter
and call :func:`maybe_run` every ``LANCEDB_COMPACT_EVERY_N`` writes
(default 100, env-tunable).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


SECONDS_PER_DAY = 86_400.0


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass
class CompactionConfig:
    enabled: bool = True
    min_age_days: float = 7.0
    similarity_threshold: float = 0.88
    min_cluster_size: int = 2
    max_memories_to_scan: int = 200
    dry_run: bool = False
    cooldown_hours: float = 24.0


@dataclass
class CompactionResult:
    scanned: int = 0
    clusters_found: int = 0
    memories_deleted: int = 0
    memories_created: int = 0
    dry_run: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "clusters_found": self.clusters_found,
            "memories_deleted": self.memories_deleted,
            "memories_created": self.memories_created,
            "dry_run": self.dry_run,
            "error": self.error,
        }


@dataclass
class _Cluster:
    member_indices: List[int]
    merged: Dict[str, Any]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
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
    val = dot / (math.sqrt(na) * math.sqrt(nb))
    return max(0.0, min(1.0, val))


# ---------------------------------------------------------------------------
# Cluster building + merging
# ---------------------------------------------------------------------------

def _build_clusters(
    entries: List[Dict[str, Any]],
    threshold: float,
    min_cluster_size: int,
) -> List[_Cluster]:
    if len(entries) < min_cluster_size:
        return []

    order = sorted(range(len(entries)), key=lambda i: -float(entries[i].get("importance", 0.5)))
    assigned = [False] * len(entries)
    plans: List[_Cluster] = []

    for seed_idx in order:
        if assigned[seed_idx]:
            continue
        seed_vec = entries[seed_idx].get("vector")
        if seed_vec is None or len(seed_vec) == 0:
            assigned[seed_idx] = True
            continue
        cluster = [seed_idx]
        assigned[seed_idx] = True
        seed_list = list(seed_vec)
        for j in range(len(entries)):
            if assigned[j]:
                continue
            jv = entries[j].get("vector")
            if jv is None or len(jv) == 0:
                continue
            if cosine_similarity(seed_list, list(jv)) >= threshold:
                cluster.append(j)
                assigned[j] = True
        if len(cluster) >= min_cluster_size:
            members = [entries[i] for i in cluster]
            plans.append(_Cluster(member_indices=cluster, merged=_build_merged(members)))
    return plans


def _build_merged(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Deduplicate lines across all members (case-insensitive).
    seen = set()
    lines: List[str] = []
    for m in members:
        for raw in str(m.get("content", "")).split("\n"):
            t = raw.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                lines.append(t)
    text = "\n".join(lines)

    # Importance: max across cluster, capped at 1.0.
    importance = min(1.0, max(float(m.get("importance", 0.5)) for m in members))

    # Category: plurality vote, ties broken by importance.
    counts: Dict[str, float] = {}
    cat_imp: Dict[str, float] = {}
    for m in members:
        cat = m.get("category") or "cases"
        counts[cat] = counts.get(cat, 0) + 1
        cat_imp[cat] = max(cat_imp.get(cat, 0.0), float(m.get("importance", 0.5)))
    if counts:
        # Sort by count desc, then importance desc, then name for determinism.
        category = sorted(counts.keys(), key=lambda c: (-counts[c], -cat_imp[c], c))[0]
    else:
        category = "cases"

    # Tier: prefer the highest tier present (core > working > peripheral).
    tier_rank = {"core": 2, "working": 1, "peripheral": 0}
    tier = max(
        (m.get("tier") or "peripheral" for m in members),
        key=lambda t: tier_rank.get(t, 0),
    )

    # Tags: union, capped.
    tags: List[str] = []
    seen_tags = set()
    for m in members:
        raw_tags = m.get("tags", [])
        if isinstance(raw_tags, str):
            try:
                raw_tags = json.loads(raw_tags)
            except Exception:
                raw_tags = []
        for t in raw_tags or []:
            ts = str(t)[:40]
            if ts and ts not in seen_tags:
                seen_tags.add(ts)
                tags.append(ts)
        if len(tags) >= 16:
            break
    tags.append("compacted")

    abstract = (lines[0] if lines else "")[:160]

    return {
        "content": text,
        "importance": importance,
        "category": category,
        "tier": tier,
        "abstract": abstract,
        "overview": "",
        "tags": tags,
        "compacted": True,
        "source_count": len(members),
    }


# ---------------------------------------------------------------------------
# Store-shaped helpers (duck-typed interface)
# ---------------------------------------------------------------------------

def _fetch_for_compaction(
    store: Any, max_timestamp: float, limit: int, user_id: str
) -> List[Dict[str, Any]]:
    """Read candidate rows from the store.

    Prefers a ``fetch_for_compaction`` method on the store; otherwise queries
    the underlying ``_table`` directly.
    """
    fetch = getattr(store, "fetch_for_compaction", None)
    if callable(fetch):
        try:
            return fetch(max_timestamp=max_timestamp, limit=limit, user_id=user_id) or []
        except Exception as e:
            logger.debug("Compactor: store.fetch_for_compaction failed: %s", e)
            return []
    table = getattr(store, "_table", None)
    if table is None:
        return []
    try:
        where = f"user_id = '{user_id}' AND timestamp <= {max_timestamp}"
        rows = table.search().where(where, prefilter=True).limit(limit).to_list()
        return rows or []
    except Exception as e:
        logger.debug("Compactor: direct table fetch failed: %s", e)
        return []


def _delete_id(store: Any, mem_id: str) -> bool:
    deleter = getattr(store, "delete_memory", None)
    if callable(deleter):
        try:
            return bool(deleter(mem_id))
        except Exception as e:
            logger.debug("Compactor: store.delete_memory failed: %s", e)
            return False
    table = getattr(store, "_table", None)
    if table is None:
        return False
    try:
        table.delete(f"id = '{mem_id}'")
        return True
    except Exception as e:
        logger.debug("Compactor: direct table delete failed: %s", e)
        return False


def _write_merged(store: Any, merged: Dict[str, Any], session_id: str, user_id: str) -> bool:
    writer = getattr(store, "_write_entries", None)
    if not callable(writer):
        return False
    entry = {
        "content": merged["content"],
        "source": "compactor",
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": time.time(),
        "tags": merged.get("tags", []),
        "category": merged.get("category", "cases"),
        "importance": merged.get("importance", 0.5),
        "tier": merged.get("tier", "working"),
        "abstract": merged.get("abstract", ""),
        "overview": merged.get("overview", ""),
        "temporal_type": "static",
    }
    try:
        writer([entry])
        return True
    except Exception as e:
        logger.debug("Compactor: store._write_entries failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compact_memories(
    store: Any,
    llm=None,  # accepted for API symmetry; not used in the heuristic merge
    *,
    config: Optional[CompactionConfig] = None,
    session_id: str = "",
    user_id: str = "andrew",
    max_iterations: int = 1,
) -> CompactionResult:
    """Run compaction passes against ``store``.

    Multiple iterations let large clusters be compacted further if newly
    merged entries themselves form clusters with what's left.
    """
    cfg = config or CompactionConfig()
    if not cfg.enabled:
        return CompactionResult(dry_run=cfg.dry_run)

    overall = CompactionResult(dry_run=cfg.dry_run)

    for _ in range(max(1, max_iterations)):
        cutoff = time.time() - cfg.min_age_days * SECONDS_PER_DAY
        entries = _fetch_for_compaction(store, cutoff, cfg.max_memories_to_scan, user_id)
        valid = [e for e in entries if e.get("vector") and len(e["vector"]) > 0]
        overall.scanned += len(valid)

        if len(valid) < cfg.min_cluster_size:
            break

        plans = _build_clusters(valid, cfg.similarity_threshold, cfg.min_cluster_size)
        overall.clusters_found += len(plans)
        if not plans:
            break

        if cfg.dry_run:
            break

        any_progress = False
        for plan in plans:
            members = [valid[i] for i in plan.member_indices]
            try:
                if not _write_merged(store, plan.merged, session_id=session_id, user_id=user_id):
                    continue
                overall.memories_created += 1
                for m in members:
                    mid = m.get("id", "")
                    if mid and _delete_id(store, mid):
                        overall.memories_deleted += 1
                any_progress = True
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Compactor: cluster of %d failed: %s", len(members), e)

        if not any_progress:
            break

    return overall


# ---------------------------------------------------------------------------
# Auto-trigger helpers (call from write paths)
# ---------------------------------------------------------------------------

class CompactionTrigger:
    """Tracks how many writes have happened and fires :func:`compact_memories`
    once a threshold is reached. Thread-safe; runs the actual compaction in
    a background thread to avoid blocking the write path."""

    def __init__(self, every_n: int = 100):
        self.every_n = max(1, every_n)
        self._counter = 0
        self._lock = threading.Lock()
        self._last_run = 0.0
        self._thread: Optional[threading.Thread] = None

    def bump(
        self,
        store: Any,
        *,
        user_id: str = "andrew",
        session_id: str = "",
        cooldown_seconds: float = 60.0,
        config: Optional[CompactionConfig] = None,
    ) -> bool:
        """Returns True if a compaction run was triggered."""
        with self._lock:
            self._counter += 1
            if self._counter < self.every_n:
                return False
            now = time.time()
            if now - self._last_run < cooldown_seconds:
                return False
            self._counter = 0
            self._last_run = now
            if self._thread and self._thread.is_alive():
                return False

        def _run():
            try:
                compact_memories(
                    store,
                    config=config,
                    session_id=session_id,
                    user_id=user_id,
                    max_iterations=1,
                )
            except Exception as e:  # pragma: no cover
                logger.debug("Compaction trigger run failed: %s", e)

        t = threading.Thread(target=_run, name="lancedb-compactor", daemon=True)
        with self._lock:
            self._thread = t
        t.start()
        return True


__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "CompactionTrigger",
    "compact_memories",
    "cosine_similarity",
]
