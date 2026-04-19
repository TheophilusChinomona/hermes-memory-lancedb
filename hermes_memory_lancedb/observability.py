"""Observability for the retrieval pipeline.

Ports `src/retrieval-stats.ts` + `src/retrieval-trace.ts` from the TS plugin.

`RetrievalTrace` records per-stage timings, input/output counts, dropped IDs,
and score ranges as a query flows through the pipeline. It is opt-in: pass
``trace=RetrievalTrace()`` to ``_hybrid_search`` to capture diagnostics, or
pass ``None`` for zero overhead.

`RetrievalStats` is a ring-buffer collector of completed traces. It exposes
``record_query``/``get_stats``/``reset`` and produces aggregate counters:
total queries, zero-result queries, average + p95 latency, average result
count, rerank usage, noise filter activations, queries by source, and the
top stages that drop the most entries.

Both classes are thread-safe enough for the provider's background access-bump
threads (a single ``threading.Lock`` guards stats writes).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------------


class StageResult:
    """Result of a single pipeline stage.

    Attributes mirror the TS interface ``RetrievalStageResult`` plus a small
    convenience ``dropped`` property.
    """

    __slots__ = (
        "name",
        "input_count",
        "output_count",
        "dropped_ids",
        "score_range",
        "duration_ms",
    )

    def __init__(
        self,
        name: str,
        input_count: int,
        output_count: int,
        dropped_ids: List[str],
        score_range: Optional[Tuple[float, float]],
        duration_ms: float,
    ):
        self.name = name
        self.input_count = input_count
        self.output_count = output_count
        self.dropped_ids = dropped_ids
        self.score_range = score_range
        self.duration_ms = duration_ms

    @property
    def dropped(self) -> int:
        return self.input_count - self.output_count

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "dropped_ids": list(self.dropped_ids),
            "score_range": list(self.score_range) if self.score_range else None,
            "duration_ms": round(self.duration_ms, 3),
        }


# ---------------------------------------------------------------------------
# RetrievalTrace
# ---------------------------------------------------------------------------


class RetrievalTrace:
    """Pipeline trace built up stage-by-stage.

    Usage:

        trace = RetrievalTrace()
        trace.start_stage("vector_search", input_ids=[])
        # ... do work ...
        trace.end_stage(surviving_ids=[h["id"] for h in v_hits])

        ...

        result = trace.finalize(query="foo", mode="hybrid")
    """

    def __init__(self):
        self._start_time = time.time()
        self._stages: List[StageResult] = []
        self._pending: Optional[Dict] = None
        self.query: str = ""
        self.mode: str = "hybrid"
        self._final_count: int = 0
        self._total_ms: float = 0.0
        self._finalized: bool = False

    # ----- stage tracking -------------------------------------------------

    def start_stage(self, name: str, input_ids: Sequence[str]) -> None:
        # Defensive: auto-close any unclosed stage.
        if self._pending is not None:
            self.end_stage(list(self._pending["input_ids"]))
        self._pending = {
            "name": name,
            "input_ids": set(input_ids),
            "start_time": time.time(),
        }

    def end_stage(
        self,
        surviving_ids: Sequence[str],
        scores: Optional[Sequence[float]] = None,
    ) -> None:
        if self._pending is None:
            return
        name = self._pending["name"]
        input_ids = self._pending["input_ids"]
        start_time = self._pending["start_time"]
        surviving_set = set(surviving_ids)

        dropped = [i for i in input_ids if i not in surviving_set]

        score_range: Optional[Tuple[float, float]] = None
        if scores:
            mn = math.inf
            mx = -math.inf
            for s in scores:
                if s < mn:
                    mn = s
                if s > mx:
                    mx = s
            if mn != math.inf:
                score_range = (mn, mx)

        self._stages.append(
            StageResult(
                name=name,
                input_count=len(input_ids),
                output_count=len(surviving_ids),
                dropped_ids=dropped,
                score_range=score_range,
                duration_ms=(time.time() - start_time) * 1000.0,
            )
        )
        self._pending = None

    # ----- finalize -------------------------------------------------------

    def finalize(self, query: str, mode: str = "hybrid") -> "RetrievalTrace":
        if self._pending is not None:
            self.end_stage(list(self._pending["input_ids"]))
        self.query = query
        self.mode = mode
        last = self._stages[-1] if self._stages else None
        self._final_count = last.output_count if last else 0
        self._total_ms = (time.time() - self._start_time) * 1000.0
        self._finalized = True
        return self

    # ----- accessors ------------------------------------------------------

    @property
    def stages(self) -> List[StageResult]:
        return list(self._stages)

    @property
    def final_count(self) -> int:
        return self._final_count

    @property
    def total_ms(self) -> float:
        return self._total_ms

    @property
    def started_at(self) -> float:
        return self._start_time

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "started_at": self._start_time,
            "stages": [s.to_dict() for s in self._stages],
            "final_count": self._final_count,
            "total_ms": round(self._total_ms, 3),
        }

    def summarize(self) -> str:
        """Human-readable pipeline summary, one line per stage.

        Output looks like::

            Retrieval trace (5 stages):
              vector_search: 0 -> 12 (-0) 18ms scores=[0.40, 0.92]
              bm25: 12 -> 8 (-4) 9ms
              ...
              total: 446ms, final count: 6
        """
        lines = [f"Retrieval trace ({len(self._stages)} stages):"]
        for s in self._stages:
            score_str = (
                f" scores=[{s.score_range[0]:.3f}, {s.score_range[1]:.3f}]"
                if s.score_range
                else ""
            )
            lines.append(
                f"  {s.name}: {s.input_count} -> {s.output_count} "
                f"(-{s.dropped}) {s.duration_ms:.0f}ms{score_str}"
            )
            if 0 < len(s.dropped_ids) <= 5:
                lines.append(f"    dropped: {', '.join(s.dropped_ids)}")
            elif len(s.dropped_ids) > 5:
                head = ", ".join(s.dropped_ids[:5])
                lines.append(
                    f"    dropped: {head} (+{len(s.dropped_ids) - 5} more)"
                )
        total_ms = self._total_ms if self._finalized else (time.time() - self._start_time) * 1000
        last = self._stages[-1] if self._stages else None
        final = last.output_count if last else 0
        lines.append(f"  total: {total_ms:.0f}ms, final count: {final}")
        return "\n".join(lines)

    def compact_summary(self) -> str:
        """One-line stage chain, e.g.

        ``vector_search:18ms,12rows -> bm25:9ms,8rows -> rrf:1ms,15rows -> final:6``
        """
        parts = [
            f"{s.name}:{s.duration_ms:.0f}ms,{s.output_count}rows"
            for s in self._stages
        ]
        parts.append(f"final:{self._final_count}")
        return " -> ".join(parts)


# ---------------------------------------------------------------------------
# RetrievalStats — aggregate ring-buffer collector
# ---------------------------------------------------------------------------


class RetrievalStats:
    """Ring-buffer aggregate stats over completed traces.

    The buffer is bounded (default 1000) so long-running processes don't grow
    unbounded. Writes are O(1).
    """

    def __init__(self, max_records: int = 1000):
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._max = max_records
        self._records: List[Optional[Dict]] = [None] * max_records
        self._head = 0
        self._count = 0
        self._lock = threading.Lock()

    # ----- recording ------------------------------------------------------

    def record_query(self, trace: RetrievalTrace, source: str = "unknown") -> None:
        """Record a finalized trace and the source that triggered it."""
        with self._lock:
            self._records[self._head] = {"trace": trace, "source": source}
            self._head = (self._head + 1) % self._max
            if self._count < self._max:
                self._count += 1

    def reset(self) -> None:
        with self._lock:
            self._records = [None] * self._max
            self._head = 0
            self._count = 0

    @property
    def count(self) -> int:
        return self._count

    # ----- aggregate ------------------------------------------------------

    def get_stats(self) -> Dict:
        with self._lock:
            n = self._count
            if n == 0:
                return {
                    "total_queries": 0,
                    "zero_result_queries": 0,
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "avg_result_count": 0.0,
                    "rerank_used": 0,
                    "noise_filtered": 0,
                    "queries_by_source": {},
                    "top_drop_stages": [],
                    "top_k_distribution": {},
                }

            total_latency = 0.0
            total_results = 0
            zero = 0
            rerank_used = 0
            noise_filtered = 0
            latencies: List[float] = []
            by_source: Dict[str, int] = {}
            drops: Dict[str, int] = {}
            top_k_dist: Dict[str, int] = {}

            start = 0 if n < self._max else self._head
            for i in range(n):
                rec = self._records[(start + i) % self._max]
                if rec is None:
                    continue
                trace: RetrievalTrace = rec["trace"]
                source: str = rec["source"]

                total_latency += trace.total_ms
                total_results += trace.final_count
                latencies.append(trace.total_ms)
                if trace.final_count == 0:
                    zero += 1

                by_source[source] = by_source.get(source, 0) + 1
                bucket = _bucket_size(trace.final_count)
                top_k_dist[bucket] = top_k_dist.get(bucket, 0) + 1

                for stage in trace.stages:
                    drop = stage.dropped
                    if drop > 0:
                        drops[stage.name] = drops.get(stage.name, 0) + drop
                    if stage.name == "rerank":
                        rerank_used += 1
                    if stage.name == "noise_filter" and drop > 0:
                        noise_filtered += 1

            latencies.sort()
            p95_index = min(max(0, math.ceil(n * 0.95) - 1), n - 1)

            top_drops = sorted(
                ({"name": k, "total_dropped": v} for k, v in drops.items()),
                key=lambda x: x["total_dropped"],
                reverse=True,
            )[:5]

            return {
                "total_queries": n,
                "zero_result_queries": zero,
                "avg_latency_ms": round(total_latency / n),
                "p95_latency_ms": round(latencies[p95_index]),
                "avg_result_count": round(total_results / n, 1),
                "rerank_used": rerank_used,
                "noise_filtered": noise_filtered,
                "queries_by_source": by_source,
                "top_drop_stages": top_drops,
                "top_k_distribution": top_k_dist,
            }


def _bucket_size(n: int) -> str:
    """Bucket result counts into coarse top-k size bands for the histogram."""
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    if n <= 20:
        return "11-20"
    return "21+"


__all__ = ["RetrievalTrace", "RetrievalStats", "StageResult"]
