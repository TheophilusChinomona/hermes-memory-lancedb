"""Admission control — port of TS `src/admission-control.ts` + `admission-stats.ts`.

Gates writes based on rolling stats so a noisy session can't flood the store.

What we keep from the TS port:
  - Per-category type priors (profile/preferences/entities/events/cases/patterns).
  - Reject vs admit thresholds with a configurable preset.
  - Novelty penalty: cosine similarity vs recent rejects (and recent admits).
  - Recency gap scoring against the most recent matched memory.

What we add (new for the Python port):
  - Persisted rolling stats (acceptance rate + recent reject vectors) under
    `<storage_path>/admission_stats.json`. Loaded lazily, saved after every
    decision.
  - A simple `evaluate(candidate_text, candidate_vector, category)` API that
    returns `(admit: bool, reason: str, score: float)` so the calling write
    pipeline can short-circuit before touching LanceDB.

This module deliberately does NOT call the LLM. Utility scoring in TS is
optional — we leave it out of the default Python flow to keep writes cheap.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — mirror the TS "balanced" preset
# ---------------------------------------------------------------------------

DEFAULT_TYPE_PRIORS: Dict[str, float] = {
    "profile": 0.95,
    "preferences": 0.9,
    "entities": 0.75,
    "events": 0.45,
    "cases": 0.8,
    "patterns": 0.85,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "novelty": 0.2,
    "recency": 0.1,
    "type_prior": 0.55,
    "rate": 0.15,
}

# Anti-flood: if more than this many writes have been admitted in the rolling
# window, the rate-limit term drops sharply.
DEFAULT_RATE_WINDOW_S = 60.0
DEFAULT_RATE_BUDGET = 8

# Cosine threshold against recent reject prototypes — anything above this drops
# straight to reject regardless of overall score.
DEFAULT_HARD_REJECT_COSINE = 0.92

DEFAULT_REJECT_THRESHOLD = 0.45
DEFAULT_ADMIT_THRESHOLD = 0.6

# How many recent reject vectors we hold in memory and persist.
DEFAULT_REJECT_POOL_SIZE = 32

# Recency half-life for the recency term.
DEFAULT_HALF_LIFE_DAYS = 14.0

STATS_FILENAME = "admission_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float, fallback: float = 0.0) -> float:
    if v != v or v in (float("inf"), float("-inf")):
        return max(0.0, min(1.0, fallback))
    return max(0.0, min(1.0, v))


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
# Stats container
# ---------------------------------------------------------------------------

@dataclass
class AdmissionStats:
    """Persisted rolling state."""
    admitted_count: int = 0
    rejected_count: int = 0
    # (timestamp, decision) tuples for windowed rate calc
    recent_decisions: List[List[float]] = field(default_factory=list)
    # Most recent reject vectors (to penalize lookalikes)
    reject_vectors: List[List[float]] = field(default_factory=list)
    # Most recent admit vectors (used as recency match pool)
    recent_admit_vectors: List[List[float]] = field(default_factory=list)
    last_admit_at: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "AdmissionStats":
        if not isinstance(data, dict):
            return cls()
        return cls(
            admitted_count=int(data.get("admitted_count") or 0),
            rejected_count=int(data.get("rejected_count") or 0),
            recent_decisions=[list(p) for p in data.get("recent_decisions") or [] if isinstance(p, (list, tuple)) and len(p) == 2],
            reject_vectors=[list(v) for v in data.get("reject_vectors") or [] if isinstance(v, list)],
            recent_admit_vectors=[list(v) for v in data.get("recent_admit_vectors") or [] if isinstance(v, list)],
            last_admit_at=float(data.get("last_admit_at") or 0.0),
        )


@dataclass
class AdmissionDecision:
    admit: bool
    score: float
    reason: str
    feature_scores: Dict[str, float] = field(default_factory=dict)
    matched_reject_cosine: float = 0.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AdmissionController:
    """Gates writes based on novelty / recency / rate / type priors.

    Lifecycle:
        ctrl = AdmissionController(storage_path)
        decision = ctrl.evaluate(text, vector, category="profile")
        if decision.admit:
            ...write to lancedb...

    Stats persist to `<storage_path>/admission_stats.json` after every call.
    Threadsafe via a lock around the in-memory state and the file write.
    """

    def __init__(
        self,
        storage_path: str,
        type_priors: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        admit_threshold: float = DEFAULT_ADMIT_THRESHOLD,
        rate_window_s: float = DEFAULT_RATE_WINDOW_S,
        rate_budget: int = DEFAULT_RATE_BUDGET,
        hard_reject_cosine: float = DEFAULT_HARD_REJECT_COSINE,
        reject_pool_size: int = DEFAULT_REJECT_POOL_SIZE,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        enabled: bool = True,
    ) -> None:
        self.storage_path = storage_path
        self.type_priors = dict(DEFAULT_TYPE_PRIORS)
        if type_priors:
            self.type_priors.update(type_priors)
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.reject_threshold = float(reject_threshold)
        self.admit_threshold = max(float(admit_threshold), float(reject_threshold))
        self.rate_window_s = float(rate_window_s)
        self.rate_budget = max(1, int(rate_budget))
        self.hard_reject_cosine = float(hard_reject_cosine)
        self.reject_pool_size = max(1, int(reject_pool_size))
        self.half_life_days = max(0.1, float(half_life_days))
        self.enabled = bool(enabled)

        self._lock = threading.RLock()
        self._stats: AdmissionStats = AdmissionStats()
        self._loaded = False

    # -- persistence -----------------------------------------------------

    @property
    def stats_path(self) -> str:
        return os.path.join(self.storage_path, STATS_FILENAME)

    def _load(self) -> None:
        if self._loaded:
            return
        path = self.stats_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._stats = AdmissionStats.from_json(data)
            except Exception as e:
                logger.debug("admission stats load failed: %s — starting fresh", e)
                self._stats = AdmissionStats()
        self._loaded = True

    def _persist(self) -> None:
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._stats.to_json(), f)
            os.replace(tmp, self.stats_path)
        except Exception as e:
            logger.debug("admission stats persist failed: %s", e)

    # -- accessors -------------------------------------------------------

    @property
    def stats(self) -> AdmissionStats:
        with self._lock:
            self._load()
            return self._stats

    def reset(self) -> None:
        with self._lock:
            self._stats = AdmissionStats()
            self._loaded = True
            self._persist()

    # -- scoring helpers --------------------------------------------------

    def _score_type_prior(self, category: str) -> float:
        return _clamp01(self.type_priors.get(category, 0.5), fallback=0.5)

    def _score_novelty(self, vector: Sequence[float]) -> tuple:
        """Return (score, max_reject_cosine).

        Score = 1 - max cosine vs recent reject vectors.
        """
        if not vector or not self._stats.reject_vectors:
            return 1.0, 0.0
        max_sim = 0.0
        for rv in self._stats.reject_vectors:
            sim = max(0.0, _cosine(vector, rv))
            if sim > max_sim:
                max_sim = sim
        return _clamp01(1.0 - max_sim, fallback=1.0), max_sim

    def _score_recency(self, now: float) -> float:
        """Higher when there's been a long gap since the last admit; lower when bursting."""
        if self._stats.last_admit_at <= 0:
            return 1.0
        gap_days = max(0.0, (now - self._stats.last_admit_at) / 86400.0)
        if gap_days <= 0:
            return 0.0
        lam = math.log(2.0) / self.half_life_days
        return _clamp01(1.0 - math.exp(-lam * gap_days), fallback=1.0)

    def _score_rate(self, now: float) -> float:
        """Drops as the rolling-window admit count approaches the budget."""
        cutoff = now - self.rate_window_s
        recent_admits = sum(
            1 for ts, decision in self._stats.recent_decisions
            if ts >= cutoff and decision >= 0.5
        )
        if recent_admits >= self.rate_budget:
            return 0.0
        return _clamp01(1.0 - (recent_admits / float(self.rate_budget)))

    # -- public API ------------------------------------------------------

    def evaluate(
        self,
        text: str,
        vector: Optional[Sequence[float]] = None,
        category: str = "cases",
        now: Optional[float] = None,
    ) -> AdmissionDecision:
        """Score the candidate and return an admit/reject decision.

        When `enabled=False`, always admits with a passthrough reason.
        """
        ts = now if now is not None else time.time()

        if not self.enabled:
            return AdmissionDecision(
                admit=True,
                score=1.0,
                reason="admission control disabled",
                feature_scores={},
                matched_reject_cosine=0.0,
            )

        with self._lock:
            self._load()

            type_prior = self._score_type_prior(category)
            novelty, max_reject_cos = self._score_novelty(vector or [])
            recency = self._score_recency(ts)
            rate = self._score_rate(ts)

            features = {
                "novelty": novelty,
                "recency": recency,
                "type_prior": type_prior,
                "rate": rate,
            }
            score = sum(features[k] * self.weights.get(k, 0.0) for k in features)
            score = _clamp01(score)

            # Hard reject: too close to a recent reject prototype.
            hard_reject = (
                bool(vector)
                and self._stats.reject_vectors
                and max_reject_cos >= self.hard_reject_cosine
            )

            if hard_reject:
                admit = False
                reason = (
                    f"hard reject (cosine={max_reject_cos:.3f} >= "
                    f"{self.hard_reject_cosine:.3f} vs recent rejects)"
                )
            elif score < self.reject_threshold:
                admit = False
                reason = f"score {score:.3f} < reject threshold {self.reject_threshold:.3f}"
            else:
                admit = True
                reason = f"score {score:.3f} >= reject threshold {self.reject_threshold:.3f}"

            self._record(admit, vector, ts)
            self._persist()

            return AdmissionDecision(
                admit=admit,
                score=score,
                reason=reason,
                feature_scores=features,
                matched_reject_cosine=max_reject_cos,
            )

    def _record(self, admit: bool, vector: Optional[Sequence[float]], ts: float) -> None:
        if admit:
            self._stats.admitted_count += 1
            self._stats.last_admit_at = ts
            if vector:
                self._stats.recent_admit_vectors.append(list(vector))
                if len(self._stats.recent_admit_vectors) > self.reject_pool_size:
                    self._stats.recent_admit_vectors = (
                        self._stats.recent_admit_vectors[-self.reject_pool_size:]
                    )
        else:
            self._stats.rejected_count += 1
            if vector:
                self._stats.reject_vectors.append(list(vector))
                if len(self._stats.reject_vectors) > self.reject_pool_size:
                    self._stats.reject_vectors = (
                        self._stats.reject_vectors[-self.reject_pool_size:]
                    )

        self._stats.recent_decisions.append([ts, 1.0 if admit else 0.0])
        # Trim decisions older than 4x rate_window to keep file small.
        cutoff = ts - (self.rate_window_s * 4)
        self._stats.recent_decisions = [
            row for row in self._stats.recent_decisions if row[0] >= cutoff
        ]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            self._load()
            total = self._stats.admitted_count + self._stats.rejected_count
            return {
                "enabled": self.enabled,
                "admitted": self._stats.admitted_count,
                "rejected": self._stats.rejected_count,
                "total": total,
                "reject_rate": (
                    self._stats.rejected_count / total if total > 0 else 0.0
                ),
                "reject_pool_size": len(self._stats.reject_vectors),
                "admit_pool_size": len(self._stats.recent_admit_vectors),
                "last_admit_at": self._stats.last_admit_at,
                "stats_path": self.stats_path,
            }


__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionStats",
    "DEFAULT_TYPE_PRIORS",
    "DEFAULT_WEIGHTS",
    "DEFAULT_REJECT_THRESHOLD",
    "DEFAULT_ADMIT_THRESHOLD",
    "DEFAULT_HARD_REJECT_COSINE",
    "DEFAULT_RATE_BUDGET",
    "DEFAULT_RATE_WINDOW_S",
    "STATS_FILENAME",
]
