"""Memory lifecycle: tier promotion/demotion + Weibull decay engine.

Ports `tier-manager.ts` + `decay-engine.ts` from memory-lancedb-pro.

Two top-level classes:

- ``DecayEngine`` — Weibull stretched-exponential decay with importance-modulated
  half-life and tier-specific beta. Computes a composite score
  (recency + frequency + intrinsic) used by the tier manager and search boost.
- ``TierManager`` — promotes/demotes memories between ``peripheral`` /
  ``working`` / ``core`` based on access count, composite score, importance,
  and age.

The legacy module-level ``_tier_evaluate`` (defined in ``athena_memory.__init__``)
is kept as a thin backwards-compat wrapper around ``TierManager.evaluate`` so v1.1.0
callers continue to work unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


SECONDS_PER_DAY = 86_400.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DecayConfig:
    recency_half_life_days: float = 30.0
    recency_weight: float = 0.4
    frequency_weight: float = 0.3
    intrinsic_weight: float = 0.3
    stale_threshold: float = 0.3
    search_boost_min: float = 0.3
    importance_modulation: float = 1.5
    beta_core: float = 0.8
    beta_working: float = 1.0
    beta_peripheral: float = 1.3
    core_decay_floor: float = 0.9
    working_decay_floor: float = 0.7
    peripheral_decay_floor: float = 0.5
    # Dynamic memories decay this many times faster than static.
    dynamic_decay_multiplier: float = 3.0


@dataclass
class TierConfig:
    core_access_threshold: int = 10
    core_composite_threshold: float = 0.7
    core_importance_threshold: float = 0.8
    peripheral_composite_threshold: float = 0.15
    peripheral_age_days: float = 60.0
    working_access_threshold: int = 3
    working_composite_threshold: float = 0.4


@dataclass
class DecayScore:
    memory_id: str
    recency: float
    frequency: float
    intrinsic: float
    composite: float


@dataclass
class TierTransition:
    memory_id: str
    from_tier: str
    to_tier: str
    reason: str


@dataclass
class DecayableMemory:
    """Minimal fields needed for decay calculation."""
    id: str
    importance: float
    confidence: float = 1.0
    tier: str = "peripheral"
    access_count: int = 0
    created_at: float = 0.0  # unix seconds
    last_accessed_at: float = 0.0  # unix seconds; 0 = never
    temporal_type: Optional[str] = None  # "static" | "dynamic" | None


@dataclass
class TierableMemory:
    id: str
    tier: str
    importance: float
    access_count: int
    created_at: float  # unix seconds


# ---------------------------------------------------------------------------
# DecayEngine
# ---------------------------------------------------------------------------

class DecayEngine:
    """Weibull stretched-exponential decay engine.

    Composite score = recency_weight * recency
                    + frequency_weight * frequency
                    + intrinsic_weight * intrinsic

    Recency uses an importance-modulated half-life
    (``effective_half_life = base_hl * exp(mu * importance)``)
    and a tier-specific Weibull beta so core memories decay sub-exponentially
    and peripheral ones super-exponentially. Dynamic memories decay 3x faster
    than static ones (their base half-life is divided by
    ``config.dynamic_decay_multiplier``).
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()

    # ---- internals ----

    def _tier_beta(self, tier: str) -> float:
        if tier == "core":
            return self.config.beta_core
        if tier == "working":
            return self.config.beta_working
        return self.config.beta_peripheral

    def _tier_floor(self, tier: str) -> float:
        if tier == "core":
            return self.config.core_decay_floor
        if tier == "working":
            return self.config.working_decay_floor
        return self.config.peripheral_decay_floor

    def _recency(self, memory: DecayableMemory, now: float) -> float:
        if memory.access_count > 0 and memory.last_accessed_at > 0:
            last_active = memory.last_accessed_at
        else:
            last_active = memory.created_at
        days_since = max(0.0, (now - last_active) / SECONDS_PER_DAY)
        base_hl = self.config.recency_half_life_days
        if memory.temporal_type == "dynamic":
            base_hl = base_hl / max(self.config.dynamic_decay_multiplier, 1e-6)
        effective_hl = base_hl * math.exp(self.config.importance_modulation * memory.importance)
        # Avoid div-by-zero
        if effective_hl <= 0:
            return 0.0
        lam = math.log(2.0) / effective_hl
        beta = self._tier_beta(memory.tier)
        try:
            return math.exp(-lam * (days_since ** beta))
        except OverflowError:
            return 0.0

    def _frequency(self, memory: DecayableMemory) -> float:
        base = 1.0 - math.exp(-memory.access_count / 5.0)
        if memory.access_count <= 1:
            return base
        last_active = memory.last_accessed_at if memory.last_accessed_at > 0 else memory.created_at
        access_span_days = max(1.0, (last_active - memory.created_at) / SECONDS_PER_DAY)
        avg_gap_days = access_span_days / max(memory.access_count - 1, 1)
        recentness_bonus = math.exp(-avg_gap_days / 30.0)
        return base * (0.5 + 0.5 * recentness_bonus)

    def _intrinsic(self, memory: DecayableMemory) -> float:
        return memory.importance * memory.confidence

    # ---- public API ----

    def score(self, memory: DecayableMemory, now: Optional[float] = None) -> DecayScore:
        n = now if now is not None else time.time()
        r = self._recency(memory, n)
        f = self._frequency(memory)
        i = self._intrinsic(memory)
        composite = (
            self.config.recency_weight * r
            + self.config.frequency_weight * f
            + self.config.intrinsic_weight * i
        )
        return DecayScore(
            memory_id=memory.id,
            recency=r,
            frequency=f,
            intrinsic=i,
            composite=composite,
        )

    def score_all(
        self, memories: Iterable[DecayableMemory], now: Optional[float] = None
    ) -> List[DecayScore]:
        n = now if now is not None else time.time()
        return [self.score(m, n) for m in memories]

    def apply_search_boost(
        self,
        results: List[Dict],
        now: Optional[float] = None,
    ) -> List[Dict]:
        """Multiply each result's ``score`` by a tier/composite-aware boost.

        Each result must be a dict with ``score`` (float) and a ``memory``
        which is either a :class:`DecayableMemory` or a dict carrying the
        same fields. Mutates the dicts in place AND returns the list.
        """
        n = now if now is not None else time.time()
        boost_min = self.config.search_boost_min
        for r in results:
            mem = r.get("memory")
            d_mem = _coerce_memory(mem)
            if d_mem is None:
                continue
            ds = self.score(d_mem, n)
            tier_floor = max(self._tier_floor(d_mem.tier), ds.composite)
            multiplier = boost_min + (1.0 - boost_min) * tier_floor
            multiplier = min(1.0, max(boost_min, multiplier))
            r["score"] = float(r.get("score", 0.0)) * multiplier
        return results

    def get_stale_memories(
        self,
        memories: Iterable[DecayableMemory],
        now: Optional[float] = None,
    ) -> List[DecayScore]:
        n = now if now is not None else time.time()
        scores = [self.score(m, n) for m in memories]
        scores = [s for s in scores if s.composite < self.config.stale_threshold]
        scores.sort(key=lambda s: s.composite)
        return scores


def _coerce_memory(obj) -> Optional[DecayableMemory]:
    if obj is None:
        return None
    if isinstance(obj, DecayableMemory):
        return obj
    if isinstance(obj, dict):
        return DecayableMemory(
            id=str(obj.get("id", "")),
            importance=float(obj.get("importance", 0.5)),
            confidence=float(obj.get("confidence", 1.0)),
            tier=str(obj.get("tier", "peripheral")),
            access_count=int(obj.get("access_count", 0)),
            created_at=float(obj.get("created_at", obj.get("timestamp", 0.0))),
            last_accessed_at=float(obj.get("last_accessed_at", 0.0)),
            temporal_type=obj.get("temporal_type"),
        )
    return None


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------

class TierManager:
    """Three-tier promotion/demotion logic.

    Promotion path:  peripheral -> working -> core
    Demotion path:   core -> working -> peripheral

    Decisions are based on access count, composite decay score, importance
    and age. See :class:`TierConfig` for thresholds.
    """

    def __init__(self, config: Optional[TierConfig] = None):
        self.config = config or TierConfig()

    def evaluate(
        self,
        memory: TierableMemory,
        decay_score: DecayScore,
        now: Optional[float] = None,
    ) -> Optional[TierTransition]:
        n = now if now is not None else time.time()
        age_days = max(0.0, (n - memory.created_at) / SECONDS_PER_DAY)
        cfg = self.config
        composite = decay_score.composite

        if memory.tier == "peripheral":
            if (memory.access_count >= cfg.working_access_threshold
                    and composite >= cfg.working_composite_threshold):
                return TierTransition(
                    memory_id=memory.id,
                    from_tier="peripheral",
                    to_tier="working",
                    reason=(
                        f"Access count ({memory.access_count}) >= "
                        f"{cfg.working_access_threshold} and composite "
                        f"({composite:.2f}) >= {cfg.working_composite_threshold}"
                    ),
                )
            return None

        if memory.tier == "working":
            if (memory.access_count >= cfg.core_access_threshold
                    and composite >= cfg.core_composite_threshold
                    and memory.importance >= cfg.core_importance_threshold):
                return TierTransition(
                    memory_id=memory.id,
                    from_tier="working",
                    to_tier="core",
                    reason=(
                        f"High access ({memory.access_count}), composite "
                        f"({composite:.2f}), importance ({memory.importance})"
                    ),
                )
            if (composite < cfg.peripheral_composite_threshold
                    or (age_days > cfg.peripheral_age_days
                        and memory.access_count < cfg.working_access_threshold)):
                return TierTransition(
                    memory_id=memory.id,
                    from_tier="working",
                    to_tier="peripheral",
                    reason=(
                        f"Low composite ({composite:.2f}) or aged {age_days:.0f} days "
                        f"with low access ({memory.access_count})"
                    ),
                )
            return None

        if memory.tier == "core":
            if (composite < cfg.peripheral_composite_threshold
                    and memory.access_count < cfg.working_access_threshold):
                return TierTransition(
                    memory_id=memory.id,
                    from_tier="core",
                    to_tier="working",
                    reason=(
                        f"Severely low composite ({composite:.2f}) and "
                        f"access ({memory.access_count})"
                    ),
                )
            return None

        return None

    def evaluate_all(
        self,
        memories: List[TierableMemory],
        decay_scores: List[DecayScore],
        now: Optional[float] = None,
    ) -> List[TierTransition]:
        n = now if now is not None else time.time()
        score_map = {s.memory_id: s for s in decay_scores}
        out: List[TierTransition] = []
        for mem in memories:
            score = score_map.get(mem.id)
            if score is None:
                continue
            t = self.evaluate(mem, score, n)
            if t is not None:
                out.append(t)
        return out


# ---------------------------------------------------------------------------
# Backwards-compat wrapper for the v1.1.0 inline _tier_evaluate function
# ---------------------------------------------------------------------------

_DEFAULT_TIER_MANAGER = TierManager()


def tier_evaluate_legacy(
    current_tier: str,
    access_count: int,
    importance: float,
    decay_weight: float,
    age_days: float,
) -> Optional[str]:
    """Legacy wrapper preserving the original ``_tier_evaluate`` signature.

    The v1.1.0 inline implementation used a simple ``decay_weight * importance``
    composite. This wrapper feeds those same values into a ``TierManager`` so
    behaviour is preserved bit-for-bit while delegating to the new module.
    """
    composite = decay_weight * importance
    # Build a synthetic decay score and tierable memory; the manager only
    # cares about composite, access_count, importance, and (for working->peri)
    # the age in days. We anchor created_at relative to now so age_days matches.
    fake_now = time.time()
    fake_created = fake_now - age_days * SECONDS_PER_DAY
    mem = TierableMemory(
        id="_legacy_",
        tier=current_tier,
        importance=importance,
        access_count=access_count,
        created_at=fake_created,
    )
    score = DecayScore(
        memory_id="_legacy_",
        recency=decay_weight,
        frequency=0.0,
        intrinsic=importance,
        composite=composite,
    )
    transition = _DEFAULT_TIER_MANAGER.evaluate(mem, score, now=fake_now)
    return transition.to_tier if transition is not None else None


__all__ = [
    "DecayConfig",
    "TierConfig",
    "DecayScore",
    "TierTransition",
    "DecayableMemory",
    "TierableMemory",
    "DecayEngine",
    "TierManager",
    "tier_evaluate_legacy",
    "SECONDS_PER_DAY",
]
