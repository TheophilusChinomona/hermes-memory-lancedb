"""Reflection-specific ranking — recency-weighted, importance-boosted.

Python port of `reflection-ranking.ts`. The model is a logistic decay around
``midpoint_days`` with steepness ``k``, scaled by a per-line ``base_weight``
and a [0, 1] ``quality`` factor, with an extra penalty when the originating
LLM call fell back to a smaller/cheaper model (``used_fallback=True``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


REFLECTION_FALLBACK_SCORE_FACTOR = 0.75


@dataclass(frozen=True)
class ReflectionScoreInput:
    age_days: float
    midpoint_days: float
    k: float
    base_weight: float
    quality: float
    used_fallback: bool


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def compute_reflection_logistic(age_days: float, midpoint_days: float, k: float) -> float:
    """Pure logistic-decay weight in (0, 1].

    Mirrors the TS implementation: invalid / non-positive params snap to safe
    defaults so callers can pass raw metadata values without pre-validation.
    """
    safe_age = max(0.0, float(age_days)) if _is_finite(age_days) else 0.0
    safe_mid = float(midpoint_days) if _is_finite(midpoint_days) and midpoint_days > 0 else 1.0
    safe_k = float(k) if _is_finite(k) and k > 0 else 0.1
    return 1.0 / (1.0 + math.exp(safe_k * (safe_age - safe_mid)))


def compute_reflection_score(input_: ReflectionScoreInput) -> float:
    """Recency × base_weight × quality × (fallback penalty)."""
    logistic = compute_reflection_logistic(input_.age_days, input_.midpoint_days, input_.k)
    base_weight = float(input_.base_weight) if _is_finite(input_.base_weight) and input_.base_weight > 0 else 1.0
    if _is_finite(input_.quality):
        quality = max(0.0, min(1.0, float(input_.quality)))
    else:
        quality = 1.0
    fallback_factor = REFLECTION_FALLBACK_SCORE_FACTOR if input_.used_fallback else 1.0
    return logistic * base_weight * quality * fallback_factor


_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_reflection_line_for_aggregation(line: str) -> str:
    """Collapse whitespace + lowercase so duplicate lines aggregate."""
    if line is None:
        return ""
    return _WHITESPACE_RUN.sub(" ", str(line).strip()).lower()


class ReflectionRanker:
    """Tiny callable wrapper exposed at the package level.

    The TS port leaves ranking as free functions; the Python port also exposes
    a class so the main provider can swap implementations or carry config
    (e.g. fallback factor overrides) without changing call sites.
    """

    def __init__(self, fallback_factor: float = REFLECTION_FALLBACK_SCORE_FACTOR) -> None:
        self.fallback_factor = float(fallback_factor)

    def score(self, input_: ReflectionScoreInput) -> float:
        # Identical to compute_reflection_score except fallback factor is configurable.
        logistic = compute_reflection_logistic(input_.age_days, input_.midpoint_days, input_.k)
        base_weight = float(input_.base_weight) if _is_finite(input_.base_weight) and input_.base_weight > 0 else 1.0
        if _is_finite(input_.quality):
            quality = max(0.0, min(1.0, float(input_.quality)))
        else:
            quality = 1.0
        fallback_factor = self.fallback_factor if input_.used_fallback else 1.0
        return logistic * base_weight * quality * fallback_factor

    @staticmethod
    def normalize_line(line: str) -> str:
        return normalize_reflection_line_for_aggregation(line)


__all__ = [
    "REFLECTION_FALLBACK_SCORE_FACTOR",
    "ReflectionScoreInput",
    "ReflectionRanker",
    "compute_reflection_logistic",
    "compute_reflection_score",
    "normalize_reflection_line_for_aggregation",
]
