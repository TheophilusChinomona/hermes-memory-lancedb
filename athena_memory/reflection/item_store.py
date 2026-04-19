"""Structured reflection item store — per-line invariants and derived deltas.

Python port of `reflection-item-store.ts`. Each line emitted by ``slices``
becomes one ``memory-reflection-item`` payload with logistic decay defaults
that depend on whether the line is an ``invariant`` (long-lived rule) or
``derived`` (session-specific delta).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

from .slices import ReflectionSliceItem


ReflectionItemKind = Literal["invariant", "derived"]


# ---------------------------------------------------------------------------
# Decay defaults (logistic model around midpoint_days, steepness k)
# ---------------------------------------------------------------------------


REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS = 45
REFLECTION_INVARIANT_DECAY_K = 0.22
REFLECTION_INVARIANT_BASE_WEIGHT = 1.1
REFLECTION_INVARIANT_QUALITY = 1.0

REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS = 7
REFLECTION_DERIVED_DECAY_K = 0.65
REFLECTION_DERIVED_BASE_WEIGHT = 1.0
REFLECTION_DERIVED_QUALITY = 0.95


@dataclass(frozen=True)
class ReflectionItemDecayDefaults:
    midpoint_days: float
    k: float
    base_weight: float
    quality: float


def get_reflection_item_decay_defaults(item_kind: ReflectionItemKind) -> ReflectionItemDecayDefaults:
    if item_kind == "invariant":
        return ReflectionItemDecayDefaults(
            midpoint_days=REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS,
            k=REFLECTION_INVARIANT_DECAY_K,
            base_weight=REFLECTION_INVARIANT_BASE_WEIGHT,
            quality=REFLECTION_INVARIANT_QUALITY,
        )
    return ReflectionItemDecayDefaults(
        midpoint_days=REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS,
        k=REFLECTION_DERIVED_DECAY_K,
        base_weight=REFLECTION_DERIVED_BASE_WEIGHT,
        quality=REFLECTION_DERIVED_QUALITY,
    )


# ---------------------------------------------------------------------------
# Payload dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReflectionItemMetadata:
    event_id: str
    item_kind: ReflectionItemKind
    section: Literal["Invariants", "Derived"]
    ordinal: int
    group_size: int
    agent_id: str
    session_key: str
    session_id: str
    stored_at: float
    used_fallback: bool
    error_signals: List[str]
    decay_midpoint_days: float
    decay_k: float
    base_weight: float
    quality: float
    type: str = "memory-reflection-item"
    reflection_version: int = 4
    stage: str = "reflect-store"
    decay_model: str = "logistic"
    source_reflection_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "reflectionVersion": self.reflection_version,
            "stage": self.stage,
            "eventId": self.event_id,
            "itemKind": self.item_kind,
            "section": self.section,
            "ordinal": self.ordinal,
            "groupSize": self.group_size,
            "agentId": self.agent_id,
            "sessionKey": self.session_key,
            "sessionId": self.session_id,
            "storedAt": self.stored_at,
            "usedFallback": self.used_fallback,
            "errorSignals": list(self.error_signals),
            "decayModel": self.decay_model,
            "decayMidpointDays": self.decay_midpoint_days,
            "decayK": self.decay_k,
            "baseWeight": self.base_weight,
            "quality": self.quality,
        }
        if self.source_reflection_path:
            out["sourceReflectionPath"] = self.source_reflection_path
        return out


@dataclass
class ReflectionItemPayload:
    text: str
    metadata: ReflectionItemMetadata
    kind: Literal["item-invariant", "item-derived"]


def build_reflection_item_payloads(
    *,
    items: Sequence[ReflectionSliceItem],
    event_id: str,
    agent_id: str,
    session_key: str,
    session_id: str,
    run_at: float,
    used_fallback: bool,
    tool_error_signals: Sequence[Any],
    source_reflection_path: Optional[str] = None,
) -> List[ReflectionItemPayload]:
    out: List[ReflectionItemPayload] = []
    for item in items:
        defaults = get_reflection_item_decay_defaults(item.item_kind)
        metadata = ReflectionItemMetadata(
            event_id=event_id,
            item_kind=item.item_kind,
            section=item.section,
            ordinal=item.ordinal,
            group_size=item.group_size,
            agent_id=agent_id,
            session_key=session_key,
            session_id=session_id,
            stored_at=run_at,
            used_fallback=used_fallback,
            error_signals=[
                getattr(s, "signature_hash", None) or s.get("signatureHash") if isinstance(s, dict)
                else getattr(s, "signature_hash", "")
                for s in tool_error_signals
            ],
            decay_midpoint_days=defaults.midpoint_days,
            decay_k=defaults.k,
            base_weight=defaults.base_weight,
            quality=defaults.quality,
            source_reflection_path=source_reflection_path,
        )
        kind: Literal["item-invariant", "item-derived"] = (
            "item-invariant" if item.item_kind == "invariant" else "item-derived"
        )
        out.append(ReflectionItemPayload(text=item.text, metadata=metadata, kind=kind))
    return out


# ---------------------------------------------------------------------------
# In-memory ReflectionItemStore
# ---------------------------------------------------------------------------


class ReflectionItemStore:
    """In-process buffer for reflection items.

    Used by the provider for fast "give me the items written by event X"
    lookups. The durable store is the LanceDB ``reflections`` table; this
    buffer just mirrors recent writes.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._capacity = max(1, int(capacity))
        self._items: List[ReflectionItemPayload] = []

    def append_many(self, payloads: Sequence[ReflectionItemPayload]) -> None:
        self._items.extend(payloads)
        if len(self._items) > self._capacity:
            self._items = self._items[-self._capacity :]

    def __len__(self) -> int:
        return len(self._items)

    def by_event(self, event_id: str) -> List[ReflectionItemPayload]:
        return [p for p in self._items if p.metadata.event_id == event_id]

    def by_kind(self, item_kind: ReflectionItemKind) -> List[ReflectionItemPayload]:
        return [p for p in self._items if p.metadata.item_kind == item_kind]

    def all(self) -> List[ReflectionItemPayload]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


__all__ = [
    "REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS",
    "REFLECTION_INVARIANT_DECAY_K",
    "REFLECTION_INVARIANT_BASE_WEIGHT",
    "REFLECTION_INVARIANT_QUALITY",
    "REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS",
    "REFLECTION_DERIVED_DECAY_K",
    "REFLECTION_DERIVED_BASE_WEIGHT",
    "REFLECTION_DERIVED_QUALITY",
    "ReflectionItemDecayDefaults",
    "ReflectionItemMetadata",
    "ReflectionItemPayload",
    "ReflectionItemStore",
    "get_reflection_item_decay_defaults",
    "build_reflection_item_payloads",
]
