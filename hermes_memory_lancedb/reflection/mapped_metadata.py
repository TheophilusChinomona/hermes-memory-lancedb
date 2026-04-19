"""Mappers between raw metadata and structured fields for ``mapped`` items.

Python port of `reflection-mapped-metadata.ts`. Mapped items are higher-level
than raw slice items — they classify each line as ``user-model``,
``agent-model``, ``lesson``, or ``decision`` and carry per-kind decay defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from .slices import ReflectionMappedMemoryItem


ReflectionMappedKind = Literal["user-model", "agent-model", "lesson", "decision"]
ReflectionMappedCategory = Literal["preference", "fact", "decision"]


@dataclass(frozen=True)
class ReflectionMappedDecayDefaults:
    midpoint_days: float
    k: float
    base_weight: float
    quality: float


_DEFAULTS: Dict[str, ReflectionMappedDecayDefaults] = {
    "decision": ReflectionMappedDecayDefaults(midpoint_days=45, k=0.25, base_weight=1.1, quality=1.0),
    "user-model": ReflectionMappedDecayDefaults(midpoint_days=21, k=0.30, base_weight=1.0, quality=0.95),
    "agent-model": ReflectionMappedDecayDefaults(midpoint_days=10, k=0.35, base_weight=0.95, quality=0.93),
    "lesson": ReflectionMappedDecayDefaults(midpoint_days=7, k=0.45, base_weight=0.9, quality=0.9),
}


def get_reflection_mapped_decay_defaults(kind: ReflectionMappedKind) -> ReflectionMappedDecayDefaults:
    return _DEFAULTS[kind]


@dataclass
class ReflectionMappedMetadata:
    event_id: str
    mapped_kind: ReflectionMappedKind
    mapped_category: ReflectionMappedCategory
    section: str
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
    type: str = "memory-reflection-mapped"
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
            "mappedKind": self.mapped_kind,
            "mappedCategory": self.mapped_category,
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


def _signal_hash(signal: Any) -> str:
    if isinstance(signal, dict):
        return str(signal.get("signatureHash") or signal.get("signature_hash") or "")
    return str(getattr(signal, "signature_hash", "") or getattr(signal, "signatureHash", ""))


def build_reflection_mapped_metadata(
    *,
    mapped_item: ReflectionMappedMemoryItem,
    event_id: str,
    agent_id: str,
    session_key: str,
    session_id: str,
    run_at: float,
    used_fallback: bool,
    tool_error_signals: Sequence[Any],
    source_reflection_path: Optional[str] = None,
) -> ReflectionMappedMetadata:
    defaults = get_reflection_mapped_decay_defaults(mapped_item.mapped_kind)
    return ReflectionMappedMetadata(
        event_id=event_id,
        mapped_kind=mapped_item.mapped_kind,
        mapped_category=mapped_item.category,
        section=mapped_item.heading,
        ordinal=mapped_item.ordinal,
        group_size=mapped_item.group_size,
        agent_id=agent_id,
        session_key=session_key,
        session_id=session_id,
        stored_at=run_at,
        used_fallback=used_fallback,
        error_signals=[_signal_hash(s) for s in tool_error_signals],
        decay_midpoint_days=defaults.midpoint_days,
        decay_k=defaults.k,
        base_weight=defaults.base_weight,
        quality=defaults.quality,
        source_reflection_path=source_reflection_path,
    )


def parse_mapped_kind(value: Any) -> Optional[ReflectionMappedKind]:
    if value in ("user-model", "agent-model", "lesson", "decision"):
        return value  # type: ignore[return-value]
    return None


__all__ = [
    "ReflectionMappedKind",
    "ReflectionMappedCategory",
    "ReflectionMappedDecayDefaults",
    "ReflectionMappedMetadata",
    "get_reflection_mapped_decay_defaults",
    "build_reflection_mapped_metadata",
    "parse_mapped_kind",
]
