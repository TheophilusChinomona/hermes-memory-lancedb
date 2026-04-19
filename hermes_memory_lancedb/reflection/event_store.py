"""Append-only event log for reflection sessions.

Python port of `reflection-event-store.ts`. Each reflection session emits one
``memory-reflection-event`` payload that anchors all derived item / mapped
payloads via a stable ``event_id``.
"""

from __future__ import annotations

import hashlib
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


REFLECTION_SCHEMA_VERSION = 4


@dataclass
class ReflectionErrorSignalLike:
    signature_hash: str


@dataclass
class ReflectionEventMetadata:
    event_id: str
    session_key: str
    session_id: str
    agent_id: str
    command: str
    stored_at: float
    used_fallback: bool
    error_signals: List[str]
    type: str = "memory-reflection-event"
    reflection_version: int = REFLECTION_SCHEMA_VERSION
    stage: str = "reflect-store"
    source_reflection_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "reflectionVersion": self.reflection_version,
            "stage": self.stage,
            "eventId": self.event_id,
            "sessionKey": self.session_key,
            "sessionId": self.session_id,
            "agentId": self.agent_id,
            "command": self.command,
            "storedAt": self.stored_at,
            "usedFallback": self.used_fallback,
            "errorSignals": list(self.error_signals),
        }
        if self.source_reflection_path:
            out["sourceReflectionPath"] = self.source_reflection_path
        return out


@dataclass
class ReflectionEventPayload:
    text: str
    metadata: ReflectionEventMetadata
    kind: str = "event"


def _safe_run_at(run_at: float) -> int:
    if not isinstance(run_at, (int, float)) or not math.isfinite(float(run_at)):
        return int(_time.time() * 1000)
    return max(0, int(run_at))


def create_reflection_event_id(
    *,
    run_at: float,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
) -> str:
    """Deterministic id derived from the (rounded) timestamp + identifiers."""
    safe_run_at = _safe_run_at(run_at)
    # Match TS behaviour: ISO timestamp with separators stripped, first 14 chars.
    iso = datetime.fromtimestamp(safe_run_at / 1000.0, tz=timezone.utc).isoformat()
    # JS Date.toISOString uses "YYYY-MM-DDTHH:MM:SS.sssZ"; Python may emit
    # "+00:00" instead of "Z" — strip the same chars either way.
    iso = iso.replace("Z", "").replace("+00:00", "")
    date_part = "".join(c for c in iso if c not in "-:.T")[:14]

    digest_input = f"{safe_run_at}|{session_key}|{session_id}|{agent_id}|{command}"
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]
    return f"refl-{date_part}-{digest}"


def build_reflection_event_payload(
    *,
    scope: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    tool_error_signals: Sequence[ReflectionErrorSignalLike],
    run_at: float,
    used_fallback: bool,
    event_id: Optional[str] = None,
    source_reflection_path: Optional[str] = None,
) -> ReflectionEventPayload:
    eid = event_id or create_reflection_event_id(
        run_at=run_at,
        session_key=session_key,
        session_id=session_id,
        agent_id=agent_id,
        command=command,
    )
    metadata = ReflectionEventMetadata(
        event_id=eid,
        session_key=session_key,
        session_id=session_id,
        agent_id=agent_id,
        command=command,
        stored_at=run_at,
        used_fallback=used_fallback,
        error_signals=[s.signature_hash for s in tool_error_signals],
        source_reflection_path=source_reflection_path,
    )
    text = "\n".join([
        f"reflection-event · {scope}",
        f"eventId={eid}",
        f"session={session_id}",
        f"agent={agent_id}",
        f"command={command}",
        f"usedFallback={'true' if used_fallback else 'false'}",
    ])
    return ReflectionEventPayload(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# In-memory ReflectionEventStore for the Python provider
# ---------------------------------------------------------------------------


@dataclass
class _StoredEvent:
    event_id: str
    payload: ReflectionEventPayload
    appended_at: float


class ReflectionEventStore:
    """Tiny append-only in-memory log used by the provider for last-N lookups.

    The TS port relies on the underlying memory table to query events; we keep
    a thin in-process buffer so callers can ask "what was the most recent
    reflection event for this session?" without round-tripping LanceDB.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = max(1, int(capacity))
        self._events: List[_StoredEvent] = []

    def append(self, payload: ReflectionEventPayload) -> None:
        self._events.append(_StoredEvent(
            event_id=payload.metadata.event_id,
            payload=payload,
            appended_at=_time.time(),
        ))
        if len(self._events) > self._capacity:
            self._events = self._events[-self._capacity :]

    def __len__(self) -> int:
        return len(self._events)

    def latest(self) -> Optional[ReflectionEventPayload]:
        return self._events[-1].payload if self._events else None

    def latest_for_session(self, session_id: str) -> Optional[ReflectionEventPayload]:
        for stored in reversed(self._events):
            if stored.payload.metadata.session_id == session_id:
                return stored.payload
        return None

    def all(self) -> List[ReflectionEventPayload]:
        return [s.payload for s in self._events]

    def clear(self) -> None:
        self._events.clear()


__all__ = [
    "REFLECTION_SCHEMA_VERSION",
    "ReflectionErrorSignalLike",
    "ReflectionEventMetadata",
    "ReflectionEventPayload",
    "ReflectionEventStore",
    "create_reflection_event_id",
    "build_reflection_event_payload",
]
