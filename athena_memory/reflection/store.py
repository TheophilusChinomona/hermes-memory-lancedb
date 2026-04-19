"""LanceDB-backed reflection store — schema, CRUD, FTS index.

Python port of `reflection-store.ts`. Reflections live in their own LanceDB
table named ``reflections`` with a separate FTS index and (optional) vector
column. Every payload built here is also a regular row in the table:

    id           string  uuid
    text         string  the line / event text (FTS-indexed)
    vector       fixed_size_list<float32, 1536>?  optional embedding
    timestamp    float64 unix epoch (seconds)
    kind         string  event | item-invariant | item-derived | mapped | combined-legacy
    category     string  always "reflection"
    scope        string  scope tag (e.g. "global" / "<agent_id>")
    importance   float32 derived from kind (event=0.55, invariant=0.82, derived=0.78)
    agent_id     string
    session_key  string
    session_id   string
    event_id     string
    metadata     string  JSON blob (full ReflectionXxxMetadata.to_dict())

The TS port writes reflections into the same ``memories`` table with
``category="reflection"`` so consumers can union-query. The Python port keeps
that contract — but ALSO writes into a dedicated ``reflections`` table so the
provider can run reflection-only queries efficiently. The shared ``id`` makes
the two views consistent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from .event_store import (
    REFLECTION_SCHEMA_VERSION,
    ReflectionErrorSignalLike,
    ReflectionEventPayload,
    build_reflection_event_payload,
    create_reflection_event_id,
)
from .item_store import (
    ReflectionItemPayload,
    REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS,
    REFLECTION_DERIVED_DECAY_K,
    REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS,
    REFLECTION_INVARIANT_DECAY_K,
    build_reflection_item_payloads,
    get_reflection_item_decay_defaults,
)
from .mapped_metadata import (
    ReflectionMappedKind,
    get_reflection_mapped_decay_defaults,
)
from .metadata import parse_reflection_metadata
from .ranking import (
    ReflectionScoreInput,
    compute_reflection_score,
    normalize_reflection_line_for_aggregation,
)
from .slices import (
    ReflectionSlices,
    extract_injectable_reflection_slices,
    extract_injectable_reflection_slice_items,
    sanitize_injectable_reflection_lines,
    sanitize_reflection_slice_lines,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


REFLECTION_TABLE_NAME = "reflections"
REFLECTION_EMBED_DIM = 1536

REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS = 3
REFLECTION_DERIVE_LOGISTIC_K = 1.2
REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT = 0.35

DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000
DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS = 60 * 24 * 60 * 60 * 1000

ReflectionStoreKind = Literal[
    "event",
    "item-invariant",
    "item-derived",
    "combined-legacy",
    "mapped",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def get_reflection_schema(*, embed_dim: int = REFLECTION_EMBED_DIM):
    """Return the pyarrow schema for the ``reflections`` table.

    Imported lazily so the rest of the subpackage can be used in environments
    without pyarrow (rare in practice but useful for tests).
    """
    import pyarrow as pa

    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embed_dim)),
        pa.field("timestamp", pa.float64()),
        pa.field("kind", pa.string()),
        pa.field("category", pa.string()),
        pa.field("scope", pa.string()),
        pa.field("importance", pa.float32()),
        pa.field("agent_id", pa.string()),
        pa.field("session_key", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("event_id", pa.string()),
        pa.field("metadata", pa.string()),
    ])


# ---------------------------------------------------------------------------
# Payload dataclasses (combined / generic)
# ---------------------------------------------------------------------------


@dataclass
class ReflectionStorePayload:
    text: str
    metadata: Dict[str, Any]
    kind: ReflectionStoreKind


@dataclass
class BuildReflectionStorePayloadsParams:
    reflection_text: str
    session_key: str
    session_id: str
    agent_id: str
    command: str
    scope: str
    tool_error_signals: List[ReflectionErrorSignalLike]
    run_at: float
    used_fallback: bool
    event_id: Optional[str] = None
    source_reflection_path: Optional[str] = None
    write_legacy_combined: bool = True


@dataclass
class BuildReflectionStorePayloadsResult:
    event_id: str
    slices: ReflectionSlices
    payloads: List[ReflectionStorePayload]


def compute_derived_line_quality(non_placeholder_line_count: int) -> float:
    try:
        n = max(0, int(non_placeholder_line_count))
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return 0.2
    return min(1.0, 0.55 + min(6, n) * 0.075)


def _build_legacy_combined_payload(
    *,
    slices: ReflectionSlices,
    scope: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    tool_error_signals: Sequence[ReflectionErrorSignalLike],
    run_at: float,
    used_fallback: bool,
    source_reflection_path: Optional[str],
) -> ReflectionStorePayload:
    date_ymd = datetime.fromtimestamp(run_at / 1000.0, tz=timezone.utc).date().isoformat()
    iso_full = datetime.fromtimestamp(run_at / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    derive_quality = compute_derived_line_quality(len(slices.derived))
    derive_base_weight = REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT if used_fallback else 1.0

    invariant_lines = (
        [f"- {x}" for x in slices.invariants] if slices.invariants else ["- (none captured)"]
    )
    derived_lines = (
        [f"- {x}" for x in slices.derived] if slices.derived else ["- (none captured)"]
    )

    text = "\n".join([
        f"reflection · {scope} · {date_ymd}",
        f"Session Reflection ({iso_full})",
        f"Session Key: {session_key}",
        f"Session ID: {session_id}",
        "",
        "Invariants:",
        *invariant_lines,
        "",
        "Derived:",
        *derived_lines,
    ])

    metadata: Dict[str, Any] = {
        "type": "memory-reflection",
        "stage": "reflect-store",
        "reflectionVersion": 3,
        "sessionKey": session_key,
        "sessionId": session_id,
        "agentId": agent_id,
        "command": command,
        "storedAt": run_at,
        "invariants": list(slices.invariants),
        "derived": list(slices.derived),
        "usedFallback": used_fallback,
        "errorSignals": [s.signature_hash for s in tool_error_signals],
        "decayModel": "logistic",
        "decayMidpointDays": REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
        "decayK": REFLECTION_DERIVE_LOGISTIC_K,
        "deriveBaseWeight": derive_base_weight,
        "deriveQuality": derive_quality,
        "deriveSource": "fallback" if used_fallback else "normal",
    }
    if source_reflection_path:
        metadata["sourceReflectionPath"] = source_reflection_path

    return ReflectionStorePayload(text=text, metadata=metadata, kind="combined-legacy")


def build_reflection_store_payloads(
    params: BuildReflectionStorePayloadsParams,
) -> BuildReflectionStorePayloadsResult:
    """Pure builder — no I/O. Mirrors `buildReflectionStorePayloads` in TS."""
    slices = extract_injectable_reflection_slices(params.reflection_text)
    event_id = params.event_id or create_reflection_event_id(
        run_at=params.run_at,
        session_key=params.session_key,
        session_id=params.session_id,
        agent_id=params.agent_id,
        command=params.command,
    )

    payloads: List[ReflectionStorePayload] = []

    event_payload = build_reflection_event_payload(
        event_id=event_id,
        scope=params.scope,
        session_key=params.session_key,
        session_id=params.session_id,
        agent_id=params.agent_id,
        command=params.command,
        tool_error_signals=params.tool_error_signals,
        run_at=params.run_at,
        used_fallback=params.used_fallback,
        source_reflection_path=params.source_reflection_path,
    )
    payloads.append(ReflectionStorePayload(
        text=event_payload.text,
        metadata=event_payload.metadata.to_dict(),
        kind="event",
    ))

    item_payloads = build_reflection_item_payloads(
        items=extract_injectable_reflection_slice_items(params.reflection_text),
        event_id=event_id,
        agent_id=params.agent_id,
        session_key=params.session_key,
        session_id=params.session_id,
        run_at=params.run_at,
        used_fallback=params.used_fallback,
        tool_error_signals=params.tool_error_signals,
        source_reflection_path=params.source_reflection_path,
    )
    for item in item_payloads:
        payloads.append(ReflectionStorePayload(
            text=item.text,
            metadata=item.metadata.to_dict(),
            kind=item.kind,  # type: ignore[arg-type]
        ))

    if params.write_legacy_combined and (slices.invariants or slices.derived):
        payloads.append(_build_legacy_combined_payload(
            slices=slices,
            scope=params.scope,
            session_key=params.session_key,
            session_id=params.session_id,
            agent_id=params.agent_id,
            command=params.command,
            tool_error_signals=params.tool_error_signals,
            run_at=params.run_at,
            used_fallback=params.used_fallback,
            source_reflection_path=params.source_reflection_path,
        ))

    return BuildReflectionStorePayloadsResult(event_id=event_id, slices=slices, payloads=payloads)


def resolve_reflection_importance(kind: ReflectionStoreKind) -> float:
    if kind == "event":
        return 0.55
    if kind == "item-invariant":
        return 0.82
    if kind == "item-derived":
        return 0.78
    if kind == "mapped":
        return 0.80
    return 0.75  # combined-legacy


# ---------------------------------------------------------------------------
# Slice / mapped loaders (decay-weighted aggregation)
# ---------------------------------------------------------------------------


@dataclass
class _WeightedLineCandidate:
    line: str
    timestamp: float
    midpoint_days: float
    k: float
    base_weight: float
    quality: float
    used_fallback: bool


def _is_reflection_metadata_type(t: Any) -> bool:
    return t in ("memory-reflection-item", "memory-reflection")


def _is_owned_by_agent(metadata: Dict[str, Any], agent_id: str) -> bool:
    owner_raw = metadata.get("agentId")
    owner = owner_raw.strip() if isinstance(owner_raw, str) else ""
    if not owner:
        return True
    return owner == agent_id or owner == "main"


def _to_string_array(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _metadata_timestamp(metadata: Dict[str, Any], fallback_ts: float) -> float:
    stored = metadata.get("storedAt")
    try:
        v = float(stored)
        if math.isfinite(v) and v > 0:
            return v
    except (TypeError, ValueError):
        pass
    if isinstance(fallback_ts, (int, float)) and math.isfinite(fallback_ts):
        return float(fallback_ts)
    return _time.time() * 1000.0


def _read_positive(value: Any, fallback: float) -> float:
    try:
        n = float(value)
        if math.isfinite(n) and n > 0:
            return n
    except (TypeError, ValueError):
        pass
    return fallback


def _read_clamped(value: Any, fallback: float, lo: float, hi: float) -> float:
    try:
        n = float(value)
        if not math.isfinite(n):
            n = fallback
    except (TypeError, ValueError):
        n = fallback
    return max(lo, min(hi, n))


def _resolve_legacy_derive_base_weight(metadata: Dict[str, Any]) -> float:
    explicit = metadata.get("deriveBaseWeight")
    try:
        n = float(explicit)
        if math.isfinite(n) and n > 0:
            return max(0.1, min(1.2, n))
    except (TypeError, ValueError):
        pass
    if metadata.get("usedFallback") is True:
        return REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT
    return 1.0


def _build_invariant_candidates(
    item_rows: List[Dict[str, Any]],
    legacy_rows: List[Dict[str, Any]],
) -> List[_WeightedLineCandidate]:
    item_candidates: List[_WeightedLineCandidate] = []
    for row in item_rows:
        metadata = row["metadata"]
        if metadata.get("itemKind") != "invariant":
            continue
        entry = row["entry"]
        text = entry.get("text", "")
        safe_lines = sanitize_injectable_reflection_lines([text])
        if not safe_lines:
            continue
        defaults = get_reflection_item_decay_defaults("invariant")
        ts = _metadata_timestamp(metadata, entry.get("timestamp", 0.0))
        for line in safe_lines:
            item_candidates.append(_WeightedLineCandidate(
                line=line,
                timestamp=ts,
                midpoint_days=_read_positive(metadata.get("decayMidpointDays"), defaults.midpoint_days),
                k=_read_positive(metadata.get("decayK"), defaults.k),
                base_weight=_read_positive(metadata.get("baseWeight"), defaults.base_weight),
                quality=_read_clamped(metadata.get("quality"), defaults.quality, 0.2, 1.0),
                used_fallback=metadata.get("usedFallback") is True,
            ))

    if item_candidates:
        return item_candidates

    out: List[_WeightedLineCandidate] = []
    for row in legacy_rows:
        metadata = row["metadata"]
        defaults = get_reflection_item_decay_defaults("invariant")
        entry = row["entry"]
        ts = _metadata_timestamp(metadata, entry.get("timestamp", 0.0))
        lines = sanitize_injectable_reflection_lines(_to_string_array(metadata.get("invariants")))
        for line in lines:
            out.append(_WeightedLineCandidate(
                line=line,
                timestamp=ts,
                midpoint_days=defaults.midpoint_days,
                k=defaults.k,
                base_weight=defaults.base_weight,
                quality=defaults.quality,
                used_fallback=metadata.get("usedFallback") is True,
            ))
    return out


def _build_derived_candidates(
    item_rows: List[Dict[str, Any]],
    legacy_rows: List[Dict[str, Any]],
) -> List[_WeightedLineCandidate]:
    item_candidates: List[_WeightedLineCandidate] = []
    for row in item_rows:
        metadata = row["metadata"]
        if metadata.get("itemKind") != "derived":
            continue
        entry = row["entry"]
        text = entry.get("text", "")
        safe_lines = sanitize_injectable_reflection_lines([text])
        if not safe_lines:
            continue
        defaults = get_reflection_item_decay_defaults("derived")
        ts = _metadata_timestamp(metadata, entry.get("timestamp", 0.0))
        for line in safe_lines:
            item_candidates.append(_WeightedLineCandidate(
                line=line,
                timestamp=ts,
                midpoint_days=_read_positive(metadata.get("decayMidpointDays"), defaults.midpoint_days),
                k=_read_positive(metadata.get("decayK"), defaults.k),
                base_weight=_read_positive(metadata.get("baseWeight"), defaults.base_weight),
                quality=_read_clamped(metadata.get("quality"), defaults.quality, 0.2, 1.0),
                used_fallback=metadata.get("usedFallback") is True,
            ))

    if item_candidates:
        return item_candidates

    out: List[_WeightedLineCandidate] = []
    for row in legacy_rows:
        metadata = row["metadata"]
        entry = row["entry"]
        ts = _metadata_timestamp(metadata, entry.get("timestamp", 0.0))
        lines = sanitize_injectable_reflection_lines(_to_string_array(metadata.get("derived")))
        if not lines:
            continue
        defaults_mid = REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS
        defaults_k = REFLECTION_DERIVE_LOGISTIC_K
        defaults_base = _resolve_legacy_derive_base_weight(metadata)
        defaults_quality = compute_derived_line_quality(len(lines))
        for line in lines:
            out.append(_WeightedLineCandidate(
                line=line,
                timestamp=ts,
                midpoint_days=_read_positive(metadata.get("decayMidpointDays"), defaults_mid),
                k=_read_positive(metadata.get("decayK"), defaults_k),
                base_weight=_read_positive(metadata.get("deriveBaseWeight"), defaults_base),
                quality=_read_clamped(metadata.get("deriveQuality"), defaults_quality, 0.2, 1.0),
                used_fallback=metadata.get("usedFallback") is True,
            ))
    return out


def _rank_reflection_lines(
    candidates: List[_WeightedLineCandidate],
    *,
    now: float,
    max_age_ms: Optional[float],
    limit: int,
) -> List[str]:
    line_scores: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        ts = cand.timestamp if math.isfinite(cand.timestamp) else now
        if max_age_ms is not None and max_age_ms >= 0 and (now - ts) > max_age_ms:
            continue
        age_days = max(0.0, (now - ts) / 86_400_000.0)
        score = compute_reflection_score(ReflectionScoreInput(
            age_days=age_days,
            midpoint_days=cand.midpoint_days,
            k=cand.k,
            base_weight=cand.base_weight,
            quality=cand.quality,
            used_fallback=cand.used_fallback,
        ))
        if not math.isfinite(score) or score <= 0:
            continue
        key = normalize_reflection_line_for_aggregation(cand.line)
        if not key:
            continue
        current = line_scores.get(key)
        if current is None:
            line_scores[key] = {"line": cand.line, "score": score, "latest_ts": ts}
            continue
        current["score"] += score
        if ts > current["latest_ts"]:
            current["latest_ts"] = ts
            current["line"] = cand.line

    ordered = sorted(
        line_scores.values(),
        key=lambda v: (-v["score"], -v["latest_ts"], v["line"]),
    )
    return [v["line"] for v in ordered[:limit]]


@dataclass
class LoadReflectionSlicesParams:
    entries: List[Dict[str, Any]]
    agent_id: str
    now: Optional[float] = None
    derive_max_age_ms: Optional[float] = None
    invariant_max_age_ms: Optional[float] = None


def load_agent_reflection_slices_from_entries(
    params: LoadReflectionSlicesParams,
) -> Dict[str, List[str]]:
    """Aggregate reflection rows into ranked invariant + derived bullets."""
    now = float(params.now) if params.now is not None and math.isfinite(params.now) else _time.time() * 1000.0
    derive_max_age_ms = (
        max(0.0, float(params.derive_max_age_ms))
        if params.derive_max_age_ms is not None and math.isfinite(params.derive_max_age_ms)
        else float(DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS)
    )
    invariant_max_age_ms: Optional[float] = (
        max(0.0, float(params.invariant_max_age_ms))
        if params.invariant_max_age_ms is not None and math.isfinite(params.invariant_max_age_ms)
        else None
    )

    rows = [
        {"entry": entry, "metadata": parse_reflection_metadata(entry.get("metadata"))}
        for entry in params.entries
    ]
    rows = [
        r for r in rows
        if _is_reflection_metadata_type(r["metadata"].get("type"))
        and _is_owned_by_agent(r["metadata"], params.agent_id)
    ]
    rows.sort(key=lambda r: r["entry"].get("timestamp", 0.0), reverse=True)
    rows = rows[:160]

    item_rows = [r for r in rows if r["metadata"].get("type") == "memory-reflection-item"]
    legacy_rows = [r for r in rows if r["metadata"].get("type") == "memory-reflection"]

    invariants = _rank_reflection_lines(
        _build_invariant_candidates(item_rows, legacy_rows),
        now=now,
        max_age_ms=invariant_max_age_ms,
        limit=8,
    )
    derived = _rank_reflection_lines(
        _build_derived_candidates(item_rows, legacy_rows),
        now=now,
        max_age_ms=derive_max_age_ms,
        limit=10,
    )

    return {"invariants": invariants, "derived": derived}


# ---------------------------------------------------------------------------
# Mapped loader
# ---------------------------------------------------------------------------


@dataclass
class LoadReflectionMappedRowsParams:
    entries: List[Dict[str, Any]]
    agent_id: str
    now: Optional[float] = None
    max_age_ms: Optional[float] = None
    max_per_kind: Optional[int] = None


def load_reflection_mapped_rows_from_entries(
    params: LoadReflectionMappedRowsParams,
) -> Dict[str, List[str]]:
    now = float(params.now) if params.now is not None and math.isfinite(params.now) else _time.time() * 1000.0
    max_age_ms = (
        max(0.0, float(params.max_age_ms))
        if params.max_age_ms is not None and math.isfinite(params.max_age_ms)
        else float(DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS)
    )
    try:
        max_per_kind = max(1, int(params.max_per_kind)) if params.max_per_kind is not None else 10
    except (TypeError, ValueError):
        max_per_kind = 10

    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in params.entries:
        metadata = parse_reflection_metadata(entry.get("metadata"))
        if metadata.get("type") != "memory-reflection-mapped":
            continue
        if not _is_owned_by_agent(metadata, params.agent_id):
            continue
        mapped_kind = metadata.get("mappedKind")
        if mapped_kind not in ("user-model", "agent-model", "lesson", "decision"):
            continue

        text = entry.get("text", "")
        lines = sanitize_reflection_slice_lines([text])
        if not lines:
            continue

        defaults = get_reflection_mapped_decay_defaults(mapped_kind)
        ts = _metadata_timestamp(metadata, entry.get("timestamp", 0.0))
        if (now - ts) > max_age_ms:
            continue

        for line in lines:
            age_days = max(0.0, (now - ts) / 86_400_000.0)
            score = compute_reflection_score(ReflectionScoreInput(
                age_days=age_days,
                midpoint_days=_read_positive(metadata.get("decayMidpointDays"), defaults.midpoint_days),
                k=_read_positive(metadata.get("decayK"), defaults.k),
                base_weight=_read_positive(metadata.get("baseWeight"), defaults.base_weight),
                quality=_read_clamped(metadata.get("quality"), defaults.quality, 0.2, 1.0),
                used_fallback=metadata.get("usedFallback") is True,
            ))
            if not math.isfinite(score) or score <= 0:
                continue
            key_norm = normalize_reflection_line_for_aggregation(line)
            if not key_norm:
                continue
            key = f"{mapped_kind}::{key_norm}"
            current = grouped.get(key)
            if current is None:
                grouped[key] = {"text": line, "score": score, "latest_ts": ts, "kind": mapped_kind}
                continue
            current["score"] += score
            if ts > current["latest_ts"]:
                current["latest_ts"] = ts
                current["text"] = line

    def sorted_for(kind: ReflectionMappedKind) -> List[str]:
        rows = [v for v in grouped.values() if v["kind"] == kind]
        rows.sort(key=lambda v: (-v["score"], -v["latest_ts"], v["text"]))
        return [v["text"] for v in rows[:max_per_kind]]

    return {
        "userModel": sorted_for("user-model"),
        "agentModel": sorted_for("agent-model"),
        "lesson": sorted_for("lesson"),
        "decision": sorted_for("decision"),
    }


# ---------------------------------------------------------------------------
# ReflectionStore — LanceDB-backed CRUD + FTS
# ---------------------------------------------------------------------------


@dataclass
class ReflectionSearchHit:
    id: str
    text: str
    score: float
    kind: str
    scope: str
    timestamp: float
    importance: float
    metadata: Dict[str, Any]
    event_id: str
    agent_id: str
    session_id: str


class ReflectionStore:
    """LanceDB-backed reflection table with optional embedder.

    Designed so the caller can wire it into ``LanceDBMemoryProvider`` lazily
    (only when ``LANCEDB_REFLECTION_ENABLED=1``). The store falls back to
    BM25/FTS-only search when no embedder is configured — vector search is
    skipped, dedup similarity check uses text equality.

    The ``connect_fn`` parameter exists so tests can inject a stub LanceDB
    client without going through the native driver.
    """

    def __init__(
        self,
        *,
        storage_path: str,
        embedder: Optional[Callable[[str], List[float]]] = None,
        embed_dim: int = REFLECTION_EMBED_DIM,
        connect_fn: Optional[Callable[[str], Any]] = None,
        table_name: str = REFLECTION_TABLE_NAME,
        dedup_threshold: float = 0.97,
    ) -> None:
        self._storage_path = storage_path
        self._embedder = embedder
        self._embed_dim = int(embed_dim)
        self._connect_fn = connect_fn
        self._table_name = table_name
        self._dedup_threshold = float(dedup_threshold)
        self._db: Any = None
        self._table: Any = None
        self._lock = threading.Lock()
        self._ready = False

    # ------------- lifecycle -------------

    def initialize(self) -> bool:
        try:
            if self._connect_fn is not None:
                self._db = self._connect_fn(self._storage_path)
            else:
                import lancedb

                import os as _os
                _os.makedirs(self._storage_path, exist_ok=True)
                self._db = lancedb.connect(self._storage_path)

            existing = set(self._db.table_names()) if hasattr(self._db, "table_names") else set()
            if self._table_name in existing:
                self._table = self._db.open_table(self._table_name)
            else:
                self._table = self._db.create_table(
                    self._table_name, schema=get_reflection_schema(embed_dim=self._embed_dim)
                )

            try:
                self._table.create_fts_index("text", replace=True)
            except Exception as e:  # noqa: BLE001
                logger.debug("Reflection FTS index skipped: %s", e)

            self._ready = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("ReflectionStore init failed: %s", e, exc_info=True)
            self._ready = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_embedder(self) -> bool:
        return self._embedder is not None

    @property
    def storage_path(self) -> str:
        return self._storage_path

    # ------------- writes -------------

    def _embed_or_zero(self, text: str) -> List[float]:
        if self._embedder is None:
            return [0.0] * self._embed_dim
        try:
            vec = list(self._embedder(text))
        except Exception as e:  # noqa: BLE001
            logger.debug("Reflection embedder failed: %s", e)
            return [0.0] * self._embed_dim
        if len(vec) != self._embed_dim:
            # Pad / truncate so the vector column shape stays consistent.
            if len(vec) < self._embed_dim:
                vec = vec + [0.0] * (self._embed_dim - len(vec))
            else:
                vec = vec[: self._embed_dim]
        return vec

    def _scope_existing_text(self, text: str, scope: str) -> Optional[Dict[str, Any]]:
        if self._table is None or not self._ready:
            return None
        try:
            # Cheap pre-check by exact text + scope.
            rows = (
                self._table.search()
                .where(f"scope = '{_escape(scope)}'", prefilter=True)
                .limit(50)
                .to_list()
            )
            for r in rows:
                if r.get("text") == text:
                    return r
        except Exception:
            pass
        return None

    def write_payload(
        self,
        payload: ReflectionStorePayload,
        *,
        scope: str,
        agent_id: str,
        session_key: str,
        session_id: str,
    ) -> Optional[str]:
        """Persist one payload, returning the row id (or None on dedup/skip)."""
        if not self._ready or self._table is None:
            return None
        try:
            metadata = payload.metadata
            event_id = str(metadata.get("eventId", ""))
            ts = metadata.get("storedAt") or _time.time() * 1000.0
            row_id = str(uuid.uuid4())

            if payload.kind == "combined-legacy":
                # Cheap text-equality dedup for the combined-legacy slot.
                if self._scope_existing_text(payload.text, scope) is not None:
                    return None

            row = {
                "id": row_id,
                "text": payload.text,
                "vector": self._embed_or_zero(payload.text),
                "timestamp": float(ts),
                "kind": payload.kind,
                "category": "reflection",
                "scope": scope,
                "importance": float(resolve_reflection_importance(payload.kind)),
                "agent_id": agent_id,
                "session_key": session_key,
                "session_id": session_id,
                "event_id": event_id,
                "metadata": json.dumps(metadata),
            }
            self._table.add([row])
            return row_id
        except Exception as e:  # noqa: BLE001
            logger.debug("ReflectionStore write failed: %s", e)
            return None

    def write_reflection(
        self,
        params: BuildReflectionStorePayloadsParams,
    ) -> Dict[str, Any]:
        """Top-level write: build payloads + persist each one."""
        result = build_reflection_store_payloads(params)
        stored_kinds: List[str] = []
        ids: List[str] = []
        for payload in result.payloads:
            row_id = self.write_payload(
                payload,
                scope=params.scope,
                agent_id=params.agent_id,
                session_key=params.session_key,
                session_id=params.session_id,
            )
            if row_id:
                ids.append(row_id)
                stored_kinds.append(payload.kind)

        return {
            "stored": bool(stored_kinds),
            "event_id": result.event_id,
            "slices": {
                "invariants": list(result.slices.invariants),
                "derived": list(result.slices.derived),
            },
            "stored_kinds": stored_kinds,
            "ids": ids,
        }

    # ------------- reads -------------

    def count(self) -> int:
        if not self._ready or self._table is None:
            return 0
        try:
            if hasattr(self._table, "count_rows"):
                return int(self._table.count_rows())
            return len(self._table.search().limit(10_000).to_list())
        except Exception:
            return 0

    def search_text(self, query: str, *, top_k: int = 6) -> List[ReflectionSearchHit]:
        """FTS-only search — works without an embedder."""
        if not self._ready or self._table is None or not query:
            return []
        try:
            rows = (
                self._table.search(query, query_type="fts")
                .limit(top_k)
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("ReflectionStore FTS failed: %s", e)
            rows = []
        return [self._row_to_hit(r) for r in rows[:top_k]]

    def search_vector(self, query: str, *, top_k: int = 6) -> List[ReflectionSearchHit]:
        if not self._ready or self._table is None or self._embedder is None or not query:
            return []
        try:
            vec = self._embedder(query)
            rows = (
                self._table.search(list(vec), vector_column_name="vector")
                .limit(top_k)
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("ReflectionStore vector search failed: %s", e)
            rows = []
        return [self._row_to_hit(r) for r in rows[:top_k]]

    def search(self, query: str, *, top_k: int = 6) -> List[ReflectionSearchHit]:
        """Hybrid (vector + FTS, simple union dedup) when an embedder is set,
        otherwise BM25-only."""
        text_hits = self.search_text(query, top_k=top_k)
        if not self.has_embedder:
            return text_hits
        vec_hits = self.search_vector(query, top_k=top_k)
        seen: set = set()
        out: List[ReflectionSearchHit] = []
        for h in (vec_hits + text_hits):
            if h.id in seen:
                continue
            seen.add(h.id)
            out.append(h)
            if len(out) >= top_k:
                break
        return out

    def get(self, row_id: str) -> Optional[ReflectionSearchHit]:
        if not self._ready or self._table is None or not row_id:
            return None
        try:
            rows = (
                self._table.search()
                .where(f"id = '{_escape(row_id)}'", prefilter=True)
                .limit(1)
                .to_list()
            )
        except Exception:
            rows = []
        if not rows:
            return None
        return self._row_to_hit(rows[0])

    def delete(self, row_id: str) -> bool:
        if not self._ready or self._table is None or not row_id:
            return False
        try:
            self._table.delete(f"id = '{_escape(row_id)}'")
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("ReflectionStore delete failed: %s", e)
            return False

    def list_by_event(self, event_id: str, *, limit: int = 50) -> List[ReflectionSearchHit]:
        if not self._ready or self._table is None or not event_id:
            return []
        try:
            rows = (
                self._table.search()
                .where(f"event_id = '{_escape(event_id)}'", prefilter=True)
                .limit(limit)
                .to_list()
            )
        except Exception:
            rows = []
        return [self._row_to_hit(r) for r in rows]

    # ------------- helpers -------------

    @staticmethod
    def _row_to_hit(row: Dict[str, Any]) -> ReflectionSearchHit:
        try:
            metadata = parse_reflection_metadata(row.get("metadata"))
        except Exception:
            metadata = {}
        return ReflectionSearchHit(
            id=str(row.get("id", "")),
            text=str(row.get("text", "")),
            score=float(row.get("_score") or row.get("_distance") or 0.0),
            kind=str(row.get("kind", "")),
            scope=str(row.get("scope", "")),
            timestamp=float(row.get("timestamp") or 0.0),
            importance=float(row.get("importance") or 0.0),
            metadata=metadata,
            event_id=str(row.get("event_id", "")),
            agent_id=str(row.get("agent_id", "")),
            session_id=str(row.get("session_id", "")),
        )


def _escape(value: str) -> str:
    return str(value).replace("'", "''")


__all__ = [
    "REFLECTION_TABLE_NAME",
    "REFLECTION_EMBED_DIM",
    "REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS",
    "REFLECTION_DERIVE_LOGISTIC_K",
    "REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT",
    "DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS",
    "DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS",
    "ReflectionStorePayload",
    "BuildReflectionStorePayloadsParams",
    "BuildReflectionStorePayloadsResult",
    "LoadReflectionSlicesParams",
    "LoadReflectionMappedRowsParams",
    "ReflectionSearchHit",
    "ReflectionStore",
    "build_reflection_store_payloads",
    "compute_derived_line_quality",
    "get_reflection_schema",
    "load_agent_reflection_slices_from_entries",
    "load_reflection_mapped_rows_from_entries",
    "resolve_reflection_importance",
]
