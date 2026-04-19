"""Smart metadata extraction — port of TS `src/smart-metadata.ts`.

Auto-extracts structured fields (memory_temporal_type, confidence, sensitivity,
modality, etc.) from candidate memory content via the LLM at write time. The
result is JSON-encoded and stored in the new `metadata` column on the
`memories` table.

Public API:
    extract_smart_metadata(content, llm, *, abstract="", category="cases", source="turn") -> Dict
    parse_smart_metadata(raw, entry=None) -> Dict
    stringify_smart_metadata(meta) -> str
    derive_fact_key(category, abstract) -> Optional[str]
    is_memory_active_at(meta, at=None) -> bool
    is_memory_expired(meta, at=None) -> bool

This is a pragmatic subset of the full TS schema — every field that LanceDB
needs to round-trip is preserved, and we leave room for arbitrary extras via
`**parsed`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TEMPORAL_TYPES = {"static", "dynamic"}
VALID_SENSITIVITIES = {"public", "internal", "private", "secret"}
VALID_MODALITIES = {"text", "code", "audio", "image", "video", "mixed"}
VALID_TIERS = {"core", "working", "peripheral"}
VALID_STATES = {"pending", "confirmed", "archived"}
VALID_SOURCES = {"manual", "auto-capture", "reflection", "session-summary", "legacy"}
VALID_LAYERS = {"durable", "working", "reflection", "archive"}

TEMPORAL_VERSIONED_CATEGORIES = {"profile", "preferences", "entities"}

_SMART_METADATA_SYSTEM = """You are a memory metadata extractor. Given the content of a memory, return a \
SHORT JSON object describing it. All fields are optional but should be filled when reasonably inferable.

Schema:
  memory_temporal_type: "static" (a fact unlikely to change, e.g. birthplace)
                       | "dynamic" (changes over time, e.g. current address)
  confidence:          float 0.0-1.0 — how confidently the fact was stated
  sensitivity:         "public" | "internal" | "private" | "secret"
  modality:            "text" | "code" | "audio" | "image" | "video" | "mixed"
  fact_key:            short slug (lowercase, snake-ish) for deduplication
  tags:                array of <= 6 short keyword strings

Return ONLY valid JSON, no commentary, no markdown fencing."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp01(v: Any, fallback: float = 0.7) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return fallback
    if n != n or n in (float("inf"), float("-inf")):
        return fallback
    return max(0.0, min(1.0, n))


def _clamp_count(v: Any, fallback: int = 0) -> int:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return fallback
    if n != n or n in (float("inf"), float("-inf")) or n < 0:
        return fallback
    return int(n)


def _normalize_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _normalize_optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_timestamp(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if n != n or n <= 0:
        return fallback
    return n


def _normalize_optional_timestamp(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n or n <= 0:
        return None
    return n


def _normalize_enum(value: Any, allowed: set, fallback: Optional[str] = None) -> Optional[str]:
    if isinstance(value, str) and value in allowed:
        return value
    return fallback


# ---------------------------------------------------------------------------
# Fact key derivation
# ---------------------------------------------------------------------------

_FACT_KEY_TRAILING = re.compile(r"[。.!?]+$")
_FACT_KEY_WHITESPACE = re.compile(r"\s+")


def derive_fact_key(category: str, abstract: str) -> Optional[str]:
    """Derive a stable fact_key for temporal categories (profile/preferences/entities)."""
    if category not in TEMPORAL_VERSIONED_CATEGORIES:
        return None
    trimmed = (abstract or "").strip()
    if not trimmed:
        return None

    topic = trimmed
    colon = re.match(r"^(.{1,120}?)[：:]", trimmed)
    arrow = re.match(r"^(.{1,120}?)(?:\s*->|\s*=>)", trimmed)
    if colon and colon.group(1):
        topic = colon.group(1)
    elif arrow and arrow.group(1):
        topic = arrow.group(1)

    normalized = topic.lower()
    normalized = _FACT_KEY_WHITESPACE.sub(" ", normalized)
    normalized = _FACT_KEY_TRAILING.sub("", normalized).strip()
    return f"{category}:{normalized}" if normalized else None


# ---------------------------------------------------------------------------
# Lifecycle predicates
# ---------------------------------------------------------------------------

def is_memory_active_at(meta: Dict[str, Any], at: Optional[float] = None) -> bool:
    """Return True if `at` is within the memory's [valid_from, invalidated_at) window."""
    moment = at if at is not None else time.time()
    valid_from = float(meta.get("valid_from") or 0.0)
    if valid_from > moment:
        return False
    invalidated = meta.get("invalidated_at")
    if invalidated is None:
        return True
    try:
        return float(invalidated) > moment
    except (TypeError, ValueError):
        return True


def is_memory_expired(meta: Dict[str, Any], at: Optional[float] = None) -> bool:
    """Return True if valid_until is set and has passed."""
    moment = at if at is not None else time.time()
    valid_until = meta.get("valid_until")
    if valid_until is None:
        return False
    try:
        return float(valid_until) <= moment
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Parse / stringify
# ---------------------------------------------------------------------------

def parse_smart_metadata(raw: Any, entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse a raw metadata JSON string (or dict) into a normalized dict.

    Always returns a dict — never raises. Missing fields fall back to defaults
    derived from the entry (when provided).
    """
    parsed: Dict[str, Any] = {}
    if isinstance(raw, dict):
        parsed = dict(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:
            parsed = {}

    entry = entry or {}
    text = entry.get("content") or entry.get("text") or ""
    timestamp = float(entry.get("timestamp") or time.time())
    category = entry.get("category") or parsed.get("memory_category") or "cases"

    l0 = _normalize_text(parsed.get("l0_abstract") or entry.get("abstract"), text[:160])
    l2 = _normalize_text(parsed.get("l2_content"), text)
    valid_from = _normalize_timestamp(parsed.get("valid_from"), timestamp)
    invalidated_at = _normalize_optional_timestamp(parsed.get("invalidated_at"))
    fact_key = (
        _normalize_optional_string(parsed.get("fact_key"))
        or derive_fact_key(category, l0)
    )

    out: Dict[str, Any] = dict(parsed)
    out.update({
        "l0_abstract": l0,
        "l1_overview": _normalize_text(parsed.get("l1_overview"), f"- {l0}"),
        "l2_content": l2,
        "memory_category": category,
        "tier": _normalize_enum(parsed.get("tier"), VALID_TIERS, "working"),
        "access_count": _clamp_count(parsed.get("access_count"), 0),
        "confidence": _clamp01(parsed.get("confidence"), 0.7),
        "last_accessed_at": _clamp_count(parsed.get("last_accessed_at"), int(timestamp)),
        "valid_from": valid_from,
        "invalidated_at": invalidated_at if invalidated_at and invalidated_at >= valid_from else None,
        "memory_temporal_type": _normalize_enum(parsed.get("memory_temporal_type"), VALID_TEMPORAL_TYPES),
        "valid_until": _normalize_optional_timestamp(parsed.get("valid_until")),
        "fact_key": fact_key,
        "supersedes": _normalize_optional_string(parsed.get("supersedes")),
        "superseded_by": _normalize_optional_string(parsed.get("superseded_by")),
        "state": _normalize_enum(parsed.get("state"), VALID_STATES, "confirmed"),
        "source": _normalize_enum(parsed.get("source"), VALID_SOURCES, "legacy"),
        "memory_layer": _normalize_enum(parsed.get("memory_layer"), VALID_LAYERS, "working"),
        "sensitivity": _normalize_enum(parsed.get("sensitivity"), VALID_SENSITIVITIES, "internal"),
        "modality": _normalize_enum(parsed.get("modality"), VALID_MODALITIES, "text"),
    })
    # Strip None values to keep the JSON tight.
    return {k: v for k, v in out.items() if v is not None}


# Soft caps to prevent the metadata blob from ballooning.
_MAX_SOURCES = 20
_MAX_HISTORY = 50
_MAX_RELATIONS = 16


def stringify_smart_metadata(meta: Dict[str, Any]) -> str:
    """Serialize metadata to JSON, capping bulky array fields."""
    capped = dict(meta)
    if isinstance(capped.get("sources"), list) and len(capped["sources"]) > _MAX_SOURCES:
        capped["sources"] = capped["sources"][-_MAX_SOURCES:]
    if isinstance(capped.get("history"), list) and len(capped["history"]) > _MAX_HISTORY:
        capped["history"] = capped["history"][-_MAX_HISTORY:]
    if isinstance(capped.get("relations"), list) and len(capped["relations"]) > _MAX_RELATIONS:
        capped["relations"] = capped["relations"][:_MAX_RELATIONS]
    return json.dumps(capped, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

def extract_smart_metadata(
    content: str,
    llm: Optional[Any] = None,
    *,
    abstract: str = "",
    category: str = "cases",
    source: str = "legacy",
    timestamp: Optional[float] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """Build the metadata dict for a candidate memory.

    If `llm` is provided and `skip_llm=False`, calls it to fill in
    `memory_temporal_type`, `confidence`, `sensitivity`, `modality`, `fact_key`,
    and `tags`. Falls back to heuristic defaults on any LLM failure.

    Always returns a fully-normalized dict (via `parse_smart_metadata`).
    """
    ts = float(timestamp) if timestamp is not None else time.time()
    base_entry = {
        "content": content,
        "abstract": abstract,
        "category": category,
        "timestamp": ts,
    }

    llm_payload: Dict[str, Any] = {}
    if llm is not None and not skip_llm and content.strip():
        try:
            user_prompt = (
                f"Memory category: {category}\n"
                f"Abstract: {abstract or '(none)'}\n"
                f"Content:\n{content[:1800]}"
            )
            raw = llm.chat(_SMART_METADATA_SYSTEM, user_prompt, max_tokens=300)
            if isinstance(raw, str) and raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    llm_payload = obj
        except Exception as e:
            logger.debug("smart metadata LLM call failed: %s", e)

    seed: Dict[str, Any] = {
        "l0_abstract": abstract or content[:160],
        "l2_content": content,
        "memory_category": category,
        "source": source if source in VALID_SOURCES else "legacy",
        "valid_from": ts,
        "last_accessed_at": int(ts),
        "memory_temporal_type": llm_payload.get("memory_temporal_type"),
        "confidence": llm_payload.get("confidence", 0.7),
        "sensitivity": llm_payload.get("sensitivity"),
        "modality": llm_payload.get("modality"),
        "fact_key": llm_payload.get("fact_key"),
        "tags": llm_payload.get("tags") if isinstance(llm_payload.get("tags"), list) else None,
    }
    seed = {k: v for k, v in seed.items() if v is not None}
    return parse_smart_metadata(seed, base_entry)


__all__ = [
    "TEMPORAL_VERSIONED_CATEGORIES",
    "VALID_TEMPORAL_TYPES",
    "VALID_SENSITIVITIES",
    "VALID_MODALITIES",
    "extract_smart_metadata",
    "parse_smart_metadata",
    "stringify_smart_metadata",
    "derive_fact_key",
    "is_memory_active_at",
    "is_memory_expired",
]
