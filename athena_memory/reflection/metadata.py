"""Reflection metadata helpers — parse + classify the JSON metadata blob.

Python port of `reflection-metadata.ts`. The TS port stores the metadata as a
JSON string on each memory entry; we keep the same on-disk shape so the two
implementations stay interchangeable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional


_REFLECTION_METADATA_TYPES = frozenset({
    "memory-reflection",
    "memory-reflection-event",
    "memory-reflection-item",
    "memory-reflection-mapped",
})


def parse_reflection_metadata(metadata_raw: Optional[str]) -> Dict[str, Any]:
    """Parse a metadata JSON blob, returning {} on any failure."""
    if not metadata_raw:
        return {}
    try:
        parsed = json.loads(metadata_raw)
    except (TypeError, ValueError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def is_reflection_entry(entry: Mapping[str, Any]) -> bool:
    """True if the entry's category or metadata.type marks it as reflection."""
    if entry.get("category") == "reflection":
        return True
    metadata = parse_reflection_metadata(entry.get("metadata"))
    return metadata.get("type") in _REFLECTION_METADATA_TYPES


def get_display_category_tag(entry: Mapping[str, Any]) -> str:
    """Human-readable category tag, collapsing reflection variants under
    a single ``reflection:<scope>`` label.
    """
    if not is_reflection_entry(entry):
        return f"{entry.get('category', '')}:{entry.get('scope', '')}"
    return f"reflection:{entry.get('scope', '')}"


__all__ = [
    "parse_reflection_metadata",
    "is_reflection_entry",
    "get_display_category_tag",
]
