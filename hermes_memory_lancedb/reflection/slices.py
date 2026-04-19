"""Slice / segment reflections for partial recall.

Python port of `reflection-slices.ts`. Reflection markdown follows a
section-driven schema:

    ## Invariants
    - <stable rule>

    ## Derived
    - <session-specific delta>

    ## User model deltas (about the human)
    - <preference change>

    ## Lessons & pitfalls (symptom / cause / fix / prevention)
    - <bullet>

    ## Decisions (durable)
    - <decision>

    ## Open loops / next actions
    - <follow-up>

    ## Learning governance candidates (.learnings / promotion / skill extraction)
    ### Entry
    **Priority**: high
    **Status**: pending
    **Area**: tooling
    ### Summary
    <one paragraph>
    ### Details
    <multi-line>
    ### Suggested Action
    <multi-line>

This module parses that markdown into typed records, with a strong focus on
sanitizing lines that look like prompt-injection attempts before they ever
get materialized as memory text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


ReflectionMappedKind = Literal["user-model", "agent-model", "lesson", "decision"]
ReflectionItemKind = Literal["invariant", "derived"]
ReflectionMappedCategory = Literal["preference", "fact", "decision"]


@dataclass
class ReflectionSlices:
    invariants: List[str]
    derived: List[str]


@dataclass
class ReflectionMappedMemory:
    text: str
    category: ReflectionMappedCategory
    heading: str


@dataclass
class ReflectionMappedMemoryItem:
    text: str
    category: ReflectionMappedCategory
    heading: str
    mapped_kind: ReflectionMappedKind
    ordinal: int
    group_size: int


@dataclass
class ReflectionSliceItem:
    text: str
    item_kind: ReflectionItemKind
    section: Literal["Invariants", "Derived"]
    ordinal: int
    group_size: int


@dataclass
class ReflectionGovernanceEntry:
    summary: str
    priority: Optional[str] = None
    status: Optional[str] = None
    area: Optional[str] = None
    details: Optional[str] = None
    suggested_action: Optional[str] = None


# ---------------------------------------------------------------------------
# Section / bullet extraction
# ---------------------------------------------------------------------------


_LINE_SPLIT = re.compile(r"\r?\n")


def extract_section_markdown(markdown: str, heading: str) -> str:
    """Return the body of the first ``## <heading>`` section, trimmed."""
    if not markdown:
        return ""
    lines = _LINE_SPLIT.split(markdown)
    needle = f"## {heading}".lower()
    in_section = False
    collected: List[str] = []
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("## "):
            if in_section and lower != needle:
                break
            in_section = lower == needle
            continue
        if not in_section:
            continue
        collected.append(raw)
    return "\n".join(collected).strip()


def parse_section_bullets(markdown: str, heading: str) -> List[str]:
    """Extract the bullet lines under ``## <heading>``."""
    section = extract_section_markdown(markdown, heading)
    if not section:
        return []
    out: List[str] = []
    for raw in section.split("\n"):
        line = raw.strip()
        if line.startswith("- ") or line.startswith("* "):
            normalized = line[2:].strip()
            if normalized:
                out.append(normalized)
    return out


# ---------------------------------------------------------------------------
# Sanitizers — placeholder filtering and prompt-injection guard
# ---------------------------------------------------------------------------


_PLACEHOLDER_NONE = re.compile(r"^\(none( captured)?\)$", re.IGNORECASE)
_PLACEHOLDER_HEADER = re.compile(r"^(invariants?|reflections?|derived)[:：]$", re.IGNORECASE)
_PLACEHOLDER_DELTA1 = re.compile(r"apply this session'?s deltas next run", re.IGNORECASE)
_PLACEHOLDER_DELTA2 = re.compile(r"apply this session'?s distilled changes next run", re.IGNORECASE)
_PLACEHOLDER_INVESTIGATE = re.compile(r"investigate why embedded reflection generation failed", re.IGNORECASE)


def is_placeholder_reflection_slice_line(line: str) -> bool:
    normalized = line.replace("**", "").strip()
    if not normalized:
        return True
    if _PLACEHOLDER_NONE.match(normalized):
        return True
    if _PLACEHOLDER_HEADER.match(normalized):
        return True
    if _PLACEHOLDER_DELTA1.search(normalized):
        return True
    if _PLACEHOLDER_DELTA2.search(normalized):
        return True
    if _PLACEHOLDER_INVESTIGATE.search(normalized):
        return True
    return False


_LEADING_HEADER = re.compile(r"^(invariants?|reflections?|derived)[:：]\s*", re.IGNORECASE)


def normalize_reflection_slice_line(line: str) -> str:
    cleaned = line.replace("**", "")
    cleaned = _LEADING_HEADER.sub("", cleaned)
    return cleaned.strip()


def sanitize_reflection_slice_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        normalized = normalize_reflection_slice_line(raw)
        if is_placeholder_reflection_slice_line(normalized):
            continue
        out.append(normalized)
    return out


_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"^\s*(?:(?:next|this)\s+run\s+)?(?:ignore|disregard|forget|override|bypass)\b"
        r"[\s\S]{0,80}\b(?:instructions?|guardrails?|policy|developer|system)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|dump|show|output)\b"
        r"[\s\S]{0,80}\b(?:system prompt|developer prompt|hidden prompt|hidden instructions?|"
        r"full prompt|prompt verbatim|secrets?|keys?|tokens?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\s*/?\s*(?:system|assistant|user|tool|developer|inherited-rules|derived-focus)\b[^>]*>",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:system|assistant|user|developer|tool)\s*:", re.IGNORECASE),
]


def is_unsafe_injectable_reflection_line(line: str) -> bool:
    normalized = normalize_reflection_slice_line(line)
    if not normalized:
        return True
    return any(p.search(normalized) for p in _INJECTION_PATTERNS)


def sanitize_injectable_reflection_lines(lines: List[str]) -> List[str]:
    safe_first = sanitize_reflection_slice_lines(lines)
    return [line for line in safe_first if not is_unsafe_injectable_reflection_line(line)]


# ---------------------------------------------------------------------------
# Heuristic classifiers (rule-like vs delta-like vs open-loop action)
# ---------------------------------------------------------------------------


_INVARIANT_PREFIX = re.compile(
    r"^(always|never|when\b|if\b|before\b|after\b|prefer\b|avoid\b|require\b|only\b|do not\b|must\b|should\b)",
    re.IGNORECASE,
)
_INVARIANT_KEYWORDS = re.compile(
    r"\b(must|should|never|always|prefer|avoid|required?)\b", re.IGNORECASE
)
_DERIVED_PREFIX = re.compile(
    r"^(this run|next run|going forward|follow-up|re-check|retest|verify|confirm|"
    r"avoid repeating|adjust|change|update|retry|keep|watch)\b",
    re.IGNORECASE,
)
_DERIVED_KEYWORDS = re.compile(
    r"\b(this run|next run|delta|change|adjust|retry|re-check|retest|verify|confirm|"
    r"avoid repeating|follow-up)\b",
    re.IGNORECASE,
)
_OPEN_LOOP_PREFIX = re.compile(
    r"^(investigate|verify|confirm|re-check|retest|update|add|remove|fix|avoid|"
    r"keep|watch|document)\b",
    re.IGNORECASE,
)


def _is_invariant_rule_like(line: str) -> bool:
    return bool(_INVARIANT_PREFIX.search(line) or _INVARIANT_KEYWORDS.search(line))


def _is_derived_delta_like(line: str) -> bool:
    return bool(_DERIVED_PREFIX.search(line) or _DERIVED_KEYWORDS.search(line))


def _is_open_loop_action(line: str) -> bool:
    return bool(_OPEN_LOOP_PREFIX.search(line))


# ---------------------------------------------------------------------------
# Lessons + governance + mapped extractors
# ---------------------------------------------------------------------------


_LESSONS_HEADING = "Lessons & pitfalls (symptom / cause / fix / prevention)"
_GOVERNANCE_HEADING = "Learning governance candidates (.learnings / promotion / skill extraction)"


def extract_reflection_lessons(reflection_text: str) -> List[str]:
    return sanitize_reflection_slice_lines(parse_section_bullets(reflection_text, _LESSONS_HEADING))


_ENTRY_BLOCK_SPLIT = re.compile(r"(?=^###\s+Entry\b)", re.IGNORECASE | re.MULTILINE)
_ENTRY_HEADING_STRIP = re.compile(r"^###\s+Entry\b[^\n]*\n?", re.IGNORECASE)


def _read_field(body: str, label: str) -> Optional[str]:
    pattern = re.compile(rf"^\*\*{re.escape(label)}\*\*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(body)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _read_section(body: str, label: str) -> Optional[str]:
    pattern = re.compile(
        rf"^###\s+{re.escape(label)}\s*\n([\s\S]*?)(?=^###\s+|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(body)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _parse_governance_entry(block: str) -> Optional[ReflectionGovernanceEntry]:
    body = _ENTRY_HEADING_STRIP.sub("", block, count=1).strip()
    if not body:
        return None
    summary = _read_section(body, "Summary")
    if not summary:
        return None
    return ReflectionGovernanceEntry(
        summary=summary,
        priority=_read_field(body, "Priority"),
        status=_read_field(body, "Status"),
        area=_read_field(body, "Area"),
        details=_read_section(body, "Details"),
        suggested_action=_read_section(body, "Suggested Action"),
    )


def extract_reflection_learning_governance_candidates(
    reflection_text: str,
) -> List[ReflectionGovernanceEntry]:
    section = extract_section_markdown(reflection_text, _GOVERNANCE_HEADING)
    if not section:
        return []

    blocks = [b.strip() for b in _ENTRY_BLOCK_SPLIT.split(section) if b.strip()]
    parsed = [entry for b in blocks if (entry := _parse_governance_entry(b)) is not None]
    if parsed:
        return parsed

    fallback = sanitize_reflection_slice_lines(parse_section_bullets(reflection_text, _GOVERNANCE_HEADING))
    if not fallback:
        return []
    return [
        ReflectionGovernanceEntry(
            summary="Reflection learning governance candidates",
            priority="medium",
            status="pending",
            area="config",
            details="\n".join(f"- {line}" for line in fallback),
            suggested_action=(
                "Review the governance candidates, promote durable rules to AGENTS.md / "
                "SOUL.md / TOOLS.md when stable, and extract a skill if the pattern becomes reusable."
            ),
        )
    ]


_MAPPED_SECTIONS: List[tuple[str, ReflectionMappedCategory, ReflectionMappedKind]] = [
    ("User model deltas (about the human)", "preference", "user-model"),
    ("Agent model deltas (about the assistant/system)", "preference", "agent-model"),
    ("Lessons & pitfalls (symptom / cause / fix / prevention)", "fact", "lesson"),
    ("Decisions (durable)", "decision", "decision"),
]


def _extract_mapped_with(
    reflection_text: str,
    sanitize: Callable[[List[str]], List[str]],
) -> List[ReflectionMappedMemoryItem]:
    out: List[ReflectionMappedMemoryItem] = []
    for heading, category, mapped_kind in _MAPPED_SECTIONS:
        lines = sanitize(parse_section_bullets(reflection_text, heading))
        group_size = len(lines)
        for ordinal, text in enumerate(lines):
            out.append(ReflectionMappedMemoryItem(
                text=text,
                category=category,
                heading=heading,
                mapped_kind=mapped_kind,
                ordinal=ordinal,
                group_size=group_size,
            ))
    return out


def extract_reflection_mapped_memory_items(reflection_text: str) -> List[ReflectionMappedMemoryItem]:
    return _extract_mapped_with(reflection_text, sanitize_reflection_slice_lines)


def extract_injectable_reflection_mapped_memory_items(
    reflection_text: str,
) -> List[ReflectionMappedMemoryItem]:
    return _extract_mapped_with(reflection_text, sanitize_injectable_reflection_lines)


def extract_reflection_mapped_memories(reflection_text: str) -> List[ReflectionMappedMemory]:
    return [
        ReflectionMappedMemory(text=item.text, category=item.category, heading=item.heading)
        for item in extract_reflection_mapped_memory_items(reflection_text)
    ]


def extract_injectable_reflection_mapped_memories(reflection_text: str) -> List[ReflectionMappedMemory]:
    return [
        ReflectionMappedMemory(text=item.text, category=item.category, heading=item.heading)
        for item in extract_injectable_reflection_mapped_memory_items(reflection_text)
    ]


# ---------------------------------------------------------------------------
# Slices (Invariants + Derived) extraction with legacy fallbacks
# ---------------------------------------------------------------------------


def _extract_slices_with(
    reflection_text: str,
    sanitize: Callable[[List[str]], List[str]],
) -> ReflectionSlices:
    invariant_section = parse_section_bullets(reflection_text, "Invariants")
    derived_section = parse_section_bullets(reflection_text, "Derived")
    merged_section = parse_section_bullets(reflection_text, "Invariants & Reflections")

    invariants_primary = [
        line for line in sanitize(invariant_section) if _is_invariant_rule_like(line)
    ]
    derived_primary = [
        line for line in sanitize(derived_section) if _is_derived_delta_like(line)
    ]

    invariant_legacy = sanitize([
        line for line in merged_section
        if re.search(r"invariant|stable|policy|rule", line, re.IGNORECASE)
    ])
    invariant_legacy = [line for line in invariant_legacy if _is_invariant_rule_like(line)]

    reflection_legacy = sanitize([
        line for line in merged_section
        if re.search(r"reflect|inherit|derive|change|apply", line, re.IGNORECASE)
    ])
    reflection_legacy = [line for line in reflection_legacy if _is_derived_delta_like(line)]

    open_loop_lines = sanitize(parse_section_bullets(reflection_text, "Open loops / next actions"))
    open_loop_lines = [
        line for line in open_loop_lines
        if _is_open_loop_action(line) and _is_derived_delta_like(line)
    ]

    durable_decisions = sanitize(parse_section_bullets(reflection_text, "Decisions (durable)"))
    durable_decisions = [line for line in durable_decisions if _is_invariant_rule_like(line)]

    if invariants_primary:
        invariants = invariants_primary
    elif invariant_legacy:
        invariants = invariant_legacy
    else:
        invariants = durable_decisions

    if derived_primary:
        derived = derived_primary
    else:
        derived = [*reflection_legacy, *open_loop_lines]

    return ReflectionSlices(invariants=invariants[:8], derived=derived[:10])


def extract_reflection_slices(reflection_text: str) -> ReflectionSlices:
    return _extract_slices_with(reflection_text, sanitize_reflection_slice_lines)


def extract_injectable_reflection_slices(reflection_text: str) -> ReflectionSlices:
    return _extract_slices_with(reflection_text, sanitize_injectable_reflection_lines)


def _build_slice_items_from_slices(slices: ReflectionSlices) -> List[ReflectionSliceItem]:
    inv_size = len(slices.invariants)
    der_size = len(slices.derived)

    items: List[ReflectionSliceItem] = []
    for ordinal, text in enumerate(slices.invariants):
        items.append(ReflectionSliceItem(
            text=text, item_kind="invariant", section="Invariants",
            ordinal=ordinal, group_size=inv_size,
        ))
    for ordinal, text in enumerate(slices.derived):
        items.append(ReflectionSliceItem(
            text=text, item_kind="derived", section="Derived",
            ordinal=ordinal, group_size=der_size,
        ))
    return items


def extract_reflection_slice_items(reflection_text: str) -> List[ReflectionSliceItem]:
    return _build_slice_items_from_slices(extract_reflection_slices(reflection_text))


def extract_injectable_reflection_slice_items(reflection_text: str) -> List[ReflectionSliceItem]:
    return _build_slice_items_from_slices(extract_injectable_reflection_slices(reflection_text))


__all__ = [
    "ReflectionMappedKind",
    "ReflectionItemKind",
    "ReflectionMappedCategory",
    "ReflectionSlices",
    "ReflectionMappedMemory",
    "ReflectionMappedMemoryItem",
    "ReflectionSliceItem",
    "ReflectionGovernanceEntry",
    "extract_section_markdown",
    "parse_section_bullets",
    "is_placeholder_reflection_slice_line",
    "normalize_reflection_slice_line",
    "sanitize_reflection_slice_lines",
    "is_unsafe_injectable_reflection_line",
    "sanitize_injectable_reflection_lines",
    "extract_reflection_lessons",
    "extract_reflection_learning_governance_candidates",
    "extract_reflection_mapped_memory_items",
    "extract_injectable_reflection_mapped_memory_items",
    "extract_reflection_mapped_memories",
    "extract_injectable_reflection_mapped_memories",
    "extract_reflection_slices",
    "extract_injectable_reflection_slices",
    "extract_reflection_slice_items",
    "extract_injectable_reflection_slice_items",
]
