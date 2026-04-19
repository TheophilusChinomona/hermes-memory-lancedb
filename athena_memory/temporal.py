"""Temporal classifier — static vs. dynamic memories.

Ports ``temporal-classifier.ts``. A memory is:

- ``"static"``: durable fact (preferences, identity, allergies, addresses, ...)
- ``"dynamic"``: time-sensitive observation that should age out fast
  (today's plan, this week's deploy, "right now", ...)

Dynamic memories decay 3x faster than static ones — see
``DecayConfig.dynamic_decay_multiplier`` in :mod:`athena_memory.lifecycle`.

Classification strategy:
  1. Fast rule-based pre-pass (regex + CJK keywords). Cheap, deterministic.
  2. If an LLM client is provided AND the rules are ambiguous, ask the LLM for
     a 1-token classification. Falls back to the rule result on any failure.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


TemporalType = str  # Literal["static", "dynamic"]


# ---------------------------------------------------------------------------
# Rule-based patterns (mirrors the TS implementation)
# ---------------------------------------------------------------------------

_DYNAMIC_PATTERNS_EN = [
    r"\btoday\b", r"\byesterday\b", r"\btomorrow\b", r"\brecently\b",
    r"\bcurrently\b", r"\bright now\b", r"\bthis week\b", r"\bthis month\b",
    r"\blast week\b", r"\bnext week\b", r"\bthis morning\b", r"\btonight\b",
    r"\blater\b",
]

_DYNAMIC_KEYWORDS_ZH = [
    "今天", "昨天", "明天", "最近", "正在", "刚才", "刚刚",
    "这周", "这个月", "上周", "下周", "目前", "现在",
    "今晚", "今早", "稍后", "待会",
]

_STATIC_PATTERNS_EN = [
    r"\bfavorite\b", r"\bprefer\b", r"\balways\b", r"\bname is\b",
    r"\bborn\b", r"\bgraduated\b", r"\blive in\b", r"\bwork at\b",
    r"\bjob\b", r"\bprofession\b", r"\bhobby\b", r"\ballergic\b",
]

_STATIC_KEYWORDS_ZH = [
    "喜欢", "偏好", "一直", "名字", "叫做", "出生",
    "毕业", "住在", "工作", "职业", "爱好", "过敏",
]

_DYNAMIC_RE = [re.compile(p, re.IGNORECASE) for p in _DYNAMIC_PATTERNS_EN]
_STATIC_RE = [re.compile(p, re.IGNORECASE) for p in _STATIC_PATTERNS_EN]


def _rule_classify(text: str) -> TemporalType:
    if not text:
        return "static"
    has_dynamic = any(r.search(text) for r in _DYNAMIC_RE) or any(k in text for k in _DYNAMIC_KEYWORDS_ZH)
    has_static = any(r.search(text) for r in _STATIC_RE) or any(k in text for k in _STATIC_KEYWORDS_ZH)
    # Dynamic always wins when both match — same precedence as TS.
    if has_dynamic:
        return "dynamic"
    if has_static:
        return "static"
    return "static"


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You classify a memory as either STATIC (durable, time-invariant fact) "
    "or DYNAMIC (time-sensitive observation that will be stale within days "
    "or weeks). Reply with EXACTLY one JSON object: "
    '{"type": "static"} or {"type": "dynamic"}. No prose, no markdown.'
)


def _llm_classify(text: str, llm) -> Optional[TemporalType]:
    if llm is None:
        return None
    try:
        raw = llm.chat(_LLM_SYSTEM, f"Memory:\n{text[:1000]}", max_tokens=20)
    except Exception as e:
        logger.debug("Temporal LLM classify failed: %s", e)
        return None
    if not raw:
        return None
    raw = raw.strip()
    # Try strict JSON first.
    try:
        obj = json.loads(raw)
        t = str(obj.get("type", "")).lower()
        if t in ("static", "dynamic"):
            return t
    except Exception:
        pass
    # Fallback: keyword match.
    low = raw.lower()
    if "dynamic" in low:
        return "dynamic"
    if "static" in low:
        return "static"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_temporal(content: str, llm=None, *, prefer_llm: bool = False) -> TemporalType:
    """Return ``"static"`` or ``"dynamic"`` for the given content.

    By default the fast rule-based classifier is used and the LLM is only
    consulted when the rules return the safe default (``"static"``) but the
    text is non-trivial. Set ``prefer_llm=True`` to always consult the LLM.

    Always degrades gracefully — any LLM failure falls back to the rule
    result.
    """
    rule_result = _rule_classify(content)

    if llm is None:
        return rule_result
    if not prefer_llm and rule_result == "dynamic":
        # Rules are confident. Skip the LLM call.
        return rule_result
    if not prefer_llm and len(content.strip()) < 12:
        return rule_result

    llm_result = _llm_classify(content, llm)
    if llm_result is None:
        return rule_result
    return llm_result


# ---------------------------------------------------------------------------
# Expiry inference (utility)
# ---------------------------------------------------------------------------

_EXPIRY_RULES = [
    (re.compile(r"day after tomorrow", re.IGNORECASE), 48 * 3600),
    (re.compile(r"后天"), 48 * 3600),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), 24 * 3600),
    (re.compile(r"明天"), 24 * 3600),
    (re.compile(r"\bnext week\b", re.IGNORECASE), 7 * 24 * 3600),
    (re.compile(r"下周"), 7 * 24 * 3600),
    (re.compile(r"\bthis week\b", re.IGNORECASE), 3 * 24 * 3600),
    (re.compile(r"这周"), 3 * 24 * 3600),
    (re.compile(r"\bnext month\b", re.IGNORECASE), 30 * 24 * 3600),
    (re.compile(r"下个月"), 30 * 24 * 3600),
    (re.compile(r"\bthis month\b", re.IGNORECASE), 15 * 24 * 3600),
    (re.compile(r"这个月"), 15 * 24 * 3600),
    (re.compile(r"\btonight\b", re.IGNORECASE), 12 * 3600),
    (re.compile(r"今晚"), 12 * 3600),
    (re.compile(r"\btoday\b", re.IGNORECASE), 18 * 3600),
    (re.compile(r"今天"), 18 * 3600),
]


def infer_expiry(text: str, now: Optional[float] = None) -> Optional[float]:
    """Return a unix-seconds expiry timestamp inferred from temporal expressions.

    Returns ``None`` if no temporal expression is found.
    """
    if not text:
        return None
    base = now if now is not None else time.time()
    for pattern, offset in _EXPIRY_RULES:
        if pattern.search(text):
            return base + offset
    return None


__all__ = [
    "classify_temporal",
    "infer_expiry",
    "TemporalType",
]
