"""Query intent analysis + expansion.

Combined port of ``intent-analyzer.ts`` and ``query-expander.ts``.

- :func:`analyze_intent` — rule-based pattern matching that classifies a query
  as ``preference`` / ``decision`` / ``entity`` / ``event`` / ``fact`` / ``broad``
  and recommends a recall depth + confidence. Optional LLM upgrade for
  ambiguous cases.
- :func:`expand_query` — produces synonym/related-term expansions for short
  queries. Also rule-based by default with optional LLM augmentation.
- :func:`apply_category_boost` — score multiplier for hits matching the
  detected intent categories. Mirrors the TS helper.

All LLM paths degrade gracefully — callers receive the rule-based result
when the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Map intent categories to the actual stored MEMORY_CATEGORIES used by this
# Python port. The TS port had different category names.
INTENT_TO_STORED_CATEGORIES = {
    "preference": ["preferences", "patterns"],
    "decision": ["events", "cases", "patterns"],
    "entity": ["entities", "profile"],
    "event": ["events", "cases"],
    "fact": ["entities", "profile", "patterns"],
    "broad": [],
    "empty": [],
}


@dataclass
class IntentSignal:
    label: str
    categories: List[str]
    depth: str = "l1"  # "l0" | "l1" | "full"
    confidence: str = "low"  # "high" | "medium" | "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "categories": list(self.categories),
            "depth": self.depth,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Intent rules (most specific first)
# ---------------------------------------------------------------------------

@dataclass
class _IntentRule:
    label: str
    patterns: List[re.Pattern]
    depth: str


_INTENT_RULES: List[_IntentRule] = [
    _IntentRule(
        label="preference",
        patterns=[
            re.compile(r"\b(prefer|preference|style|convention|like|dislike|favorite|habit)\b", re.IGNORECASE),
            re.compile(r"\b(how do (i|we) usually|what('s| is) (my|our) (style|convention|approach))\b", re.IGNORECASE),
            re.compile(r"(偏好|喜欢|习惯|风格|惯例|常用|不喜欢|不要用|别用)"),
        ],
        depth="l0",
    ),
    _IntentRule(
        label="decision",
        patterns=[
            re.compile(r"\b(why did (we|i)|decision|decided|chose|rationale|trade-?off|reason for)\b", re.IGNORECASE),
            re.compile(r"\b(what was the (reason|rationale|decision))\b", re.IGNORECASE),
            re.compile(r"(为什么选|决定|选择了|取舍|权衡|原因是|当时决定)"),
        ],
        depth="l1",
    ),
    _IntentRule(
        label="entity",
        patterns=[
            re.compile(r"\b(who is|who are|tell me about|info on|details about|contact info)\b", re.IGNORECASE),
            re.compile(r"\b(who('s| is) (the|our|my)|what team|which (person|team))\b", re.IGNORECASE),
            re.compile(r"(谁是|告诉我关于|详情|联系方式|哪个团队)"),
        ],
        depth="l1",
    ),
    _IntentRule(
        label="event",
        patterns=[
            re.compile(r"\b(when did|what happened|timeline|incident|outage|deploy|release|shipped)\b", re.IGNORECASE),
            re.compile(r"\b(last (week|month|time|sprint)|recently|yesterday|today)\b", re.IGNORECASE),
            re.compile(r"(什么时候|发生了什么|时间线|事件|上线|部署|发布|上次|最近)"),
        ],
        depth="full",
    ),
    _IntentRule(
        label="fact",
        patterns=[
            re.compile(r"\b(how (does|do|to)|what (does|do|is)|explain|documentation|spec)\b", re.IGNORECASE),
            re.compile(r"\b(config|configuration|setup|install|architecture|api|endpoint)\b", re.IGNORECASE),
            re.compile(r"(怎么|如何|是什么|解释|文档|规范|配置|安装|架构|接口)"),
        ],
        depth="l1",
    ),
    _IntentRule(
        label="contradiction",
        patterns=[
            re.compile(r"\b(contradiction|conflict|inconsistency|inconsistent|disagree|did i (say|claim))\b", re.IGNORECASE),
            re.compile(r"(矛盾|冲突|不一致)"),
        ],
        depth="full",
    ),
    _IntentRule(
        label="lookup",
        patterns=[
            re.compile(r"\b(lookup|look up|find|search for|where is)\b", re.IGNORECASE),
        ],
        depth="l1",
    ),
    _IntentRule(
        label="recall",
        patterns=[
            re.compile(r"\b(recall|remember|do you remember|what did (i|we) (say|tell|mention))\b", re.IGNORECASE),
            re.compile(r"(还记得|记得吗|我说过|告诉过你)"),
        ],
        depth="l1",
    ),
]


# ---------------------------------------------------------------------------
# analyze_intent
# ---------------------------------------------------------------------------

_LLM_INTENT_SYSTEM = (
    "You classify a user query for a memory retrieval system. Reply with "
    "STRICT JSON: {\"label\": \"<lookup|recall|contradiction|preference|"
    "decision|entity|event|fact|broad>\", \"depth\": \"<l0|l1|full>\", "
    "\"confidence\": \"<high|medium|low>\"}. No prose, no markdown."
)


def _llm_intent(query: str, llm) -> Optional[IntentSignal]:
    if llm is None:
        return None
    try:
        raw = llm.chat(_LLM_INTENT_SYSTEM, f"Query:\n{query[:600]}", max_tokens=80)
    except Exception as e:
        logger.debug("analyze_intent LLM failed: %s", e)
        return None
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        label = str(obj.get("label", "broad")).lower()
        depth = str(obj.get("depth", "l1")).lower()
        conf = str(obj.get("confidence", "low")).lower()
        cats = INTENT_TO_STORED_CATEGORIES.get(label, [])
        if depth not in ("l0", "l1", "full"):
            depth = "l1"
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return IntentSignal(label=label, categories=cats, depth=depth, confidence=conf)
    except Exception as e:
        logger.debug("analyze_intent LLM parse failed: %s", e)
        return None


def analyze_intent(query: str, llm=None) -> IntentSignal:
    """Classify ``query`` into an :class:`IntentSignal`. Always returns a value.

    Strategy: rules first; if rules return ``broad`` and an LLM is provided
    AND the query is non-trivial, ask the LLM. Any LLM failure falls back
    to the rule-based broad signal.
    """
    trimmed = (query or "").strip()
    if not trimmed:
        return IntentSignal(label="empty", categories=[], depth="l0", confidence="low")

    for rule in _INTENT_RULES:
        if any(p.search(trimmed) for p in rule.patterns):
            return IntentSignal(
                label=rule.label,
                categories=INTENT_TO_STORED_CATEGORIES.get(rule.label, []),
                depth=rule.depth,
                confidence="high",
            )

    if llm is not None and len(trimmed) >= 8:
        llm_result = _llm_intent(trimmed, llm)
        if llm_result is not None:
            return llm_result

    return IntentSignal(label="broad", categories=[], depth="l0", confidence="low")


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

# Synonym map — mirrors the TS file plus a small EN-first set.
_SYNONYM_MAP: List[Dict[str, List[str]]] = [
    {"keys": ["挂了", "挂掉", "宕机", "shutdown", "crashed"], "expansions": ["崩溃", "crash", "error", "报错", "宕机", "失败"]},
    {"keys": ["卡住", "卡死", "没反应", "hung", "frozen"], "expansions": ["hang", "timeout", "超时", "无响应", "stuck"]},
    {"keys": ["炸了", "爆了", "oom"], "expansions": ["崩溃", "crash", "OOM", "内存溢出", "error"]},
    {"keys": ["配置", "设置", "config", "configuration"], "expansions": ["config", "configuration", "settings", "setup"]},
    {"keys": ["部署", "上线", "deploy", "deployment"], "expansions": ["deploy", "deployment", "release", "ship", "rollout"]},
    {"keys": ["容器", "docker", "container"], "expansions": ["docker", "container", "compose", "docker-compose"]},
    {"keys": ["报错", "出错", "错误", "error", "exception"], "expansions": ["error", "exception", "failure", "bug", "fault"]},
    {"keys": ["修复", "修了", "修好", "bugfix", "hotfix"], "expansions": ["fix", "patch", "repair", "resolve"]},
    {"keys": ["踩坑", "troubleshoot"], "expansions": ["troubleshoot", "debug", "investigate", "diagnose"]},
    {"keys": ["记忆", "memory", "记忆系统"], "expansions": ["memory", "recall", "lancedb", "embedding"]},
    {"keys": ["搜索", "查找", "找不到", "search", "retrieval"], "expansions": ["search", "retrieval", "lookup", "find"]},
    {"keys": ["推送", "git push"], "expansions": ["push", "commit", "merge", "pr"]},
    {"keys": ["日志", "logs", "logfile", "logging"], "expansions": ["log", "logs", "logging", "stderr", "stdout"]},
    {"keys": ["权限", "permission", "authorization"], "expansions": ["permission", "auth", "access", "role", "rbac"]},
    {"keys": ["数据库", "database", "db"], "expansions": ["database", "db", "sql", "postgres", "schema"]},
    {"keys": ["环境变量", "env", "environment"], "expansions": ["env", "environment", "variable", "config"]},
]

_MAX_EXPANSIONS = 5


def _rule_expand(query: str) -> List[str]:
    if not query or len(query.strip()) < 2:
        return []
    low = query.lower()
    additions: List[str] = []
    seen: set = set()
    for entry in _SYNONYM_MAP:
        if any(k.lower() in low for k in entry["keys"]):
            for exp in entry["expansions"]:
                low_exp = exp.lower()
                if low_exp in low or low_exp in seen:
                    continue
                seen.add(low_exp)
                additions.append(exp)
                if len(additions) >= _MAX_EXPANSIONS:
                    return additions
    return additions


_LLM_EXPAND_SYSTEM = (
    "You expand a search query with related terms to improve recall. "
    "Reply with STRICT JSON: {\"expansions\": [\"term1\", \"term2\", ...]}. "
    "Provide 2-5 related terms (synonyms, related concepts, alternative "
    "phrasings). No prose, no markdown."
)


def _llm_expand(query: str, llm) -> List[str]:
    if llm is None:
        return []
    try:
        raw = llm.chat(_LLM_EXPAND_SYSTEM, f"Query:\n{query[:300]}", max_tokens=120)
    except Exception as e:
        logger.debug("expand_query LLM failed: %s", e)
        return []
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        items = obj.get("expansions", [])
        if not isinstance(items, list):
            return []
        out: List[str] = []
        seen: set = set()
        for it in items:
            s = str(it).strip()
            if 0 < len(s) <= 60 and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
            if len(out) >= _MAX_EXPANSIONS:
                break
        return out
    except Exception as e:
        logger.debug("expand_query LLM parse failed: %s", e)
        return []


def expand_query(query: str, llm=None, *, prefer_llm: bool = False) -> List[str]:
    """Return a list of expanded query strings (including the original).

    The first element is always the original query. Subsequent elements are
    augmented variants like ``"<query> <terms...>"``. Always returns at
    least the original query.
    """
    base = (query or "").strip()
    if not base:
        return []

    rule_terms = _rule_expand(base)
    out: List[str] = [base]

    if prefer_llm and llm is not None:
        llm_terms = _llm_expand(base, llm)
    else:
        llm_terms = []
    if not llm_terms and llm is not None and not rule_terms:
        llm_terms = _llm_expand(base, llm)

    seen_terms: set = set()
    combined: List[str] = []
    for t in rule_terms + llm_terms:
        low = t.lower()
        if low in seen_terms or low in base.lower():
            continue
        seen_terms.add(low)
        combined.append(t)
        if len(combined) >= _MAX_EXPANSIONS:
            break

    if combined:
        out.append(f"{base} {' '.join(combined)}")
        # Also include the bare expansions as a separate query so BM25 can
        # match documents that contain only the synonyms.
        out.append(" ".join(combined))
    return out


# ---------------------------------------------------------------------------
# Score boosting
# ---------------------------------------------------------------------------

def apply_category_boost(
    hits: List[Dict[str, Any]],
    intent: IntentSignal,
    *,
    boost_factor: float = 1.15,
) -> List[Dict[str, Any]]:
    """Boost scores of hits whose category matches ``intent.categories``.

    Mutates the hit dicts and returns them re-sorted descending by score.
    """
    if not intent.categories or intent.confidence == "low":
        return hits
    priority = set(intent.categories)
    for h in hits:
        if h.get("category") in priority:
            h["score"] = min(1.0, float(h.get("score", 0.0)) * boost_factor)
    hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return hits


__all__ = [
    "IntentSignal",
    "INTENT_TO_STORED_CATEGORIES",
    "analyze_intent",
    "expand_query",
    "apply_category_boost",
]
