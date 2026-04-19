"""Session compression + recovery.

Loose port of ``session-compressor.ts`` (text scoring) merged with
``session-recovery.ts``-style behaviour: on session end, summarize the full
conversation into one dense memory entry; on session start with a known
``session_id``, re-inflate that summary into the recall context.

Two layers:

- :func:`compress_texts` — pure scoring + budget-aware selection. No LLM.
- :func:`compress_session` — uses an LLM (when available) to produce a single
  dense summary from the selected texts. Falls back to a heuristic
  concatenation when the LLM call fails.
- :func:`recover_session` — reads compressed entries from a store-like
  object and returns them as an injectable context block.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_TOOL_CALL_INDICATORS = [
    re.compile(r"\btool_use\b", re.IGNORECASE),
    re.compile(r"\btool_result\b", re.IGNORECASE),
    re.compile(r"\bfunction_call\b", re.IGNORECASE),
    re.compile(r"\b(memory_store|memory_recall|memory_forget|memory_update)\b", re.IGNORECASE),
    re.compile(r"\b(lancedb_search|lancedb_remember|lancedb_forget|lancedb_stats|lancedb_compact)\b", re.IGNORECASE),
]

_CORRECTION_INDICATORS = [
    re.compile(r"^no[,.\s]", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"\bcorrect(ion)?\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
    re.compile(r"不对"), re.compile(r"应该是"), re.compile(r"應該是"),
    re.compile(r"错了"), re.compile(r"錯了"), re.compile(r"改成"),
]

_DECISION_INDICATORS = [
    re.compile(r"\blet'?s go with\b", re.IGNORECASE),
    re.compile(r"\bconfirmed?\b", re.IGNORECASE),
    re.compile(r"\bapproved?\b", re.IGNORECASE),
    re.compile(r"\bdecided?\b", re.IGNORECASE),
    re.compile(r"\bwe'?ll use\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bfrom now on\b", re.IGNORECASE),
    re.compile(r"\bagreed\b", re.IGNORECASE),
    re.compile(r"决定"), re.compile(r"決定"),
    re.compile(r"确认"), re.compile(r"確認"),
]

_ACK_PATTERNS = [
    re.compile(
        r"^(ok|okay|k|sure|fine|thanks|thank you|thx|ty|got it|understood|cool|"
        r"nice|great|good|perfect|awesome|alright|yep|yup|yeah|right)\s*[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^好的?\s*[。！]?$"),
    re.compile(r"^嗯\s*[。]?$"),
    re.compile(r"^收到\s*[。！]?$"),
    re.compile(r"^了解\s*[。！]?$"),
]

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


@dataclass
class ScoredText:
    index: int
    text: str
    score: float
    reason: str


@dataclass
class CompressResult:
    texts: List[str]
    scored: List[ScoredText]
    dropped: int
    total_chars: int


def score_text(text: str, index: int) -> ScoredText:
    """Score a text segment in [0, 1] by information density."""
    trimmed = text.strip()
    if not trimmed:
        return ScoredText(index, text, 0.0, "empty")
    if any(p.search(trimmed) for p in _TOOL_CALL_INDICATORS):
        return ScoredText(index, text, 1.0, "tool_call")
    if any(p.search(trimmed) for p in _CORRECTION_INDICATORS):
        return ScoredText(index, text, 0.95, "correction")
    if any(p.search(trimmed) for p in _DECISION_INDICATORS):
        return ScoredText(index, text, 0.85, "decision")
    if any(p.search(trimmed) for p in _ACK_PATTERNS):
        return ScoredText(index, text, 0.1, "acknowledgment")
    has_cjk = bool(_CJK_RE.search(trimmed))
    threshold = 30 if has_cjk else 80
    if len(trimmed) > threshold:
        if trimmed.startswith("<") and trimmed.endswith(">") and "</" in trimmed:
            return ScoredText(index, text, 0.3, "system_xml")
        return ScoredText(index, text, 0.7, "substantive")
    if "?" in trimmed or "\uff1f" in trimmed:
        return ScoredText(index, text, 0.5, "short_question")
    return ScoredText(index, text, 0.4, "short_statement")


# ---------------------------------------------------------------------------
# Budget-aware compression (pure / no LLM)
# ---------------------------------------------------------------------------

def compress_texts(
    texts: List[str],
    max_chars: int,
    *,
    min_texts: int = 3,
    min_score_to_keep: float = 0.3,
) -> CompressResult:
    if not texts:
        return CompressResult(texts=[], scored=[], dropped=0, total_chars=0)

    scored = [score_text(t, i) for i, t in enumerate(texts)]
    all_chars = sum(len(t) for t in texts)
    if all_chars <= max_chars:
        return CompressResult(texts=list(texts), scored=scored, dropped=0, total_chars=all_chars)

    selected: set = set()
    used = 0

    def _add(idx: int) -> bool:
        nonlocal used
        if idx in selected or idx < 0 or idx >= len(texts):
            return False
        ln = len(texts[idx])
        if used + ln > max_chars:
            return False
        selected.add(idx)
        used += ln
        return True

    _add(0)
    if len(texts) > 1:
        _add(len(texts) - 1)

    candidates = [s for s in scored if s.index != 0 and s.index != len(texts) - 1]
    candidates.sort(key=lambda s: (-s.score, s.index))

    paired_with: Dict[int, int] = {}
    for s in scored:
        if s.reason != "tool_call":
            continue
        i, j = s.index, s.index + 1
        if j >= len(texts):
            continue
        if i in paired_with or j in paired_with:
            continue
        paired_with[i] = j
        paired_with[j] = i

    for cand in candidates:
        if used >= max_chars:
            break
        if _add(cand.index):
            partner = paired_with.get(cand.index)
            if partner is not None:
                _add(partner)

    all_low = all(s.score < min_score_to_keep for s in scored)
    if all_low and len(selected) < min(min_texts, len(texts)):
        for i in range(len(texts) - 1, -1, -1):
            if len(selected) >= min(min_texts, len(texts)):
                break
            _add(i)

    sorted_idx = sorted(selected)
    out_texts = [texts[i] for i in sorted_idx]
    return CompressResult(
        texts=out_texts,
        scored=scored,
        dropped=len(texts) - len(sorted_idx),
        total_chars=sum(len(t) for t in out_texts),
    )


def estimate_conversation_value(texts: List[str]) -> float:
    """Return an overall value in [0, 1] used by the throttle."""
    if not texts:
        return 0.0
    joined = " ".join(texts)
    value = 0.0
    intent_re = re.compile(
        r"\b(remember|recall|don'?t forget|note that|keep in mind)\b", re.IGNORECASE
    )
    intent_cjk = re.compile(r"(记住|記住|别忘|不要忘|记一下|記一下)")
    if intent_re.search(joined) or intent_cjk.search(joined):
        value += 0.5
    if any(p.search(joined) for p in _TOOL_CALL_INDICATORS):
        value += 0.4
    if any(p.search(joined) for p in _CORRECTION_INDICATORS) or any(
        p.search(joined) for p in _DECISION_INDICATORS
    ):
        value += 0.3
    substantive = sum(len(t) for t in texts if len(t.strip()) > 20)
    if substantive > 200:
        value += 0.2
    if len(texts) > 6:
        value += 0.1
    return min(value, 1.0)


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------

_COMPRESSION_SYSTEM = (
    "You compress a chat session into a single dense memory entry that an "
    "agent can re-read in a future session to fully reconstruct context. "
    "Output STRICT JSON: {\"abstract\": \"<one line, <=80 chars>\", "
    "\"overview\": \"<3-7 markdown bullets>\", "
    "\"content\": \"<full narrative summary, <=1500 chars>\", "
    "\"importance\": <0.0-1.0>, \"tags\": [\"<keyword>\", ...]}. "
    "Prioritize: decisions made, problems solved, tasks open, names/IDs, "
    "preferences, contradictions. Skip greetings and noise."
)


def _messages_to_texts(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            continue
        out.append(f"{role.capitalize()}: {content.strip()}")
    return out


def compress_session(
    messages: List[Dict[str, Any]],
    llm=None,
    *,
    session_id: str = "",
    max_chars: int = 6000,
    user_id: str = "andrew",
) -> Optional[Dict[str, Any]]:
    """Compress an entire session into one memory entry.

    Returns ``None`` if there's nothing to compress. Otherwise returns a
    dict ready to feed into ``LanceDBMemoryProvider._write_entries``.

    The returned entry carries ``source = "session_compressed"`` and
    ``category = "events"`` so it surfaces naturally in recovery.
    """
    if not messages:
        return None

    texts = _messages_to_texts(messages)
    if not texts:
        return None

    compressed = compress_texts(texts, max_chars=max_chars)
    if not compressed.texts:
        return None

    joined = "\n".join(compressed.texts)

    summary: Dict[str, Any] = {
        "abstract": "",
        "overview": "",
        "content": joined,
        "importance": 0.6,
        "tags": ["session_summary"],
    }

    if llm is not None:
        try:
            import json as _json

            raw = llm.chat(_COMPRESSION_SYSTEM, joined[:max_chars], max_tokens=900)
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                summary["abstract"] = str(parsed.get("abstract", ""))[:160]
                summary["overview"] = str(parsed.get("overview", ""))[:1200]
                summary["content"] = str(parsed.get("content", joined))[:2000]
                imp = parsed.get("importance", 0.6)
                try:
                    summary["importance"] = max(0.0, min(1.0, float(imp)))
                except (TypeError, ValueError):
                    summary["importance"] = 0.6
                tags = parsed.get("tags", [])
                if isinstance(tags, list):
                    summary["tags"] = [str(t)[:40] for t in tags[:10]] + ["session_summary"]
        except Exception as e:
            logger.debug("compress_session LLM call failed; using heuristic summary: %s", e)

    if not summary["abstract"]:
        # Heuristic: first ~80 chars of the most-substantive line.
        candidates = [s for s in compressed.scored if s.reason in ("substantive", "decision", "correction", "tool_call")]
        if candidates:
            best = sorted(candidates, key=lambda s: -s.score)[0]
            summary["abstract"] = best.text.strip()[:80]
        else:
            summary["abstract"] = (compressed.texts[0] if compressed.texts else "")[:80]

    entry = {
        "source": "session_compressed",
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": time.time(),
        "category": "events",
        "tier": "working",
        "importance": summary["importance"],
        "abstract": summary["abstract"],
        "overview": summary["overview"],
        "content": summary["content"],
        "tags": summary["tags"],
        "temporal_type": "static",
    }
    return entry


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def recover_session(
    session_id: str,
    store: Any,
    *,
    user_id: str = "andrew",
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Look up compressed entries for ``session_id`` and return them.

    ``store`` is duck-typed: it must expose either:
      - a ``find_session_summaries(session_id, user_id, limit)`` method, OR
      - a low-level ``_table`` LanceDB table (we'll query it directly).
    """
    if not session_id:
        return []

    # Preferred: a dedicated method on the store.
    finder = getattr(store, "find_session_summaries", None)
    if callable(finder):
        try:
            rows = finder(session_id=session_id, user_id=user_id, limit=limit) or []
            return list(rows)
        except Exception as e:
            logger.debug("recover_session: find_session_summaries failed: %s", e)
            return []

    # Fallback: direct table query.
    table = getattr(store, "_table", None)
    if table is None:
        return []
    try:
        where = (
            f"session_id = '{session_id}' AND "
            f"user_id = '{user_id}' AND "
            f"source = 'session_compressed'"
        )
        rows = table.search().where(where, prefilter=True).limit(limit).to_list()
        return rows
    except Exception as e:
        logger.debug("recover_session: table query failed: %s", e)
        return []


def format_recovered(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = ["## Recovered Session Context"]
    for e in entries:
        abstract = e.get("abstract") or e.get("content", "")[:80]
        overview = e.get("overview", "")
        lines.append(f"- **{abstract}**")
        if overview:
            for ov_line in str(overview).splitlines():
                if ov_line.strip():
                    lines.append(f"  {ov_line.strip()}")
    return "\n".join(lines)


__all__ = [
    "ScoredText",
    "CompressResult",
    "score_text",
    "compress_texts",
    "estimate_conversation_value",
    "compress_session",
    "recover_session",
    "format_recovered",
]
