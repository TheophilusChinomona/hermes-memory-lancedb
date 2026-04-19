"""Retry policy for transient LLM failures inside the reflection pipeline.

Python port of `reflection-retry.ts`. Single-shot retry only — the goal is to
absorb upstream blips (502/503/504, socket hang up, connection reset, etc.)
without masking real bugs (auth errors, quota exhaustion, content-policy
refusals, oversize prompts).
"""

from __future__ import annotations

import json
import random as _random
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Literal, Optional, Tuple


_TRANSIENT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"unexpected eof", re.IGNORECASE),
    re.compile(r"\beconnreset\b", re.IGNORECASE),
    re.compile(r"\beconnaborted\b", re.IGNORECASE),
    re.compile(r"\betimedout\b", re.IGNORECASE),
    re.compile(r"\bepipe\b", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
    re.compile(r"socket hang up", re.IGNORECASE),
    re.compile(r"socket (?:closed|disconnected)", re.IGNORECASE),
    re.compile(r"connection (?:closed|aborted|dropped)", re.IGNORECASE),
    re.compile(r"early close", re.IGNORECASE),
    re.compile(r"stream (?:ended|closed) unexpectedly", re.IGNORECASE),
    re.compile(r"temporar(?:y|ily).*unavailable", re.IGNORECASE),
    re.compile(r"upstream.*unavailable", re.IGNORECASE),
    re.compile(r"service unavailable", re.IGNORECASE),
    re.compile(r"bad gateway", re.IGNORECASE),
    re.compile(r"gateway timeout", re.IGNORECASE),
    re.compile(r"\b(?:http|status)\s*(?:502|503|504)\b", re.IGNORECASE),
    re.compile(r"\btimed out\b", re.IGNORECASE),
    re.compile(r"\btimeout\b", re.IGNORECASE),
    re.compile(r"\bund_err_(?:socket|headers_timeout|body_timeout)\b", re.IGNORECASE),
    re.compile(r"network error", re.IGNORECASE),
    re.compile(r"fetch failed", re.IGNORECASE),
]


_NON_RETRY_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b401\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"invalid api key", re.IGNORECASE),
    re.compile(r"invalid[_ -]?token", re.IGNORECASE),
    re.compile(r"\bauth(?:entication)?_?unavailable\b", re.IGNORECASE),
    re.compile(r"insufficient (?:credit|credits|balance)", re.IGNORECASE),
    re.compile(r"\bbilling\b", re.IGNORECASE),
    re.compile(r"\bquota exceeded\b", re.IGNORECASE),
    re.compile(r"payment required", re.IGNORECASE),
    re.compile(r"model .*not found", re.IGNORECASE),
    re.compile(r"no such model", re.IGNORECASE),
    re.compile(r"unknown model", re.IGNORECASE),
    re.compile(r"context length", re.IGNORECASE),
    re.compile(r"context window", re.IGNORECASE),
    re.compile(r"request too large", re.IGNORECASE),
    re.compile(r"payload too large", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit", re.IGNORECASE),
    re.compile(r"prompt too long", re.IGNORECASE),
    re.compile(r"session expired", re.IGNORECASE),
    re.compile(r"invalid session", re.IGNORECASE),
    re.compile(r"refusal", re.IGNORECASE),
    re.compile(r"content policy", re.IGNORECASE),
    re.compile(r"safety policy", re.IGNORECASE),
    re.compile(r"content filter", re.IGNORECASE),
    re.compile(r"disallowed", re.IGNORECASE),
]


RetryReason = Literal[
    "not_reflection_scope",
    "retry_already_used",
    "useful_output_present",
    "non_retry_error",
    "non_transient_error",
    "transient_upstream_failure",
]


@dataclass
class RetryClassifierInput:
    in_reflection_scope: bool
    retry_count: int
    useful_output_chars: int
    error: Any


@dataclass
class RetryClassifierResult:
    retryable: bool
    reason: RetryReason
    normalized_error: str


@dataclass
class RetryState:
    count: int = 0


_WHITESPACE_RUN = re.compile(r"\s+")


def _to_error_message(error: Any) -> str:
    if isinstance(error, BaseException):
        msg = f"{type(error).__name__}: {error}".strip()
        return msg or "Error"
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error)
    except (TypeError, ValueError):
        return str(error)


def _clip_single_line(text: str, max_len: int = 260) -> str:
    one_line = _WHITESPACE_RUN.sub(" ", text).strip()
    if len(one_line) <= max_len:
        return one_line
    return f"{one_line[: max_len - 3]}..."


def is_transient_reflection_upstream_error(error: Any) -> bool:
    msg = _to_error_message(error)
    return any(p.search(msg) for p in _TRANSIENT_PATTERNS)


def is_reflection_non_retry_error(error: Any) -> bool:
    msg = _to_error_message(error)
    return any(p.search(msg) for p in _NON_RETRY_PATTERNS)


def classify_reflection_retry(input_: RetryClassifierInput) -> RetryClassifierResult:
    normalized_error = _clip_single_line(_to_error_message(input_.error), 260)

    if not input_.in_reflection_scope:
        return RetryClassifierResult(False, "not_reflection_scope", normalized_error)
    if input_.retry_count > 0:
        return RetryClassifierResult(False, "retry_already_used", normalized_error)
    if input_.useful_output_chars > 0:
        return RetryClassifierResult(False, "useful_output_present", normalized_error)
    if is_reflection_non_retry_error(input_.error):
        return RetryClassifierResult(False, "non_retry_error", normalized_error)
    if is_transient_reflection_upstream_error(input_.error):
        return RetryClassifierResult(True, "transient_upstream_failure", normalized_error)
    return RetryClassifierResult(False, "non_transient_error", normalized_error)


def compute_reflection_retry_delay_ms(random_fn: Callable[[], float] = _random.random) -> int:
    raw = random_fn()
    try:
        clamped = min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        clamped = 0.0
    return 1000 + int(clamped * 2000)


def run_with_reflection_transient_retry_once(
    *,
    scope: Literal["reflection", "distiller"],
    runner: Literal["embedded", "cli"],
    retry_state: RetryState,
    execute: Callable[[], Any],
    on_log: Optional[Callable[[Literal["info", "warn"], str], None]] = None,
    random_fn: Callable[[], float] = _random.random,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> Any:
    """Synchronous single-retry runner.

    Mirrors the TS async helper but runs synchronously — the embedding /
    chat clients we use in Python are sync. Callers that need async should
    wrap this in their own ``run_in_executor``.
    """
    sleep = sleep_fn if sleep_fn is not None else (lambda ms: _time.sleep(ms / 1000.0))
    try:
        return execute()
    except BaseException as error:  # noqa: BLE001 — classifier handles all
        decision = classify_reflection_retry(RetryClassifierInput(
            in_reflection_scope=scope in ("reflection", "distiller"),
            retry_count=retry_state.count,
            useful_output_chars=0,
            error=error,
        ))
        if not decision.retryable:
            raise

        delay_ms = compute_reflection_retry_delay_ms(random_fn)
        retry_state.count += 1
        if on_log is not None:
            on_log(
                "warn",
                f"memory-{scope}: transient upstream failure detected ({runner}); "
                f"retrying once in {delay_ms}ms ({decision.reason}). "
                f"error={decision.normalized_error}",
            )
        sleep(delay_ms)

        try:
            result = execute()
            if on_log is not None:
                on_log("info", f"memory-{scope}: retry succeeded ({runner})")
            return result
        except BaseException as retry_error:  # noqa: BLE001
            if on_log is not None:
                on_log(
                    "warn",
                    f"memory-{scope}: retry exhausted ({runner}). "
                    f"error={_clip_single_line(_to_error_message(retry_error), 260)}",
                )
            raise


__all__ = [
    "RetryClassifierInput",
    "RetryClassifierResult",
    "RetryState",
    "is_transient_reflection_upstream_error",
    "is_reflection_non_retry_error",
    "classify_reflection_retry",
    "compute_reflection_retry_delay_ms",
    "run_with_reflection_transient_retry_once",
]
