"""Embedding-based noise prototype filter — port of TS `src/noise-prototypes.ts`.

Vector-based companion to the existing regex `_is_noise`. Maintains a small
bank of "noise prototype" embeddings (greetings, denials, recall queries) and
rejects any new write whose embedding has cosine >= 0.92 with one of them.

Bundled prototypes are inline; we compute their embeddings on first use and
cache the result to disk under `<storage_path>/noise_prototypes.json`. The
cache is keyed on the bundled-list hash so changes invalidate it cleanly.

Public API:
    NoisePrototypeFilter(storage_path)
        .load_or_init(embedder)
        .is_noise(vector, threshold=0.92) -> bool
        .learn(vector)                     -> None  (LLM-feedback grow)
        .size                              -> int
        .initialized                       -> bool

Combine with the regex filter at the call site — either matcher rejects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundled noise prototypes (multilingual)
# ---------------------------------------------------------------------------

BUILTIN_NOISE_TEXTS: List[str] = [
    # Recall queries (English)
    "Do you remember what I told you?",
    "Can you recall my preferences?",
    "What did I say about that?",
    "Have you stored anything about me?",
    "What's in your memory?",
    # Recall queries (CJK)
    "你还记得我喜欢什么吗",
    "你知道我之前说过什么吗",
    "記得我上次提到的嗎",
    "我之前跟你说过吗",
    # Agent denials
    "I don't have any information about that",
    "I don't recall any previous conversation",
    "I cannot access external databases",
    "As an AI I don't have memory of previous conversations",
    "我没有相关的记忆",
    # Greetings / boilerplate
    "Hello, how are you doing today?",
    "Hi there, what's up",
    "Thanks, that's helpful",
    "Sure, sounds good",
    "Got it, no problem",
    "新的一天开始了",
]

DEFAULT_THRESHOLD = 0.92
DEFAULT_DEDUP_THRESHOLD = 0.90
MAX_LEARNED_PROTOTYPES = 200

CACHE_FILENAME = "noise_prototypes.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _bundle_hash(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class NoisePrototypeFilter:
    """Vector-based noise prototype bank with disk-backed cache."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        builtin_texts: Optional[Sequence[str]] = None,
    ) -> None:
        self.storage_path = storage_path
        self.threshold = float(threshold)
        self._builtin_texts = list(builtin_texts) if builtin_texts else list(BUILTIN_NOISE_TEXTS)
        self._lock = threading.RLock()
        self._vectors: List[List[float]] = []
        self._builtin_count: int = 0
        self._initialized: bool = False

    # -- properties ------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def cache_path(self) -> Optional[str]:
        if not self.storage_path:
            return None
        return os.path.join(self.storage_path, CACHE_FILENAME)

    # -- cache I/O -------------------------------------------------------

    def _try_load_cache(self) -> bool:
        path = self.cache_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug("noise prototype cache load failed: %s", e)
            return False
        if not isinstance(data, dict):
            return False
        if data.get("bundle_hash") != _bundle_hash(self._builtin_texts):
            return False
        vectors = data.get("vectors") or []
        builtin_count = data.get("builtin_count")
        if not isinstance(vectors, list) or not isinstance(builtin_count, int):
            return False
        clean = []
        for v in vectors:
            if isinstance(v, list) and v:
                clean.append([float(x) for x in v])
        if not clean:
            return False
        self._vectors = clean
        self._builtin_count = max(0, min(builtin_count, len(clean)))
        return True

    def _persist_cache(self) -> None:
        path = self.cache_path
        if not path:
            return
        try:
            os.makedirs(self.storage_path, exist_ok=True)  # type: ignore[arg-type]
            payload = {
                "bundle_hash": _bundle_hash(self._builtin_texts),
                "builtin_count": self._builtin_count,
                "vectors": self._vectors,
            }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("noise prototype cache persist failed: %s", e)

    # -- init ------------------------------------------------------------

    def load_or_init(self, embedder: Optional[Callable[[str], Sequence[float]]]) -> None:
        """Load cached vectors if available; otherwise embed the bundled texts.

        `embedder(text) -> list[float]` is invoked once per bundled text on a
        cache miss. Failures are tolerated — the bank works with whatever
        embeddings succeed (down to zero, in which case `is_noise` becomes a
        no-op).
        """
        with self._lock:
            if self._initialized:
                return
            if self._try_load_cache():
                self._initialized = True
                return
            if embedder is None:
                # No embedder yet — leave empty; callers can retry later.
                return
            vectors: List[List[float]] = []
            for text in self._builtin_texts:
                try:
                    v = embedder(text)
                    if v:
                        vectors.append([float(x) for x in v])
                except Exception as e:
                    logger.debug("noise proto embed failed for %r: %s", text[:40], e)
            self._vectors = vectors
            self._builtin_count = len(vectors)
            # Degeneracy check — refuse to initialize if all vectors are
            # ~identical (a deterministic mock would flag every input).
            if len(self._vectors) >= 2:
                sim = _cosine(self._vectors[0], self._vectors[1])
                if sim > 0.98:
                    logger.debug(
                        "noise proto degenerate (cos=%.4f) — disabling", sim,
                    )
                    self._vectors = []
                    self._builtin_count = 0
                    return
            self._initialized = bool(self._vectors)
            if self._initialized:
                self._persist_cache()

    # -- queries ---------------------------------------------------------

    def is_noise(self, vector: Optional[Sequence[float]], threshold: Optional[float] = None) -> bool:
        if not vector or not self._initialized or not self._vectors:
            return False
        thr = float(threshold) if threshold is not None else self.threshold
        for proto in self._vectors:
            if _cosine(proto, vector) >= thr:
                return True
        return False

    def max_similarity(self, vector: Sequence[float]) -> float:
        if not vector or not self._vectors:
            return 0.0
        best = 0.0
        for proto in self._vectors:
            s = _cosine(proto, vector)
            if s > best:
                best = s
        return best

    # -- grow ------------------------------------------------------------

    def learn(self, vector: Sequence[float]) -> bool:
        """Add a learned prototype (e.g. when LLM extraction returned nothing).

        Returns True when the prototype was added, False when it duplicated an
        existing one (or the bank is uninitialized).
        """
        if not vector or not self._initialized:
            return False
        with self._lock:
            for proto in self._vectors:
                if _cosine(proto, vector) >= DEFAULT_DEDUP_THRESHOLD:
                    return False
            self._vectors.append([float(x) for x in vector])
            # Evict the oldest LEARNED prototype past the cap (preserve builtins).
            if len(self._vectors) > self._builtin_count + MAX_LEARNED_PROTOTYPES:
                # Drop the first learned slot.
                del self._vectors[self._builtin_count]
            self._persist_cache()
            return True


__all__ = [
    "NoisePrototypeFilter",
    "BUILTIN_NOISE_TEXTS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_DEDUP_THRESHOLD",
    "MAX_LEARNED_PROTOTYPES",
    "CACHE_FILENAME",
]
