"""Long-context chunking — port of TS `src/chunker.ts`.

Split documents that exceed embedding context limits into smaller, semantically
coherent chunks with overlap. Uses character counts as a conservative proxy for
tokens.

Public API:
    chunk_text(text, max_chars, overlap)              -> List[str]
    chunk_document(text, config=DEFAULT_CHUNKER_CONFIG) -> ChunkResult
    smart_chunk(text, embedder_model=None)            -> ChunkResult

Each non-trivial chunk should share a `parent_id` at the call site so retrieval
can collapse them. The chunker itself only produces text + per-chunk metadata.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Config / dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkerConfig:
    max_chunk_size: int = 4000
    overlap_size: int = 200
    min_chunk_size: int = 200
    semantic_split: bool = True
    max_lines_per_chunk: int = 50


@dataclass(frozen=True)
class ChunkMetadata:
    start_index: int
    end_index: int
    length: int


@dataclass
class ChunkResult:
    chunks: List[str] = field(default_factory=list)
    metadatas: List[ChunkMetadata] = field(default_factory=list)
    total_original_length: int = 0
    chunk_count: int = 0


DEFAULT_CHUNKER_CONFIG = ChunkerConfig()


# Mirror of the TS embedding context map. Values are the *advertised* token
# limits for each model — we apply a 70% char heuristic on top.
EMBEDDING_CONTEXT_LIMITS = {
    "jina-embeddings-v5-text-small": 8192,
    "jina-embeddings-v5-text-nano": 8192,
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-004": 8192,
    "gemini-embedding-001": 2048,
    "nomic-embed-text": 8192,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 512,
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_SENTENCE_ENDING = re.compile(r"[.!?。！？]")
_WHITESPACE = re.compile(r"\s")
_CJK_RE = re.compile(
    "[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)
_CJK_CHAR_TOKEN_DIVISOR = 2.5
_CJK_RATIO_THRESHOLD = 0.3


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _count_lines(s: str) -> int:
    # Count split positions; CRLF counts as one break (matches TS)
    return len(re.split(r"\r\n|\n|\r", s))


def _find_split_end(text: str, start: int, max_end: int, min_end: int, config: ChunkerConfig) -> int:
    safe_min = _clamp(min_end, start + 1, max_end)
    safe_max = _clamp(max_end, safe_min, len(text))

    # Respect line cap.
    if config.max_lines_per_chunk > 0:
        candidate = text[start:safe_max]
        if _count_lines(candidate) > config.max_lines_per_chunk:
            breaks = 0
            for i in range(start, safe_max):
                if text[i] == "\n":
                    breaks += 1
                    if breaks >= config.max_lines_per_chunk:
                        return max(i + 1, safe_min)

    if config.semantic_split:
        # Prefer sentence boundary near the end.
        for i in range(safe_max - 1, safe_min - 1, -1):
            if _SENTENCE_ENDING.match(text[i]):
                # Include trailing whitespace after punctuation.
                j = i + 1
                while j < safe_max and _WHITESPACE.match(text[j]):
                    j += 1
                return j

        # Next-best: newline boundary.
        for i in range(safe_max - 1, safe_min - 1, -1):
            if text[i] == "\n":
                return i + 1

    # Fallback: last whitespace boundary.
    for i in range(safe_max - 1, safe_min - 1, -1):
        if _WHITESPACE.match(text[i]):
            return i

    return safe_max


def _slice_trim_with_indices(text: str, start: int, end: int) -> tuple:
    raw = text[start:end]
    leading_match = re.match(r"^\s*", raw)
    trailing_match = re.search(r"\s*$", raw)
    leading = len(leading_match.group(0)) if leading_match else 0
    trailing = len(trailing_match.group(0)) if trailing_match else 0
    chunk = raw.strip()

    trimmed_start = start + leading
    trimmed_end = end - trailing
    meta = ChunkMetadata(
        start_index=trimmed_start,
        end_index=max(trimmed_start, trimmed_end),
        length=len(chunk),
    )
    return chunk, meta


def _get_cjk_ratio(text: str) -> float:
    cjk = 0
    total = 0
    for ch in text:
        if _WHITESPACE.match(ch):
            continue
        total += 1
        if _CJK_RE.match(ch):
            cjk += 1
    return 0.0 if total == 0 else cjk / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(text: str, config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG) -> ChunkResult:
    """Split text into semantically-coherent chunks with overlap."""
    if not text or not text.strip():
        return ChunkResult(chunks=[], metadatas=[], total_original_length=0, chunk_count=0)

    total_original_length = len(text)
    chunks: List[str] = []
    metadatas: List[ChunkMetadata] = []

    pos = 0
    step = max(1, config.max_chunk_size - config.overlap_size)
    max_guard = max(4, math.ceil(len(text) / step) + 5)
    guard = 0

    while pos < len(text) and guard < max_guard:
        guard += 1

        remaining = len(text) - pos
        if remaining <= config.max_chunk_size:
            chunk, meta = _slice_trim_with_indices(text, pos, len(text))
            if chunk:
                chunks.append(chunk)
                metadatas.append(meta)
            break

        max_end = min(pos + config.max_chunk_size, len(text))
        min_end = min(pos + config.min_chunk_size, max_end)

        end = _find_split_end(text, pos, max_end, min_end, config)
        chunk, meta = _slice_trim_with_indices(text, pos, end)

        if len(chunk) < config.min_chunk_size:
            # Hard split fallback.
            hard_end = min(pos + config.max_chunk_size, len(text))
            hchunk, hmeta = _slice_trim_with_indices(text, pos, hard_end)
            if hchunk:
                chunks.append(hchunk)
                metadatas.append(hmeta)
            if hard_end >= len(text):
                break
            pos = max(hard_end - config.overlap_size, pos + 1)
            continue

        chunks.append(chunk)
        metadatas.append(meta)

        if end >= len(text):
            break

        pos = max(end - config.overlap_size, pos + 1)

    return ChunkResult(
        chunks=chunks,
        metadatas=metadatas,
        total_original_length=total_original_length,
        chunk_count=len(chunks),
    )


def smart_chunk(text: str, embedder_model: Optional[str] = None) -> ChunkResult:
    """Pick a config based on the embedding model's advertised context limit.

    Uses 70% of the limit for max_chunk_size, 5% for overlap, 10% for min — and
    divides everything by 2.5 if the text is predominantly CJK.
    """
    base = EMBEDDING_CONTEXT_LIMITS.get(embedder_model or "", 8192)
    cjk_heavy = _get_cjk_ratio(text) > _CJK_RATIO_THRESHOLD
    divisor = _CJK_CHAR_TOKEN_DIVISOR if cjk_heavy else 1

    config = ChunkerConfig(
        max_chunk_size=max(200, int(base * 0.7 / divisor)),
        overlap_size=max(0, int(base * 0.05 / divisor)),
        min_chunk_size=max(100, int(base * 0.1 / divisor)),
        semantic_split=True,
        max_lines_per_chunk=50,
    )
    return chunk_document(text, config)


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Convenience wrapper — returns just the chunk strings.

    Intended as the main entry point used by the write pipeline. If the text is
    short enough to fit in a single chunk, returns `[text.strip()]` (or `[]` for
    empty input).
    """
    if not text or not text.strip():
        return []
    if len(text) <= max_chars:
        return [text.strip()]
    config = ChunkerConfig(
        max_chunk_size=max(200, max_chars),
        overlap_size=max(0, min(overlap, max_chars // 2)),
        min_chunk_size=max(100, min(max_chars // 4, 200)),
        semantic_split=True,
        max_lines_per_chunk=50,
    )
    result = chunk_document(text, config)
    return result.chunks


__all__ = [
    "ChunkerConfig",
    "ChunkMetadata",
    "ChunkResult",
    "DEFAULT_CHUNKER_CONFIG",
    "EMBEDDING_CONTEXT_LIMITS",
    "chunk_document",
    "smart_chunk",
    "chunk_text",
]
