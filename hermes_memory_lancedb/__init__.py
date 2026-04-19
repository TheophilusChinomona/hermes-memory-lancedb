"""Backwards-compat shim — the real package is now ``athena_memory``.

The repo renamed from ``hermes-memory-lancedb`` to ``athena-memory`` in
v3.0.0 to reflect that it's no longer LanceDB-only (it now supports
Postgres + pgvector + pg_search as a sibling backend).

Existing consumers that do ``import hermes_memory_lancedb`` continue to
work via this shim. New code should import from ``athena_memory`` directly.
The shim will be removed in a future major version.
"""

from __future__ import annotations

import sys
import warnings

import athena_memory as _athena_memory

warnings.warn(
    "`hermes_memory_lancedb` has been renamed to `athena_memory`. "
    "Update your imports: `from athena_memory import ...`. "
    "The old name will be removed in a future major version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export every public symbol so ``from hermes_memory_lancedb import X``
# keeps working identically to ``from athena_memory import X``.
for _attr in dir(_athena_memory):
    if not _attr.startswith("_") or _attr in {
        # Also re-export the private helpers tests import directly.
        "_apply_length_normalization",
        "_apply_mmr_diversity",
        "_clamp01",
        "_cosine_similarity",
        "_normalize_to_top",
        "_rerank_cosine_fallback",
        "_rerank_jina",
        "_LENGTH_NORM_ANCHOR",
        "_MMR_SIMILARITY_THRESHOLD",
        "_EmbedClient",
        "_LLMClient",
        "_merge_rrf",
        "_rrf_score",
        "_weibull_weight",
        "_age_days",
        "_tier_evaluate",
        "_build_extraction_prompt",
        "_build_session_extraction_prompt",
        "_build_dedup_prompt",
        "_is_noise",
        "_llm_dedup",
        "_llm_extract_memories",
        "_with_lock",
    }:
        globals()[_attr] = getattr(_athena_memory, _attr)

# Submodule aliasing: ``hermes_memory_lancedb.backends`` etc. should resolve
# to the real module so isinstance checks and subclassing keep working.
for _sub in (
    "backends",
    "backends.base",
    "backends.pgvector",
    "backends.lancedb_store",
    "embedders",
    "scopes",
    "chunker",
    "dedup",
    "admission",
    "smart_metadata",
    "noise_proto",
    "reflection",
    "lifecycle",
    "temporal",
    "sessions",
    "compactor",
    "auto_capture",
    "query",
    "observability",
    "import_md",
    "cli",
):
    _full = f"athena_memory.{_sub}"
    try:
        __import__(_full)
    except ImportError:
        continue
    sys.modules[f"hermes_memory_lancedb.{_sub}"] = sys.modules[_full]

del _athena_memory, sys, warnings, _attr, _sub, _full  # type: ignore[misc]
