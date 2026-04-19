"""Reflection subsystem — agent-generated meta-memories.

Python port of the `reflection-*` modules in
`memory-lancedb-pro` (TypeScript). Reflections are structured notes about
what happened during a session — invariants (durable rules), derived deltas
(session-specific changes), governance candidates, lessons, and decisions.

They live alongside regular memories but in their own LanceDB table named
``reflections``, with a separate FTS index and (optional) vector column.

Public API:

    ReflectionStore          # LanceDB-backed CRUD + FTS index for the table
    ReflectionEventStore     # In-memory append-only log of session events
    ReflectionItemStore      # In-memory log of structured items
    ReflectionRanker         # Recency × importance × quality scorer
    RetryState / run_with_reflection_transient_retry_once  # Single-retry policy

Plus a number of pure-function builders / loaders that mirror the TS port
1:1 — see ``store.py`` for the full list.
"""

from .event_store import (
    REFLECTION_SCHEMA_VERSION,
    ReflectionErrorSignalLike,
    ReflectionEventMetadata,
    ReflectionEventPayload,
    ReflectionEventStore,
    build_reflection_event_payload,
    create_reflection_event_id,
)
from .item_store import (
    REFLECTION_DERIVED_BASE_WEIGHT,
    REFLECTION_DERIVED_DECAY_K,
    REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS,
    REFLECTION_DERIVED_QUALITY,
    REFLECTION_INVARIANT_BASE_WEIGHT,
    REFLECTION_INVARIANT_DECAY_K,
    REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS,
    REFLECTION_INVARIANT_QUALITY,
    ReflectionItemDecayDefaults,
    ReflectionItemMetadata,
    ReflectionItemPayload,
    ReflectionItemStore,
    build_reflection_item_payloads,
    get_reflection_item_decay_defaults,
)
from .mapped_metadata import (
    ReflectionMappedCategory,
    ReflectionMappedDecayDefaults,
    ReflectionMappedKind,
    ReflectionMappedMetadata,
    build_reflection_mapped_metadata,
    get_reflection_mapped_decay_defaults,
    parse_mapped_kind,
)
from .metadata import (
    get_display_category_tag,
    is_reflection_entry,
    parse_reflection_metadata,
)
from .ranking import (
    REFLECTION_FALLBACK_SCORE_FACTOR,
    ReflectionRanker,
    ReflectionScoreInput,
    compute_reflection_logistic,
    compute_reflection_score,
    normalize_reflection_line_for_aggregation,
)
from .retry import (
    RetryClassifierInput,
    RetryClassifierResult,
    RetryState,
    classify_reflection_retry,
    compute_reflection_retry_delay_ms,
    is_reflection_non_retry_error,
    is_transient_reflection_upstream_error,
    run_with_reflection_transient_retry_once,
)
from .slices import (
    ReflectionGovernanceEntry,
    ReflectionMappedMemory,
    ReflectionMappedMemoryItem,
    ReflectionSliceItem,
    ReflectionSlices,
    extract_injectable_reflection_mapped_memories,
    extract_injectable_reflection_mapped_memory_items,
    extract_injectable_reflection_slice_items,
    extract_injectable_reflection_slices,
    extract_reflection_learning_governance_candidates,
    extract_reflection_lessons,
    extract_reflection_mapped_memories,
    extract_reflection_mapped_memory_items,
    extract_reflection_slice_items,
    extract_reflection_slices,
    extract_section_markdown,
    is_placeholder_reflection_slice_line,
    is_unsafe_injectable_reflection_line,
    normalize_reflection_slice_line,
    parse_section_bullets,
    sanitize_injectable_reflection_lines,
    sanitize_reflection_slice_lines,
)
from .store import (
    BuildReflectionStorePayloadsParams,
    BuildReflectionStorePayloadsResult,
    DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS,
    DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS,
    LoadReflectionMappedRowsParams,
    LoadReflectionSlicesParams,
    REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT,
    REFLECTION_DERIVE_LOGISTIC_K,
    REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
    REFLECTION_EMBED_DIM,
    REFLECTION_TABLE_NAME,
    ReflectionSearchHit,
    ReflectionStore,
    ReflectionStorePayload,
    build_reflection_store_payloads,
    compute_derived_line_quality,
    get_reflection_schema,
    load_agent_reflection_slices_from_entries,
    load_reflection_mapped_rows_from_entries,
    resolve_reflection_importance,
)


__all__ = [
    # store.py
    "REFLECTION_TABLE_NAME",
    "REFLECTION_EMBED_DIM",
    "REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS",
    "REFLECTION_DERIVE_LOGISTIC_K",
    "REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT",
    "DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS",
    "DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS",
    "BuildReflectionStorePayloadsParams",
    "BuildReflectionStorePayloadsResult",
    "LoadReflectionMappedRowsParams",
    "LoadReflectionSlicesParams",
    "ReflectionSearchHit",
    "ReflectionStore",
    "ReflectionStorePayload",
    "build_reflection_store_payloads",
    "compute_derived_line_quality",
    "get_reflection_schema",
    "load_agent_reflection_slices_from_entries",
    "load_reflection_mapped_rows_from_entries",
    "resolve_reflection_importance",
    # event_store.py
    "REFLECTION_SCHEMA_VERSION",
    "ReflectionErrorSignalLike",
    "ReflectionEventMetadata",
    "ReflectionEventPayload",
    "ReflectionEventStore",
    "build_reflection_event_payload",
    "create_reflection_event_id",
    # item_store.py
    "REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS",
    "REFLECTION_INVARIANT_DECAY_K",
    "REFLECTION_INVARIANT_BASE_WEIGHT",
    "REFLECTION_INVARIANT_QUALITY",
    "REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS",
    "REFLECTION_DERIVED_DECAY_K",
    "REFLECTION_DERIVED_BASE_WEIGHT",
    "REFLECTION_DERIVED_QUALITY",
    "ReflectionItemDecayDefaults",
    "ReflectionItemMetadata",
    "ReflectionItemPayload",
    "ReflectionItemStore",
    "build_reflection_item_payloads",
    "get_reflection_item_decay_defaults",
    # mapped_metadata.py
    "ReflectionMappedCategory",
    "ReflectionMappedDecayDefaults",
    "ReflectionMappedKind",
    "ReflectionMappedMetadata",
    "build_reflection_mapped_metadata",
    "get_reflection_mapped_decay_defaults",
    "parse_mapped_kind",
    # metadata.py
    "get_display_category_tag",
    "is_reflection_entry",
    "parse_reflection_metadata",
    # ranking.py
    "REFLECTION_FALLBACK_SCORE_FACTOR",
    "ReflectionRanker",
    "ReflectionScoreInput",
    "compute_reflection_logistic",
    "compute_reflection_score",
    "normalize_reflection_line_for_aggregation",
    # retry.py
    "RetryClassifierInput",
    "RetryClassifierResult",
    "RetryState",
    "classify_reflection_retry",
    "compute_reflection_retry_delay_ms",
    "is_reflection_non_retry_error",
    "is_transient_reflection_upstream_error",
    "run_with_reflection_transient_retry_once",
    # slices.py
    "ReflectionGovernanceEntry",
    "ReflectionMappedMemory",
    "ReflectionMappedMemoryItem",
    "ReflectionSliceItem",
    "ReflectionSlices",
    "extract_injectable_reflection_mapped_memories",
    "extract_injectable_reflection_mapped_memory_items",
    "extract_injectable_reflection_slice_items",
    "extract_injectable_reflection_slices",
    "extract_reflection_learning_governance_candidates",
    "extract_reflection_lessons",
    "extract_reflection_mapped_memories",
    "extract_reflection_mapped_memory_items",
    "extract_reflection_slice_items",
    "extract_reflection_slices",
    "extract_section_markdown",
    "is_placeholder_reflection_slice_line",
    "is_unsafe_injectable_reflection_line",
    "normalize_reflection_slice_line",
    "parse_section_bullets",
    "sanitize_injectable_reflection_lines",
    "sanitize_reflection_slice_lines",
]
