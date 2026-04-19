# hermes-memory-lancedb

LanceDB-backed persistent memory for [Hermes Agent](https://github.com/TheophilusChinomona/hermes-agent) (Athena fork). Hybrid BM25 + vector recall, cross-encoder reranking, MMR diversity, three-tier lifecycle, multi-provider embeddings, multi-scope isolation, and LLM smart extraction with dedup.

This is the Python port of [memory-lancedb-pro](https://github.com/TheophilusChinomona/memory-lancedb-pro) (TypeScript). Drop-in for the bundled `lancedb` plugin in Athena's `plugins/memory/lancedb/`.

## Features

**Multi-tenancy (v1.3.0)**
- Multi-scope isolation: `agent_id`, `user_id`, `project_id`, `team_id`, `workspace_id` columns compose orthogonally in the search predicate
- Built-in scope patterns (`agent:*`, `user:*`, `project:*`, `team:*`, `workspace:*`, `custom:*`, `reflection:*`) plus the `global` scope
- ClawTeam shared scopes via `CLAWTEAM_MEMORY_SCOPE` env var (CSV)
- Multi-provider embeddings: OpenAI (default), Jina, Gemini, Ollama, plus any OpenAI-compatible endpoint via `LANCEDB_EMBED_BASE_URL`. Vector dimension is provider-driven and persisted in the schema.
- Backward compatible: legacy tables without scope columns automatically migrate; reads still scope to `user_id` until migration completes.

**Retrieval pipeline (v1.2.0)**
- Hybrid search: parallel vector (cosine, OpenAI `text-embedding-3-small`) + BM25 (tantivy)
- RRF fusion → score normalization → tier/recency boost
- Cross-encoder reranking via Jina (`jina-reranker-v3`), with cosine fallback when no API key is set
- Length normalization (log2 penalty for entries longer than 500 chars)
- Hard min-score cutoff (default 0.35, env-tunable)
- MMR diversity (defers near-duplicates with cosine > 0.85 to the tail)

**Write pipeline (v1.4.0 — P2 upgrades)**
- Long-context chunker: oversized contents split on sentence boundaries with overlap; chunks share a `parent_id`
- Batch dedup: pairwise cosine within the candidate batch + a single LLM call vs the existing pool (was 1 call per pair)
- Admission control: rolling acceptance-rate / novelty / recency / type-prior gate, persisted to `admission_stats.json`
- Smart metadata: per-write LLM extraction of `memory_temporal_type`, `confidence`, `sensitivity`, `modality`, `fact_key`, `tags` — JSON in `metadata` column
- Noise prototype filter: ~20 bundled multilingual noise prototypes; rejects writes with cosine ≥ 0.92 to any prototype (combined with the regex filter)

**Write pipeline (v1.1.0)**
- LLM smart extraction: gpt-4o-mini classifies turns into 6 categories (profile, preferences, entities, events, cases, patterns)
- 3-level structure per entry: `abstract` (≤80 chars) / `overview` (≤800 chars bullets) / `content` (full narrative)
- Dedup with 7 decisions: skip / support / merge / contextualize / contradict / supersede / create
- Single-NN vector probe (L2 < 0.50) for dedup pre-filter
- Noise filter for denials, meta-questions, boilerplate, diagnostic artifacts

**Lifecycle**
- 3 tiers: `peripheral` → `working` → `core`, each with a decay floor (0.0 / 0.7 / 0.9)
- Promotion: `peripheral → working` at access ≥ 3 & decay×importance ≥ 0.4; `working → core` at access ≥ 10 & composite ≥ 0.7 & importance ≥ 0.8
- Demotion: composite < 0.15 OR (age > 60 days AND access < 3)
- Weibull decay: `exp(-((age_days/30)^0.7))`

## Install

Not on PyPI. Install from git:

```bash
pip install "hermes-memory-lancedb @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@main"
```

Or pin to a tag:

```bash
pip install "hermes-memory-lancedb @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@v1.2.0"
```

Runtime requirements: Python ≥ 3.10, `lancedb ≥ 0.20`, `tantivy ≥ 0.21`, `openai ≥ 1.0`, `pyarrow ≥ 12`, `httpx` (transitively via `openai`).

## Activate in Hermes

In `~/.hermes/config.yaml`:

```yaml
memory:
  provider: lancedb
```

The bundled plugin shim at `plugins/memory/lancedb/__init__.py` (in `hermes-agent`) wraps this package and registers it with the `MemoryProvider` ABC.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required for default provider | OpenAI embeddings + LLM extraction (`gpt-4o-mini`) |
| `LANCEDB_PATH` | `$HERMES_HOME/lancedb` | Storage directory |
| `LANCEDB_EMBED_PROVIDER` | `openai` | One of `openai` / `jina` / `gemini` / `ollama` / `openai-compatible` |
| `LANCEDB_EMBED_MODEL` | provider default | Override embedding model id |
| `LANCEDB_EMBED_DIM` | from model id | Explicit embedding dimension override |
| `LANCEDB_EMBED_BASE_URL` | unset | Custom base URL (required for `openai-compatible`) |
| `LANCEDB_EMBED_API_KEY` | unset | Generic key override (otherwise reads provider-specific keys) |
| `JINA_API_KEY` | unset | Required for Jina embeddings |
| `GEMINI_API_KEY` | unset | Required for Gemini embeddings |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server base URL |
| `LANCEDB_AGENT_ID` / `LANCEDB_PROJECT_ID` / `LANCEDB_TEAM_ID` / `LANCEDB_WORKSPACE_ID` | unset | Scope identifiers (compose orthogonally) |
| `CLAWTEAM_MEMORY_SCOPE` | unset | CSV of extra scopes granted to every agent |
| `LANCEDB_RERANK_PROVIDER` | `auto` | `auto` (Jina if key set, else cosine) or `none` to skip rerank |
| `LANCEDB_RERANK_API_KEY` / `JINA_API_KEY` | unset | Enables Jina cross-encoder rerank |
| `LANCEDB_RERANK_MODEL` | `jina-reranker-v3` | Jina model name |
| `LANCEDB_RERANK_ENDPOINT` | `https://api.jina.ai/v1/rerank` | Jina-compatible endpoint |
| `LANCEDB_RERANK_TIMEOUT_S` | `5.0` | Rerank API timeout in seconds |
| `LANCEDB_HARD_MIN_SCORE` | `0.35` | Final score cutoff after pipeline |
| `LANCEDB_ADMISSION_ENABLED` | `1` | Set to `0` to disable the P2 admission controller |
| `LANCEDB_SMART_METADATA` | `1` | Set to `0` to skip per-write smart metadata LLM calls |

Persistent config in `$HERMES_HOME/lancedb.json`:

```json
{
  "storage_path": "/home/me/.hermes/lancedb",
  "user_id": "andrew",
  "extraction_model": "gpt-4o-mini"
}
```

## Storage schema

Single LanceDB table `memories` with FTS index on `content`:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | string | UUID |
| `content` | string | Full narrative (≤2000 chars) |
| `vector` | fixed_size_list<float, _N_> | _N_ = active embedder's dimension (default 1536 for OpenAI 3-small) |
| `timestamp` | float64 | Unix epoch |
| `source` | string | e.g. `discord:dm:<id>` |
| `session_id` | string | Hermes session id at write time |
| `user_id` | string | Scoping key (default `andrew`) |
| `tags` | list<string> | Free-form labels |
| `tier` | string | `peripheral` / `working` / `core` |
| `importance` | float | 0–1, set by extractor |
| `access_count` | int | Bumped on retrieval |
| `category` | string | One of `MEMORY_CATEGORIES` |
| `abstract` | string | L0, ≤80 chars |
| `overview` | string | L1, ≤800 chars bullets |
| `agent_id` | string | P1 — empty if not scoped per-agent |
| `project_id` | string | P1 — composable column filter |
| `team_id` | string | P1 — composable column filter |
| `workspace_id` | string | P1 — composable column filter |
| `scope` | string | P1 — canonical scope id (e.g. `agent:andrew`) |
| `metadata` | string | P2 — JSON-encoded smart metadata |
| `parent_id` | string | P2 — set on chunked rows; empty for atomic writes |

## Tools exposed to the agent

| Tool | Purpose |
| --- | --- |
| `lancedb_search` | Hybrid retrieval, returns markdown bullets |
| `lancedb_remember` | Explicit write of a durable fact |
| `lancedb_forget` | Soft-delete by id |
| `lancedb_stats` | Counts per tier / category, storage path |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

170+ unit tests cover noise filtering, decay, RRF fusion, tier evaluation, LLM extraction, dedup, prompt builders, the v1.2.0 retrieval helpers (length norm, MMR, rerank, cosine), and the v1.4.0 P2 write-pipeline modules (chunker, batch dedup, admission control, smart metadata, noise prototypes). Tests that touch the LanceDB native runtime are gated behind a real `lancedb` import — skip them on CPUs without AVX support.

## Status

| Phase | Status |
| --- | --- |
| P0 — Cross-encoder rerank, MMR, length norm, hard min-score | Done (v1.2.0) |
| P1 — Multi-scope (agent/user/project/team/workspace), multi-provider embeddings (Jina/Gemini/Ollama) | Done (v1.3.0) |
| P2 — Chunker, batch dedup, admission control, smart metadata, noise prototypes | Done (v1.4.0) |
| P3 — Reflection subsystem (event store, item store, ranking, retry, slices) | Pending |
| P4 — Session compactor, memory compactor, temporal classifier, auto-capture cleanup, query expansion | Pending |
| P5 — Management CLI, retrieval observability, markdown import, A/B reembed | Pending |

OAuth is intentionally out of scope — `OPENAI_API_KEY` and (optionally) `JINA_API_KEY` are sufficient.
