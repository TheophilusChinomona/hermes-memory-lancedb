# hermes-memory-lancedb

Pluggable persistent memory for [Hermes Agent](https://github.com/TheophilusChinomona/hermes-agent) (Athena fork). Hybrid BM25 + vector recall, cross-encoder reranking, MMR diversity, three-tier lifecycle, multi-provider embeddings, multi-scope isolation, LLM smart extraction with dedup, management CLI, and per-query observability — over either **LanceDB** (default, embedded) or **Postgres + pgvector + pg_search** (server-side, runs on any CPU).

Python port of [memory-lancedb-pro](https://github.com/TheophilusChinomona/memory-lancedb-pro) (TypeScript). Drop-in for the bundled `lancedb` plugin in Athena's `plugins/memory/lancedb/`.

> **Heads up — name vs. scope:** the package is still named `hermes-memory-lancedb` for backwards compatibility, but as of v2.0.0 it's a multi-backend memory plugin. A rename is planned once both backends are battle-tested.

## Features

**Pluggable storage backends (v2.0.0 — P6)**
- `MemoryStore` ABC with two impls: `LanceDBStore` (embedded LanceDB; needs AVX2 host) and `PgvectorStore` (Postgres + pgvector + pg_search via the [ParadeDB](https://www.paradedb.com/) image; runs on any CPU)
- Selection via `HERMES_MEMORY_BACKEND={lancedb,pgvector}` (auto-detect: if `HERMES_MEMORY_DATABASE_URL` is set, defaults to pgvector)
- LanceDB and pgvector ship as **optional extras** — base install pulls neither, so AVX2-less hosts can `pip install ".[pgvector]"` without LanceDB's native binary
- Pipeline (rerank, MMR, length-norm, dedup, scope filtering) is backend-agnostic — same retrieval quality on both
- Pgvector backend uses `pg_search` BM25 (tantivy under the hood — same engine LanceDB's FTS uses) when available, falls back to `tsvector` + `ts_rank_cd` on vanilla pgvector

**Management & observability (v1.7.0 — P5)**
- Console CLI: `hermes-memory-lancedb {list,search,stats,delete,delete-bulk,export,import,import-markdown,reembed,migrate,reindex-fts,version}`
- Per-query `RetrievalTrace` records timings, score ranges, and dropped IDs at every pipeline stage
- Rolling `RetrievalStats` ring buffer aggregates queries-by-source, p95 latency, top drop stages, and result-count histograms
- Markdown ingest for `MEMORY.md` and dated `memory/YYYY-MM-DD.md` files
- A/B re-embedding via `reembed --target /path/to/alt.db` for embedder upgrades
- `portalocker` cross-process file lock around writes, schema migrations, and FTS reindex so multiple Hermes processes can share one LanceDB safely

**Multi-tenancy (v1.3.0)**
- Multi-scope isolation: `agent_id`, `user_id`, `project_id`, `team_id`, `workspace_id` columns compose orthogonally in the search predicate
- Built-in scope patterns (`agent:*`, `user:*`, `project:*`, `team:*`, `workspace:*`, `custom:*`, `reflection:*`) plus the `global` scope
- ClawTeam shared scopes via `CLAWTEAM_MEMORY_SCOPE` env var (CSV)
- Multi-provider embeddings: OpenAI (default), OpenRouter (`openai/`, `cohere/`, `voyage/`, `jina/` model ids), Jina, Gemini, Ollama, plus any OpenAI-compatible endpoint via `LANCEDB_EMBED_BASE_URL`. Vector dimension is provider-driven and persisted in the schema.
- Backward compatible: legacy tables without scope columns automatically migrate; reads still scope to `user_id` until migration completes.

**Lifecycle & ops (v1.6.0 — P4)**
- `TierManager` + `DecayEngine` extracted into `lifecycle.py` with importance-modulated half-life and `apply_search_boost`
- Temporal classifier: `static` vs `dynamic` memories; dynamic ones decay 3x faster
- Session compressor: end-of-session summary written as a single dense entry
- Session recovery: re-inflate compressed entries when a session_id reopens
- Memory compactor: clusters near-duplicates (cosine ≥ 0.88) and merges them; auto-trigger every N writes plus a `lancedb_compact` tool
- Auto-capture cleanup at session start: deletes/demotes the previous session's noisy auto-captures
- Intent analyzer + query expansion: rule-based pre-search routing + BM25 fan-out across synonyms

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

Not on PyPI. Install from git, picking the backend extras you want:

```bash
# LanceDB backend (default, requires host CPU with AVX2 — Intel ≥ Haswell 2013, AMD ≥ Excavator 2015)
pip install "hermes-memory-lancedb[lancedb] @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@main"

# Postgres + pgvector backend (any CPU; needs a Postgres server with the vector extension — ParadeDB image strongly recommended for pg_search BM25)
pip install "hermes-memory-lancedb[pgvector] @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@main"

# Both
pip install "hermes-memory-lancedb[lancedb,pgvector] @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@main"
```

Pin to a tag with `@v2.0.0` instead of `@main`.

Base requirements: Python ≥ 3.10, `openai`, `pyarrow`, `httpx`, `click`, `portalocker`. Backend extras add `lancedb`+`tantivy` or `psycopg[binary,pool]`.

### Backend selection matrix

| Want | Set | Notes |
|---|---|---|
| LanceDB (embedded, default) | nothing — or `HERMES_MEMORY_BACKEND=lancedb` | Storage at `$LANCEDB_PATH` |
| Postgres + pgvector | `HERMES_MEMORY_DATABASE_URL=postgresql://...` | Auto-detected; backend becomes `pgvector` |
| Force pgvector even with URL unset | `HERMES_MEMORY_BACKEND=pgvector` + `HERMES_MEMORY_DATABASE_URL=...` | Same as above; explicit |
| Force LanceDB even with URL set | `HERMES_MEMORY_BACKEND=lancedb` | Useful for A/B tests |

### Pgvector backend setup

The plugin creates its own schema (`hermes_memory.memories`) inside whatever DB you point it at — no name clashes with existing tables. On first connect it runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_search;  -- ParadeDB BM25 (skipped silently if not available)
CREATE SCHEMA IF NOT EXISTS hermes_memory;
CREATE TABLE hermes_memory.memories (...);
CREATE INDEX ... USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ... USING bm25 (id, content) WITH (key_field='id');  -- or GIN tsvector fallback
```

Concurrency is handled by Postgres advisory locks (no `portalocker` files for pgvector mode).

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
| **Backend selection** | | |
| `HERMES_MEMORY_BACKEND` | auto | `lancedb` or `pgvector`. Auto-detects pgvector when `HERMES_MEMORY_DATABASE_URL` is set |
| `HERMES_MEMORY_DATABASE_URL` | unset | Postgres connection URL (URL-encode special chars in the password). Required for pgvector backend |
| **Embedding** | | |
| `OPENAI_API_KEY` | required for default provider | OpenAI embeddings + LLM extraction (`gpt-4o-mini`) |
| `LANCEDB_PATH` | `$HERMES_HOME/lancedb` | LanceDB storage directory (lancedb backend only) |
| `LANCEDB_EMBED_PROVIDER` | `openai` | One of `openai` / `openrouter` / `jina` / `gemini` / `ollama` / `openai-compatible` |
| `LANCEDB_EMBED_MODEL` | provider default | Override embedding model id |
| `LANCEDB_EMBED_DIM` | from model id | Explicit embedding dimension override |
| `LANCEDB_EMBED_BASE_URL` | unset | Custom base URL (required for `openai-compatible`) |
| `LANCEDB_EMBED_API_KEY` | unset | Generic key override (otherwise reads provider-specific keys) |
| `JINA_API_KEY` | unset | Required for Jina embeddings |
| `GEMINI_API_KEY` | unset | Required for Gemini embeddings |
| `OPENROUTER_API_KEY` | unset | Required for OpenRouter embeddings (default model `openai/text-embedding-3-small`) |
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
| `LANCEDB_COMPACT_EVERY_N` | `100` | Auto-trigger memory compaction every N writes (P4) |

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
| `temporal_type` | string | P4 — `static` (default) or `dynamic` (decays 3x faster) |

## Tools exposed to the agent

| Tool | Purpose |
| --- | --- |
| `lancedb_search` | Hybrid retrieval (intent-routed, query-expanded), returns markdown bullets |
| `lancedb_remember` | Explicit write of a durable fact |
| `lancedb_forget` | Soft-delete by id |
| `lancedb_stats` | Counts per tier / category, storage path |
| `lancedb_compact` | Cluster + merge near-duplicate memories (also auto-runs every N writes) |

## CLI

Installing the package registers a `hermes-memory-lancedb` console script (also runnable as `python -m hermes_memory_lancedb.cli`). All commands accept `--storage-path` and `--user-id` at the top level.

| Command | Purpose |
| --- | --- |
| `version` | Print the package version |
| `list [--limit N] [--tier TIER] [--category CAT] [--json]` | List entries |
| `search <query> [--limit N] [--trace] [--json]` | Hybrid search via the existing pipeline |
| `stats [--json]` | Counts per tier/category, storage path, table size, retrieval stats |
| `delete <id>` | Delete one entry by id |
| `delete-bulk --ids <id1,id2,...>` or `--filter "tier='peripheral'"` | Bulk delete (supports `--dry-run`) |
| `export [--format json\|jsonl] [--out FILE]` | Dump all entries |
| `import <file> [--dry-run]` | Load entries from JSON or JSONL |
| `import-markdown [GLOB...] [--base-dir DIR] [--dry-run]` | Ingest `MEMORY.md` / `memory/YYYY-MM-DD.md` files |
| `reembed [--target PATH] [--dry-run] [--batch-size N]` | Re-embed all entries; `--target` writes to a parallel DB for A/B retrieval comparison |
| `migrate check\|run\|verify` | Schema migration ops |
| `reindex-fts` | Drop and rebuild the FTS index |

Examples:

```bash
# Show what's in the store
hermes-memory-lancedb list --tier core --limit 50

# Inspect a search end-to-end with the per-query pipeline trace
hermes-memory-lancedb search "andrew leeds plumbing" --trace

# A/B compare embedders by re-embedding into a parallel DB
hermes-memory-lancedb reembed --target ~/.hermes/lancedb-jina

# Pull MEMORY.md into the store from any directory
hermes-memory-lancedb import-markdown 'docs/**/*.md' --base-dir .

# Run schema migrations (e.g. after upgrading)
hermes-memory-lancedb migrate run
```

Both `LanceDBMemoryProvider._hybrid_search` and the CLI `search` command accept an optional `RetrievalTrace`. Pass one in to capture per-stage timings, score ranges, and dropped IDs:

```python
from hermes_memory_lancedb import LanceDBMemoryProvider, RetrievalTrace
provider = LanceDBMemoryProvider()
provider.initialize("session")
trace = RetrievalTrace()
hits = provider._hybrid_search("hello", top_k=5, trace=trace)
print(trace.summarize())
print(provider.get_stats())  # rolling aggregate over the last 1000 queries
```

Concurrent writers (multiple Hermes processes sharing the same LanceDB) are made safe by an exclusive `portalocker` file lock around `_write_entries`, schema migrations, and `reindex-fts`. The lock file lives next to the table directory at `<storage_path>/memories.lock`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

330+ unit tests across all phases — noise filtering, decay, RRF fusion, tier evaluation, LLM extraction, dedup, prompt builders, the v1.2.0 retrieval helpers (length norm, MMR, rerank, cosine), v1.3.0 P1 (scope manager + embedder factory), v1.4.0 P2 (chunker, batch dedup, admission control, smart metadata, noise prototypes), v1.5.0 P3 (reflection subsystem), v1.6.0 P4 (lifecycle, temporal classifier, session compress/recover, compactor, query expansion), and v1.7.0 P5 (management CLI via `click.testing.CliRunner`, `RetrievalTrace`/`RetrievalStats`, markdown parser, file-lock context manager). Tests that touch the LanceDB native runtime are gated behind a real `lancedb` import — skip them on CPUs without AVX support.

## Status

| Phase | Status |
| --- | --- |
| P0 — Cross-encoder rerank, MMR, length norm, hard min-score | Done (v1.2.0) |
| P1 — Multi-scope (agent/user/project/team/workspace), multi-provider embeddings (Jina/Gemini/Ollama) | Done (v1.3.0) |
| P2 — Chunker, batch dedup, admission control, smart metadata, noise prototypes | Done (v1.4.0) |
| P3 — Reflection subsystem (event store, item store, ranking, retry, slices) | Done (v1.5.0) |
| P4 — Lifecycle module, temporal classifier, session compactor, memory compactor, auto-capture cleanup, intent + query expansion | Done (v1.6.0) |
| P5 — Management CLI, retrieval observability, markdown import, A/B reembed, file locking | Done (v1.7.0) |
| P6 — Pluggable storage backends (`MemoryStore` ABC, `LanceDBStore`, `PgvectorStore` with pg_search BM25) | Done (v2.0.0) |

### Deployment notes

| Host hardware | Recommended backend | Why |
|---|---|---|
| Modern CPU with AVX2 (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015) | LanceDB (default) | Embedded, no external DB, lower latency |
| Older CPU without AVX2 (Sandy Bridge, etc.) | pgvector (ParadeDB) | LanceDB native binary SIGILLs on import; route storage server-side |
| Multiple Athena instances sharing memory | pgvector | Postgres ACID + advisory locks vs file lock; cleaner concurrency story |
| Embedded/single-process, no DB to manage | LanceDB | Zero infra |

For pgvector mode the recommended Postgres image is `paradedb/paradedb:latest` (ships `pg_search` for true BM25). On vanilla pgvector the plugin transparently falls back to `tsvector` ranking.

### P3 — Reflection subsystem (v1.5.0)

Reflections are agent-generated meta-memories with their own LanceDB table (`reflections`) and FTS index. They live in `hermes_memory_lancedb.reflection` and expose `ReflectionStore`, `ReflectionEventStore`, `ReflectionItemStore`, and `ReflectionRanker`.

Disabled by default — opt in with:

```bash
export LANCEDB_REFLECTION_ENABLED=1
# Optional knobs:
export LANCEDB_REFLECTION_TOP_K=3              # reflections merged into hybrid_search
export LANCEDB_REFLECTION_SCOPE=global         # scope tag for new reflections
export LANCEDB_REFLECTION_COMMAND=session_end  # command field on auto-captures
```

Two new agent tools:

| Tool | Purpose |
| --- | --- |
| `lancedb_reflect` | Explicit write of a reflection markdown blob (Invariants / Derived / Lessons / Decisions sections) |
| `lancedb_reflections` | Search ONLY the reflection store (separate from `lancedb_search`) |

The reflection markdown schema follows the TS port:

```markdown
## Invariants
- Always confirm production deploys via the dashboard before declaring done.

## Derived
- Next run, retry the failing migration with `--no-stats` to bypass the lock.

## Lessons & pitfalls (symptom / cause / fix / prevention)
- Symptom: 503 on container start. Cause: missing env var. Fix: load via Infisical. Prevention: add a startup smoke test.

## Decisions (durable)
- Use Compose + Traefik instead of Swarm on single-server deploys.
```

OAuth is intentionally out of scope — `OPENAI_API_KEY` and (optionally) `JINA_API_KEY` are sufficient.
