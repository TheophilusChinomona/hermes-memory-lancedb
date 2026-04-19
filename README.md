# athena-memory

Persistent memory plugin for [Hermes Agent](https://github.com/TheophilusChinomona/hermes-agent) (Athena fork) — hybrid BM25 + vector retrieval, three-tier lifecycle, LLM-driven extraction and dedup, multi-scope isolation, multi-provider embeddings, cross-encoder reranking, MMR diversity, reflection subsystem, management CLI, per-query observability.

Two interchangeable backends:

- **LanceDB** — embedded, zero external services. Fast, but the native binary requires AVX2 (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015).
- **Postgres + pgvector + pg_search** ([ParadeDB](https://www.paradedb.com/)) — server-side, runs on any CPU, ACID writes, easy cross-process sharing.

Same pipeline, same retrieval quality, same agent-facing tools — pick whichever fits your hardware and ops story.

Python port of [memory-lancedb-pro](https://github.com/TheophilusChinomona/memory-lancedb-pro) (TypeScript).

---

## Quickstart

### A. Postgres + pgvector (any CPU, recommended for self-host)

```bash
# 1. Install with the pgvector extra
pip install "athena-memory[pgvector] @ git+https://gitlab.com/chinomonatinotenda19/athena-memory@main"

# 2. Point at any Postgres with `vector` (and optionally `pg_search`) installed.
#    The plugin creates its own `hermes_memory` schema on first connect.
export HERMES_MEMORY_DATABASE_URL='postgresql://user:pass@host:5432/yourdb'

# 3. Pick an embedding provider — OpenRouter routes to OpenAI, Cohere, Voyage, Jina from one key
export OPENROUTER_API_KEY=sk-or-v1-...
export LANCEDB_EMBED_PROVIDER=openrouter
export LANCEDB_EMBED_MODEL=openai/text-embedding-3-small
```

```python
from athena_memory import LanceDBMemoryProvider

p = LanceDBMemoryProvider()
p.initialize("session-1", user_id="andrew")
p.handle_tool_call("lancedb_remember", {"content": "Theo runs SpecCon Holdings."})
p.handle_tool_call("lancedb_remember", {"content": "Athena is the Discord bot fork of Hermes."})
print(p.handle_tool_call("lancedb_search", {"query": "who is Theo?", "top_k": 3}))
```

### B. LanceDB (embedded, AVX2-capable hardware)

```bash
pip install "athena-memory[lancedb] @ git+https://gitlab.com/chinomonatinotenda19/athena-memory@main"
export OPENAI_API_KEY=sk-...
# That's it — LANCEDB_PATH defaults to $HERMES_HOME/lancedb.
```

Same Python code as above; the backend is auto-detected (LanceDB when `HERMES_MEMORY_DATABASE_URL` is unset).

---

## Why athena-memory

Most "vector DB" packages assume modern CPU SIMD (AVX2 / x86-64-v3). On older boxes their native binaries crash on import. This plugin separates the **storage layer** (`MemoryStore` ABC) from the **retrieval pipeline** (rerank, MMR, length-norm, dedup, scope filtering, reflection, observability) so the same logic runs against either:

- LanceDB's embedded Rust core, or
- A Postgres server doing the heavy lifting on whatever hardware it lives on.

For Athena specifically: the `ai-dev` LXD host runs on a 2011 Sandy Bridge CPU. LanceDB SIGILLs on `import lancedb`. The pgvector backend is the workaround — and is fully feature-equivalent.

---

## Backend selection

| Want | Set | Notes |
|---|---|---|
| LanceDB (embedded) | nothing — or `HERMES_MEMORY_BACKEND=lancedb` | Storage at `$LANCEDB_PATH` |
| Postgres + pgvector | `HERMES_MEMORY_DATABASE_URL=postgresql://...` | Auto-detected; backend becomes `pgvector` |
| Force pgvector explicitly | `HERMES_MEMORY_BACKEND=pgvector` + URL | Useful for clarity in CI |
| Force LanceDB even with URL set | `HERMES_MEMORY_BACKEND=lancedb` | A/B testing, dev override |

### Deployment matrix

| Host | Recommended backend | Why |
|---|---|---|
| Modern CPU with AVX2 | LanceDB | Embedded; lowest latency; no DB to operate |
| Older CPU (no AVX2) | pgvector / ParadeDB | LanceDB binary won't run; route storage server-side |
| Multiple agent processes sharing memory | pgvector | Postgres ACID + advisory locks vs file locks |
| Stateless workers / serverless | pgvector | LanceDB needs a writable filesystem |
| Single-process embedded, no infra to manage | LanceDB | Zero ops |

### Pgvector backend setup

The plugin creates its own schema in whatever DB you point it at. On first connect:

```sql
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_search;  -- ParadeDB BM25 (skipped silently if missing)
CREATE SCHEMA IF NOT EXISTS hermes_memory;
CREATE TABLE hermes_memory.memories (...);
CREATE INDEX ... USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ... USING bm25 (id, content) WITH (key_field='id');  -- or GIN on tsvector
```

- The `hermes_memory.memories` schema name is intentional — isolated from any application tables in the host database.
- Concurrency uses Postgres advisory locks (no `portalocker` files in pgvector mode).
- Connection pool defaults to `min_size=1, max_size=5` per provider instance.

### Available extension support

| Postgres image | `vector` | `pg_search` BM25 |
|---|---|---|
| `paradedb/paradedb:latest` | ✅ | ✅ (recommended) |
| `pgvector/pgvector:pg16` (or pg15/14) | ✅ | tsvector fallback |
| Vanilla `postgres:18` + `apt install postgresql-18-pgvector` | ✅ | tsvector fallback |
| Vanilla Postgres without pgvector | ❌ | won't initialize |

ParadeDB ships with `pg_search` (real BM25 via tantivy — same engine LanceDB uses internally). Without it, the plugin falls back to Postgres native `tsvector` + `ts_rank_cd` (TF-IDF flavor); recall is comparable for chat memory, slightly worse for long technical docs.

---

## Architecture

### Retrieval pipeline

```
                ┌──────────────┐
   query ──────▶│  embedder    │──▶ vector
                │ (multi-prov) │
                └──────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼                               ▼
  vector ANN                       BM25 / FTS
  (pgvector HNSW                   (pg_search or
   or LanceDB)                     LanceDB tantivy)
       │                               │
       └──────────┬────────────────────┘
                  ▼
            RRF fusion
                  │
                  ▼
         normalize to top
                  │
                  ▼
       tier × decay weight boost
                  │
                  ▼
        early min-score filter
                  │
                  ▼
       cross-encoder rerank
       (Jina if key, else cosine)
                  │
                  ▼
        length normalization
       (penalize sprawling rows)
                  │
                  ▼
        hard min-score cutoff
                  │
                  ▼
         MMR diversity
       (defer cosine > 0.85 dups)
                  │
                  ▼
              top_k
```

### Write pipeline

```
turn / explicit content
        │
        ▼
   noise filter (regex + vector prototype)
        │
        ▼
   admission control (rolling stats)
        │
        ▼
   smart metadata extraction (LLM)
        │
        ▼
   long-content chunker (parent_id linkage)
        │
        ▼
   batch dedup (cosine pre-filter + LLM judge)
        │
        ▼
   write to backend (advisory lock OR file lock)
        │
        ▼
   compactor trigger every N writes
```

---

## Features

**Retrieval**
- Hybrid: parallel vector ANN + BM25 (pg_search / LanceDB tantivy / tsvector)
- RRF fusion → score normalization → tier/recency boost
- Cross-encoder rerank via Jina (`jina-reranker-v3`); cosine fallback when no key
- Length normalization (log2 penalty for entries > 500 chars)
- Hard min-score cutoff (default 0.35, env-tunable)
- MMR diversity — defers near-duplicates (cosine > 0.85) instead of dropping
- Intent analyzer + query expansion (BM25 fan-out across synonyms)

**Write pipeline**
- LLM smart extraction: 6 categories (`profile`, `preferences`, `entities`, `events`, `cases`, `patterns`)
- 3-level entry: `abstract` (≤80 chars) / `overview` (≤800 chars bullets) / `content` (full)
- Dedup with 7 outcomes: `skip` / `support` / `merge` / `contextualize` / `contradict` / `supersede` / `create`
- Long-content chunker — sentence-boundary splits with overlap, chunks share `parent_id`
- Admission controller with rolling acceptance / novelty / recency stats persisted to disk
- Smart metadata: per-write LLM extraction of `temporal_type`, `confidence`, `sensitivity`, `modality`, `fact_key`, `tags`
- Noise prototype filter — bundled multilingual prototypes; rejects writes with cosine ≥ 0.92 to any

**Lifecycle**
- 3 tiers: `peripheral` → `working` → `core`, each with a decay floor
- Promotion: `peripheral → working` at access ≥ 3 & decay×importance ≥ 0.4; `working → core` at access ≥ 10 & composite ≥ 0.7 & importance ≥ 0.8
- Demotion: composite < 0.15 OR (age > 60 days AND access < 3)
- Weibull decay: `exp(-((age_days/30)^0.7))`; `dynamic` memories decay 3× faster
- Session compressor: end-of-session summary written as a single dense entry
- Session recovery: re-inflate compressed entries when a `session_id` reopens
- Memory compactor: clusters near-duplicates and merges (auto-trigger every N writes; also a `lancedb_compact` tool)
- Auto-capture cleanup at session start: deletes / demotes the previous session's noisy auto-captures

**Multi-tenancy**
- Multi-scope isolation: `agent_id`, `user_id`, `project_id`, `team_id`, `workspace_id` columns compose orthogonally in the search WHERE clause
- Built-in scope patterns (`agent:*`, `user:*`, `project:*`, `team:*`, `workspace:*`, `custom:*`, `reflection:*`) plus `global`
- ClawTeam shared scopes via `CLAWTEAM_MEMORY_SCOPE` env (CSV)
- Backwards compatible: legacy tables without scope columns auto-migrate

**Multi-provider embeddings**
- OpenAI (default), OpenRouter (`openai/`, `cohere/`, `voyage/`, `jina/` prefixed model ids), Jina, Gemini, Ollama, plus any OpenAI-compatible endpoint via `LANCEDB_EMBED_BASE_URL`
- Vector dimension is provider-driven and persisted in the schema

**Reflection subsystem** (opt-in via `LANCEDB_REFLECTION_ENABLED=1`)
- Separate `reflections` table with its own FTS index
- 8 modules: `store`, `event_store`, `item_store`, `metadata`, `mapped_metadata`, `ranking`, `retry`, `slices`
- Captures session summaries as structured markdown (Invariants / Derived / Lessons / Decisions)
- New tools: `lancedb_reflect`, `lancedb_reflections`
- Currently LanceDB-backed only; pgvector port pending

**Management & observability**
- Console CLI (see [CLI](#cli))
- `RetrievalTrace`: per-stage timings, score ranges, dropped IDs
- `RetrievalStats`: rolling ring buffer, p95 latency, queries-by-source, top drop stages
- Markdown ingest for `MEMORY.md` and dated `memory/YYYY-MM-DD.md` files
- A/B re-embedding via `reembed --target /path/to/alt.db`

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| **Backend selection** | | |
| `HERMES_MEMORY_BACKEND` | auto | `lancedb` or `pgvector`; auto-detects pgvector when `HERMES_MEMORY_DATABASE_URL` is set |
| `HERMES_MEMORY_DATABASE_URL` | unset | Postgres URL (URL-encode special chars in the password) |
| `LANCEDB_PATH` | `$HERMES_HOME/lancedb` | LanceDB storage directory (lancedb backend only) |
| **Embedding** | | |
| `LANCEDB_EMBED_PROVIDER` | `openai` | One of `openai` / `openrouter` / `jina` / `gemini` / `ollama` / `openai-compatible` |
| `LANCEDB_EMBED_MODEL` | provider default | Override embedding model id |
| `LANCEDB_EMBED_DIM` | from model id | Explicit dimension override |
| `LANCEDB_EMBED_BASE_URL` | unset | Custom base URL (required for `openai-compatible`) |
| `LANCEDB_EMBED_API_KEY` | unset | Generic key override (otherwise reads provider-specific keys) |
| `OPENAI_API_KEY` | required for OpenAI | OpenAI embeddings + `gpt-4o-mini` for LLM extraction |
| `OPENROUTER_API_KEY` | unset | OpenRouter embeddings (default model `openai/text-embedding-3-small`) |
| `JINA_API_KEY` | unset | Jina embeddings |
| `GEMINI_API_KEY` | unset | Gemini embeddings |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| **Scope** | | |
| `LANCEDB_AGENT_ID` / `LANCEDB_PROJECT_ID` / `LANCEDB_TEAM_ID` / `LANCEDB_WORKSPACE_ID` | unset | Per-dimension scope identifiers (compose orthogonally) |
| `CLAWTEAM_MEMORY_SCOPE` | unset | CSV of extra scopes granted to every agent |
| **Rerank** | | |
| `LANCEDB_RERANK_PROVIDER` | `auto` | `auto` (Jina if key set, else cosine) or `none` to skip |
| `LANCEDB_RERANK_API_KEY` / `JINA_API_KEY` | unset | Enables Jina cross-encoder rerank |
| `LANCEDB_RERANK_MODEL` | `jina-reranker-v3` | Reranker model |
| `LANCEDB_RERANK_ENDPOINT` | `https://api.jina.ai/v1/rerank` | Jina-compatible endpoint (Cohere/OpenRouter also work) |
| `LANCEDB_RERANK_TIMEOUT_S` | `5.0` | Rerank API timeout |
| **Pipeline tuning** | | |
| `LANCEDB_HARD_MIN_SCORE` | `0.35` | Final score cutoff after rerank |
| `LANCEDB_ADMISSION_ENABLED` | `1` | Set `0` to disable admission control |
| `LANCEDB_SMART_METADATA` | `1` | Set `0` to skip per-write smart metadata LLM calls |
| `LANCEDB_COMPACT_EVERY_N` | `100` | Auto-trigger memory compaction every N writes |
| `LANCEDB_REFLECTION_ENABLED` | unset | Set `1` to enable the reflection subsystem (LanceDB backend only) |

The `LANCEDB_*` prefix is historical and somewhat misleading post-v3.0.0 (these knobs apply across both backends). Renaming to `ATHENA_*` is on the roadmap.

Persistent config is also accepted in `$HERMES_HOME/lancedb.json`:

```json
{
  "storage_path": "/home/me/.hermes/lancedb",
  "user_id": "andrew",
  "extraction_model": "gpt-4o-mini"
}
```

---

## Storage schema

Single `memories` table (LanceDB) or `hermes_memory.memories` (pgvector). Identical column set on both backends:

| Column | Type | Notes |
|---|---|---|
| `id` | string | UUID |
| `content` | string | Full narrative (≤2000 chars) |
| `vector` | vector(_N_) | _N_ = active embedder's dim (default 1536 for OpenAI 3-small) |
| `timestamp` | float | Unix epoch |
| `source` | string | e.g. `discord:dm:<id>`, `explicit`, `turn` |
| `session_id` | string | Hermes session id at write time |
| `user_id` | string | Scoping key (default `andrew`) |
| `tags` | string | JSON-encoded list |
| `tier` | string | `peripheral` / `working` / `core` |
| `importance` | float | 0–1, set by extractor |
| `access_count` | int | Bumped on retrieval |
| `category` | string | One of `MEMORY_CATEGORIES` |
| `abstract` | string | L0 — ≤80 chars |
| `overview` | string | L1 — ≤800 chars bullets |
| `agent_id` | string | Scope column |
| `project_id` | string | Scope column |
| `team_id` | string | Scope column |
| `workspace_id` | string | Scope column |
| `scope` | string | Canonical scope id (e.g. `agent:andrew`) |
| `metadata` | string | JSON-encoded smart metadata |
| `parent_id` | string | Set on chunked rows; empty for atomic writes |
| `temporal_type` | string | `static` (default) or `dynamic` |

---

## Tools exposed to the agent

| Tool | Purpose |
|---|---|
| `lancedb_search` | Hybrid retrieval (intent-routed, query-expanded), returns markdown bullets |
| `lancedb_remember` | Explicit write of a durable fact |
| `lancedb_forget` | Soft-delete by id |
| `lancedb_stats` | Counts per tier / category, storage path, table size |
| `lancedb_compact` | Cluster + merge near-duplicate memories (also auto-runs every N writes) |
| `lancedb_reflect` | Write a reflection markdown blob (when reflection subsystem is enabled) |
| `lancedb_reflections` | Search the reflection store (separate from regular memories) |

Tool names retain the `lancedb_*` prefix for backwards compat with deployed Hermes configs; they work on either backend.

---

## CLI

The package registers a `athena-memory` console script (the legacy `hermes-memory-lancedb` name is also kept as an alias). Also runnable as `python -m athena_memory.cli`.

| Command | Purpose |
|---|---|
| `version` | Print package version |
| `list [--limit N] [--tier TIER] [--category CAT] [--json]` | List entries |
| `search <query> [--limit N] [--trace] [--json]` | Hybrid search |
| `stats [--json]` | Counts per tier/category, storage path, table size, retrieval stats |
| `delete <id>` | Delete one entry |
| `delete-bulk --ids <id1,id2,...>` or `--filter "tier='peripheral'"` | Bulk delete (`--dry-run` supported) |
| `export [--format json\|jsonl] [--out FILE]` | Dump all entries |
| `import <file> [--dry-run]` | Load entries from JSON / JSONL |
| `import-markdown [GLOB...] [--base-dir DIR] [--dry-run]` | Ingest `MEMORY.md` / `memory/YYYY-MM-DD.md` |
| `reembed [--target PATH] [--dry-run] [--batch-size N]` | Re-embed all entries; `--target` writes to a parallel DB for A/B comparison |
| `migrate check\|run\|verify` | Schema migration ops |
| `reindex-fts` | Drop and rebuild the FTS index |

```bash
# Show what's in the store
athena-memory list --tier core --limit 50

# Inspect a search end-to-end with the per-query pipeline trace
athena-memory search "andrew leeds plumbing" --trace

# A/B compare embedders by re-embedding into a parallel DB
athena-memory reembed --target ~/.hermes/lancedb-jina

# Pull MEMORY.md into the store from any directory
athena-memory import-markdown 'docs/**/*.md' --base-dir .

# Run schema migrations after upgrading
athena-memory migrate run
```

### Programmatic observability

```python
from athena_memory import LanceDBMemoryProvider, RetrievalTrace

provider = LanceDBMemoryProvider()
provider.initialize("session")

trace = RetrievalTrace()
hits = provider._hybrid_search("hello", top_k=5, trace=trace)
print(trace.summarize())

# Rolling aggregate over the last 1000 queries:
print(provider.get_stats())
```

---

## Activate in Hermes Agent

In `~/.hermes/config.yaml`:

```yaml
memory:
  provider: lancedb
```

The bundled plugin shim at `plugins/memory/lancedb/__init__.py` (in [hermes-agent](https://github.com/TheophilusChinomona/hermes-agent)) wraps this package and registers it with Hermes's `MemoryProvider` ABC. The shim works against both backends — `HERMES_MEMORY_BACKEND` selects which one runs.

---

## Reflection subsystem

Reflections are agent-generated meta-memories — structured notes about what happened, what was learned, what to retry — stored in their own table separate from regular memories. Enable with:

```bash
export LANCEDB_REFLECTION_ENABLED=1
# Optional knobs:
export LANCEDB_REFLECTION_TOP_K=3              # reflections merged into hybrid_search
export LANCEDB_REFLECTION_SCOPE=global         # scope tag for new reflections
export LANCEDB_REFLECTION_COMMAND=session_end  # command field on auto-captures
```

Reflection markdown schema:

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

Reflections are currently LanceDB-only — the subsystem is silently skipped on the pgvector backend pending a `PgvectorReflectionStore` port.

---

## Tests

```bash
pip install -e ".[lancedb,pgvector,dev]"
pytest tests/

# Live integration tests (env-gated by HERMES_MEMORY_DATABASE_URL):
HERMES_MEMORY_DATABASE_URL='postgresql://...' pytest tests/test_backend_pgvector_live.py
```

400+ tests across the codebase: noise filtering, decay, RRF fusion, tier evaluation, LLM extraction, dedup, prompt builders, retrieval helpers (length norm, MMR, rerank, cosine), scope manager + embedder factory, write pipeline (chunker, batch dedup, admission, smart metadata, noise prototypes), reflection subsystem, lifecycle (tier manager, decay engine, temporal classifier, session compress/recover, compactor, query expansion), management CLI, observability, markdown parser, file lock, and pgvector store integration.

Tests that touch LanceDB's native runtime are gated behind a real `import lancedb` and skip on CPUs without AVX2 support.

---

## Status

| Phase | Status |
|---|---|
| P0 — Cross-encoder rerank, MMR, length norm, hard min-score | ✅ v1.2.0 |
| P1 — Multi-scope (agent/user/project/team/workspace), multi-provider embeddings | ✅ v1.3.0 |
| P2 — Chunker, batch dedup, admission control, smart metadata, noise prototypes | ✅ v1.4.0 |
| P3 — Reflection subsystem (8 modules) | ✅ v1.5.0 |
| P4 — Lifecycle module, temporal classifier, session compactor, memory compactor, auto-capture cleanup, intent + query expansion | ✅ v1.6.0 |
| P5 — Management CLI, retrieval observability, markdown import, A/B reembed, file locking | ✅ v1.7.0 |
| P6 — Pluggable storage backends (`MemoryStore` ABC, `LanceDBStore`, `PgvectorStore` with pg_search BM25) | ✅ v2.0.0 |
| Repo + package rename (`hermes-memory-lancedb` → `athena-memory`) | ✅ v3.0.0 |

### Roadmap

- `PgvectorReflectionStore` — port the reflection subsystem to the pgvector backend
- `compactor` / `sessions` / `auto_capture` — route through `MemoryStore` instead of introspecting `getattr(store, "_table")`
- Rename `LANCEDB_*` env vars to `ATHENA_*` (with old names as fallbacks)
- Rename `LanceDBMemoryProvider` class to `AthenaMemoryProvider` (with old name as alias)
- OAuth flows for managed embedding providers (intentionally out of scope for now)

---

## Compatibility

The repo and package were renamed in v3.0.0:

- GitHub: `TheophilusChinomona/hermes-memory-lancedb` → `TheophilusChinomona/athena-memory` (GitHub auto-redirects the old URL)
- Distribution: `hermes-memory-lancedb` → `athena-memory`
- Python module: `hermes_memory_lancedb` → `athena_memory`
- Console script: `hermes-memory-lancedb` → `athena-memory` (legacy name retained as an alias)

Old code that does `import hermes_memory_lancedb` keeps working through a thin shim that re-exports the entire `athena_memory` surface and aliases all submodules in `sys.modules`. The shim emits a `DeprecationWarning` pointing at the new name and will be removed in a future major version.

---

## License

MIT.
