# hermes-memory-lancedb

LanceDB-backed persistent memory for [Hermes Agent](https://github.com/TheophilusChinomona/hermes-agent) (Athena fork). Hybrid BM25 + vector recall, cross-encoder reranking, MMR diversity, three-tier lifecycle, LLM smart extraction with dedup, plus a management CLI and per-query observability.

This is the Python port of [memory-lancedb-pro](https://github.com/TheophilusChinomona/memory-lancedb-pro) (TypeScript). Drop-in for the bundled `lancedb` plugin in Athena's `plugins/memory/lancedb/`.

## Features

**Management & observability (v1.7.0)**
- Console CLI: `hermes-memory-lancedb {list,search,stats,delete,delete-bulk,export,import,import-markdown,reembed,migrate,reindex-fts,version}`
- Per-query `RetrievalTrace` records timings, score ranges, and dropped IDs at every pipeline stage
- Rolling `RetrievalStats` ring buffer aggregates queries-by-source, p95 latency, top drop stages, and result-count histograms
- Markdown ingest for `MEMORY.md` and dated `memory/YYYY-MM-DD.md` files
- A/B re-embedding via `reembed --target /path/to/alt.db` for embedder upgrades
- `portalocker` cross-process file lock around writes, schema migrations, and FTS reindex so multiple Hermes processes can share one LanceDB safely

**Retrieval pipeline (v1.2.0)**
- Hybrid search: parallel vector (cosine, OpenAI `text-embedding-3-small`) + BM25 (tantivy)
- RRF fusion → score normalization → tier/recency boost
- Cross-encoder reranking via Jina (`jina-reranker-v3`), with cosine fallback when no API key is set
- Length normalization (log2 penalty for entries longer than 500 chars)
- Hard min-score cutoff (default 0.35, env-tunable)
- MMR diversity (defers near-duplicates with cosine > 0.85 to the tail)

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
pip install "hermes-memory-lancedb @ git+https://github.com/TheophilusChinomona/hermes-memory-lancedb@v1.7.0"
```

Runtime requirements: Python ≥ 3.10, `lancedb ≥ 0.20`, `tantivy ≥ 0.21`, `openai ≥ 1.0`, `pyarrow ≥ 12`, `click ≥ 8.0`, `portalocker ≥ 2.7`, `httpx` (transitively via `openai`).

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
| `OPENAI_API_KEY` | required | Embeddings (`text-embedding-3-small`) and LLM extraction (`gpt-4o-mini`) |
| `LANCEDB_PATH` | `$HERMES_HOME/lancedb` | Storage directory |
| `LANCEDB_RERANK_PROVIDER` | `auto` | `auto` (Jina if key set, else cosine) or `none` to skip rerank |
| `LANCEDB_RERANK_API_KEY` / `JINA_API_KEY` | unset | Enables Jina cross-encoder rerank |
| `LANCEDB_RERANK_MODEL` | `jina-reranker-v3` | Jina model name |
| `LANCEDB_RERANK_ENDPOINT` | `https://api.jina.ai/v1/rerank` | Jina-compatible endpoint |
| `LANCEDB_RERANK_TIMEOUT_S` | `5.0` | Rerank API timeout in seconds |
| `LANCEDB_HARD_MIN_SCORE` | `0.35` | Final score cutoff after pipeline |

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
| `vector` | fixed_size_list<float, 1536> | text-embedding-3-small |
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

## Tools exposed to the agent

| Tool | Purpose |
| --- | --- |
| `lancedb_search` | Hybrid retrieval, returns markdown bullets |
| `lancedb_remember` | Explicit write of a durable fact |
| `lancedb_forget` | Soft-delete by id |
| `lancedb_stats` | Counts per tier / category, storage path |

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

138+ unit tests cover noise filtering, decay, RRF fusion, tier evaluation, LLM extraction, dedup, prompt builders, the v1.2.0 retrieval helpers (length norm, MMR, rerank, cosine), the v1.7.0 management CLI (via `click.testing.CliRunner`), `RetrievalTrace`/`RetrievalStats` aggregation, the markdown parser, and the file-lock context manager. Tests that touch the LanceDB native runtime are gated behind a real `lancedb` import — skip them on CPUs without AVX support.

## Status

| Phase | Status |
| --- | --- |
| P0 — Cross-encoder rerank, MMR, length norm, hard min-score | Done (v1.2.0) |
| P1 — Multi-scope (agent/user/project/team), multi-provider embeddings (Jina/Gemini/Ollama) | Pending |
| P2 — Chunker, batch dedup, admission control, smart metadata, noise prototypes | Pending |
| P3 — Reflection subsystem (event store, item store, ranking, retry, slices) | Pending |
| P4 — Session compactor, memory compactor, temporal classifier, auto-capture cleanup, query expansion | Pending |
| P5 — Management CLI, retrieval observability, markdown import, A/B reembed, file locking | Done (v1.7.0) |

OAuth is intentionally out of scope — `OPENAI_API_KEY` and (optionally) `JINA_API_KEY` are sufficient.
