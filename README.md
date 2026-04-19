# hermes-memory-lancedb

LanceDB-backed persistent memory for [Hermes Agent](https://github.com/TheophilusChinomona/hermes-agent) (Athena fork). Hybrid BM25 + vector recall, cross-encoder reranking, MMR diversity, three-tier lifecycle, and LLM smart extraction with dedup.

This is the Python port of [memory-lancedb-pro](https://github.com/TheophilusChinomona/memory-lancedb-pro) (TypeScript). Drop-in for the bundled `lancedb` plugin in Athena's `plugins/memory/lancedb/`.

## Features

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

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

84 unit tests cover noise filtering, decay, RRF fusion, tier evaluation, LLM extraction, dedup, prompt builders, and the v1.2.0 retrieval helpers (length norm, MMR, rerank, cosine). Tests that touch the LanceDB native runtime are gated behind a real `lancedb` import — skip them on CPUs without AVX support.

## Status

| Phase | Status |
| --- | --- |
| P0 — Cross-encoder rerank, MMR, length norm, hard min-score | Done (v1.2.0) |
| P1 — Multi-scope (agent/user/project/team), multi-provider embeddings (Jina/Gemini/Ollama) | In flight |
| P2 — Chunker, batch dedup, admission control, smart metadata, noise prototypes | In flight |
| P3 — Reflection subsystem (event store, item store, ranking, retry, slices) | Done (v1.5.0) |
| P4 — Session compactor, memory compactor, temporal classifier, auto-capture cleanup, query expansion | Pending |
| P5 — Management CLI, retrieval observability, markdown import, A/B reembed | Pending |

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
