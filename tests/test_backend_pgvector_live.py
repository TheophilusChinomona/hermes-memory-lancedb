"""Live integration test for the pgvector backend.

Skipped by default. Set ``HERMES_MEMORY_DATABASE_URL`` plus
``HERMES_MEMORY_BACKEND=pgvector`` to enable; the test will then
initialize the provider against the real Postgres+pgvector instance,
write a few entries and run a search.

Embedding is mocked so the test doesn't depend on a real network call.

Run with::

    HERMES_MEMORY_DATABASE_URL='postgresql://user:pass@host/db' \\
    HERMES_MEMORY_BACKEND=pgvector \\
    pytest tests/test_backend_pgvector_live.py -v
"""

from __future__ import annotations

import hashlib
import os
import time
import unittest
import uuid
from typing import List

import pytest


_LIVE = bool(os.environ.get("HERMES_MEMORY_DATABASE_URL"))


def _hash_embed(text: str, *, dim: int = 1536) -> List[float]:
    """Deterministic hash-based embedder so the test has no network deps.

    Uses SHA-256 expanded to ``dim`` floats in [-1, 1]. Same text → same
    vector, so a search for ``text`` should find rows containing ``text``.
    """
    if not text:
        return [0.0] * dim
    out: List[float] = []
    counter = 0
    while len(out) < dim:
        h = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for i in range(0, len(h), 4):
            if len(out) >= dim:
                break
            chunk = h[i : i + 4]
            n = int.from_bytes(chunk, "big", signed=False)
            out.append(((n / 0xFFFFFFFF) * 2.0) - 1.0)
        counter += 1
    return out


class _StubEmbedder:
    """Minimal Embedder shim with the attributes the provider expects."""

    dimensions = 1536
    model = "stub-hash-embedder"
    provider = "stub"

    def embed(self, text: str) -> List[float]:
        return _hash_embed(text, dim=self.dimensions)

    def embed_batch(self, texts):  # noqa: D401 — match Embedder duck-type
        return [self.embed(t) for t in texts]


@pytest.mark.skipif(not _LIVE, reason="HERMES_MEMORY_DATABASE_URL not set")
class TestPgvectorLive(unittest.TestCase):
    """End-to-end smoke test of the pgvector backend through the provider."""

    @classmethod
    def setUpClass(cls):
        # Force pgvector backend for this run (covers the case where someone
        # has the env URL set but no explicit override).
        os.environ["HERMES_MEMORY_BACKEND"] = "pgvector"
        # Disable admission control + smart metadata so the writes go through
        # without an LLM key.
        os.environ.setdefault("LANCEDB_ADMISSION_ENABLED", "0")
        os.environ.setdefault("LANCEDB_SMART_METADATA", "0")
        # Avoid background prefetch trying to hit anything fancy.
        cls._session_id = f"pg-live-{uuid.uuid4()}"
        cls._user_id = f"pg-live-user-{uuid.uuid4().hex[:8]}"

    def setUp(self):
        # Build a provider whose embedder is the stub. We patch make_embedder
        # at the module-level so initialize() picks it up.
        from unittest.mock import patch

        import athena_memory as hl

        self._embedder_patch = patch.object(
            hl, "make_embedder", return_value=_StubEmbedder()
        )
        self._embedder_patch.start()
        self.provider = hl.LanceDBMemoryProvider()
        self.provider.initialize(self._session_id, user_id=self._user_id)

    def tearDown(self):
        try:
            self._embedder_patch.stop()
        except Exception:
            pass
        # Clean up only the rows this test wrote, by user_id.
        try:
            store = getattr(self.provider, "_store", None)
            if store is not None and getattr(store, "is_ready", False):
                # Use a where-list-then-delete loop to keep the test self-cleaning.
                rows = store.list_rows(
                    where=f"user_id = '{self._user_id}'", limit=500
                )
                for r in rows:
                    rid = r.get("id")
                    if rid:
                        store.delete_by_id(str(rid))
        except Exception:
            pass
        try:
            self.provider.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_provider_is_ready_against_pgvector(self):
        self.assertTrue(self.provider._ready, "provider failed to initialize")
        self.assertIsNotNone(self.provider._store)
        self.assertEqual(self.provider._backend_name, "pgvector")
        self.assertTrue(self.provider._store.is_ready)

    def test_provider_initialize_writes_search(self):
        # Write 3 entries via the lancedb_remember tool — the public write path.
        contents = [
            "Theo prefers concise communication and direct feedback.",
            "Andrew is the AI sales agent built for SpecCon.",
            "PostgreSQL with pgvector backs the v2.0.0 hermes memory layer.",
        ]
        for c in contents:
            result = self.provider.handle_tool_call(
                "lancedb_remember",
                {"content": c, "category": "cases", "importance": 0.7},
            )
            self.assertIn("stored", result)

        # Give the writes a beat in case anything is async.
        time.sleep(0.2)

        # Search should hit at least one of the rows we just wrote.
        result = self.provider.handle_tool_call(
            "lancedb_search", {"query": "pgvector hermes memory", "top_k": 6}
        )
        self.assertIn("results", result)
        # Decode and assert at least one result is non-empty.
        import json as _json
        payload = _json.loads(result)
        self.assertIsInstance(payload.get("results"), list)
        self.assertGreaterEqual(
            len(payload["results"]),
            1,
            f"no results returned: {payload}",
        )

    def test_lancedb_stats_runs_against_pgvector(self):
        # Write one row so the count is nonzero.
        self.provider.handle_tool_call(
            "lancedb_remember",
            {"content": "Stats test entry for pgvector backend.", "importance": 0.5},
        )
        time.sleep(0.1)
        result = self.provider.handle_tool_call("lancedb_stats", {})
        import json as _json
        payload = _json.loads(result)
        self.assertNotIn("error", payload)
        self.assertGreaterEqual(payload.get("total", 0), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
