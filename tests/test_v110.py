"""Comprehensive tests for hermes-memory-lancedb v1.1.0."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pyarrow as pa

from hermes_memory_lancedb import (
    LanceDBMemoryProvider,
    MEMORY_CATEGORIES,
    _age_days,
    _build_extraction_prompt,
    _is_noise,
    _llm_dedup,
    _llm_extract_memories,
    _merge_rrf,
    _rrf_score,
    _tier_evaluate,
    _weibull_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(tmpdir: str) -> LanceDBMemoryProvider:
    """Create an initialized provider backed by a temp directory."""
    p = LanceDBMemoryProvider()
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANCEDB_PATH": tmpdir}):
        with patch("hermes_memory_lancedb._EmbedClient") as MockEmbed, \
             patch("hermes_memory_lancedb._LLMClient") as MockLLM:
            MockEmbed.return_value.embed = lambda text: [0.1] * 1536
            MockLLM.return_value.chat = lambda *a, **kw: "[]"
            p.initialize("test-session-1")
    return p


def _dummy_embed(text: str) -> List[float]:
    import hashlib
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
    vec = [0.0] * 1536
    vec[h % 1536] = 1.0
    return vec


# ---------------------------------------------------------------------------
# 1. Noise filter
# ---------------------------------------------------------------------------

class TestNoiseFilter(unittest.TestCase):

    def test_empty_string_is_noise(self):
        self.assertTrue(_is_noise(""))

    def test_very_short_is_noise(self):
        self.assertTrue(_is_noise("ok"))

    def test_hi_is_noise(self):
        self.assertTrue(_is_noise("hi"))

    def test_hello_is_noise(self):
        self.assertTrue(_is_noise("Hello"))

    def test_thanks_is_noise(self):
        self.assertTrue(_is_noise("Thanks"))

    def test_sure_is_noise(self):
        self.assertTrue(_is_noise("Sure"))

    def test_ai_denial_is_noise(self):
        self.assertTrue(_is_noise("I don't have access to that information."))

    def test_as_an_ai_is_noise(self):
        self.assertTrue(_is_noise("As an AI, I cannot do that."))

    def test_meta_question_is_noise(self):
        self.assertTrue(_is_noise("Do you remember what we discussed?"))

    def test_can_you_recall_is_noise(self):
        self.assertTrue(_is_noise("Can you recall the last meeting?"))

    def test_test_message_is_noise(self):
        self.assertTrue(_is_noise("test message please ignore"))

    def test_real_content_not_noise(self):
        self.assertFalse(_is_noise("Theo wants to focus on UK care operators for TAP Business."))

    def test_long_ai_response_not_noise(self):
        # Starts with denial but has substantive content — still filtered (denial pattern)
        self.assertTrue(_is_noise("As an AI I can help you with this task."))

    def test_company_name_not_noise(self):
        self.assertFalse(_is_noise("SpecCon Holdings targets NHS-commissioned care providers."))

    def test_decision_not_noise(self):
        self.assertFalse(_is_noise("Decided to prioritise LinkedIn outreach over cold email this quarter."))

    def test_case_insensitive_denial(self):
        self.assertTrue(_is_noise("I CANNOT ACCESS external databases."))


# ---------------------------------------------------------------------------
# 2. Weibull decay
# ---------------------------------------------------------------------------

class TestWeibullDecay(unittest.TestCase):

    def test_zero_age_is_one(self):
        self.assertAlmostEqual(_weibull_weight(0), 1.0, places=5)

    def test_negative_age_is_one(self):
        self.assertAlmostEqual(_weibull_weight(-5), 1.0, places=5)

    def test_decay_over_time(self):
        w0 = _weibull_weight(0)
        w30 = _weibull_weight(30)
        w90 = _weibull_weight(90)
        self.assertGreater(w0, w30)
        self.assertGreater(w30, w90)

    def test_decay_between_zero_and_one(self):
        for days in [1, 7, 30, 90, 365]:
            w = _weibull_weight(days)
            self.assertGreater(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_heavy_tail_slow_decay(self):
        # shape < 1 means slower decay at long timescales
        w30 = _weibull_weight(30)
        w365 = _weibull_weight(365)
        # Even after a year, should retain some weight with heavy-tail Weibull
        self.assertGreater(w365, 0.001)

    def test_age_days_recent(self):
        ts = time.time() - 3600  # 1 hour ago
        self.assertAlmostEqual(_age_days(ts), 1 / 24, delta=0.01)


# ---------------------------------------------------------------------------
# 3. RRF merging
# ---------------------------------------------------------------------------

class TestRRF(unittest.TestCase):

    def test_rrf_score_decreases_with_rank(self):
        self.assertGreater(_rrf_score(0), _rrf_score(5))
        self.assertGreater(_rrf_score(5), _rrf_score(20))

    def test_merge_rrf_deduplicates(self):
        vec_hits = [{"id": "a", "content": "alpha"}, {"id": "b", "content": "beta"}]
        bm25_hits = [{"id": "a", "content": "alpha"}, {"id": "c", "content": "gamma"}]
        merged = _merge_rrf(vec_hits, bm25_hits, top_k=10)
        ids = [h["id"] for h in merged]
        self.assertEqual(len(ids), len(set(ids)))  # no dups
        self.assertEqual(sorted(ids), ["a", "b", "c"])

    def test_merge_rrf_top_k(self):
        vec_hits = [{"id": str(i), "content": f"item{i}"} for i in range(10)]
        bm25_hits = [{"id": str(i + 5), "content": f"item{i+5}"} for i in range(10)]
        merged = _merge_rrf(vec_hits, bm25_hits, top_k=3)
        self.assertEqual(len(merged), 3)

    def test_merge_rrf_shared_id_gets_higher_score(self):
        # id "shared" appears in both lists at rank 0 → should win
        vec_hits = [{"id": "shared", "content": "shared"}, {"id": "vec_only", "content": "v"}]
        bm25_hits = [{"id": "shared", "content": "shared"}, {"id": "bm25_only", "content": "b"}]
        merged = _merge_rrf(vec_hits, bm25_hits, top_k=3)
        self.assertEqual(merged[0]["id"], "shared")

    def test_merge_rrf_empty_inputs(self):
        self.assertEqual(_merge_rrf([], [], top_k=5), [])
        self.assertEqual(_merge_rrf([{"id": "a", "content": "x"}], [], top_k=5)[0]["id"], "a")


# ---------------------------------------------------------------------------
# 4. Tier evaluation
# ---------------------------------------------------------------------------

class TestTierEvaluation(unittest.TestCase):

    def test_peripheral_to_working_on_access(self):
        result = _tier_evaluate("peripheral", access_count=3, importance=0.8, decay_weight=0.9, age_days=1.0)
        self.assertEqual(result, "working")

    def test_peripheral_stays_if_low_access(self):
        result = _tier_evaluate("peripheral", access_count=2, importance=0.9, decay_weight=0.9, age_days=1.0)
        self.assertIsNone(result)

    def test_peripheral_stays_if_low_composite(self):
        result = _tier_evaluate("peripheral", access_count=3, importance=0.1, decay_weight=0.1, age_days=1.0)
        self.assertIsNone(result)  # composite = 0.01 < 0.4

    def test_working_to_core(self):
        result = _tier_evaluate("working", access_count=10, importance=0.9, decay_weight=0.85, age_days=2.0)
        self.assertEqual(result, "core")

    def test_working_to_core_requires_all_thresholds(self):
        # High access but low importance — no promo
        result = _tier_evaluate("working", access_count=15, importance=0.5, decay_weight=0.9, age_days=1.0)
        self.assertIsNone(result)

    def test_working_demotion_low_composite(self):
        result = _tier_evaluate("working", access_count=0, importance=0.1, decay_weight=0.1, age_days=10.0)
        self.assertEqual(result, "peripheral")

    def test_working_demotion_old_and_unaccessed(self):
        result = _tier_evaluate("working", access_count=1, importance=0.5, decay_weight=0.5, age_days=70.0)
        self.assertEqual(result, "peripheral")

    def test_core_demotion_very_low_composite(self):
        result = _tier_evaluate("core", access_count=0, importance=0.1, decay_weight=0.1, age_days=100.0)
        self.assertEqual(result, "working")

    def test_core_stays_with_high_access(self):
        result = _tier_evaluate("core", access_count=20, importance=0.9, decay_weight=0.95, age_days=5.0)
        self.assertIsNone(result)

    def test_unknown_tier_returns_none(self):
        result = _tier_evaluate("unknown_tier", 0, 0.0, 0.0, 0.0)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 5. Memory categories
# ---------------------------------------------------------------------------

class TestMemoryCategories(unittest.TestCase):

    def test_six_categories_defined(self):
        self.assertEqual(len(MEMORY_CATEGORIES), 6)

    def test_expected_category_names(self):
        for cat in ["profile", "preferences", "entities", "events", "cases", "patterns"]:
            self.assertIn(cat, MEMORY_CATEGORIES)


# ---------------------------------------------------------------------------
# 6. LLM extraction (mocked)
# ---------------------------------------------------------------------------

class TestLLMExtraction(unittest.TestCase):

    def _make_llm(self, response: str) -> Any:
        llm = MagicMock()
        llm.chat.return_value = response
        return llm

    def test_extract_valid_candidates(self):
        response = json.dumps([
            {
                "category": "profile",
                "abstract": "Theo is founder of SpecCon",
                "overview": "- Founder/operator\n- AI-powered sales agency",
                "content": "Theophilus Chinomona is the founder of SpecCon Holdings, building an AI-powered revenue agency.",
                "importance": 0.9,
                "tags": ["founder", "speccon"],
            }
        ])
        llm = self._make_llm(response)
        results = _llm_extract_memories("User: I run SpecCon.\nAssistant: Got it.", llm)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["category"], "profile")
        self.assertAlmostEqual(results[0]["importance"], 0.9)
        self.assertIn("founder", results[0]["tags"])

    def test_extract_empty_response(self):
        llm = self._make_llm("[]")
        results = _llm_extract_memories("User: ok\nAssistant: sure", llm)
        self.assertEqual(results, [])

    def test_extract_caps_at_five(self):
        many = [{"category": "cases", "abstract": f"item {i}", "overview": "", "content": f"content {i}", "importance": 0.5, "tags": []} for i in range(10)]
        llm = self._make_llm(json.dumps(many))
        results = _llm_extract_memories("User: lots of stuff\nAssistant: ok", llm)
        self.assertLessEqual(len(results), 5)

    def test_extract_invalid_json_returns_empty(self):
        llm = self._make_llm("this is not json")
        results = _llm_extract_memories("anything", llm)
        self.assertEqual(results, [])

    def test_extract_unknown_category_corrected(self):
        response = json.dumps([{
            "category": "INVALID_CAT",
            "abstract": "something",
            "overview": "",
            "content": "something useful",
            "importance": 0.5,
            "tags": [],
        }])
        llm = self._make_llm(response)
        results = _llm_extract_memories("anything", llm)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["category"], "cases")  # fallback

    def test_extract_truncates_long_content(self):
        long_content = "x" * 5000
        response = json.dumps([{
            "category": "events",
            "abstract": "long event",
            "overview": "",
            "content": long_content,
            "importance": 0.5,
            "tags": [],
        }])
        llm = self._make_llm(response)
        results = _llm_extract_memories("anything", llm)
        self.assertLessEqual(len(results[0]["content"]), 2001)


# ---------------------------------------------------------------------------
# 7. Dedup (mocked)
# ---------------------------------------------------------------------------

class TestDedup(unittest.TestCase):

    def _make_llm(self, decision: str, merged: str = "") -> Any:
        llm = MagicMock()
        llm.chat.return_value = json.dumps({"decision": decision, "merged_content": merged})
        return llm

    def test_dedup_skip(self):
        decision, merged = _llm_dedup("existing memory", "same thing again", self._make_llm("skip"))
        self.assertEqual(decision, "skip")
        self.assertEqual(merged, "")

    def test_dedup_merge(self):
        llm = self._make_llm("merge", "combined memory content")
        decision, merged = _llm_dedup("existing", "candidate", llm)
        self.assertEqual(decision, "merge")
        self.assertEqual(merged, "combined memory content")

    def test_dedup_create(self):
        decision, merged = _llm_dedup("unrelated A", "unrelated B", self._make_llm("create"))
        self.assertEqual(decision, "create")

    def test_dedup_supersede(self):
        decision, merged = _llm_dedup("old info", "updated info", self._make_llm("supersede", "updated info"))
        self.assertEqual(decision, "supersede")

    def test_dedup_invalid_json_returns_create(self):
        llm = MagicMock()
        llm.chat.return_value = "not json"
        decision, merged = _llm_dedup("a", "b", llm)
        self.assertEqual(decision, "create")


# ---------------------------------------------------------------------------
# 8. Provider lifecycle (mocked LanceDB + embeddings)
# ---------------------------------------------------------------------------

class TestProviderLifecycle(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_name(self):
        p = LanceDBMemoryProvider()
        self.assertEqual(p.name, "lancedb")

    def test_is_available_with_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            p = LanceDBMemoryProvider()
            self.assertTrue(p.is_available())

    def test_is_available_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            p = LanceDBMemoryProvider()
            self.assertFalse(p.is_available())

    def test_initialize_creates_table(self):
        import lancedb
        p = _make_provider(self.tmpdir)
        self.assertTrue(p._ready)
        db = lancedb.connect(self.tmpdir)
        self.assertIn("memories", db.table_names())

    def test_schema_has_v110_fields(self):
        import lancedb
        _make_provider(self.tmpdir)
        db = lancedb.connect(self.tmpdir)
        table = db.open_table("memories")
        col_names = {f.name for f in table.schema}
        for col in ("tier", "importance", "access_count", "category", "abstract", "overview"):
            self.assertIn(col, col_names, f"Missing column: {col}")

    def test_system_prompt_block_when_ready(self):
        p = _make_provider(self.tmpdir)
        block = p.system_prompt_block()
        self.assertIn("LanceDB Memory", block)

    def test_system_prompt_block_when_not_ready(self):
        p = LanceDBMemoryProvider()
        self.assertEqual(p.system_prompt_block(), "")

    def test_get_tool_schemas_returns_four_tools(self):
        p = LanceDBMemoryProvider()
        schemas = p.get_tool_schemas()
        names = [s["name"] for s in schemas]
        self.assertIn("lancedb_search", names)
        self.assertIn("lancedb_remember", names)
        self.assertIn("lancedb_forget", names)
        self.assertIn("lancedb_stats", names)

    def test_get_config_schema(self):
        p = LanceDBMemoryProvider()
        keys = [c["key"] for c in p.get_config_schema()]
        self.assertIn("storage_path", keys)
        self.assertIn("user_id", keys)
        self.assertIn("extraction_model", keys)


# ---------------------------------------------------------------------------
# 9. Write and search
# ---------------------------------------------------------------------------

class TestWriteAndSearch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.p = _make_provider(self.tmpdir)
        # Use deterministic embed for real vector search
        self.p._embedder.embed = _dummy_embed

    def _write(self, content: str, category: str = "cases", importance: float = 0.7, tier: str = "peripheral") -> None:
        self.p._write_entries([{
            "content": content,
            "source": "test",
            "session_id": "test-session-1",
            "user_id": "andrew",
            "timestamp": time.time(),
            "category": category,
            "importance": importance,
            "tier": tier,
            "abstract": content[:80],
            "overview": "",
        }])

    def test_write_single_entry(self):
        self._write("Theo focuses on NHS care market in the UK.")
        rows = self.p._table.search().limit(100).to_list()
        contents = [r["content"] for r in rows]
        self.assertIn("Theo focuses on NHS care market in the UK.", contents)

    def test_write_preserves_tier(self):
        self._write("Core strategic decision", tier="core")
        rows = self.p._table.search().where("tier = 'core'", prefilter=True).limit(10).to_list()
        self.assertTrue(any("Core strategic decision" in r["content"] for r in rows))

    def test_write_preserves_category(self):
        self._write("Andrew prefers concise emails.", category="preferences")
        rows = self.p._table.search().where("category = 'preferences'", prefilter=True).limit(10).to_list()
        self.assertTrue(any("concise emails" in r["content"] for r in rows))

    def test_hybrid_search_returns_results(self):
        self._write("SpecCon targets UK care homes and nursing agencies.")
        self._write("TAP Business international workforce platform for care operators.")
        time.sleep(0.1)
        hits = self.p._hybrid_search("UK care operators", top_k=5)
        self.assertGreater(len(hits), 0)

    def test_hybrid_search_returns_decay_weight(self):
        self._write("Recent important fact about TAP Recruitment.")
        hits = self.p._hybrid_search("TAP Recruitment", top_k=3)
        for h in hits:
            self.assertIn("decay_weight", h)
            self.assertGreater(h["decay_weight"], 0)

    def test_hybrid_search_core_tier_floor(self):
        # Write a very old core memory — should still surface due to floor
        old_ts = time.time() - (365 * 86400)  # 1 year ago
        self.p._write_entries([{
            "content": "Critical core fact from a year ago.",
            "source": "test",
            "session_id": "test-session-1",
            "user_id": "andrew",
            "timestamp": old_ts,
            "category": "profile",
            "importance": 1.0,
            "tier": "core",
            "abstract": "Critical core fact",
            "overview": "",
        }])
        hits = self.p._hybrid_search("Critical core fact", top_k=5)
        core_hits = [h for h in hits if h.get("tier") == "core"]
        self.assertGreater(len(core_hits), 0)
        # Decay weight should be at or above core floor (0.9)
        self.assertGreaterEqual(core_hits[0]["decay_weight"], 0.9)

    def test_empty_content_not_written(self):
        before = len(self.p._table.search().limit(1000).to_list())
        self.p._write_entries([{"content": "   ", "source": "test", "session_id": "s", "user_id": "andrew"}])
        after = len(self.p._table.search().limit(1000).to_list())
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# 10. Tool handlers
# ---------------------------------------------------------------------------

class TestToolHandlers(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.p = _make_provider(self.tmpdir)
        self.p._embedder.embed = _dummy_embed
        # Seed some data
        self.p._write_entries([{
            "content": "Theo runs TAP Business International.",
            "source": "test",
            "session_id": "s1",
            "user_id": "andrew",
            "timestamp": time.time(),
            "category": "profile",
            "importance": 0.9,
            "tier": "working",
            "abstract": "Theo runs TAP Business",
            "overview": "",
        }])

    def test_lancedb_stats(self):
        result = json.loads(self.p.handle_tool_call("lancedb_stats", {}))
        self.assertIn("total", result)
        self.assertIn("tiers", result)
        self.assertIn("categories", result)
        self.assertGreaterEqual(result["total"], 1)

    def test_lancedb_stats_tier_breakdown(self):
        result = json.loads(self.p.handle_tool_call("lancedb_stats", {}))
        self.assertIn("working", result["tiers"])

    def test_lancedb_remember(self):
        result = json.loads(self.p.handle_tool_call("lancedb_remember", {
            "content": "Theo's primary LinkedIn strategy is warm intro requests.",
            "category": "preferences",
            "importance": 0.8,
        }))
        self.assertTrue(result.get("stored"))

    def test_lancedb_remember_empty_content_errors(self):
        result = json.loads(self.p.handle_tool_call("lancedb_remember", {"content": ""}))
        self.assertIn("error", result)

    def test_lancedb_remember_sets_working_tier(self):
        self.p.handle_tool_call("lancedb_remember", {"content": "Explicit important fact."})
        rows = self.p._table.search().where("tier = 'working'", prefilter=True).limit(10).to_list()
        self.assertTrue(any("Explicit important fact" in r.get("content", "") for r in rows))

    def test_lancedb_search_returns_results(self):
        result = json.loads(self.p.handle_tool_call("lancedb_search", {"query": "TAP Business"}))
        self.assertIn("results", result)

    def test_lancedb_search_includes_tier_and_category(self):
        result = json.loads(self.p.handle_tool_call("lancedb_search", {"query": "Theo TAP Business"}))
        if result["results"]:
            r = result["results"][0]
            self.assertIn("tier", r)
            self.assertIn("category", r)

    def test_lancedb_search_category_filter(self):
        # Add a preferences entry
        self.p._write_entries([{
            "content": "Prefers bullet-point summaries over paragraphs.",
            "source": "test", "session_id": "s1", "user_id": "andrew",
            "timestamp": time.time(), "category": "preferences",
            "importance": 0.6, "tier": "peripheral", "abstract": "", "overview": "",
        }])
        result = json.loads(self.p.handle_tool_call("lancedb_search", {
            "query": "summary style preferences",
            "category": "preferences",
        }))
        for r in result.get("results", []):
            self.assertEqual(r["category"], "preferences")

    def test_lancedb_forget_by_query(self):
        before = json.loads(self.p.handle_tool_call("lancedb_stats", {}))["total"]
        result = json.loads(self.p.handle_tool_call("lancedb_forget", {"query": "TAP Business International"}))
        self.assertTrue(result.get("deleted"))
        after = json.loads(self.p.handle_tool_call("lancedb_stats", {}))["total"]
        self.assertLess(after, before)

    def test_lancedb_forget_no_args_errors(self):
        result = json.loads(self.p.handle_tool_call("lancedb_forget", {}))
        self.assertIn("error", result)

    def test_unknown_tool_errors(self):
        result = json.loads(self.p.handle_tool_call("nonexistent_tool", {}))
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# 11. sync_turn noise filtering
# ---------------------------------------------------------------------------

class TestSyncTurn(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.p = _make_provider(self.tmpdir)
        self.p._embedder.embed = _dummy_embed

    def test_sync_turn_noise_skips_extraction(self):
        """Noisy turns should not queue anything."""
        queue_before = len(self.p._extract_queue)
        self.p.sync_turn("hi", "Hello! How can I help?")
        # Give thread a moment
        time.sleep(0.05)
        # Queue should remain empty (both sides are noise)
        self.assertEqual(len(self.p._extract_queue), queue_before)

    def test_sync_turn_substantive_queues_extraction(self):
        """A real turn should queue extraction."""
        # Mock LLM to return a memory
        self.p._llm = MagicMock()
        self.p._llm.chat.return_value = json.dumps([{
            "category": "cases",
            "abstract": "Discussed UK market entry",
            "overview": "- UK NHS care operators\n- Care home staffing",
            "content": "Discussed strategy for entering UK care operator market via staffing contracts.",
            "importance": 0.7,
            "tags": ["uk", "care"],
        }])
        self.p.sync_turn(
            "We need to focus on UK care homes this quarter.",
            "Understood — I'll prioritise NHS-commissioned operators.",
        )
        # Wait for background thread
        if self.p._extract_thread:
            self.p._extract_thread.join(timeout=3.0)
        rows = self.p._table.search().limit(100).to_list()
        self.assertTrue(any("UK" in r.get("content", "") for r in rows))

    def test_sync_turn_not_ready_does_nothing(self):
        p = LanceDBMemoryProvider()  # not initialized
        p.sync_turn("some content", "some response")  # should not raise


# ---------------------------------------------------------------------------
# 12. on_session_end
# ---------------------------------------------------------------------------

class TestSessionEnd(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.p = _make_provider(self.tmpdir)
        self.p._embedder.embed = _dummy_embed

    def test_session_end_extracts_memories(self):
        self.p._llm = MagicMock()
        self.p._llm.chat.return_value = json.dumps([{
            "category": "events",
            "abstract": "Decided on UK-first GTM strategy",
            "overview": "- UK over SA for Q2\n- Focus on care operators",
            "content": "Session decision: UK-first go-to-market strategy adopted for Q2 2026.",
            "importance": 0.85,
            "tags": ["gtm", "uk"],
        }])
        messages = [
            {"role": "user", "content": "Let's go UK-first this quarter."},
            {"role": "assistant", "content": "Confirmed, UK-first GTM for Q2."},
        ]
        self.p.on_session_end(messages)
        # Wait for background thread
        time.sleep(0.5)
        rows = self.p._table.search().limit(100).to_list()
        self.assertTrue(any("UK" in r.get("content", "") for r in rows))

    def test_session_end_not_ready_does_nothing(self):
        p = LanceDBMemoryProvider()
        p.on_session_end([{"role": "user", "content": "test"}])  # should not raise


# ---------------------------------------------------------------------------
# 13. Schema migration
# ---------------------------------------------------------------------------

class TestSchemaMigration(unittest.TestCase):

    def test_migration_adds_missing_columns(self):
        """Simulate a v1.0.0 table and verify migration adds new columns."""
        import lancedb
        import pyarrow as pa

        tmpdir = tempfile.mkdtemp()

        # Create old-style table without v1.1.0 fields
        old_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1536)),
            pa.field("timestamp", pa.float64()),
            pa.field("source", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("tags", pa.string()),
        ])
        db = lancedb.connect(tmpdir)
        db.create_table("memories", schema=old_schema)

        # Now initialize provider — should migrate
        p = LanceDBMemoryProvider()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANCEDB_PATH": tmpdir}):
            with patch("hermes_memory_lancedb._EmbedClient") as ME, \
                 patch("hermes_memory_lancedb._LLMClient") as ML:
                ME.return_value.embed = lambda t: [0.1] * 1536
                ML.return_value.chat = lambda *a, **kw: "[]"
                p.initialize("migrated-session")

        self.assertTrue(p._ready)
        col_names = {f.name for f in p._table.schema}
        for col in ("tier", "importance", "access_count", "category", "abstract", "overview"):
            self.assertIn(col, col_names, f"Migration failed to add column: {col}")


# ---------------------------------------------------------------------------
# 14. Shutdown
# ---------------------------------------------------------------------------

class TestShutdown(unittest.TestCase):

    def test_shutdown_completes_cleanly(self):
        tmpdir = tempfile.mkdtemp()
        p = _make_provider(tmpdir)
        p.shutdown()  # should not raise or hang

    def test_shutdown_not_ready_does_nothing(self):
        p = LanceDBMemoryProvider()
        p.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# 15. Extraction prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilders(unittest.TestCase):

    def test_extraction_prompt_includes_conversation(self):
        text = "User: hi\nAssistant: hello"
        prompt = _build_extraction_prompt(text)
        self.assertIn("hi", prompt)

    def test_extraction_prompt_truncates_long_input(self):
        long_text = "x" * 10000
        prompt = _build_extraction_prompt(long_text)
        self.assertLessEqual(len(prompt), 7000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
