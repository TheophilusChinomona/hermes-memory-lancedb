"""Unit tests for P4: lifecycle, temporal, sessions, compactor, auto-capture, query.

All tests mock LLM/store/embedder calls so they run without LanceDB native code.
"""

from __future__ import annotations

import json
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from athena_memory import lifecycle, temporal, sessions, compactor, auto_capture, query


# ---------------------------------------------------------------------------
# 1. Lifecycle — DecayEngine
# ---------------------------------------------------------------------------

class TestDecayEngine(unittest.TestCase):
    def setUp(self):
        self.engine = lifecycle.DecayEngine()
        self.now = time.time()

    def _mem(self, **kw) -> lifecycle.DecayableMemory:
        defaults = dict(
            id="m1",
            importance=0.5,
            confidence=1.0,
            tier="working",
            access_count=0,
            created_at=self.now,
            last_accessed_at=0.0,
            temporal_type="static",
        )
        defaults.update(kw)
        return lifecycle.DecayableMemory(**defaults)

    def test_fresh_memory_has_high_recency(self):
        s = self.engine.score(self._mem(), now=self.now)
        self.assertGreater(s.recency, 0.95)

    def test_old_memory_recency_drops(self):
        old = self._mem(created_at=self.now - 365 * lifecycle.SECONDS_PER_DAY)
        s = self.engine.score(old, now=self.now)
        self.assertLess(s.recency, 0.5)

    def test_dynamic_decays_faster_than_static(self):
        age = 30 * lifecycle.SECONDS_PER_DAY
        static = self._mem(created_at=self.now - age, temporal_type="static")
        dynamic = self._mem(created_at=self.now - age, temporal_type="dynamic", id="m2")
        s_static = self.engine.score(static, now=self.now)
        s_dynamic = self.engine.score(dynamic, now=self.now)
        self.assertGreater(s_static.recency, s_dynamic.recency)

    def test_importance_modulation_increases_half_life(self):
        age = 60 * lifecycle.SECONDS_PER_DAY
        low = self._mem(importance=0.1, created_at=self.now - age)
        high = self._mem(importance=0.9, created_at=self.now - age, id="m2")
        self.assertGreater(self.engine.score(high).recency, self.engine.score(low).recency)

    def test_frequency_saturates(self):
        few = self._mem(access_count=1)
        many = self._mem(access_count=100, id="m2")
        s_few = self.engine.score(few)
        s_many = self.engine.score(many)
        self.assertLess(s_few.frequency, s_many.frequency)
        # Saturation: 100 accesses shouldn't be 100x score of 1
        self.assertLess(s_many.frequency, 1.01)

    def test_intrinsic_is_importance_times_confidence(self):
        m = self._mem(importance=0.7, confidence=0.5)
        self.assertAlmostEqual(self.engine.score(m).intrinsic, 0.35, places=4)

    def test_score_all_returns_one_per_memory(self):
        mems = [self._mem(id=f"m{i}") for i in range(5)]
        scores = self.engine.score_all(mems)
        self.assertEqual(len(scores), 5)
        self.assertEqual({s.memory_id for s in scores}, {m.id for m in mems})

    def test_apply_search_boost_multiplies_scores(self):
        m = self._mem(tier="core", importance=0.9, access_count=20,
                      created_at=self.now - 5 * lifecycle.SECONDS_PER_DAY)
        results = [{"memory": m, "score": 0.5}]
        self.engine.apply_search_boost(results, now=self.now)
        # Core tier with high composite → boost ~ near 1.0; multiplier > min
        self.assertGreater(results[0]["score"], 0.5 * 0.3)
        self.assertLessEqual(results[0]["score"], 0.5 * 1.0001)

    def test_apply_search_boost_accepts_dict_memory(self):
        results = [{"memory": {"id": "m1", "importance": 0.9, "tier": "core",
                                "access_count": 10, "created_at": self.now}, "score": 0.5}]
        self.engine.apply_search_boost(results, now=self.now)
        self.assertGreater(results[0]["score"], 0.0)

    def test_get_stale_memories_filters_below_threshold(self):
        mems = [
            self._mem(id="fresh", importance=0.9, access_count=10),
            self._mem(id="stale", importance=0.05, access_count=0,
                      created_at=self.now - 365 * lifecycle.SECONDS_PER_DAY,
                      tier="peripheral"),
        ]
        stale = self.engine.get_stale_memories(mems, now=self.now)
        self.assertEqual([s.memory_id for s in stale], ["stale"])


# ---------------------------------------------------------------------------
# 2. Lifecycle — TierManager
# ---------------------------------------------------------------------------

class TestTierManager(unittest.TestCase):
    def setUp(self):
        self.mgr = lifecycle.TierManager()
        self.now = time.time()

    def _mem(self, **kw) -> lifecycle.TierableMemory:
        defaults = dict(id="m1", tier="peripheral", importance=0.5,
                        access_count=0, created_at=self.now)
        defaults.update(kw)
        return lifecycle.TierableMemory(**defaults)

    def _score(self, composite: float) -> lifecycle.DecayScore:
        return lifecycle.DecayScore(memory_id="m1", recency=composite,
                                     frequency=composite, intrinsic=composite,
                                     composite=composite)

    def test_promote_peripheral_to_working(self):
        mem = self._mem(tier="peripheral", access_count=5)
        t = self.mgr.evaluate(mem, self._score(0.6), now=self.now)
        self.assertIsNotNone(t)
        self.assertEqual(t.to_tier, "working")

    def test_no_promote_low_access(self):
        mem = self._mem(tier="peripheral", access_count=1)
        self.assertIsNone(self.mgr.evaluate(mem, self._score(0.6), now=self.now))

    def test_promote_working_to_core(self):
        mem = self._mem(tier="working", access_count=15, importance=0.9)
        t = self.mgr.evaluate(mem, self._score(0.8), now=self.now)
        self.assertIsNotNone(t)
        self.assertEqual(t.to_tier, "core")

    def test_demote_working_to_peripheral_low_composite(self):
        mem = self._mem(tier="working", access_count=2)
        t = self.mgr.evaluate(mem, self._score(0.05), now=self.now)
        self.assertIsNotNone(t)
        self.assertEqual(t.to_tier, "peripheral")

    def test_demote_working_to_peripheral_aged(self):
        old = self.now - 70 * lifecycle.SECONDS_PER_DAY
        mem = self._mem(tier="working", access_count=1, created_at=old)
        t = self.mgr.evaluate(mem, self._score(0.5), now=self.now)
        self.assertIsNotNone(t)
        self.assertEqual(t.to_tier, "peripheral")

    def test_demote_core_to_working(self):
        mem = self._mem(tier="core", access_count=1)
        t = self.mgr.evaluate(mem, self._score(0.05), now=self.now)
        self.assertIsNotNone(t)
        self.assertEqual(t.to_tier, "working")

    def test_evaluate_all_filters_missing_scores(self):
        mems = [self._mem(id="m1"), self._mem(id="m2")]
        scores = [self._score(0.6)]  # only score for m1
        scores[0].memory_id = "m1"
        out = self.mgr.evaluate_all(mems, scores, now=self.now)
        # m1 may or may not transition based on access; m2 is skipped
        self.assertLessEqual(len(out), 1)


class TestTierEvaluateLegacy(unittest.TestCase):
    def test_legacy_promote_peripheral(self):
        # access >= 3, composite (decay*importance) >= 0.4 → working
        out = lifecycle.tier_evaluate_legacy("peripheral", 5, 0.8, 0.6, age_days=10)
        self.assertEqual(out, "working")

    def test_legacy_no_promote_below_threshold(self):
        out = lifecycle.tier_evaluate_legacy("peripheral", 5, 0.3, 0.3, age_days=10)
        self.assertIsNone(out)

    def test_legacy_demote_working_aged(self):
        out = lifecycle.tier_evaluate_legacy("working", 0, 0.5, 0.5, age_days=70)
        self.assertEqual(out, "peripheral")


# ---------------------------------------------------------------------------
# 3. Temporal classifier
# ---------------------------------------------------------------------------

class TestTemporalClassifier(unittest.TestCase):
    def test_dynamic_keyword_today(self):
        self.assertEqual(temporal.classify_temporal("I deployed the service today."), "dynamic")

    def test_dynamic_keyword_tomorrow(self):
        self.assertEqual(temporal.classify_temporal("Meeting tomorrow at 3pm"), "dynamic")

    def test_dynamic_chinese_keyword(self):
        self.assertEqual(temporal.classify_temporal("今天和Theo开会"), "dynamic")

    def test_static_keyword_favorite(self):
        self.assertEqual(temporal.classify_temporal("My favorite editor is vim"), "static")

    def test_static_keyword_allergic(self):
        self.assertEqual(temporal.classify_temporal("Allergic to peanuts"), "static")

    def test_default_static(self):
        self.assertEqual(temporal.classify_temporal("Random unrelated content"), "static")

    def test_empty_returns_static(self):
        self.assertEqual(temporal.classify_temporal(""), "static")

    def test_dynamic_wins_on_both_match(self):
        # Has both 'favorite' (static) and 'today' (dynamic)
        self.assertEqual(
            temporal.classify_temporal("My favorite restaurant is open today"),
            "dynamic",
        )

    def test_substring_false_positive_avoided(self):
        # "later" is dynamic, but "collateral" should NOT match (word boundary)
        self.assertEqual(temporal.classify_temporal("This is collateral damage"), "static")

    def test_llm_fallback_on_failure(self):
        bad_llm = MagicMock()
        bad_llm.chat.side_effect = RuntimeError("API down")
        # Rule says static; LLM raises — should still return static
        result = temporal.classify_temporal("Plain text with no signal", llm=bad_llm, prefer_llm=True)
        self.assertEqual(result, "static")

    def test_llm_overrides_when_prefer_llm(self):
        good_llm = MagicMock()
        good_llm.chat.return_value = '{"type": "dynamic"}'
        result = temporal.classify_temporal("Some plain content here", llm=good_llm, prefer_llm=True)
        self.assertEqual(result, "dynamic")

    def test_infer_expiry_tomorrow(self):
        now = 1_000_000_000.0
        out = temporal.infer_expiry("Meeting tomorrow", now=now)
        self.assertAlmostEqual(out, now + 24 * 3600, places=1)

    def test_infer_expiry_returns_none_when_no_match(self):
        self.assertIsNone(temporal.infer_expiry("Plain static fact"))


# ---------------------------------------------------------------------------
# 4. Sessions — compress / recover
# ---------------------------------------------------------------------------

class TestSessionCompression(unittest.TestCase):
    def test_score_text_acknowledgment(self):
        s = sessions.score_text("ok", 0)
        self.assertEqual(s.reason, "acknowledgment")
        self.assertLess(s.score, 0.2)

    def test_score_text_decision(self):
        s = sessions.score_text("Let's go with option B from now on", 0)
        self.assertEqual(s.reason, "decision")
        self.assertGreater(s.score, 0.8)

    def test_score_text_correction(self):
        s = sessions.score_text("No, actually that's wrong", 0)
        self.assertEqual(s.reason, "correction")
        self.assertGreater(s.score, 0.9)

    def test_score_text_substantive(self):
        s = sessions.score_text("x" * 200, 0)
        self.assertEqual(s.reason, "substantive")

    def test_compress_texts_within_budget(self):
        texts = ["short message"] * 3
        out = sessions.compress_texts(texts, max_chars=10000)
        self.assertEqual(out.dropped, 0)
        self.assertEqual(out.texts, texts)

    def test_compress_texts_drops_over_budget(self):
        texts = [f"text {i}" for i in range(20)]
        out = sessions.compress_texts(texts, max_chars=20)
        self.assertGreater(out.dropped, 0)
        # First and last always kept
        self.assertIn(texts[0], out.texts)
        self.assertIn(texts[-1], out.texts)

    def test_compress_session_no_messages_returns_none(self):
        self.assertIsNone(sessions.compress_session([]))

    def test_compress_session_heuristic_summary(self):
        msgs = [
            {"role": "user", "content": "Let's deploy v2 of the service today"},
            {"role": "assistant", "content": "Confirmed, deploying now."},
        ]
        out = sessions.compress_session(msgs, llm=None, session_id="sess-1")
        self.assertIsNotNone(out)
        self.assertEqual(out["source"], "session_compressed")
        self.assertEqual(out["session_id"], "sess-1")
        self.assertTrue(out["abstract"])

    def test_compress_session_with_llm(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({
            "abstract": "Deployed v2",
            "overview": "- Decision: ship v2\n- No blockers",
            "content": "Full narrative here",
            "importance": 0.85,
            "tags": ["deploy", "v2"],
        })
        msgs = [{"role": "user", "content": "Deploy v2 service today please"}]
        out = sessions.compress_session(msgs, llm=llm, session_id="s")
        self.assertEqual(out["abstract"], "Deployed v2")
        self.assertAlmostEqual(out["importance"], 0.85, places=2)
        self.assertIn("session_summary", out["tags"])

    def test_compress_session_llm_failure_falls_back(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API timeout")
        msgs = [{"role": "user", "content": "Some content here that is meaningful"}]
        out = sessions.compress_session(msgs, llm=llm, session_id="s")
        self.assertIsNotNone(out)  # heuristic fallback
        self.assertTrue(out["abstract"])

    def test_recover_session_no_id_returns_empty(self):
        store = MagicMock()
        self.assertEqual(sessions.recover_session("", store), [])

    def test_recover_session_uses_finder_method(self):
        store = MagicMock()
        store.find_session_summaries.return_value = [{"id": "x", "abstract": "summary"}]
        out = sessions.recover_session("sess-1", store)
        self.assertEqual(len(out), 1)
        store.find_session_summaries.assert_called_once()

    def test_format_recovered_renders_block(self):
        entries = [{"abstract": "Deployed v2", "overview": "- Decision\n- Done"}]
        block = sessions.format_recovered(entries)
        self.assertIn("Recovered Session Context", block)
        self.assertIn("Deployed v2", block)


# ---------------------------------------------------------------------------
# 5. Memory compactor
# ---------------------------------------------------------------------------

class TestCompactor(unittest.TestCase):
    def test_cosine_similarity_identical(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(compactor.cosine_similarity(v, v), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        self.assertAlmostEqual(compactor.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_similarity_handles_zero_vector(self):
        self.assertEqual(compactor.cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_compact_disabled_short_circuits(self):
        store = MagicMock()
        cfg = compactor.CompactionConfig(enabled=False)
        result = compactor.compact_memories(store, config=cfg)
        self.assertEqual(result.scanned, 0)

    def test_compact_below_min_cluster_size(self):
        store = MagicMock()
        # Return 1 entry but min_cluster_size=2 — no clusters possible
        store.fetch_for_compaction.return_value = [
            {"id": "a", "content": "x", "vector": [1.0, 0.0], "importance": 0.5},
        ]
        cfg = compactor.CompactionConfig(min_cluster_size=2, min_age_days=0)
        result = compactor.compact_memories(store, config=cfg)
        self.assertEqual(result.clusters_found, 0)

    def test_compact_clusters_near_duplicates(self):
        store = MagicMock()
        store.fetch_for_compaction.return_value = [
            {"id": "a", "content": "Theo prefers vim", "vector": [1.0, 0.0, 0.0], "importance": 0.7, "category": "preferences", "tier": "working"},
            {"id": "b", "content": "Theo likes vim",   "vector": [0.99, 0.01, 0.0], "importance": 0.5, "category": "preferences", "tier": "peripheral"},
            {"id": "c", "content": "Unrelated",        "vector": [0.0, 1.0, 0.0], "importance": 0.6, "category": "events", "tier": "working"},
        ]
        # Track delete + write calls
        deleted = []
        written = []
        store.delete_memory.side_effect = lambda mid: deleted.append(mid) or True
        store._write_entries = MagicMock(side_effect=lambda entries: written.extend(entries))

        cfg = compactor.CompactionConfig(min_age_days=0, similarity_threshold=0.95, min_cluster_size=2)
        result = compactor.compact_memories(store, config=cfg)

        self.assertGreaterEqual(result.clusters_found, 1)
        self.assertEqual(result.memories_created, 1)
        self.assertEqual(set(deleted), {"a", "b"})
        self.assertEqual(written[0]["category"], "preferences")
        # Importance is max across cluster
        self.assertAlmostEqual(written[0]["importance"], 0.7, places=4)
        # Tier is highest present (working > peripheral)
        self.assertEqual(written[0]["tier"], "working")

    def test_compact_dry_run_makes_no_changes(self):
        store = MagicMock()
        store.fetch_for_compaction.return_value = [
            {"id": "a", "content": "x", "vector": [1.0, 0.0], "importance": 0.5},
            {"id": "b", "content": "y", "vector": [0.99, 0.01], "importance": 0.5},
        ]
        store._write_entries = MagicMock()
        cfg = compactor.CompactionConfig(min_age_days=0, similarity_threshold=0.9, dry_run=True)
        result = compactor.compact_memories(store, config=cfg)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.memories_created, 0)
        store._write_entries.assert_not_called()

    def test_compaction_trigger_fires_at_threshold(self):
        trig = compactor.CompactionTrigger(every_n=3)
        store = MagicMock()
        store.fetch_for_compaction.return_value = []
        # First two bumps shouldn't trigger
        self.assertFalse(trig.bump(store, cooldown_seconds=0))
        self.assertFalse(trig.bump(store, cooldown_seconds=0))
        # Third does
        fired = trig.bump(store, cooldown_seconds=0)
        self.assertTrue(fired)
        # Wait for background thread to settle
        if trig._thread:
            trig._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# 6. Auto-capture cleanup
# ---------------------------------------------------------------------------

class TestAutoCaptureCleanup(unittest.TestCase):
    def test_strip_addressing_prefix(self):
        out = auto_capture.strip_auto_capture_prefix("user", "@theo hello world")
        self.assertEqual(out, "hello world")

    def test_strip_relevant_memories_block(self):
        text = "<relevant-memories>\nstuff\n</relevant-memories>\nactual content"
        out = auto_capture.strip_auto_capture_prefix("user", text)
        self.assertEqual(out, "actual content")

    def test_strip_session_reset_prefix(self):
        text = (
            "A new session was started via /new or /reset. Execute your Session Startup sequence now\n\n"
            "What's up?"
        )
        out = auto_capture.strip_auto_capture_prefix("user", text)
        self.assertEqual(out, "What's up?")

    def test_strip_inbound_metadata_block(self):
        text = (
            "Sender (untrusted metadata):\n```json\n{\"who\": \"x\"}\n```\n\nReal message"
        )
        out = auto_capture.strip_auto_capture_prefix("user", text)
        self.assertIn("Real message", out)
        self.assertNotIn("untrusted metadata", out)

    def test_is_pure_metadata_true_for_only_metadata(self):
        text = "<relevant-memories>\nstuff\n</relevant-memories>\n"
        self.assertTrue(auto_capture.is_pure_metadata(text))

    def test_is_pure_metadata_false_for_real_content(self):
        self.assertFalse(auto_capture.is_pure_metadata("This is a real message about deployment"))

    def test_assistant_role_passthrough(self):
        out = auto_capture.strip_auto_capture_prefix("assistant", "@theo hi")
        # Non-user role is not stripped
        self.assertEqual(out, "@theo hi")

    def test_cleanup_no_session_id_noop(self):
        store = MagicMock()
        report = auto_capture.cleanup_auto_captures("", store)
        self.assertEqual(report.scanned, 0)

    def test_cleanup_deletes_pure_metadata(self):
        # Build a fake store with a _table
        store = MagicMock()
        store._table = MagicMock()
        store._table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
            {
                "id": "junk",
                "session_id": "s1",
                "user_id": "andrew",
                "source": "session_end",
                "content": "<relevant-memories>\nstuff\n</relevant-memories>",
                "tier": "peripheral",
                "importance": 0.1,
                "abstract": "",
                "access_count": 0,
            },
            {
                "id": "good",
                "session_id": "s1",
                "user_id": "andrew",
                "source": "session_end",
                "content": "Theo decided to use vim as the default editor for the team going forward",
                "tier": "working",
                "importance": 0.7,
                "abstract": "vim default",
                "access_count": 5,
            },
        ]
        report = auto_capture.cleanup_auto_captures("s1", store)
        self.assertEqual(report.scanned, 2)
        self.assertEqual(report.deleted, 1)
        # delete called with pure-metadata id
        deletes = [c.args[0] for c in store._table.delete.call_args_list]
        self.assertTrue(any("junk" in d for d in deletes))


# ---------------------------------------------------------------------------
# 7. Query — intent + expansion
# ---------------------------------------------------------------------------

class TestIntentAnalyzer(unittest.TestCase):
    def test_preference_intent(self):
        sig = query.analyze_intent("What's my style for commit messages?")
        self.assertEqual(sig.label, "preference")
        self.assertEqual(sig.confidence, "high")
        self.assertIn("preferences", sig.categories)

    def test_decision_intent(self):
        sig = query.analyze_intent("Why did we choose Postgres?")
        self.assertEqual(sig.label, "decision")
        self.assertEqual(sig.confidence, "high")

    def test_entity_intent(self):
        sig = query.analyze_intent("Tell me about the Acme account")
        self.assertEqual(sig.label, "entity")

    def test_event_intent(self):
        sig = query.analyze_intent("When did we deploy v2?")
        self.assertEqual(sig.label, "event")
        self.assertEqual(sig.depth, "full")

    def test_fact_intent(self):
        sig = query.analyze_intent("How does the auth setup work?")
        self.assertEqual(sig.label, "fact")

    def test_recall_intent(self):
        sig = query.analyze_intent("Do you remember what I said about pricing?")
        self.assertEqual(sig.label, "recall")

    def test_contradiction_intent(self):
        sig = query.analyze_intent("Is there a contradiction here?")
        self.assertEqual(sig.label, "contradiction")

    def test_empty_query(self):
        self.assertEqual(query.analyze_intent("").label, "empty")

    def test_broad_fallback(self):
        sig = query.analyze_intent("xyz random words abc")
        self.assertEqual(sig.label, "broad")
        self.assertEqual(sig.confidence, "low")
        self.assertEqual(sig.categories, [])

    def test_llm_fallback_on_broad(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps(
            {"label": "fact", "depth": "l1", "confidence": "high"}
        )
        sig = query.analyze_intent("xyz random words abc", llm=llm)
        self.assertEqual(sig.label, "fact")

    def test_llm_failure_returns_broad(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API down")
        sig = query.analyze_intent("xyz random words abc", llm=llm)
        self.assertEqual(sig.label, "broad")


class TestQueryExpander(unittest.TestCase):
    def test_expand_includes_original(self):
        out = query.expand_query("docker container restart")
        self.assertEqual(out[0], "docker container restart")

    def test_expand_with_synonyms(self):
        out = query.expand_query("docker container")
        # The container/docker key triggers expansions
        self.assertGreater(len(out), 1)
        self.assertTrue(any("compose" in s.lower() for s in out))

    def test_expand_empty_returns_empty(self):
        self.assertEqual(query.expand_query(""), [])

    def test_expand_too_short_returns_just_original(self):
        out = query.expand_query("a")
        self.assertEqual(out, ["a"])

    def test_expand_no_synonym_match_returns_just_original(self):
        out = query.expand_query("the quick brown fox")
        self.assertEqual(out, ["the quick brown fox"])

    def test_expand_with_llm(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({"expansions": ["alpha", "beta", "gamma"]})
        out = query.expand_query("xenon yttrium", llm=llm)
        # Should include LLM expansions
        self.assertGreater(len(out), 1)

    def test_expand_llm_failure_falls_back_to_rules(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("nope")
        out = query.expand_query("xenon yttrium", llm=llm)
        # No rule match, no LLM result → just original
        self.assertEqual(out, ["xenon yttrium"])


class TestApplyCategoryBoost(unittest.TestCase):
    def test_no_boost_when_low_confidence(self):
        sig = query.IntentSignal(label="broad", categories=[], confidence="low")
        hits = [{"category": "events", "score": 0.5}]
        out = query.apply_category_boost(hits, sig)
        self.assertEqual(out[0]["score"], 0.5)

    def test_boost_applied_to_matching_category(self):
        sig = query.IntentSignal(label="event", categories=["events", "cases"], confidence="high")
        hits = [
            {"category": "events", "score": 0.5},
            {"category": "profile", "score": 0.5},
        ]
        out = query.apply_category_boost(hits, sig, boost_factor=1.2)
        events_hit = next(h for h in out if h["category"] == "events")
        profile_hit = next(h for h in out if h["category"] == "profile")
        self.assertGreater(events_hit["score"], profile_hit["score"])
        self.assertAlmostEqual(events_hit["score"], 0.6, places=4)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
