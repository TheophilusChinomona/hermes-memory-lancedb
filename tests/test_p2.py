"""Tests for hermes-memory-lancedb v1.4.0 — P2 write-pipeline upgrades.

Covers:
  - chunker.py             (boundaries, overlap, cjk, edge cases)
  - dedup.py               (cosine batch dedup, LLM batch dispatch)
  - admission.py           (gate logic, persistence, hard reject)
  - smart_metadata.py      (parse, stringify, fact_key, lifecycle)
  - noise_proto.py         (init, cosine match, learn, cache)
  - __init__.py wiring     (schema column presence, helper exposure)

LanceDB native runtime is NOT touched — the worktree CPU may not have AVX, so
those code paths are exercised via mocks.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from typing import List
from unittest.mock import MagicMock

from hermes_memory_lancedb import (
    AdmissionController,
    NoisePrototypeFilter,
    batch_dedup,
    chunk_text,
    extract_smart_metadata,
    stringify_smart_metadata,
)
from hermes_memory_lancedb.admission import AdmissionDecision, AdmissionStats
from hermes_memory_lancedb.chunker import (
    DEFAULT_CHUNKER_CONFIG,
    ChunkerConfig,
    chunk_document,
    smart_chunk,
)
from hermes_memory_lancedb.dedup import (
    cosine_batch_dedup,
    DEFAULT_BATCH_THRESHOLD,
    VALID_DECISIONS,
)
from hermes_memory_lancedb.noise_proto import (
    BUILTIN_NOISE_TEXTS,
    DEFAULT_THRESHOLD,
)
from hermes_memory_lancedb.smart_metadata import (
    derive_fact_key,
    is_memory_active_at,
    is_memory_expired,
    parse_smart_metadata,
)


# ---------------------------------------------------------------------------
# Vector helpers (deterministic; no embedder needed)
# ---------------------------------------------------------------------------

def _v(*values: float) -> List[float]:
    return list(values)


def _unit(angle_deg: float, dim: int = 1536) -> List[float]:
    """Build a unit vector at a particular angle in the xy plane."""
    a = math.radians(angle_deg)
    out = [0.0] * dim
    out[0] = math.cos(a)
    out[1] = math.sin(a)
    return out


def _seeded_embed(seed: int, dim: int = 1536) -> List[float]:
    """Cheap deterministic embedding — seed maps to a single 1.0 slot."""
    out = [0.0] * dim
    out[seed % dim] = 1.0
    return out


# ---------------------------------------------------------------------------
# 1. Chunker
# ---------------------------------------------------------------------------

class TestChunker(unittest.TestCase):

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("Hello world.", max_chars=4000, overlap=200)
        self.assertEqual(chunks, ["Hello world."])

    def test_empty_text_returns_empty(self):
        self.assertEqual(chunk_text("", 4000, 200), [])
        self.assertEqual(chunk_text("   \n\t  ", 4000, 200), [])

    def test_long_text_splits_into_multiple_chunks(self):
        # 12000 chars of repeating sentences
        text = ("This is sentence number {n}. ".format(n=i) for i in range(800))
        big = "".join(text)
        chunks = chunk_text(big, max_chars=2000, overlap=200)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c), 2200)  # max + a little slack

    def test_chunker_respects_sentence_boundaries(self):
        # Build text with very obvious sentence boundaries.
        sent = "The quick brown fox jumps over the lazy dog. "
        text = sent * 100  # ~4500 chars
        result = chunk_document(
            text,
            ChunkerConfig(
                max_chunk_size=600, overlap_size=50, min_chunk_size=200,
                semantic_split=True, max_lines_per_chunk=50,
            ),
        )
        self.assertGreater(result.chunk_count, 1)
        # Most chunks should end with a period (sentence boundary)
        endings = sum(1 for c in result.chunks if c.rstrip().endswith("."))
        self.assertGreaterEqual(endings, result.chunk_count - 1)

    def test_chunker_overlap_is_present(self):
        # Adjacent chunks should share their overlap window.
        sent = "Sentence ABC DEF GHI XYZ. "
        text = sent * 200
        result = chunk_document(
            text,
            ChunkerConfig(
                max_chunk_size=500, overlap_size=100, min_chunk_size=200,
                semantic_split=True, max_lines_per_chunk=50,
            ),
        )
        self.assertGreater(result.chunk_count, 2)
        # End of chunk N should appear at the start of chunk N+1 (within overlap)
        for i in range(result.chunk_count - 1):
            tail = result.chunks[i][-50:]
            head = result.chunks[i + 1][:200]
            self.assertTrue(any(tok in head for tok in tail.split() if len(tok) > 3),
                            f"No overlap evident between chunk {i} and {i+1}")

    def test_chunker_handles_no_punctuation(self):
        # Single long word — should still split at max boundary.
        text = "a" * 5000
        chunks = chunk_text(text, max_chars=1000, overlap=100)
        self.assertGreater(len(chunks), 2)

    def test_smart_chunk_uses_known_model_limit(self):
        text = "x. " * 5000
        result = smart_chunk(text, "all-MiniLM-L6-v2")
        # 512 token limit → max chunk ~ 358 chars
        for c in result.chunks:
            self.assertLessEqual(len(c), 600)

    def test_smart_chunk_unknown_model_uses_8192(self):
        text = "Word. " * 1000
        result = smart_chunk(text, "totally-fake-model-name")
        self.assertGreater(result.chunk_count, 0)

    def test_chunker_metadata_indices_match_chunks(self):
        text = ("Sentence number {n}. ".format(n=i) for i in range(300))
        big = "".join(text)
        result = chunk_document(big, ChunkerConfig(
            max_chunk_size=400, overlap_size=80, min_chunk_size=200,
            semantic_split=True, max_lines_per_chunk=50,
        ))
        for chunk, meta in zip(result.chunks, result.metadatas):
            self.assertEqual(meta.length, len(chunk))


# ---------------------------------------------------------------------------
# 2. Cosine batch dedup
# ---------------------------------------------------------------------------

class TestCosineBatchDedup(unittest.TestCase):

    def test_empty_returns_empty(self):
        r = cosine_batch_dedup([], [], threshold=0.85)
        self.assertEqual(r.surviving_indices, [])
        self.assertEqual(r.input_count, 0)

    def test_single_item_survives(self):
        r = cosine_batch_dedup(["solo"], [_v(1.0, 0.0)], threshold=0.85)
        self.assertEqual(r.surviving_indices, [0])
        self.assertEqual(r.duplicate_indices, [])

    def test_identical_vectors_dedup(self):
        v = _v(1.0, 0.0, 0.0)
        r = cosine_batch_dedup(["a", "b", "c"], [v, v, v], threshold=0.85)
        self.assertEqual(r.surviving_indices, [0])
        self.assertEqual(set(r.duplicate_indices), {1, 2})
        self.assertEqual(r.duplicate_of[1], 0)

    def test_orthogonal_vectors_all_survive(self):
        r = cosine_batch_dedup(
            ["a", "b", "c"],
            [_v(1, 0, 0), _v(0, 1, 0), _v(0, 0, 1)],
            threshold=0.85,
        )
        self.assertEqual(r.surviving_indices, [0, 1, 2])

    def test_threshold_respected(self):
        # 30deg apart -> cos ~ 0.866. With threshold 0.9, should survive.
        a = _unit(0, dim=4)[:4]
        b = _unit(30, dim=4)[:4]
        # use the planar version directly
        a = [math.cos(0), math.sin(0), 0, 0]
        b = [math.cos(math.radians(30)), math.sin(math.radians(30)), 0, 0]
        r_strict = cosine_batch_dedup(["a", "b"], [a, b], threshold=0.9)
        self.assertEqual(r_strict.surviving_indices, [0, 1])
        r_loose = cosine_batch_dedup(["a", "b"], [a, b], threshold=0.5)
        self.assertEqual(r_loose.surviving_indices, [0])


# ---------------------------------------------------------------------------
# 3. LLM batch dedup wrapper
# ---------------------------------------------------------------------------

class TestBatchDedup(unittest.TestCase):

    def _llm_returning(self, payload):
        m = MagicMock()
        m.chat.return_value = json.dumps(payload)
        return m

    def test_no_pool_defaults_to_create(self):
        cands = [{"abstract": "a", "content": "alpha"}, {"abstract": "b", "content": "beta"}]
        out = batch_dedup(cands, [], llm=None)
        self.assertEqual([d["decision"] for d in out], ["create", "create"])
        self.assertEqual([d["index"] for d in out], [0, 1])

    def test_llm_decisions_parsed(self):
        cands = [
            {"abstract": "a", "content": "alpha"},
            {"abstract": "b", "content": "beta"},
            {"abstract": "c", "content": "gamma"},
        ]
        pool = [{"id": "x", "content": "old", "abstract": "old"}]
        llm = self._llm_returning([
            {"index": 0, "decision": "skip", "merged_content": "", "matched_existing_id": "x"},
            {"index": 1, "decision": "merge", "merged_content": "ALPHA+OLD",
             "matched_existing_id": "x"},
            {"index": 2, "decision": "create", "merged_content": "", "matched_existing_id": ""},
        ])
        out = batch_dedup(cands, pool, llm=llm)
        decisions = {d["index"]: d["decision"] for d in out}
        self.assertEqual(decisions[0], "skip")
        self.assertEqual(decisions[1], "merge")
        self.assertEqual(decisions[2], "create")
        merged = next(d for d in out if d["index"] == 1)
        self.assertEqual(merged["merged_content"], "ALPHA+OLD")

    def test_llm_invalid_json_falls_back_to_create(self):
        cands = [{"abstract": "a", "content": "alpha"}]
        pool = [{"id": "x", "content": "old", "abstract": "old"}]
        llm = MagicMock()
        llm.chat.return_value = "not valid json {"
        out = batch_dedup(cands, pool, llm=llm)
        self.assertEqual(out[0]["decision"], "create")

    def test_unknown_decision_normalized_to_create(self):
        cands = [{"abstract": "a", "content": "alpha"}]
        pool = [{"id": "x", "content": "old", "abstract": "old"}]
        llm = self._llm_returning([
            {"index": 0, "decision": "WHATEVER", "merged_content": "", "matched_existing_id": ""},
        ])
        out = batch_dedup(cands, pool, llm=llm)
        self.assertEqual(out[0]["decision"], "create")

    def test_cosine_dedup_within_batch(self):
        # Two near-identical candidates — embedder returns the same vector,
        # so the second is marked skip without any LLM call.
        cands = [
            {"abstract": "Theo founded SpecCon", "content": "Theo founded SpecCon."},
            {"abstract": "Theo founded SpecCon", "content": "Theo founded SpecCon."},
        ]
        embedder = lambda t: _v(1.0, 0.0, 0.0)
        out = batch_dedup(cands, [], llm=None, embedder=embedder)
        decisions = {d["index"]: d["decision"] for d in out}
        # Only one will be skip (the duplicate); the other becomes create.
        self.assertEqual(sum(1 for v in decisions.values() if v == "skip"), 1)

    def test_valid_decisions_set_complete(self):
        for d in ("skip", "support", "merge", "contextualize", "contradict",
                  "supersede", "create"):
            self.assertIn(d, VALID_DECISIONS)


# ---------------------------------------------------------------------------
# 4. Admission control
# ---------------------------------------------------------------------------

class TestAdmissionControl(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_disabled_always_admits(self):
        ctrl = AdmissionController(self.tmpdir, enabled=False)
        d = ctrl.evaluate("any text", _seeded_embed(1), category="cases")
        self.assertTrue(d.admit)
        self.assertIn("disabled", d.reason)

    def test_enabled_admits_first_write(self):
        ctrl = AdmissionController(self.tmpdir, enabled=True)
        d = ctrl.evaluate("Theo founded SpecCon Holdings", _seeded_embed(7), category="profile")
        self.assertTrue(d.admit, f"got reason: {d.reason}")
        self.assertGreater(d.score, 0.0)

    def test_rate_limit_kicks_in(self):
        ctrl = AdmissionController(
            self.tmpdir, enabled=True,
            rate_budget=2, rate_window_s=60,
            reject_threshold=0.5, admit_threshold=0.5,
        )
        # Push two admits then expect the third to score lower.
        for i in range(2):
            ctrl.evaluate(f"Fact {i}", _seeded_embed(100 + i), category="profile")
        third = ctrl.evaluate("Yet another fact", _seeded_embed(200), category="profile")
        # rate term is 0 once budget is hit
        self.assertEqual(third.feature_scores["rate"], 0.0)

    def test_hard_reject_against_recent_reject(self):
        ctrl = AdmissionController(
            self.tmpdir, enabled=True,
            reject_threshold=0.99,  # force first to be rejected
            admit_threshold=0.99,
            hard_reject_cosine=0.9,
        )
        bad_vec = _seeded_embed(42)
        rejected = ctrl.evaluate("Trash text", bad_vec, category="events")
        self.assertFalse(rejected.admit)
        # Now another candidate with the same vector should hard-reject.
        again = ctrl.evaluate("Different surface form", bad_vec, category="profile")
        self.assertFalse(again.admit)
        self.assertIn("hard reject", again.reason)

    def test_persists_stats_to_disk(self):
        ctrl = AdmissionController(self.tmpdir, enabled=True)
        ctrl.evaluate("hello world", _seeded_embed(11), category="profile")
        path = os.path.join(self.tmpdir, "admission_stats.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("admitted_count", data)

    def test_reload_reads_persisted_stats(self):
        ctrl1 = AdmissionController(self.tmpdir, enabled=True)
        ctrl1.evaluate("hello", _seeded_embed(12), category="profile")
        admitted = ctrl1.stats.admitted_count
        # Fresh controller should pick up the same count
        ctrl2 = AdmissionController(self.tmpdir, enabled=True)
        self.assertEqual(ctrl2.stats.admitted_count, admitted)

    def test_summary_shape(self):
        ctrl = AdmissionController(self.tmpdir, enabled=True)
        ctrl.evaluate("seed", _seeded_embed(13), category="profile")
        s = ctrl.summary()
        for key in ("enabled", "admitted", "rejected", "total", "reject_rate", "stats_path"):
            self.assertIn(key, s)

    def test_reset_clears_stats(self):
        ctrl = AdmissionController(self.tmpdir, enabled=True)
        ctrl.evaluate("seed", _seeded_embed(14), category="profile")
        self.assertGreater(ctrl.stats.admitted_count, 0)
        ctrl.reset()
        self.assertEqual(ctrl.stats.admitted_count, 0)
        self.assertEqual(ctrl.stats.rejected_count, 0)

    def test_type_prior_affects_score(self):
        ctrl = AdmissionController(self.tmpdir, enabled=True)
        d_profile = ctrl.evaluate("F", _seeded_embed(20), category="profile", now=time.time())
        # Reset so prior reject doesn't pollute the next call
        ctrl.reset()
        d_events = ctrl.evaluate("F", _seeded_embed(21), category="events", now=time.time())
        # Events have lower type prior -> lower type_prior contribution
        self.assertGreater(d_profile.feature_scores["type_prior"], d_events.feature_scores["type_prior"])


# ---------------------------------------------------------------------------
# 5. Smart metadata
# ---------------------------------------------------------------------------

class TestSmartMetadata(unittest.TestCase):

    def test_extract_smart_metadata_with_llm(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({
            "memory_temporal_type": "static",
            "confidence": 0.95,
            "sensitivity": "private",
            "modality": "text",
            "fact_key": "founder_speccon",
            "tags": ["founder", "speccon", "uk"],
        })
        meta = extract_smart_metadata(
            "Theo Chinomona founded SpecCon Holdings.",
            llm,
            abstract="Theo founded SpecCon",
            category="profile",
        )
        self.assertEqual(meta["memory_temporal_type"], "static")
        self.assertAlmostEqual(meta["confidence"], 0.95)
        self.assertEqual(meta["sensitivity"], "private")
        self.assertEqual(meta["modality"], "text")
        self.assertEqual(meta["memory_category"], "profile")
        # fact_key should be set (either from LLM or derived)
        self.assertIn("fact_key", meta)

    def test_extract_smart_metadata_without_llm_uses_defaults(self):
        meta = extract_smart_metadata("Some content", None, abstract="abs", category="cases")
        self.assertEqual(meta["memory_category"], "cases")
        self.assertEqual(meta["confidence"], 0.7)
        self.assertEqual(meta["modality"], "text")  # default

    def test_extract_handles_llm_invalid_json(self):
        llm = MagicMock()
        llm.chat.return_value = "{ this is broken"
        meta = extract_smart_metadata("Some content", llm, category="cases")
        # Falls back gracefully
        self.assertEqual(meta["memory_category"], "cases")

    def test_invalid_temporal_type_dropped(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({"memory_temporal_type": "WHENEVER"})
        meta = extract_smart_metadata("X", llm, category="cases")
        self.assertNotIn("memory_temporal_type", meta)

    def test_stringify_round_trip(self):
        meta = {
            "l0_abstract": "Theo founded SpecCon",
            "memory_category": "profile",
            "confidence": 0.9,
        }
        s = stringify_smart_metadata(meta)
        self.assertEqual(json.loads(s)["l0_abstract"], "Theo founded SpecCon")

    def test_stringify_caps_arrays(self):
        meta = {"sources": list(range(50)), "history": list(range(100)),
                "relations": list(range(40))}
        s = stringify_smart_metadata(meta)
        d = json.loads(s)
        self.assertLessEqual(len(d["sources"]), 20)
        self.assertLessEqual(len(d["history"]), 50)
        self.assertLessEqual(len(d["relations"]), 16)

    def test_derive_fact_key_for_profile(self):
        key = derive_fact_key("profile", "Birthplace: Harare")
        self.assertEqual(key, "profile:birthplace")

    def test_derive_fact_key_skips_non_temporal(self):
        self.assertIsNone(derive_fact_key("cases", "anything"))
        self.assertIsNone(derive_fact_key("events", "decision: launch"))

    def test_parse_metadata_handles_empty(self):
        meta = parse_smart_metadata("", {"content": "x", "category": "cases", "timestamp": 100.0})
        self.assertEqual(meta["memory_category"], "cases")
        self.assertEqual(meta["valid_from"], 100.0)

    def test_parse_metadata_handles_dict_input(self):
        meta = parse_smart_metadata({"confidence": 0.5}, {"content": "x", "timestamp": 100.0})
        self.assertEqual(meta["confidence"], 0.5)

    def test_is_memory_active_at(self):
        now = 1_000_000.0
        meta = {"valid_from": 999_000.0, "invalidated_at": 1_001_000.0}
        self.assertTrue(is_memory_active_at(meta, now))
        self.assertFalse(is_memory_active_at(meta, 1_002_000.0))
        self.assertFalse(is_memory_active_at({"valid_from": 1_001_000.0}, now))

    def test_is_memory_expired(self):
        now = 1_000_000.0
        self.assertTrue(is_memory_expired({"valid_until": 999_000.0}, now))
        self.assertFalse(is_memory_expired({"valid_until": 1_001_000.0}, now))
        self.assertFalse(is_memory_expired({}, now))


# ---------------------------------------------------------------------------
# 6. Noise prototype filter
# ---------------------------------------------------------------------------

class TestNoisePrototypeFilter(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_uninitialized_says_not_noise(self):
        f = NoisePrototypeFilter(self.tmpdir)
        self.assertFalse(f.is_noise(_seeded_embed(1)))

    def test_init_embeds_bundled_texts(self):
        f = NoisePrototypeFilter(self.tmpdir)
        # use distinct embeddings keyed on text length so they aren't degenerate
        f.load_or_init(lambda t: _seeded_embed(len(t) + hash(t) % 1000))
        self.assertTrue(f.initialized)
        self.assertGreater(f.size, 0)
        self.assertLessEqual(f.size, len(BUILTIN_NOISE_TEXTS))

    def test_degenerate_embedder_disables_filter(self):
        f = NoisePrototypeFilter(self.tmpdir)
        # Constant embedder -> all prototypes identical -> degeneracy guard trips
        f.load_or_init(lambda t: [1.0] + [0.0] * 1535)
        self.assertFalse(f.initialized)

    def test_is_noise_above_threshold(self):
        f = NoisePrototypeFilter(self.tmpdir)
        # Embedder maps each text to a unique slot; first prototype gets slot N0
        f.load_or_init(lambda t: _seeded_embed(hash(t) % 1500 + 1))
        # The vector identical to the first prototype should match
        first = BUILTIN_NOISE_TEXTS[0]
        first_vec = _seeded_embed(hash(first) % 1500 + 1)
        self.assertTrue(f.is_noise(first_vec, threshold=0.99))

    def test_is_noise_below_threshold(self):
        f = NoisePrototypeFilter(self.tmpdir)
        f.load_or_init(lambda t: _seeded_embed(hash(t) % 1500 + 1))
        # An orthogonal vector (1.0 in slot 0, prototypes use slots > 0) should not match
        other = [0.0] * 1536
        other[0] = 1.0
        # Compute max similarity — should be very small
        sim = f.max_similarity(other)
        self.assertLess(sim, DEFAULT_THRESHOLD)
        self.assertFalse(f.is_noise(other))

    def test_learn_grows_bank(self):
        f = NoisePrototypeFilter(self.tmpdir)
        f.load_or_init(lambda t: _seeded_embed(hash(t) % 1500 + 1))
        before = f.size
        # A novel-enough vector should be learned
        novel = _seeded_embed(99999)
        added = f.learn(novel)
        self.assertTrue(added)
        self.assertEqual(f.size, before + 1)

    def test_learn_dedups_against_existing(self):
        f = NoisePrototypeFilter(self.tmpdir)
        f.load_or_init(lambda t: _seeded_embed(hash(t) % 1500 + 1))
        first = BUILTIN_NOISE_TEXTS[0]
        first_vec = _seeded_embed(hash(first) % 1500 + 1)
        before = f.size
        # A vector identical to an existing prototype should be deduped out
        added = f.learn(first_vec)
        self.assertFalse(added)
        self.assertEqual(f.size, before)

    def test_cache_is_persisted_and_reloaded(self):
        f1 = NoisePrototypeFilter(self.tmpdir)
        f1.load_or_init(lambda t: _seeded_embed(hash(t) % 1500 + 1))
        size = f1.size
        cache = os.path.join(self.tmpdir, "noise_prototypes.json")
        self.assertTrue(os.path.exists(cache))
        # Fresh filter should load from cache without invoking the embedder again
        called = []
        f2 = NoisePrototypeFilter(self.tmpdir)
        f2.load_or_init(lambda t: (called.append(t), _seeded_embed(0))[1])
        self.assertEqual(f2.size, size)
        self.assertEqual(called, [], "embedder should not be called on cache hit")


# ---------------------------------------------------------------------------
# 7. Wiring (verify P2 schema columns appear in the in-package schema)
# ---------------------------------------------------------------------------

class TestSchemaP2Columns(unittest.TestCase):

    def test_schema_has_metadata_and_parent_id(self):
        from hermes_memory_lancedb import _get_schema
        col_names = {f.name for f in _get_schema()}
        self.assertIn("metadata", col_names)
        self.assertIn("parent_id", col_names)


# ---------------------------------------------------------------------------
# 8. Stats dataclass round-trip
# ---------------------------------------------------------------------------

class TestAdmissionStatsDataclass(unittest.TestCase):

    def test_roundtrip(self):
        s = AdmissionStats(
            admitted_count=3, rejected_count=2,
            recent_decisions=[[1.0, 1.0], [2.0, 0.0]],
            reject_vectors=[[0.1, 0.2]], recent_admit_vectors=[[0.3, 0.4]],
            last_admit_at=42.0,
        )
        d = s.to_json()
        s2 = AdmissionStats.from_json(d)
        self.assertEqual(s2.admitted_count, 3)
        self.assertEqual(s2.rejected_count, 2)
        self.assertEqual(s2.last_admit_at, 42.0)
        self.assertEqual(s2.reject_vectors, [[0.1, 0.2]])

    def test_from_json_handles_garbage(self):
        s = AdmissionStats.from_json("not a dict")  # type: ignore[arg-type]
        self.assertEqual(s.admitted_count, 0)


# ---------------------------------------------------------------------------
# 9. Provider wiring (no lancedb runtime — purely tests the helper plumbing)
# ---------------------------------------------------------------------------

class TestProviderWiringP2(unittest.TestCase):

    def _bare_provider(self, tmpdir: str):
        """Build a provider with admission/noise wired but no real LanceDB table."""
        from hermes_memory_lancedb import LanceDBMemoryProvider
        p = LanceDBMemoryProvider()
        p._storage_path = tmpdir
        p._admission = AdmissionController(tmpdir, enabled=True)
        p._noise_proto = NoisePrototypeFilter(tmpdir)
        p._embedder = MagicMock()
        p._embedder.embed = lambda t: _seeded_embed(hash(t) % 1500 + 1)
        p._noise_proto.load_or_init(p._embedder.embed)
        return p

    def test_should_admit_rejects_regex_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._bare_provider(tmpdir)
            admit, reason = p._should_admit("hi", _seeded_embed(1), "cases")
            self.assertFalse(admit)
            self.assertIn("regex", reason)

    def test_should_admit_passes_real_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._bare_provider(tmpdir)
            admit, reason = p._should_admit(
                "Theo founded SpecCon Holdings in 2024.",
                _seeded_embed(9999),
                "profile",
            )
            self.assertTrue(admit, f"should admit but got: {reason}")

    def test_should_admit_rejects_via_admission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._bare_provider(tmpdir)
            # Force the admission controller to always reject by tightening thresholds.
            p._admission = AdmissionController(
                tmpdir, enabled=True,
                reject_threshold=0.99, admit_threshold=0.99,
            )
            admit, reason = p._should_admit(
                "Real content that bypasses regex.",
                _seeded_embed(123),
                "events",
            )
            self.assertFalse(admit)
            self.assertIn("admission", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
