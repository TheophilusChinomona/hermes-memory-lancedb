"""Tests for hermes-memory-lancedb v1.2.0 retrieval pipeline.

Covers the P0 retrieval-quality features ported from memory-lancedb-pro:
cross-encoder rerank, MMR diversity, length normalization, hard min-score.
"""

from __future__ import annotations

import math
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from hermes_memory_lancedb import (
    _apply_length_normalization,
    _apply_mmr_diversity,
    _clamp01,
    _cosine_similarity,
    _normalize_to_top,
    _rerank_cosine_fallback,
    _rerank_jina,
    _LENGTH_NORM_ANCHOR,
    _MMR_SIMILARITY_THRESHOLD,
)


class TestClamp01(unittest.TestCase):
    def test_within_range(self):
        self.assertEqual(_clamp01(0.5), 0.5)

    def test_above_one(self):
        self.assertEqual(_clamp01(1.7), 1.0)

    def test_below_zero(self):
        self.assertEqual(_clamp01(-0.3), 0.0)

    def test_nan_uses_fallback(self):
        self.assertEqual(_clamp01(float("nan"), fallback=0.4), 0.4)

    def test_inf_uses_fallback(self):
        self.assertEqual(_clamp01(float("inf"), fallback=0.7), 0.7)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_mismatched_lengths_returns_zero(self):
        self.assertEqual(_cosine_similarity([1.0, 2.0], [1.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(_cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_empty_returns_zero(self):
        self.assertEqual(_cosine_similarity([], []), 0.0)


class TestNormalizeToTop(unittest.TestCase):
    def test_top_becomes_one(self):
        hits = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.4}]
        out = _normalize_to_top(hits)
        self.assertEqual(out[0]["score"], 1.0)
        self.assertAlmostEqual(out[1]["score"], 0.5)

    def test_empty_passthrough(self):
        self.assertEqual(_normalize_to_top([]), [])

    def test_zero_top_passthrough(self):
        hits = [{"id": "a", "score": 0.0}]
        out = _normalize_to_top(hits)
        self.assertEqual(out[0]["score"], 0.0)


class TestLengthNormalization(unittest.TestCase):
    def test_short_entries_unchanged(self):
        hits = [{"id": "a", "content": "x" * 100, "score": 0.9}]
        out = _apply_length_normalization(hits)
        self.assertAlmostEqual(out[0]["score"], 0.9, places=4)

    def test_at_anchor_unchanged(self):
        hits = [{"id": "a", "content": "x" * _LENGTH_NORM_ANCHOR, "score": 0.9}]
        out = _apply_length_normalization(hits)
        self.assertAlmostEqual(out[0]["score"], 0.9, places=4)

    def test_long_entries_penalized(self):
        # 2000 chars vs 500 anchor → ratio 4 → log2(4)=2 → factor 1/(1+1)=0.5
        hits = [{"id": "a", "content": "x" * 2000, "score": 0.9}]
        out = _apply_length_normalization(hits)
        self.assertAlmostEqual(out[0]["score"], 0.45, places=2)

    def test_resort_by_score(self):
        hits = [
            {"id": "long", "content": "x" * 4000, "score": 0.9},
            {"id": "short", "content": "x" * 100, "score": 0.7},
        ]
        out = _apply_length_normalization(hits)
        # After norm: long ≈ 0.9 * 1/(1+0.5*log2(8)) = 0.9 * 1/2.5 = 0.36
        # short stays at 0.7 → should now rank first.
        self.assertEqual(out[0]["id"], "short")

    def test_anchor_zero_passthrough(self):
        hits = [{"id": "a", "content": "x" * 5000, "score": 0.5}]
        out = _apply_length_normalization(hits, anchor=0)
        self.assertEqual(out[0]["score"], 0.5)


class TestMMRDiversity(unittest.TestCase):
    def test_singleton_passthrough(self):
        hits = [{"id": "a", "score": 0.9, "vector": [1.0, 0.0]}]
        self.assertEqual(_apply_mmr_diversity(hits), hits)

    def test_orthogonal_all_selected(self):
        hits = [
            {"id": "a", "score": 0.9, "vector": [1.0, 0.0, 0.0]},
            {"id": "b", "score": 0.8, "vector": [0.0, 1.0, 0.0]},
            {"id": "c", "score": 0.7, "vector": [0.0, 0.0, 1.0]},
        ]
        out = _apply_mmr_diversity(hits)
        # Order preserved (none deferred)
        self.assertEqual([h["id"] for h in out], ["a", "b", "c"])

    def test_near_duplicates_deferred(self):
        # b is near-duplicate of a (cosine ~0.999) — should be deferred
        hits = [
            {"id": "a", "score": 0.9, "vector": [1.0, 0.0, 0.0]},
            {"id": "b", "score": 0.85, "vector": [0.99, 0.01, 0.0]},
            {"id": "c", "score": 0.7, "vector": [0.0, 1.0, 0.0]},
        ]
        out = _apply_mmr_diversity(hits)
        # a + c selected first, b deferred to end
        self.assertEqual([h["id"] for h in out], ["a", "c", "b"])

    def test_missing_vector_kept_in_selected(self):
        hits = [
            {"id": "a", "score": 0.9, "vector": [1.0, 0.0]},
            {"id": "b", "score": 0.8},  # no vector
        ]
        out = _apply_mmr_diversity(hits)
        # Both kept, no deferral
        self.assertEqual(len(out), 2)
        self.assertEqual([h["id"] for h in out], ["a", "b"])

    def test_threshold_respected(self):
        # cosine of these two ≈ 0.5 — under default 0.85 threshold
        hits = [
            {"id": "a", "score": 0.9, "vector": [1.0, 1.0]},
            {"id": "b", "score": 0.8, "vector": [1.0, -1.0 / 3]},
        ]
        out = _apply_mmr_diversity(hits, threshold=0.85)
        # Both kept in selected (not deferred)
        self.assertEqual([h["id"] for h in out[:2]], ["a", "b"])


class TestRerankJina(unittest.TestCase):
    def test_no_api_key_returns_none(self):
        hits = [{"id": "a", "content": "foo", "score": 0.5}]
        self.assertIsNone(_rerank_jina("query", hits, api_key=""))

    def test_empty_hits_returns_none(self):
        self.assertIsNone(_rerank_jina("query", [], api_key="key"))

    def test_blends_with_original(self):
        hits = [
            {"id": "a", "content": "foo", "score": 0.5},
            {"id": "b", "content": "bar", "score": 0.4},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        with patch("httpx.post", return_value=mock_resp):
            out = _rerank_jina("query", hits, api_key="key")
        self.assertIsNotNone(out)
        # b: 0.6*0.9 + 0.4*0.4 = 0.70 → ranks first
        # a: 0.6*0.2 + 0.4*0.5 = 0.32
        self.assertEqual(out[0]["id"], "b")
        self.assertAlmostEqual(out[0]["score"], 0.70, places=2)

    def test_http_error_returns_none(self):
        hits = [{"id": "a", "content": "foo", "score": 0.5}]
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        with patch("httpx.post", return_value=mock_resp):
            self.assertIsNone(_rerank_jina("query", hits, api_key="key"))

    def test_unreturned_candidates_penalized(self):
        hits = [
            {"id": "a", "content": "foo", "score": 0.5},
            {"id": "b", "content": "bar", "score": 0.4},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Only index 0 returned; index 1 should appear with score * 0.8 = 0.32
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
        with patch("httpx.post", return_value=mock_resp):
            out = _rerank_jina("query", hits, api_key="key")
        self.assertEqual(len(out), 2)
        b_hit = next(h for h in out if h["id"] == "b")
        self.assertAlmostEqual(b_hit["score"], 0.32, places=2)


class TestRerankCosineFallback(unittest.TestCase):
    def test_blends_70_30(self):
        query_vec = [1.0, 0.0, 0.0]
        hits = [
            {"id": "a", "score": 0.5, "vector": [1.0, 0.0, 0.0]},  # cos=1.0
            {"id": "b", "score": 0.5, "vector": [0.0, 1.0, 0.0]},  # cos=0.0
        ]
        out = _rerank_cosine_fallback(query_vec, hits)
        # a: 0.7*0.5 + 0.3*1.0 = 0.65; b: 0.7*0.5 + 0.3*0 = 0.35
        a_hit = next(h for h in out if h["id"] == "a")
        b_hit = next(h for h in out if h["id"] == "b")
        self.assertAlmostEqual(a_hit["score"], 0.65, places=2)
        self.assertAlmostEqual(b_hit["score"], 0.35, places=2)
        self.assertEqual(out[0]["id"], "a")  # higher score wins

    def test_missing_vector_passes_through(self):
        query_vec = [1.0, 0.0]
        hits = [{"id": "a", "score": 0.5}]  # no vector
        out = _rerank_cosine_fallback(query_vec, hits)
        self.assertEqual(out[0]["score"], 0.5)

    def test_empty_query_vec_passes_through(self):
        hits = [{"id": "a", "score": 0.5, "vector": [1.0, 0.0]}]
        self.assertEqual(_rerank_cosine_fallback([], hits), hits)


if __name__ == "__main__":
    unittest.main()
