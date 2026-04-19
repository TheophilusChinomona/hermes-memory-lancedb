"""Tests for hermes-memory-lancedb v1.3.0 (P1).

Covers:
  - Multi-scope isolation (ScopeManager): composable filters, per-agent
    accessibility, ClawTeam shared scopes, schema migration.
  - Multi-provider embeddings: factory selection, dim resolution,
    dim-mismatch detection, provider availability.
"""

from __future__ import annotations

import os
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from hermes_memory_lancedb import (
    EMBEDDING_DIMENSIONS,
    GLOBAL_SCOPE,
    PROVIDER_DEFAULT_MODEL,
    SCOPE_COLUMNS,
    SCOPE_COLUMN_DEFAULTS,
    EmbeddingError,
    LanceDBMemoryProvider,
    ScopeManager,
    clawteam_scopes_from_env,
    get_provider_from_env,
    is_provider_available,
    make_embedder,
    parse_agent_id_from_session_key,
)
from hermes_memory_lancedb.embedders import (
    GeminiEmbedder,
    JinaEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
)
from hermes_memory_lancedb.scopes import (
    ScopeConfig,
    ScopeDefinition,
    agent_scope,
    is_system_bypass_id,
    parse_clawteam_scopes,
    parse_scope_id,
    project_scope,
    reflection_scope,
    team_scope,
    user_scope,
    workspace_scope,
)


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


class TestScopeHelpers(unittest.TestCase):
    def test_agent_scope_format(self):
        self.assertEqual(agent_scope("andrew"), "agent:andrew")

    def test_user_scope_format(self):
        self.assertEqual(user_scope("theo"), "user:theo")

    def test_project_scope_format(self):
        self.assertEqual(project_scope("p1"), "project:p1")

    def test_team_scope_format(self):
        self.assertEqual(team_scope("ops"), "team:ops")

    def test_workspace_scope_format(self):
        self.assertEqual(workspace_scope("ws1"), "workspace:ws1")

    def test_reflection_scope_format(self):
        self.assertEqual(reflection_scope("a"), "reflection:agent:a")

    def test_parse_scope_id_global(self):
        self.assertEqual(parse_scope_id("global"), ("global", ""))

    def test_parse_scope_id_typed(self):
        self.assertEqual(parse_scope_id("project:abc"), ("project", "abc"))

    def test_parse_scope_id_invalid(self):
        self.assertIsNone(parse_scope_id("noprefix"))

    def test_is_system_bypass_id(self):
        self.assertTrue(is_system_bypass_id("system"))
        self.assertTrue(is_system_bypass_id("undefined"))
        self.assertFalse(is_system_bypass_id("andrew"))
        self.assertFalse(is_system_bypass_id(None))

    def test_parse_agent_id_from_session_key_two_segment(self):
        self.assertEqual(parse_agent_id_from_session_key("agent:main"), "main")

    def test_parse_agent_id_from_session_key_with_trailing(self):
        self.assertEqual(
            parse_agent_id_from_session_key("agent:main:discord:channel:123"),
            "main",
        )

    def test_parse_agent_id_from_session_key_non_agent(self):
        self.assertIsNone(parse_agent_id_from_session_key("user:theo"))

    def test_parse_agent_id_from_session_key_bypass_id(self):
        self.assertIsNone(parse_agent_id_from_session_key("agent:system"))

    def test_parse_agent_id_from_session_key_empty(self):
        self.assertIsNone(parse_agent_id_from_session_key(""))
        self.assertIsNone(parse_agent_id_from_session_key(None))


# ---------------------------------------------------------------------------
# ScopeManager
# ---------------------------------------------------------------------------


class TestScopeManagerBasics(unittest.TestCase):
    def test_default_config_has_global(self):
        sm = ScopeManager()
        self.assertIn(GLOBAL_SCOPE, sm.get_all_scopes())
        self.assertEqual(sm.get_default_scope(), GLOBAL_SCOPE)

    def test_validate_scope_global(self):
        sm = ScopeManager()
        self.assertTrue(sm.validate_scope("global"))
        self.assertTrue(sm.validate_scope("agent:foo"))
        self.assertTrue(sm.validate_scope("project:bar"))
        self.assertFalse(sm.validate_scope(""))

    def test_get_accessible_scopes_no_agent_returns_all(self):
        sm = ScopeManager()
        self.assertEqual(set(sm.get_accessible_scopes()), set(sm.get_all_scopes()))

    def test_get_accessible_scopes_includes_own_agent_and_reflection(self):
        sm = ScopeManager()
        scopes = sm.get_accessible_scopes("alice")
        self.assertIn(GLOBAL_SCOPE, scopes)
        self.assertIn("agent:alice", scopes)
        self.assertIn("reflection:agent:alice", scopes)

    def test_default_scope_per_agent_returns_agent_private(self):
        sm = ScopeManager()
        self.assertEqual(sm.get_default_scope("alice"), "agent:alice")

    def test_default_scope_bypass_id_raises(self):
        sm = ScopeManager()
        with self.assertRaises(ValueError):
            sm.get_default_scope("system")

    def test_set_agent_access_explicit(self):
        sm = ScopeManager()
        sm.add_scope_definition("custom:secret", ScopeDefinition(description="x"))
        sm.set_agent_access("alice", ["global", "custom:secret"])
        accessible = sm.get_accessible_scopes("alice")
        self.assertIn("custom:secret", accessible)
        self.assertIn("global", accessible)
        # Reflection is auto-granted even with explicit ACL
        self.assertIn("reflection:agent:alice", accessible)

    def test_set_agent_access_bypass_id_rejected(self):
        sm = ScopeManager()
        with self.assertRaises(ValueError):
            sm.set_agent_access("system", ["global"])

    def test_get_scope_filter_bypass(self):
        sm = ScopeManager()
        self.assertIsNone(sm.get_scope_filter())
        self.assertIsNone(sm.get_scope_filter("system"))

    def test_get_scope_filter_agent(self):
        sm = ScopeManager()
        result = sm.get_scope_filter("alice")
        self.assertIsNotNone(result)
        self.assertIn("agent:alice", result)

    def test_remove_global_scope_raises(self):
        sm = ScopeManager()
        with self.assertRaises(ValueError):
            sm.remove_scope_definition(GLOBAL_SCOPE)

    def test_remove_unknown_scope_returns_false(self):
        sm = ScopeManager()
        self.assertFalse(sm.remove_scope_definition("custom:nope"))

    def test_clawteam_scopes_grant_to_all(self):
        sm = ScopeManager()
        sm.apply_clawteam_scopes(["custom:teamA", "custom:teamB"])
        self.assertIn("custom:teamA", sm.get_accessible_scopes("alice"))
        self.assertIn("custom:teamB", sm.get_accessible_scopes("bob"))
        # Definitions should also be registered
        self.assertIsNotNone(sm.get_scope_definition("custom:teamA"))

    def test_clawteam_empty_noop(self):
        sm = ScopeManager()
        before = list(sm.get_all_scopes())
        sm.apply_clawteam_scopes([])
        self.assertEqual(sorted(before), sorted(sm.get_all_scopes()))


# ---------------------------------------------------------------------------
# Composable WHERE clause builder
# ---------------------------------------------------------------------------


class TestScopeWhereClause(unittest.TestCase):
    def test_legacy_only_user_id(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(
            user_id="andrew",
            scope_columns_present=False,
            legacy_user_id="andrew",
        )
        self.assertEqual(clause, "user_id = 'andrew'")

    def test_legacy_no_user(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(scope_columns_present=False)
        self.assertEqual(clause, "")

    def test_modern_single_column(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(user_id="andrew", scope_columns_present=True)
        self.assertEqual(clause, "user_id = 'andrew'")

    def test_modern_multi_column_compose(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(
            agent_id="main",
            user_id="andrew",
            project_id="hermes",
            scope_columns_present=True,
        )
        self.assertIn("agent_id = 'main'", clause)
        self.assertIn("user_id = 'andrew'", clause)
        self.assertIn("project_id = 'hermes'", clause)
        self.assertEqual(clause.count(" AND "), 2)

    def test_modern_all_five_dimensions(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(
            agent_id="a",
            user_id="u",
            project_id="p",
            team_id="t",
            workspace_id="w",
            scope_columns_present=True,
        )
        for col, val in (("agent_id", "a"), ("user_id", "u"), ("project_id", "p"),
                         ("team_id", "t"), ("workspace_id", "w")):
            self.assertIn(f"{col} = '{val}'", clause)
        # 5 clauses → 4 ANDs
        self.assertEqual(clause.count(" AND "), 4)

    def test_modern_empty_filters_omitted(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(
            agent_id="a",
            user_id=None,
            project_id="",
            scope_columns_present=True,
        )
        self.assertEqual(clause, "agent_id = 'a'")

    def test_sql_injection_escaped(self):
        sm = ScopeManager()
        clause = sm.build_where_clause(
            user_id="andrew' OR '1'='1",
            scope_columns_present=True,
        )
        # Single quotes escaped via doubling — no unescaped quote sneaks out
        self.assertEqual(clause, "user_id = 'andrew'' OR ''1''=''1'")


# ---------------------------------------------------------------------------
# ClawTeam env parsing
# ---------------------------------------------------------------------------


class TestClawteamEnvParsing(unittest.TestCase):
    def test_parse_clawteam_scopes_csv(self):
        self.assertEqual(parse_clawteam_scopes("a,b, c "), ["a", "b", "c"])

    def test_parse_clawteam_scopes_empty(self):
        self.assertEqual(parse_clawteam_scopes(""), [])
        self.assertEqual(parse_clawteam_scopes(None), [])

    def test_clawteam_scopes_from_env(self):
        with patch.dict(os.environ, {"CLAWTEAM_MEMORY_SCOPE": "x,y"}, clear=False):
            self.assertEqual(clawteam_scopes_from_env(), ["x", "y"])
        # When unset
        env = dict(os.environ)
        env.pop("CLAWTEAM_MEMORY_SCOPE", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(clawteam_scopes_from_env(), [])


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------


class TestEmbedderFactory(unittest.TestCase):
    def test_default_provider_is_openai(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LANCEDB_EMBED")}
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env, clear=True):
            emb = make_embedder()
        self.assertIsInstance(emb, OpenAIEmbedder)
        self.assertEqual(emb.model, "text-embedding-3-small")
        self.assertEqual(emb.dimensions, 1536)

    def test_jina_provider_selection(self):
        with patch.dict(os.environ, {"LANCEDB_EMBED_PROVIDER": "jina", "JINA_API_KEY": "jk"}, clear=False):
            emb = make_embedder()
        self.assertIsInstance(emb, JinaEmbedder)
        self.assertEqual(emb.dimensions, 1024)

    def test_gemini_provider_selection(self):
        with patch.dict(os.environ, {"LANCEDB_EMBED_PROVIDER": "gemini", "GEMINI_API_KEY": "gk"}, clear=False):
            emb = make_embedder()
        self.assertIsInstance(emb, GeminiEmbedder)
        self.assertEqual(emb.dimensions, 768)

    def test_ollama_provider_no_key_required(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LANCEDB_EMBED")}
        env["LANCEDB_EMBED_PROVIDER"] = "ollama"
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            emb = make_embedder()
        self.assertIsInstance(emb, OllamaEmbedder)

    def test_openai_compatible_requires_base_url(self):
        env = {"LANCEDB_EMBED_PROVIDER": "openai-compatible", "OPENAI_API_KEY": "sk"}
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(EmbeddingError):
                make_embedder()

    def test_openai_compatible_with_base_url(self):
        env = {
            "LANCEDB_EMBED_PROVIDER": "openai-compatible",
            "LANCEDB_EMBED_BASE_URL": "https://api.example.com/v1",
            "LANCEDB_EMBED_MODEL": "text-embedding-3-small",
            "OPENAI_API_KEY": "sk",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertIsInstance(emb, OpenAIEmbedder)
        self.assertEqual(emb.dimensions, 1536)

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LANCEDB_EMBED_PROVIDER": "bogus"}, clear=False):
            with self.assertRaises(EmbeddingError):
                make_embedder()

    def test_openrouter_provider_selection(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LANCEDB_EMBED")}
        env["LANCEDB_EMBED_PROVIDER"] = "openrouter"
        env["OPENROUTER_API_KEY"] = "or-test"
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            emb = make_embedder()
        self.assertIsInstance(emb, OpenAIEmbedder)
        self.assertEqual(emb.model, "openai/text-embedding-3-small")
        # Prefix-stripped lookup → 1536 (text-embedding-3-small).
        self.assertEqual(emb.dimensions, 1536)

    def test_openrouter_uses_openrouter_base_by_default(self):
        env = {"LANCEDB_EMBED_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "or-key"}
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertEqual(emb._base_url, "https://openrouter.ai/api/v1")

    def test_openrouter_custom_base_url_override(self):
        env = {
            "LANCEDB_EMBED_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "or-key",
            "LANCEDB_EMBED_BASE_URL": "https://my-proxy.example.com/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertEqual(emb._base_url, "https://my-proxy.example.com/v1")

    def test_openrouter_custom_model_dim_resolution(self):
        # cohere/embed-english-v3.0 isn't in EMBEDDING_DIMENSIONS — falls back
        # to provider default (1536).
        env = {
            "LANCEDB_EMBED_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "or-key",
            "LANCEDB_EMBED_MODEL": "cohere/embed-english-v3.0",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertEqual(emb.model, "cohere/embed-english-v3.0")
        self.assertEqual(emb.dimensions, 1536)  # openrouter fallback

    def test_openrouter_voyage_model_dim_via_prefix_strip(self):
        # voyage-3 IS in EMBEDDING_DIMENSIONS (1024) — prefix strip should hit.
        env = {
            "LANCEDB_EMBED_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "or-key",
            "LANCEDB_EMBED_MODEL": "voyage/voyage-3",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertEqual(emb.dimensions, 1024)

    def test_openrouter_sends_dimensions_for_v3_family(self):
        from hermes_memory_lancedb.embedders import _is_openai_v3_family
        # Both prefixed and bare model ids should be recognised.
        self.assertTrue(_is_openai_v3_family("text-embedding-3-small"))
        self.assertTrue(_is_openai_v3_family("openai/text-embedding-3-large"))
        self.assertFalse(_is_openai_v3_family("text-embedding-ada-002"))
        self.assertFalse(_is_openai_v3_family("openai/text-embedding-ada-002"))
        self.assertFalse(_is_openai_v3_family("voyage/voyage-3"))

    def test_openrouter_is_provider_available(self):
        from hermes_memory_lancedb.embedders import is_provider_available
        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(is_provider_available("openrouter"))
        env["OPENROUTER_API_KEY"] = "or"
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(is_provider_available("openrouter"))

    def test_dim_override_via_env(self):
        env = {
            "LANCEDB_EMBED_PROVIDER": "openai",
            "LANCEDB_EMBED_MODEL": "text-embedding-3-large",
            "LANCEDB_EMBED_DIM": "2048",
            "OPENAI_API_KEY": "sk",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        self.assertEqual(emb.dimensions, 2048)

    def test_unknown_model_uses_provider_default_dim(self):
        env = {
            "LANCEDB_EMBED_PROVIDER": "jina",
            "LANCEDB_EMBED_MODEL": "jina-mystery-model-x",
            "JINA_API_KEY": "jk",
        }
        with patch.dict(os.environ, env, clear=False):
            emb = make_embedder()
        # Falls back to Jina default 1024
        self.assertEqual(emb.dimensions, 1024)

    def test_get_provider_from_env_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LANCEDB_EMBED_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_provider_from_env(), "openai")

    def test_is_provider_available(self):
        env = {"OPENAI_API_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(is_provider_available("openai"))
            self.assertFalse(is_provider_available("jina"))
            self.assertFalse(is_provider_available("gemini"))
            self.assertTrue(is_provider_available("ollama"))


# ---------------------------------------------------------------------------
# Embedder dimension validation
# ---------------------------------------------------------------------------


class TestEmbedderDimValidation(unittest.TestCase):
    def test_dim_mismatch_raises(self):
        emb = OpenAIEmbedder(api_key="sk", model="text-embedding-3-small", dimensions=4)

        def fake_embed(text):
            return [0.1, 0.2]  # wrong length

        emb._embed_uncached = fake_embed  # type: ignore[assignment]
        with self.assertRaises(EmbeddingError):
            emb.embed("hello")

    def test_correct_dim_passes(self):
        emb = OpenAIEmbedder(api_key="sk", model="text-embedding-3-small", dimensions=3)
        emb._embed_uncached = lambda t: [0.0, 0.0, 1.0]  # type: ignore[assignment]
        out = emb.embed("hello")
        self.assertEqual(out, [0.0, 0.0, 1.0])

    def test_cache_avoids_duplicate_calls(self):
        emb = OpenAIEmbedder(api_key="sk", model="text-embedding-3-small", dimensions=2)
        calls = []

        def fake_embed(text):
            calls.append(text)
            return [1.0, 2.0]

        emb._embed_uncached = fake_embed  # type: ignore[assignment]
        emb.embed("foo")
        emb.embed("foo")
        self.assertEqual(len(calls), 1)


# ---------------------------------------------------------------------------
# Per-provider HTTP request shape (mocked)
# ---------------------------------------------------------------------------


class TestJinaEmbedderHTTP(unittest.TestCase):
    def test_jina_calls_post_with_correct_body(self):
        emb = JinaEmbedder(api_key="jk", model="jina-embeddings-v3", dimensions=1024)
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"embedding": [0.0] * 1024}]}
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            vec = emb.embed("hello")
        self.assertEqual(len(vec), 1024)
        self.assertEqual(captured["url"], "https://api.jina.ai/v1/embeddings")
        self.assertEqual(captured["json"]["model"], "jina-embeddings-v3")
        self.assertEqual(captured["json"]["task"], "retrieval.passage")
        self.assertTrue(captured["json"]["normalized"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer jk")

    def test_jina_http_error_raises(self):
        emb = JinaEmbedder(api_key="jk", model="jina-embeddings-v3", dimensions=1024)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "auth error"
        with patch("httpx.post", return_value=mock_resp):
            with self.assertRaises(EmbeddingError):
                emb.embed("hi")


class TestGeminiEmbedderHTTP(unittest.TestCase):
    def test_gemini_payload_shape(self):
        emb = GeminiEmbedder(api_key="gk", model="text-embedding-004", dimensions=768)
        captured = {}

        def fake_post(url, params=None, json=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["json"] = json
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embedding": {"values": [0.0] * 768}}
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            vec = emb.embed("hello")
        self.assertEqual(len(vec), 768)
        self.assertIn("text-embedding-004:embedContent", captured["url"])
        self.assertEqual(captured["params"]["key"], "gk")
        self.assertEqual(captured["json"]["taskType"], "RETRIEVAL_DOCUMENT")


class TestOllamaEmbedderHTTP(unittest.TestCase):
    def test_ollama_uses_native_endpoint(self):
        emb = OllamaEmbedder("nomic-embed-text", base_url="http://127.0.0.1:11434/v1", dimensions=768)
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embedding": [0.0] * 768}
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            emb.embed("hello")
        # Should rewrite away from /v1/embeddings to /api/embeddings
        self.assertTrue(captured["url"].endswith("/api/embeddings"))
        # Should use 'prompt' not 'input'
        self.assertEqual(captured["json"]["prompt"], "hello")


# ---------------------------------------------------------------------------
# Schema helpers — verify the new scope columns are part of the schema
# ---------------------------------------------------------------------------


class TestSchemaFields(unittest.TestCase):
    def test_scope_columns_constant(self):
        # All five scope columns plus the canonical scope string
        for col in ("agent_id", "project_id", "team_id", "workspace_id", "scope"):
            self.assertIn(col, SCOPE_COLUMNS)
            self.assertIn(col, SCOPE_COLUMN_DEFAULTS)

    def test_get_schema_includes_scope_columns(self):
        from hermes_memory_lancedb import _get_schema

        schema = _get_schema(embed_dim=128)
        names = {f.name for f in schema}
        for col in SCOPE_COLUMNS:
            self.assertIn(col, names)
        # Sanity: vector dim wired through
        vec_field = next(f for f in schema if f.name == "vector")
        # pyarrow fixed_size_list type carries .list_size or value_field
        self.assertEqual(vec_field.type.list_size, 128)

    def test_get_schema_default_dim(self):
        from hermes_memory_lancedb import _get_schema

        schema = _get_schema()
        vec_field = next(f for f in schema if f.name == "vector")
        self.assertEqual(vec_field.type.list_size, 1536)


# ---------------------------------------------------------------------------
# Schema migration — simulate add_columns on a legacy table
# ---------------------------------------------------------------------------


class TestSchemaMigrationAddsScopeColumns(unittest.TestCase):
    def _make_provider_with_fake_table(self, existing_columns):
        provider = LanceDBMemoryProvider()
        # Build a fake table that responds to .schema and .add_columns
        fake_schema_fields = [MagicMock(name=col) for col in existing_columns]
        # Use spec'd objects so attr access works
        for f, name in zip(fake_schema_fields, existing_columns):
            f.name = name
        fake_table = MagicMock()
        fake_table.schema = fake_schema_fields
        # Track add_columns calls
        added = {}

        def fake_add(cols):
            added.update(cols)
            # Update schema to include the new fields
            for new_col in cols.keys():
                m = MagicMock()
                m.name = new_col
                fake_table.schema.append(m)

        fake_table.add_columns.side_effect = fake_add
        provider._table = fake_table
        provider._added = added  # type: ignore[attr-defined]
        return provider, fake_table, added

    def test_migration_adds_scope_columns_to_legacy(self):
        legacy_cols = [
            "id", "content", "vector", "timestamp", "source",
            "session_id", "user_id", "tags",
        ]
        provider, table, added = self._make_provider_with_fake_table(legacy_cols)
        provider._migrate_schema_if_needed()
        # All P1 scope columns should be queued
        for col in SCOPE_COLUMNS:
            self.assertIn(col, added)
        # And the v1.1.0 fields too
        for col in ("tier", "importance", "access_count", "category", "abstract", "overview"):
            self.assertIn(col, added)
        self.assertTrue(provider._has_scope_columns)

    def test_migration_skips_already_present(self):
        all_cols = [
            "id", "content", "vector", "timestamp", "source", "session_id",
            "user_id", "tags", "tier", "importance", "access_count",
            "category", "abstract", "overview",
            "agent_id", "project_id", "team_id", "workspace_id", "scope",
            # P2 columns also present
            "metadata", "parent_id",
            # P4 column also present
            "temporal_type",
        ]
        provider, table, added = self._make_provider_with_fake_table(all_cols)
        provider._migrate_schema_if_needed()
        # Nothing new to add
        table.add_columns.assert_not_called()
        self.assertTrue(provider._has_scope_columns)

    def test_migration_partial_legacy_v110_only(self):
        # Has v1.1.0 fields but no scope columns
        cols = [
            "id", "content", "vector", "timestamp", "source", "session_id",
            "user_id", "tags", "tier", "importance", "access_count",
            "category", "abstract", "overview",
        ]
        provider, table, added = self._make_provider_with_fake_table(cols)
        provider._migrate_schema_if_needed()
        # Only scope cols added
        for col in SCOPE_COLUMNS:
            self.assertIn(col, added)
        for col in ("tier", "importance"):
            self.assertNotIn(col, added)


# ---------------------------------------------------------------------------
# Provider integration — _build_scope_where uses configured ids
# ---------------------------------------------------------------------------


class TestProviderScopeWiring(unittest.TestCase):
    def test_build_where_legacy_when_no_scope_columns(self):
        provider = LanceDBMemoryProvider()
        provider._user_id = "andrew"
        provider._has_scope_columns = False
        provider._scope_manager = ScopeManager()
        clause = provider._build_scope_where()
        self.assertEqual(clause, "user_id = 'andrew'")

    def test_build_where_modern_compose(self):
        provider = LanceDBMemoryProvider()
        provider._user_id = "andrew"
        provider._agent_id = "main"
        provider._project_id = "hermes"
        provider._has_scope_columns = True
        provider._scope_manager = ScopeManager()
        clause = provider._build_scope_where()
        self.assertIn("agent_id = 'main'", clause)
        self.assertIn("user_id = 'andrew'", clause)
        self.assertIn("project_id = 'hermes'", clause)

    def test_initialize_picks_up_scope_kwargs(self):
        # Embedder + lancedb stub
        provider = LanceDBMemoryProvider()
        env = {
            "OPENAI_API_KEY": "sk",
            "LANCEDB_PATH": "/tmp/__hermes_p1_no_table__",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("hermes_memory_lancedb.make_embedder") as m_emb:
                fake_emb = MagicMock()
                fake_emb.dimensions = 1536
                fake_emb.embed = lambda t: [0.0] * 1536
                m_emb.return_value = fake_emb
                # Stub out lancedb so initialize doesn't need the native lib.
                fake_lancedb = MagicMock()
                fake_db = MagicMock()
                fake_db.table_names.return_value = []
                fake_table = MagicMock()
                fake_table.schema = []
                fake_db.create_table.return_value = fake_table
                fake_lancedb.connect.return_value = fake_db
                with patch.dict("sys.modules", {"lancedb": fake_lancedb}):
                    provider.initialize(
                        "test-session-x",
                        agent_id="ag1",
                        project_id="pj1",
                        team_id="tm1",
                        workspace_id="ws1",
                    )
        self.assertEqual(provider._agent_id, "ag1")
        self.assertEqual(provider._project_id, "pj1")
        self.assertEqual(provider._team_id, "tm1")
        self.assertEqual(provider._workspace_id, "ws1")

    def test_initialize_extracts_agent_from_session_key(self):
        provider = LanceDBMemoryProvider()
        env = {
            "OPENAI_API_KEY": "sk",
            "LANCEDB_PATH": "/tmp/__hermes_p1_no_table_2__",
        }
        # Don't pollute env with a pre-set LANCEDB_AGENT_ID
        for k in ("LANCEDB_AGENT_ID", "LANCEDB_PROJECT_ID", "LANCEDB_TEAM_ID", "LANCEDB_WORKSPACE_ID"):
            env[k] = ""
        with patch.dict(os.environ, env, clear=False):
            with patch("hermes_memory_lancedb.make_embedder") as m_emb:
                fake_emb = MagicMock()
                fake_emb.dimensions = 1536
                fake_emb.embed = lambda t: [0.0] * 1536
                m_emb.return_value = fake_emb
                fake_lancedb = MagicMock()
                fake_db = MagicMock()
                fake_db.table_names.return_value = []
                fake_table = MagicMock()
                fake_table.schema = []
                fake_db.create_table.return_value = fake_table
                fake_lancedb.connect.return_value = fake_db
                with patch.dict("sys.modules", {"lancedb": fake_lancedb}):
                    provider.initialize("agent:bob:discord:99")
        self.assertEqual(provider._agent_id, "bob")


# ---------------------------------------------------------------------------
# is_available with provider-specific keys
# ---------------------------------------------------------------------------


class TestProviderAvailable(unittest.TestCase):
    def test_is_available_openai(self):
        env = {k: v for k, v in os.environ.items() if k != "LANCEDB_EMBED_PROVIDER"}
        env["OPENAI_API_KEY"] = "sk"
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(LanceDBMemoryProvider().is_available())

    def test_is_available_jina(self):
        env = {"LANCEDB_EMBED_PROVIDER": "jina", "JINA_API_KEY": "jk"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(LanceDBMemoryProvider().is_available())

    def test_is_available_no_keys(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY",
                            "LANCEDB_EMBED_API_KEY", "LANCEDB_EMBED_PROVIDER")}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(LanceDBMemoryProvider().is_available())


# ---------------------------------------------------------------------------
# Sanity: dim lookup table
# ---------------------------------------------------------------------------


class TestEmbeddingDimensions(unittest.TestCase):
    def test_known_models(self):
        self.assertEqual(EMBEDDING_DIMENSIONS["text-embedding-3-small"], 1536)
        self.assertEqual(EMBEDDING_DIMENSIONS["text-embedding-3-large"], 3072)
        self.assertEqual(EMBEDDING_DIMENSIONS["jina-embeddings-v3"], 1024)
        self.assertEqual(EMBEDDING_DIMENSIONS["text-embedding-004"], 768)

    def test_provider_default_models_have_known_dims(self):
        # OpenRouter defaults use provider-prefixed model ids (e.g.
        # "openai/text-embedding-3-small"); allow lookup via prefix-strip.
        from hermes_memory_lancedb.embedders import _strip_provider_prefix
        for provider, model in PROVIDER_DEFAULT_MODEL.items():
            stripped = _strip_provider_prefix(model)
            self.assertTrue(
                model in EMBEDDING_DIMENSIONS or stripped in EMBEDDING_DIMENSIONS,
                f"Provider {provider} default model {model} missing from EMBEDDING_DIMENSIONS",
            )


if __name__ == "__main__":
    unittest.main()
