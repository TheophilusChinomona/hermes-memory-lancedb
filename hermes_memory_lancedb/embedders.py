"""Multi-provider embedding clients.

Ported from memory-lancedb-pro TypeScript src/embedder.ts.

Supported providers (selected via ``LANCEDB_EMBED_PROVIDER`` env var):

  - ``openai``   — OpenAI text-embedding-3-* (default)
  - ``jina``     — Jina embeddings v3/v5 family
  - ``gemini``   — Google Generative AI embeddings (REST)
  - ``ollama``   — Local Ollama server (``/api/embeddings``)
  - ``openai-compatible`` — Any OpenAI-shaped endpoint via custom
    ``LANCEDB_EMBED_BASE_URL``.

Each provider has its own API key env var:

  - ``OPENAI_API_KEY``
  - ``JINA_API_KEY``
  - ``GEMINI_API_KEY``
  - ``OLLAMA_BASE_URL`` (URL, not a key — defaults to
    ``http://127.0.0.1:11434``)

Optional overrides:

  - ``LANCEDB_EMBED_MODEL``    — provider-specific model id
  - ``LANCEDB_EMBED_DIM``      — explicit dimensions override
  - ``LANCEDB_EMBED_BASE_URL`` — custom base URL (openai-compatible)
  - ``LANCEDB_EMBED_API_KEY``  — generic key override (any provider)
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known model dimensions (kept in sync with EMBEDDING_DIMENSIONS in embedder.ts)
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Google
    "text-embedding-004": 768,
    "gemini-embedding-001": 3072,
    # Jina
    "jina-embeddings-v3": 1024,
    "jina-embeddings-v5-text-small": 1024,
    "jina-embeddings-v5-text-nano": 768,
    # Ollama / local
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "BAAI/bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 512,
    # Voyage (compat)
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-4-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
}

# Provider defaults
PROVIDER_DEFAULT_MODEL: Dict[str, str] = {
    "openai": "text-embedding-3-small",
    "jina": "jina-embeddings-v3",
    "gemini": "text-embedding-004",
    "ollama": "nomic-embed-text",
    "openai-compatible": "text-embedding-3-small",
}

# When dimensions can't be inferred from model id, fall back to this.
_DEFAULT_DIM_FALLBACK = 1536

_CACHE_MAX = 2000
_CACHE_EVICT = 200


class EmbeddingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Embedder(abc.ABC):
    """Provider-neutral embedding interface.

    Concrete implementations only need to implement :meth:`_embed_uncached`.
    The base class handles caching, dimension validation, and basic error
    wrapping.
    """

    def __init__(self, model: str, dimensions: int):
        self._model = model
        self._dimensions = int(dimensions)
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    @abc.abstractmethod
    def _embed_uncached(self, text: str) -> List[float]:
        """Provider-specific HTTP call. Returns a single embedding vector."""

    def embed(self, text: str) -> List[float]:
        if not text:
            text = " "
        # Trim defensively to avoid sending megabytes through provider APIs.
        text = text[:8000]
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        vec = self._embed_uncached(text)
        if not isinstance(vec, list):
            raise EmbeddingError(f"Embedding from {self._model} is not a list")
        if len(vec) != self._dimensions:
            raise EmbeddingError(
                f"Embedding dimension mismatch from {self._model}: "
                f"expected {self._dimensions}, got {len(vec)}"
            )
        with self._lock:
            if len(self._cache) > _CACHE_MAX:
                for k in list(self._cache.keys())[:_CACHE_EVICT]:
                    del self._cache[k]
            self._cache[key] = vec
        return vec


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-* and OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            raise EmbeddingError("OpenAI embedder requires api_key")
        dim = dimensions or EMBEDDING_DIMENSIONS.get(model, _DEFAULT_DIM_FALLBACK)
        super().__init__(model, dim)
        self._api_key = api_key
        self._base_url = base_url

    def _embed_uncached(self, text: str) -> List[float]:
        if self._base_url:
            return self._embed_via_httpx(text)
        # Use the OpenAI SDK when no custom base_url — preserves existing
        # behaviour for the default OpenAI provider.
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)
            kwargs = {"model": self._model, "input": text}
            # Only request dimensions for models that support it (3-* family).
            if self._model.startswith("text-embedding-3") and self._dimensions in (256, 512, 1024, 1536, 3072):
                kwargs["dimensions"] = self._dimensions
            resp = client.embeddings.create(**kwargs)
            return list(resp.data[0].embedding)
        except Exception as exc:
            # Fall back to httpx if the SDK is unavailable in some test envs.
            logger.debug("OpenAI SDK path failed (%s); retrying via httpx", exc)
            return self._embed_via_httpx(text)

    def _embed_via_httpx(self, text: str) -> List[float]:
        import httpx

        url = (self._base_url or "https://api.openai.com/v1").rstrip("/") + "/embeddings"
        payload = {"model": self._model, "input": text}
        if self._model.startswith("text-embedding-3"):
            payload["dimensions"] = self._dimensions
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        if resp.status_code >= 400:
            raise EmbeddingError(f"OpenAI-compatible embed failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        try:
            return list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(f"OpenAI-compatible response malformed: {data}") from exc


# ---------------------------------------------------------------------------
# Jina
# ---------------------------------------------------------------------------


class JinaEmbedder(Embedder):
    """Jina embeddings v3/v5.

    Uses the OpenAI-shaped ``/v1/embeddings`` endpoint at
    ``https://api.jina.ai`` and supports the Jina-specific ``task`` and
    ``normalized`` fields.
    """

    DEFAULT_BASE = "https://api.jina.ai/v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        task: Optional[str] = None,
        normalized: bool = True,
    ):
        if not api_key:
            raise EmbeddingError("Jina embedder requires api_key")
        dim = dimensions or EMBEDDING_DIMENSIONS.get(model, 1024)
        super().__init__(model, dim)
        self._api_key = api_key
        self._base = (base_url or self.DEFAULT_BASE).rstrip("/")
        self._task = task or "retrieval.passage"
        self._normalized = normalized

    def _embed_uncached(self, text: str) -> List[float]:
        import httpx

        url = self._base + "/embeddings"
        payload = {
            "model": self._model,
            "input": [text],
            "task": self._task,
            "normalized": self._normalized,
            "dimensions": self._dimensions,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        if resp.status_code >= 400:
            raise EmbeddingError(f"Jina embed failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        try:
            return list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(f"Jina response malformed: {data}") from exc


# ---------------------------------------------------------------------------
# Gemini (Google Generative Language)
# ---------------------------------------------------------------------------


class GeminiEmbedder(Embedder):
    """Google Generative Language embeddings via REST API.

    Uses the v1beta REST endpoint
    ``https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent``.
    """

    DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        if not api_key:
            raise EmbeddingError("Gemini embedder requires api_key")
        dim = dimensions or EMBEDDING_DIMENSIONS.get(model, 768)
        super().__init__(model, dim)
        self._api_key = api_key
        self._base = (base_url or self.DEFAULT_BASE).rstrip("/")
        self._task_type = task_type

    def _embed_uncached(self, text: str) -> List[float]:
        import httpx

        # Strip any "models/" prefix for URL construction.
        model_name = self._model.split("/")[-1]
        url = f"{self._base}/models/{model_name}:embedContent"
        params = {"key": self._api_key}
        payload = {
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": self._task_type,
        }
        resp = httpx.post(url, params=params, json=payload, timeout=30.0)
        if resp.status_code >= 400:
            raise EmbeddingError(f"Gemini embed failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        try:
            return list(data["embedding"]["values"])
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Gemini response malformed: {data}") from exc


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaEmbedder(Embedder):
    """Local Ollama server using ``/api/embeddings``.

    Ollama's OpenAI-compat ``/v1/embeddings`` endpoint has a known bug
    (returns empty arrays) — we always use the native ``/api/embeddings``
    endpoint instead.
    """

    DEFAULT_BASE = "http://127.0.0.1:11434"

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        # Resolve dim — Ollama doesn't return it ahead of time, so we may
        # need to probe (defer to first call if not set and not in table).
        dim = dimensions or EMBEDDING_DIMENSIONS.get(model, _DEFAULT_DIM_FALLBACK)
        super().__init__(model, dim)
        self._base = (base_url or self.DEFAULT_BASE).rstrip("/").rstrip("/v1")

    def _embed_uncached(self, text: str) -> List[float]:
        import httpx

        url = self._base + "/api/embeddings"
        payload = {"model": self._model, "prompt": text}
        resp = httpx.post(url, json=payload, timeout=30.0)
        if resp.status_code >= 400:
            raise EmbeddingError(f"Ollama embed failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        try:
            return list(data["embedding"])
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Ollama response malformed: {data}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_dimensions(provider: str, model: str, override: Optional[int]) -> int:
    if override and override > 0:
        return int(override)
    dim = EMBEDDING_DIMENSIONS.get(model)
    if dim is not None:
        return dim
    # Provider-typical fallback
    fallback = {
        "openai": 1536,
        "openai-compatible": 1536,
        "jina": 1024,
        "gemini": 768,
        "ollama": 768,
    }.get(provider, _DEFAULT_DIM_FALLBACK)
    logger.debug(
        "Unknown embedding model '%s' for provider '%s'; defaulting to %d dims (override via LANCEDB_EMBED_DIM)",
        model,
        provider,
        fallback,
    )
    return fallback


def make_embedder(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> Embedder:
    """Build an :class:`Embedder` for the configured provider.

    Resolution order for each parameter:

    1. Explicit argument
    2. Provider-specific env var (``OPENAI_API_KEY`` etc.)
    3. Generic env var (``LANCEDB_EMBED_API_KEY`` / ``LANCEDB_EMBED_MODEL`` ...)
    4. Provider default
    """
    provider = (provider or os.environ.get("LANCEDB_EMBED_PROVIDER") or "openai").lower().strip()
    if provider == "openai_compatible":
        provider = "openai-compatible"

    if provider not in PROVIDER_DEFAULT_MODEL:
        raise EmbeddingError(
            f"Unknown embedding provider: {provider!r}. "
            f"Supported: {sorted(PROVIDER_DEFAULT_MODEL.keys())}"
        )

    model = model or os.environ.get("LANCEDB_EMBED_MODEL") or PROVIDER_DEFAULT_MODEL[provider]
    base_url = base_url or os.environ.get("LANCEDB_EMBED_BASE_URL") or None
    dim_override = dimensions
    if dim_override is None:
        env_dim = os.environ.get("LANCEDB_EMBED_DIM")
        if env_dim:
            try:
                dim_override = int(env_dim)
            except ValueError:
                logger.warning("Invalid LANCEDB_EMBED_DIM=%r — ignoring", env_dim)
                dim_override = None

    dim = _resolve_dimensions(provider, model, dim_override)

    if provider == "openai":
        key = api_key or os.environ.get("LANCEDB_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        return OpenAIEmbedder(key, model, dimensions=dim, base_url=base_url)

    if provider == "openai-compatible":
        key = api_key or os.environ.get("LANCEDB_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-noop"
        if not base_url:
            raise EmbeddingError(
                "openai-compatible provider requires LANCEDB_EMBED_BASE_URL"
            )
        return OpenAIEmbedder(key, model, dimensions=dim, base_url=base_url)

    if provider == "jina":
        key = api_key or os.environ.get("LANCEDB_EMBED_API_KEY") or os.environ.get("JINA_API_KEY") or ""
        return JinaEmbedder(key, model, dimensions=dim, base_url=base_url)

    if provider == "gemini":
        key = api_key or os.environ.get("LANCEDB_EMBED_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        return GeminiEmbedder(key, model, dimensions=dim, base_url=base_url)

    if provider == "ollama":
        # Ollama doesn't require an API key; base URL falls back to env var.
        ollama_base = base_url or os.environ.get("OLLAMA_BASE_URL") or OllamaEmbedder.DEFAULT_BASE
        return OllamaEmbedder(model, base_url=ollama_base, dimensions=dim)

    # Unreachable
    raise EmbeddingError(f"Unhandled provider: {provider}")


def get_provider_from_env() -> str:
    return (os.environ.get("LANCEDB_EMBED_PROVIDER") or "openai").lower().strip()


def is_provider_available(provider: str) -> bool:
    """Best-effort availability check for `is_available()` in the plugin."""
    provider = provider.lower().strip()
    if provider in ("openai", "openai-compatible"):
        return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("LANCEDB_EMBED_API_KEY"))
    if provider == "jina":
        return bool(os.environ.get("JINA_API_KEY") or os.environ.get("LANCEDB_EMBED_API_KEY"))
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("LANCEDB_EMBED_API_KEY"))
    if provider == "ollama":
        # Ollama always considered available when chosen — runtime check only.
        return True
    return False


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Embedder",
    "EmbeddingError",
    "GeminiEmbedder",
    "JinaEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "PROVIDER_DEFAULT_MODEL",
    "get_provider_from_env",
    "is_provider_available",
    "make_embedder",
]
