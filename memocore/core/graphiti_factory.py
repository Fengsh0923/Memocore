"""
Memocore Graphiti Factory — inject configurable LLM Client and Embedder.

LLM Provider (MEMOCORE_LLM_PROVIDER):
  anthropic  — graphiti built-in AnthropicClient (requires ANTHROPIC_API_KEY)
  openai     — graphiti built-in OpenAIClient (requires OPENAI_API_KEY)
  auto       — prefer Anthropic, fallback to OpenAI (default)

Embed Provider (MEMOCORE_EMBED_PROVIDER):
  openai     — OpenAIEmbedder (default, requires OPENAI_API_KEY)
  local      — local embedding (fastembed or sentence-transformers, no API key)
  auto       — use openai if OPENAI_API_KEY is set, otherwise local

Usage:
    from memocore.core.graphiti_factory import build_graphiti
    graphiti = await build_graphiti(uri=..., user=..., password=...)
"""

import logging
import os
from typing import Optional

from memocore.core.config import get_neo4j_config

logger = logging.getLogger("memocore.graphiti_factory")


def _llm_provider() -> str:
    explicit = os.getenv("MEMOCORE_LLM_PROVIDER", "auto").lower()
    if explicit == "auto":
        return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    return explicit


def _embed_provider() -> str:
    explicit = os.getenv("MEMOCORE_EMBED_PROVIDER", "auto").lower()
    if explicit == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "local"
    return explicit


def _build_llm_client():
    """
    Return LLMClient instance for Graphiti based on MEMOCORE_LLM_PROVIDER.
    Graphiti natively supports AnthropicClient and OpenAIClient.
    """
    provider = _llm_provider()

    if provider == "anthropic":
        try:
            from graphiti_core.llm_client.anthropic_client import AnthropicClient
            from graphiti_core.llm_client.config import LLMConfig
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("MEMOCORE_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
            model = os.getenv("MEMOCORE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            logger.info(f"[graphiti_factory] LLM client: AnthropicClient model={model}")
            return AnthropicClient(config=LLMConfig(api_key=api_key, model=model))
        except ImportError:
            logger.warning("[graphiti_factory] AnthropicClient unavailable, falling back to OpenAI")

    # OpenAI (default)
    try:
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.llm_client.config import LLMConfig
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and no LLM provider is available")
        model = os.getenv("MEMOCORE_OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"[graphiti_factory] LLM client: OpenAIClient model={model}")
        return OpenAIClient(config=LLMConfig(api_key=api_key, model=model))
    except ImportError as e:
        raise RuntimeError(f"Cannot initialize LLM client: {e}")


def _build_embedder():
    """
    Return EmbedderClient instance for Graphiti based on MEMOCORE_EMBED_PROVIDER.
    """
    provider = _embed_provider()

    if provider == "local":
        from memocore.core.embedder import LocalEmbedder
        logger.info("[graphiti_factory] Embedder: LocalEmbedder (local inference, no API key needed)")
        return LocalEmbedder()

    # OpenAI (default)
    try:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("[graphiti_factory] OPENAI_API_KEY not set, switching to local Embedder")
            from memocore.core.embedder import LocalEmbedder
            return LocalEmbedder()
        model = os.getenv("MEMOCORE_EMBED_MODEL", "text-embedding-3-small")
        logger.info(f"[graphiti_factory] Embedder: OpenAIEmbedder model={model}")
        return OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=api_key, embedding_model=model))
    except ImportError:
        logger.warning("[graphiti_factory] OpenAIEmbedder unavailable, falling back to local Embedder")
        from memocore.core.embedder import LocalEmbedder
        return LocalEmbedder()


async def build_graphiti(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    build_indices: bool = False,
) -> "Graphiti":
    """
    Create a Graphiti instance with configurable LLM client and Embedder.

    Args:
        uri, user, password: Neo4j connection params (reads from config if None)
        build_indices: if True, call build_indices_and_constraints() (first init)

    Returns:
        Configured Graphiti instance
    """
    from graphiti_core import Graphiti

    cfg = get_neo4j_config()
    neo4j_uri = uri or cfg["uri"]
    neo4j_user = user or cfg["user"]
    neo4j_password = password or cfg["password"]

    llm_client = _build_llm_client()
    embedder = _build_embedder()

    graphiti = Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
    )

    if build_indices:
        await graphiti.build_indices_and_constraints()

    return graphiti
