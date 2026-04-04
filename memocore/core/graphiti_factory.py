"""
Graphiti 工厂模块 — 注入可配置的 LLM Client 和 Embedder

解决的问题：
  Graphiti 默认全部走 OpenAI（LLM + Embedding），没有 OpenAI key 时系统完全不可用。

支持的组合：
  LLM Provider (MEMOCORE_LLM_PROVIDER):
    anthropic  — 使用 graphiti 内置 AnthropicClient（需要 ANTHROPIC_API_KEY）
    openai     — 使用 graphiti 内置 OpenAIClient（需要 OPENAI_API_KEY）
    auto       — 优先 Anthropic，fallback OpenAI（默认）

  Embed Provider (MEMOCORE_EMBED_PROVIDER):
    openai     — OpenAIEmbedder（默认，需要 OPENAI_API_KEY）
    local      — 本地 embedding（fastembed 或 sentence-transformers，无需 API key）
    auto       — 有 OPENAI_API_KEY 用 openai，否则 local

用法：
    from memocore.core.graphiti_factory import build_graphiti
    graphiti = await build_graphiti(uri=..., user=..., password=...)
"""

import logging
import os
from typing import Optional

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
    根据 MEMOCORE_LLM_PROVIDER 返回 Graphiti 用的 LLMClient 实例。
    Graphiti 原生支持 AnthropicClient 和 OpenAIClient。
    """
    provider = _llm_provider()

    if provider == "anthropic":
        try:
            from graphiti_core.llm_client.anthropic_client import AnthropicClient
            from graphiti_core.llm_client.config import LLMConfig
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("MEMOCORE_LLM_PROVIDER=anthropic 但 ANTHROPIC_API_KEY 未设置")
            model = os.getenv("MEMOCORE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            logger.info(f"[graphiti_factory] LLM client: AnthropicClient model={model}")
            return AnthropicClient(config=LLMConfig(api_key=api_key, model=model))
        except ImportError:
            logger.warning("[graphiti_factory] AnthropicClient 不可用，fallback 到 OpenAI")

    # OpenAI (default)
    try:
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.llm_client.config import LLMConfig
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置，且无可用的 LLM provider")
        model = os.getenv("MEMOCORE_OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"[graphiti_factory] LLM client: OpenAIClient model={model}")
        return OpenAIClient(config=LLMConfig(api_key=api_key, model=model))
    except ImportError as e:
        raise RuntimeError(f"无法初始化 LLM client: {e}")


def _build_embedder():
    """
    根据 MEMOCORE_EMBED_PROVIDER 返回 Graphiti 用的 EmbedderClient 实例。
    """
    provider = _embed_provider()

    if provider == "local":
        from memocore.core.embedder import LocalEmbedder
        logger.info("[graphiti_factory] Embedder: LocalEmbedder（本地推理，无需 API key）")
        return LocalEmbedder()

    # OpenAI (default)
    try:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("[graphiti_factory] OPENAI_API_KEY 未设置，自动切换到本地 Embedder")
            from memocore.core.embedder import LocalEmbedder
            return LocalEmbedder()
        model = os.getenv("MEMOCORE_EMBED_MODEL", "text-embedding-3-small")
        logger.info(f"[graphiti_factory] Embedder: OpenAIEmbedder model={model}")
        return OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=api_key, embedding_model=model))
    except ImportError:
        logger.warning("[graphiti_factory] OpenAIEmbedder 不可用，fallback 到本地 Embedder")
        from memocore.core.embedder import LocalEmbedder
        return LocalEmbedder()


async def build_graphiti(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    build_indices: bool = False,
) -> "Graphiti":
    """
    创建注入了可配置 LLM client 和 Embedder 的 Graphiti 实例

    Args:
        uri, user, password: Neo4j 连接参数（None 时从环境变量读取）
        build_indices: True 时调用 build_indices_and_constraints()（首次初始化时用）

    Returns:
        配置好的 Graphiti 实例
    """
    from graphiti_core import Graphiti

    neo4j_uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = password or os.getenv("NEO4J_PASSWORD", "")

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
