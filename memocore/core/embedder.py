"""
本地 Embedding 实现 — 无需 OpenAI API Key

实现 graphiti_core.embedder.client.EmbedderClient 接口，
支持两种后端（按可用性自动选择）：

  1. fastembed  — 首选（轻量，~50MB 模型，纯 Python）
     pip install fastembed
     默认模型: BAAI/bge-small-en-v1.5（384 维，支持中英双语）

  2. sentence-transformers — 备选（功能全，但依赖 torch，体积大）
     pip install sentence-transformers
     默认模型: BAAI/bge-small-en-v1.5

环境变量:
  MEMOCORE_LOCAL_EMBED_MODEL  — 模型名称
                                fastembed 默认: BAAI/bge-small-en-v1.5
                                sentence-transformers 默认: BAAI/bge-small-en-v1.5
"""

import logging
import os
from typing import Iterable

logger = logging.getLogger("memocore.embedder")

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    """
    本地 Embedding，实现 graphiti EmbedderClient 接口。
    优先使用 fastembed，不可用时 fallback 到 sentence-transformers。
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("MEMOCORE_LOCAL_EMBED_MODEL", _DEFAULT_MODEL)
        self._backend: str | None = None
        self._model = None

    def _load(self):
        """延迟加载模型（首次调用时初始化）"""
        if self._model is not None:
            return

        # 尝试 fastembed
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            self._backend = "fastembed"
            logger.info(f"[embedder] 使用 fastembed backend，model={self.model_name}")
            return
        except ImportError:
            logger.debug("[embedder] fastembed 未安装，尝试 sentence-transformers")

        # 尝试 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
            logger.info(f"[embedder] 使用 sentence-transformers backend，model={self.model_name}")
            return
        except ImportError:
            pass

        raise RuntimeError(
            "本地 embedding 需要安装依赖：\n"
            "  pip install fastembed          # 推荐（轻量）\n"
            "  pip install sentence-transformers  # 备选（需要 torch）\n"
            "或者设置 MEMOCORE_EMBED_PROVIDER=openai 并配置 OPENAI_API_KEY"
        )

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        """实现 EmbedderClient.create — 对单条文本生成 embedding"""
        self._load()

        # 统一转为字符串
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            # 字符串列表 → 取第一条（graphiti 通常单条调用）
            text = input_data[0]
        else:
            # int token list → 无法处理，返回零向量（防止崩溃）
            logger.warning("[embedder] 收到 token id 列表，无法处理，返回零向量")
            return [0.0] * 384

        if self._backend == "fastembed":
            embeddings = list(self._model.embed([text]))
            return embeddings[0].tolist()
        else:
            # sentence-transformers
            import asyncio
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(
                None, lambda: self._model.encode(text, normalize_embeddings=True)
            )
            return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """批量 embedding（可选实现，提升多条文本写入效率）"""
        self._load()

        if self._backend == "fastembed":
            embeddings = list(self._model.embed(input_data_list))
            return [e.tolist() for e in embeddings]
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            vecs = await loop.run_in_executor(
                None,
                lambda: self._model.encode(input_data_list, normalize_embeddings=True),
            )
            return [v.tolist() for v in vecs]
