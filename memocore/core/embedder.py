"""
Memocore Local Embedder — no OpenAI API Key required.

Implements the graphiti_core EmbedderClient interface.
Two backends (auto-selected by availability):

  1. fastembed  — preferred (lightweight, ~50MB model, pure Python)
     pip install fastembed

  2. sentence-transformers — fallback (full-featured, requires torch)
     pip install sentence-transformers

Default model: BAAI/bge-small-en-v1.5 (384-dim, bilingual)
Override via: MEMOCORE_LOCAL_EMBED_MODEL env var
"""

import logging
import os
import threading
from typing import Iterable

logger = logging.getLogger("memocore.embedder")

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    """
    Local Embedder implementing the graphiti EmbedderClient interface.
    Prefers fastembed; falls back to sentence-transformers if unavailable.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("MEMOCORE_LOCAL_EMBED_MODEL", _DEFAULT_MODEL)
        self._backend: str | None = None
        self._model = None
        self._dim: int = 384  # updated after model load
        self._load_lock = threading.Lock()

    def _load(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            # try fastembed
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
                self._backend = "fastembed"
                # Detect embedding dimension
                try:
                    test_vec = list(self._model.embed(["test"]))[0]
                    self._dim = len(test_vec)
                except Exception:
                    pass
                logger.info(f"[embedder] fastembed backend, model={self.model_name}, dim={self._dim}")
                return
            except ImportError:
                logger.debug("[embedder] fastembed not installed, trying sentence-transformers")

            # try sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence_transformers"
                try:
                    self._dim = self._model.get_sentence_embedding_dimension()
                except Exception:
                    pass
                logger.info(f"[embedder] sentence-transformers backend, model={self.model_name}, dim={self._dim}")
                return
            except ImportError:
                pass

            raise RuntimeError(
                "Local embedding requires one of:\n"
                "  pip install fastembed                # recommended (lightweight)\n"
                "  pip install sentence-transformers    # fallback (requires torch)\n"
                "Or set MEMOCORE_EMBED_PROVIDER=openai with OPENAI_API_KEY"
            )

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        """Implement EmbedderClient.create — generate embedding for a single text."""
        self._load()

        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            raise TypeError("Expected string input, got token ID list")

        import asyncio
        loop = asyncio.get_running_loop()

        if self._backend == "fastembed":
            embeddings = await loop.run_in_executor(
                None, lambda: list(self._model.embed([text]))
            )
            return embeddings[0].tolist()
        else:
            vec = await loop.run_in_executor(
                None, lambda: self._model.encode(text, normalize_embeddings=True)
            )
            return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch embedding for improved throughput on multi-text writes."""
        self._load()

        import asyncio
        loop = asyncio.get_running_loop()

        if self._backend == "fastembed":
            embeddings = await loop.run_in_executor(
                None, lambda: list(self._model.embed(input_data_list))
            )
            return [e.tolist() for e in embeddings]
        else:
            vecs = await loop.run_in_executor(
                None,
                lambda: self._model.encode(input_data_list, normalize_embeddings=True),
            )
            return [v.tolist() for v in vecs]
