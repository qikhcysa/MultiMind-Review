"""RAG infrastructure: embeddings and vector store management."""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for generating text embeddings.
    Falls back to a deterministic hash-based embedding when the model
    cannot be loaded (e.g. in offline/sandboxed environments).
    """

    FALLBACK_DIM = 128

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._model = None
        self._fallback = False

    def _load(self) -> None:
        if self._model is not None or self._fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._fallback = True

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        self._load()
        if self._fallback:
            return self._hash_encode(texts, normalize)
        embeddings = self._model.encode(texts, normalize_embeddings=normalize)
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], normalize=normalize)[0]

    # ------------------------------------------------------------------
    # Fallback: deterministic hash-based embeddings
    # ------------------------------------------------------------------

    def _hash_encode(self, texts: list[str], normalize: bool) -> np.ndarray:
        embs = np.array([self._hash_vec(t) for t in texts], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs = embs / norms
        return embs

    def _hash_vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(
            seed=int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)
        )
        return rng.standard_normal(self.FALLBACK_DIM).astype(np.float32)

