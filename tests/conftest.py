"""Shared test utilities: mock embedding model for offline testing."""
from __future__ import annotations

import hashlib
import numpy as np

from src.rag import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """
    A deterministic embedding model that does NOT require network access.
    Embeddings are produced by hashing the text and projecting to a fixed-dim vector.
    """

    DIM = 64

    def __init__(self) -> None:
        # Skip parent __init__ (avoids loading model_name from env)
        self.model_name = "mock"
        self._model = None

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        embs = np.array([self._hash_embed(t) for t in texts], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs = embs / norms
        return embs

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        """Produce a pseudo-random but deterministic vector from text."""
        rng = np.random.default_rng(
            seed=int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)
        )
        return rng.standard_normal(self.DIM).astype(np.float32)
