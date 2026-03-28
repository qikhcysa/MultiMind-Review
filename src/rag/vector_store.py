"""Vector store management using ChromaDB for product and review collections."""
from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np


class VectorStore:
    """Manages ChromaDB collections for product knowledge base and review storage."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def get_or_create_collection(self, name: str):
        """Get or create a named ChromaDB collection."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents with embeddings into a collection."""
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{} for _ in ids],
        )

    def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query a collection and return top-k results with metadata and distances."""
        collection = self.get_or_create_collection(collection_name)
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                }
            )
        return output

    def count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""
        client = self._get_client()
        client.delete_collection(collection_name)

    def list_collections(self) -> list[str]:
        """List all collection names."""
        client = self._get_client()
        return [c.name for c in client.list_collections()]
