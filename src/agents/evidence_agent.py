"""Evidence Retrieval Agent: retrieves relevant review sentences per dimension."""
from __future__ import annotations

import uuid
from typing import Any

from src.models import Dimension, AuditEntry
from src.rag import EmbeddingModel, VectorStore

REVIEW_COLLECTION = "review_sentences"


class EvidenceRetrievalAgent:
    """
    For each detected dimension, retrieves the most relevant sentences/segments
    from the review text using vector similarity search.

    Sentences are indexed on-the-fly into a per-review ephemeral collection and
    queried using the dimension's description as the query.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 3,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.name = "EvidenceRetrievalAgent"

    def retrieve(
        self,
        review_text: str,
        review_id: str,
        dimensions: list[Dimension],
    ) -> tuple[dict[str, list[str]], list[AuditEntry]]:
        """
        Retrieve evidence for each dimension.

        Returns:
            evidence_map: {dimension_id: [evidence_sentence, ...]}
            audit_entries: one AuditEntry per dimension
        """
        sentences = self._split_sentences(review_text)
        collection_name = f"review_{review_id}"

        # Index sentences into a temporary collection
        if sentences:
            sent_embs = self.embedding_model.encode(sentences)
            self.vector_store.upsert(
                collection_name,
                ids=[f"s{i}" for i in range(len(sentences))],
                embeddings=[e.tolist() for e in sent_embs],
                documents=sentences,
                metadatas=[{"idx": i} for i in range(len(sentences))],
            )

        evidence_map: dict[str, list[str]] = {}
        audit_entries: list[AuditEntry] = []

        for dim in dimensions:
            query = f"{dim.name} {dim.description}"
            query_emb = self.embedding_model.encode_single(query)

            if not sentences:
                evidence_map[dim.id] = [review_text]
                audit_entries.append(
                    self._build_audit(review_id, dim, [review_text], [], [])
                )
                continue

            neighbors = self.vector_store.query(
                collection_name,
                query_embeddings=[query_emb.tolist()],
                top_k=min(self.top_k, len(sentences)),
            )
            evidence = [n["document"] for n in neighbors]
            sims = [round(n["similarity"], 4) for n in neighbors]

            evidence_map[dim.id] = evidence
            audit_entries.append(
                self._build_audit(review_id, dim, evidence, neighbors, sims)
            )

        # Clean up ephemeral collection
        try:
            self.vector_store.delete_collection(collection_name)
        except Exception:
            pass

        return evidence_map, audit_entries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter for Chinese + English text."""
        import re

        parts = re.split(r"[。！？；\n.!?;]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_audit(
        self,
        review_id: str,
        dim: Dimension,
        evidence: list[str],
        neighbors: list[Any],
        sims: list[float],
    ) -> AuditEntry:
        return AuditEntry(
            entry_id=str(uuid.uuid4()),
            review_id=review_id,
            stage="evidence_retrieval",
            agent_name=self.name,
            input_data={"dimension_id": dim.id, "dimension_name": dim.name},
            output_data={"evidence": evidence},
            retrieved_neighbors=[
                {"id": n["id"], "document": n["document"], "similarity": round(n["similarity"], 4)}
                for n in neighbors
            ],
            similarities=sims,
            reasoning=f"Retrieved top-{len(evidence)} sentences for dimension '{dim.name}'.",
        )
