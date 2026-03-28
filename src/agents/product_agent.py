"""Product Recognition Agent: identifies the product being reviewed via entity linking."""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.models import ProductInfo, ProductMatch, AuditEntry
from src.rag import EmbeddingModel, VectorStore


PRODUCT_COLLECTION = "products"


class ProductRecognitionAgent:
    """
    Recognises the product referenced in a user review using vector similarity
    search over the product knowledge base.

    The agent embeds the review text and retrieves the most similar product
    entry from ChromaDB.  If the top-1 similarity exceeds the configured
    threshold the product is considered matched; otherwise the review is
    treated as 'unknown product'.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.name = "ProductRecognitionAgent"

    # ------------------------------------------------------------------
    # Knowledge base management
    # ------------------------------------------------------------------

    def index_products(self, products: list[ProductInfo]) -> None:
        """Embed and index all products in the knowledge base."""
        ids, embeddings, documents, metadatas = [], [], [], []
        for product in products:
            doc = self._build_product_document(product)
            emb = self.embedding_model.encode_single(doc)
            ids.append(product.id)
            embeddings.append(emb.tolist())
            documents.append(doc)
            metadatas.append(
                {
                    "id": product.id,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "keywords": json.dumps(product.keywords, ensure_ascii=False),
                }
            )
        self.vector_store.upsert(PRODUCT_COLLECTION, ids, embeddings, documents, metadatas)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recognize(self, review_text: str, review_id: str) -> tuple[ProductMatch | None, AuditEntry]:
        """
        Recognise the product in *review_text*.

        Returns:
            A (ProductMatch | None, AuditEntry) tuple.
        """
        query_emb = self.embedding_model.encode_single(review_text)
        neighbors = self.vector_store.query(
            PRODUCT_COLLECTION,
            query_embeddings=[query_emb.tolist()],
            top_k=self.top_k,
        )

        product_match: ProductMatch | None = None
        if neighbors and neighbors[0]["similarity"] >= self.similarity_threshold:
            top = neighbors[0]
            meta = top["metadata"]
            try:
                kw = json.loads(meta.get("keywords", "[]"))
            except (json.JSONDecodeError, TypeError):
                kw = []
            product_match = ProductMatch(
                product_id=meta["id"],
                product_name=meta["name"],
                brand=meta["brand"],
                category=meta["category"],
                similarity=round(top["similarity"], 4),
                matched_keywords=kw,
            )

        audit = AuditEntry(
            entry_id=str(uuid.uuid4()),
            review_id=review_id,
            stage="product_recognition",
            agent_name=self.name,
            input_data={"review_text": review_text},
            output_data=(
                product_match.model_dump() if product_match else {"matched": False}
            ),
            retrieved_neighbors=[
                {
                    "id": n["id"],
                    "document": n["document"],
                    "similarity": round(n["similarity"], 4),
                }
                for n in neighbors
            ],
            similarities=[round(n["similarity"], 4) for n in neighbors],
            reasoning=(
                f"Top-1 similarity={neighbors[0]['similarity']:.4f} "
                f"(threshold={self.similarity_threshold}). "
                + (
                    f"Matched product: {product_match.product_name}"
                    if product_match
                    else "No product matched above threshold."
                )
                if neighbors
                else "No products in knowledge base."
            ),
        )
        return product_match, audit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_product_document(product: ProductInfo) -> str:
        parts = [
            product.name,
            product.brand,
            product.category,
            product.description,
            " ".join(product.features),
            " ".join(product.keywords),
        ]
        return " | ".join(filter(None, parts))
