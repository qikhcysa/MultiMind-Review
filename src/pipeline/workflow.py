"""Three-stage review analysis workflow orchestrating all agents."""
from __future__ import annotations

import uuid
from typing import Any

from src.models import (
    ProductInfo,
    Dimension,
    ReviewAnalysisResult,
    AuditEntry,
)
from src.rag import EmbeddingModel, VectorStore
from src.agents import (
    ProductRecognitionAgent,
    DimensionDetectionAgent,
    EvidenceRetrievalAgent,
    SentimentScoringAgent,
)
from src.audit import AuditTrail


# Sentiment classification thresholds (score range 1–5)
_POSITIVE_THRESHOLD = 4.0
_NEGATIVE_THRESHOLD = 2.5


class ReviewAnalysisPipeline:
    """
    Orchestrates the three-stage multi-agent review analysis workflow:

    Stage 1 – Product Recognition:  identify the product being reviewed.
    Stage 2 – Dimension Detection:  identify which evaluation dimensions are present.
    Stage 3 – Evidence & Scoring:   retrieve evidence per dimension and score sentiment.
    """

    def __init__(
        self,
        products: list[ProductInfo],
        dimensions: list[Dimension],
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        audit_trail: AuditTrail | None = None,
        use_llm: bool = True,
        top_k_products: int = 5,
        similarity_threshold: float = 0.5,
        evidence_top_k: int = 3,
    ) -> None:
        self.products = products
        self.dimensions = dimensions

        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.audit_trail = audit_trail or AuditTrail()

        # Initialise agents
        self.product_agent = ProductRecognitionAgent(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=top_k_products,
            similarity_threshold=similarity_threshold,
        )
        self.dimension_agent = DimensionDetectionAgent(
            dimensions=dimensions,
            use_llm=use_llm,
        )
        self.evidence_agent = EvidenceRetrievalAgent(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=evidence_top_k,
        )
        self.scoring_agent = SentimentScoringAgent(
            dimensions=dimensions,
            use_llm=use_llm,
        )

        # Index products on first run
        self._products_indexed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Index products and prepare the pipeline (idempotent)."""
        if not self._products_indexed and self.products:
            self.product_agent.index_products(self.products)
            self._products_indexed = True

    def analyze(self, review_text: str, review_id: str | None = None) -> ReviewAnalysisResult:
        """
        Run the full three-stage analysis on a single review.

        Args:
            review_text: The raw review text.
            review_id:   Optional review identifier; auto-generated if omitted.

        Returns:
            A :class:`ReviewAnalysisResult` with full analysis and audit trail.
        """
        if not self._products_indexed:
            self.setup()

        if review_id is None:
            review_id = str(uuid.uuid4())

        all_audit_entries: list[AuditEntry] = []

        # ---- Stage 1: Product Recognition --------------------------------
        product_match, prod_audit = self.product_agent.recognize(review_text, review_id)
        all_audit_entries.append(prod_audit)

        # ---- Stage 2: Dimension Detection --------------------------------
        detected_dim_ids, dim_audit = self.dimension_agent.detect(review_text, review_id)
        all_audit_entries.append(dim_audit)

        detected_dims = [d for d in self.dimensions if d.id in detected_dim_ids]

        # ---- Stage 3a: Evidence Retrieval --------------------------------
        evidence_map: dict[str, list[str]] = {}
        if detected_dims:
            evidence_map, ev_audits = self.evidence_agent.retrieve(
                review_text, review_id, detected_dims
            )
            all_audit_entries.extend(ev_audits)

        # ---- Stage 3b: Sentiment Scoring ---------------------------------
        dim_scores = []
        score_audits: list[AuditEntry] = []
        if evidence_map:
            dim_scores, score_audits = self.scoring_agent.score(review_id, evidence_map)
            all_audit_entries.extend(score_audits)

        # ---- Aggregate overall score --------------------------------------
        overall_score = None
        overall_sentiment = "neutral"
        if dim_scores:
            overall_score = round(sum(s.score for s in dim_scores) / len(dim_scores), 2)
            if overall_score >= _POSITIVE_THRESHOLD:
                overall_sentiment = "positive"
            elif overall_score <= _NEGATIVE_THRESHOLD:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

        # ---- Persist audit trail -----------------------------------------
        for entry in all_audit_entries:
            self.audit_trail.record(entry)

        return ReviewAnalysisResult(
            review_id=review_id,
            review_text=review_text,
            product_match=product_match,
            detected_dimensions=detected_dim_ids,
            dimension_scores=dim_scores,
            overall_sentiment=overall_sentiment,
            overall_score=overall_score,
        )

    def analyze_batch(
        self, reviews: list[str], review_ids: list[str] | None = None
    ) -> list[ReviewAnalysisResult]:
        """Analyze a batch of reviews."""
        if not self._products_indexed:
            self.setup()
        if review_ids is None:
            review_ids = [str(uuid.uuid4()) for _ in reviews]
        return [
            self.analyze(text, rid) for text, rid in zip(reviews, review_ids)
        ]

    def update_products(self, products: list[ProductInfo]) -> None:
        """Replace the product knowledge base and re-index."""
        self.products = products
        self.product_agent.index_products(products)
        self._products_indexed = True

    def update_dimensions(self, dimensions: list[Dimension]) -> None:
        """Update the active dimensions."""
        self.dimensions = dimensions
        self.dimension_agent.dimensions = {d.id: d for d in dimensions}
        self.scoring_agent.dimensions = {d.id: d for d in dimensions}
