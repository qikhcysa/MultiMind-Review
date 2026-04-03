"""LangGraph StateGraph-based review analysis pipeline.

Replaces the hand-coded :class:`~src.pipeline.workflow.ReviewAnalysisPipeline`
with a proper LangGraph ``StateGraph``, making data flow between analysis
stages explicit and extensible.

Graph topology::

    START
      │
      ▼
    recognize_product  ──►  detect_dimensions
                                │
                    ┌───────────┴───────────┐
              dims detected?              no dims
                    │                       │
                    ▼                       │
            retrieve_evidence               │
                    │                       │
                    ▼                       │
            score_sentiment                 │
                    │                       │
                    └────────┬──────────────┘
                             ▼
                          aggregate
                             │
                             ▼
                            END

Each node returns a partial state dict; the ``Annotated[list, operator.add]``
reducer on ``audit_entries`` accumulates entries from all nodes without
overwriting.
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.dimension_agent import DimensionDetectionAgent
from src.agents.evidence_agent import EvidenceRetrievalAgent
from src.agents.product_agent import ProductRecognitionAgent
from src.agents.scoring_agent import SentimentScoringAgent
from src.audit import AuditTrail
from src.models import (
    AuditEntry,
    Dimension,
    DimensionScore,
    ProductInfo,
    ProductMatch,
    ReviewAnalysisResult,
)
from src.rag import EmbeddingModel, VectorStore

# Sentiment classification thresholds (score range 1–5)
_POSITIVE_THRESHOLD = 4.0
_NEGATIVE_THRESHOLD = 2.5


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    """State flowing through the LangGraph analysis pipeline.

    ``audit_entries`` uses ``operator.add`` as its reducer so that every node
    can *append* entries without overwriting the list written by previous nodes.
    All other fields use the default "last-write-wins" replacement behaviour.
    """

    review_text: str
    review_id: str
    product_match: ProductMatch | None
    detected_dim_ids: list[str]
    detected_dims: list[Dimension]
    evidence_map: dict[str, list[str]]
    dim_scores: list[DimensionScore]
    audit_entries: Annotated[list[AuditEntry], operator.add]
    overall_score: float | None
    overall_sentiment: str


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class LangGraphPipeline:
    """
    LangGraph StateGraph implementation of the multi-agent review analysis
    pipeline.

    Provides the same public API as
    :class:`~src.pipeline.workflow.ReviewAnalysisPipeline` (``analyze``,
    ``analyze_batch``, ``setup``, ``update_products``, ``update_dimensions``)
    while using a compiled ``StateGraph`` for execution, making the data flow
    between the four specialist agents explicit and easy to extend.

    Usage::

        from src.pipeline.langgraph_pipeline import LangGraphPipeline
        from src.config_loader import load_products, load_dimensions

        pipeline = LangGraphPipeline(
            products=load_products(),
            dimensions=load_dimensions(),
        )
        result = pipeline.analyze("这款手机质量很好，但快递慢")
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

        # Initialise specialist agents
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

        self._products_indexed = False
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph ``StateGraph``."""
        builder: StateGraph = StateGraph(ReviewState)

        # Register nodes
        builder.add_node("recognize_product", self._node_recognize_product)
        builder.add_node("detect_dimensions", self._node_detect_dimensions)
        builder.add_node("retrieve_evidence", self._node_retrieve_evidence)
        builder.add_node("score_sentiment", self._node_score_sentiment)
        builder.add_node("aggregate", self._node_aggregate)

        # Sequential edges
        builder.add_edge(START, "recognize_product")
        builder.add_edge("recognize_product", "detect_dimensions")

        # Conditional: skip evidence/scoring if no dimensions detected
        builder.add_conditional_edges(
            "detect_dimensions",
            lambda s: "retrieve_evidence" if s["detected_dim_ids"] else "aggregate",
        )
        # Conditional: skip scoring if evidence retrieval yielded nothing
        builder.add_conditional_edges(
            "retrieve_evidence",
            lambda s: "score_sentiment" if s["evidence_map"] else "aggregate",
        )

        builder.add_edge("score_sentiment", "aggregate")
        builder.add_edge("aggregate", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _node_recognize_product(self, state: ReviewState) -> dict[str, Any]:
        """Stage 1: identify the product via vector similarity search."""
        match, audit = self.product_agent.recognize(
            state["review_text"], state["review_id"]
        )
        return {
            "product_match": match,
            "audit_entries": [audit],
        }

    def _node_detect_dimensions(self, state: ReviewState) -> dict[str, Any]:
        """Stage 2: detect which sentiment dimensions appear in the review."""
        dim_ids, audit = self.dimension_agent.detect(
            state["review_text"], state["review_id"]
        )
        detected_dims = [d for d in self.dimensions if d.id in dim_ids]
        return {
            "detected_dim_ids": dim_ids,
            "detected_dims": detected_dims,
            "audit_entries": [audit],
        }

    def _node_retrieve_evidence(self, state: ReviewState) -> dict[str, Any]:
        """Stage 3a: retrieve sentence-level evidence per dimension (RAG)."""
        evidence_map, audits = self.evidence_agent.retrieve(
            state["review_text"], state["review_id"], state["detected_dims"]
        )
        return {
            "evidence_map": evidence_map,
            "audit_entries": audits,
        }

    def _node_score_sentiment(self, state: ReviewState) -> dict[str, Any]:
        """Stage 3b: score sentiment (1–5) for each dimension's evidence."""
        dim_scores, audits = self.scoring_agent.score(
            state["review_id"], state["evidence_map"]
        )
        return {
            "dim_scores": dim_scores,
            "audit_entries": audits,
        }

    def _node_aggregate(self, state: ReviewState) -> dict[str, Any]:
        """Compute overall score / sentiment from per-dimension scores."""
        dim_scores: list[DimensionScore] = state.get("dim_scores", [])
        overall_score: float | None = None
        overall_sentiment = "neutral"
        if dim_scores:
            overall_score = round(
                sum(s.score for s in dim_scores) / len(dim_scores), 2
            )
            if overall_score >= _POSITIVE_THRESHOLD:
                overall_sentiment = "positive"
            elif overall_score <= _NEGATIVE_THRESHOLD:
                overall_sentiment = "negative"
        return {
            "overall_score": overall_score,
            "overall_sentiment": overall_sentiment,
        }

    # ------------------------------------------------------------------
    # Public API  (mirrors ReviewAnalysisPipeline)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Index products into the vector store (idempotent)."""
        if not self._products_indexed and self.products:
            self.product_agent.index_products(self.products)
            self._products_indexed = True

    def analyze(
        self, review_text: str, review_id: str | None = None
    ) -> ReviewAnalysisResult:
        """Run the full pipeline on a single review and return a structured result.

        Args:
            review_text: The raw review text.
            review_id:   Optional identifier; auto-generated when omitted.

        Returns:
            A :class:`~src.models.ReviewAnalysisResult` with full analysis.
        """
        if not self._products_indexed:
            self.setup()

        if review_id is None:
            review_id = str(uuid.uuid4())

        initial_state: ReviewState = {
            "review_text": review_text,
            "review_id": review_id,
            "product_match": None,
            "detected_dim_ids": [],
            "detected_dims": [],
            "evidence_map": {},
            "dim_scores": [],
            "audit_entries": [],
            "overall_score": None,
            "overall_sentiment": "neutral",
        }

        final_state: ReviewState = self._graph.invoke(initial_state)

        # Persist audit trail
        for entry in final_state["audit_entries"]:
            self.audit_trail.record(entry)

        return ReviewAnalysisResult(
            review_id=review_id,
            review_text=review_text,
            product_match=final_state["product_match"],
            detected_dimensions=final_state["detected_dim_ids"],
            dimension_scores=final_state["dim_scores"],
            overall_sentiment=final_state["overall_sentiment"],
            overall_score=final_state["overall_score"],
        )

    def analyze_batch(
        self,
        reviews: list[str],
        review_ids: list[str] | None = None,
    ) -> list[ReviewAnalysisResult]:
        """Analyze a batch of reviews sequentially."""
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
        """Update the active dimensions for detection and scoring."""
        self.dimensions = dimensions
        self.dimension_agent.dimensions = {d.id: d for d in dimensions}
        self.scoring_agent.dimensions = {d.id: d for d in dimensions}
