"""Data models and schemas for the review analysis system."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ProductInfo(BaseModel):
    """Represents a product in the knowledge base."""

    id: str
    name: str
    brand: str
    category: str
    description: str
    features: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class Dimension(BaseModel):
    """Represents a sentiment analysis dimension."""

    id: str
    name: str
    name_en: str
    description: str
    keywords: list[str] = Field(default_factory=list)


class DimensionScore(BaseModel):
    """Sentiment score for a specific dimension."""

    dimension_id: str
    dimension_name: str
    score: float = Field(ge=1.0, le=5.0)
    sentiment: str  # positive / neutral / negative
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ProductMatch(BaseModel):
    """Result of product recognition."""

    product_id: str
    product_name: str
    brand: str
    category: str
    similarity: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)


class ReviewAnalysisResult(BaseModel):
    """Full result of a review analysis."""

    review_id: str
    review_text: str
    product_match: ProductMatch | None = None
    detected_dimensions: list[str] = Field(default_factory=list)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    overall_sentiment: str = ""
    overall_score: float | None = None
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEntry(BaseModel):
    """Audit trail entry for a single decision step."""

    entry_id: str
    review_id: str
    stage: str  # product_recognition / dimension_detection / evidence_retrieval / scoring
    agent_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    retrieved_neighbors: list[dict[str, Any]] = Field(default_factory=list)
    similarities: list[float] = Field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClusterResult(BaseModel):
    """Result of review clustering for a dimension."""

    dimension_id: str
    dimension_name: str
    total_reviews: int
    num_clusters: int
    noise_count: int
    clusters: list[ClusterDetail] = Field(default_factory=list)


class ClusterDetail(BaseModel):
    """Details of a single review cluster."""

    cluster_id: int
    size: int
    summary: str
    representative_reviews: list[str] = Field(default_factory=list)
    avg_sentiment_score: float | None = None
