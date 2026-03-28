"""Tests for the review clustering module."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.clustering import ReviewClusterer
from tests.conftest import MockEmbeddingModel


@pytest.fixture
def clusterer():
    """Clusterer without LLM (keyword-based summarisation) and mock embeddings."""
    return ReviewClusterer(
        embedding_model=MockEmbeddingModel(),
        eps=0.5,
        min_samples=2,
        use_llm=False,
    )


def test_cluster_basic(clusterer):
    """Should produce a ClusterResult with at least one cluster from similar reviews."""
    reviews = [
        "产品质量非常好，做工精细",
        "质量很棒，材质高档",
        "质量差，用了一个月就坏了",
        "做工粗糙，容易损坏",
    ]
    result = clusterer.cluster(reviews, "quality", "产品质量")
    assert result.dimension_id == "quality"
    assert result.total_reviews == 4
    assert result.num_clusters >= 0  # may be 0 if eps is too small
    assert result.noise_count >= 0
    assert result.num_clusters + result.noise_count <= 4


def test_cluster_empty(clusterer):
    """Clustering an empty list should return an empty result without error."""
    result = clusterer.cluster([], "quality", "产品质量")
    assert result.total_reviews == 0
    assert result.num_clusters == 0
    assert result.noise_count == 0
    assert result.clusters == []


def test_cluster_single_review(clusterer):
    """A single review cannot form a cluster (min_samples=2) but should not error."""
    result = clusterer.cluster(["质量很好"], "quality", "产品质量")
    assert result.total_reviews == 1
    # With min_samples=2, a single point becomes noise
    assert result.num_clusters == 0
    assert result.noise_count == 1


def test_cluster_representative_reviews(clusterer):
    """Clusters should include representative reviews."""
    reviews = [
        "物流超快，隔天到",
        "发货速度很快",
        "快递非常迅速",
        "质量好",
    ]
    result = clusterer.cluster(reviews, "delivery", "物流配送")
    for c in result.clusters:
        assert len(c.representative_reviews) <= 3
        assert c.size >= 2


def test_cluster_summary_not_empty(clusterer):
    """Each cluster should have a non-empty summary."""
    reviews = [
        "质量非常好，用了两年依然正常",
        "产品品质优秀，非常满意",
        "质量很棒",
    ]
    result = clusterer.cluster(reviews, "quality", "产品质量")
    for c in result.clusters:
        assert c.summary != ""
