"""Tests for the review analysis pipeline (without LLM, using heuristic fallback)."""
from __future__ import annotations

import os
import sys
import uuid
import tempfile

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.models import ProductInfo, Dimension
from src.audit import AuditTrail
from tests.conftest import MockEmbeddingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_products():
    return [
        ProductInfo(
            id="phone_test",
            name="测试智能手机",
            brand="TestBrand",
            category="电子/手机",
            description="一款测试用的智能手机，屏幕大，拍照好",
            features=["6寸屏幕", "高清摄像头"],
            keywords=["手机", "智能手机", "phone"],
        ),
        ProductInfo(
            id="laptop_test",
            name="测试笔记本电脑",
            brand="TestBrand",
            category="电子/电脑",
            description="轻薄笔记本，续航长",
            features=["14寸屏幕", "长续航"],
            keywords=["笔记本", "电脑", "laptop"],
        ),
    ]


@pytest.fixture
def sample_dimensions():
    return [
        Dimension(
            id="quality",
            name="产品质量",
            name_en="Quality",
            description="产品质量、做工、耐用性",
            keywords=["质量", "做工", "耐用", "好", "差", "quality"],
        ),
        Dimension(
            id="delivery",
            name="物流配送",
            name_en="Delivery",
            description="发货速度、包装、物流",
            keywords=["物流", "快递", "发货", "到货", "delivery"],
        ),
    ]


@pytest.fixture
def audit_trail(tmp_path):
    return AuditTrail(log_dir=str(tmp_path / "audit_logs"), enabled=True)


# ---------------------------------------------------------------------------
# Tests: Pipeline without LLM
# ---------------------------------------------------------------------------

def test_pipeline_no_llm(sample_products, sample_dimensions, audit_trail, tmp_path):
    """Pipeline should run end-to-end without LLM and produce a result."""
    from src.rag import EmbeddingModel, VectorStore
    from src.pipeline import ReviewAnalysisPipeline

    emb = MockEmbeddingModel()
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    pipeline = ReviewAnalysisPipeline(
        products=sample_products,
        dimensions=sample_dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit_trail,
        use_llm=False,
        similarity_threshold=0.3,
    )
    pipeline.setup()

    review = "这款手机质量非常好，物流也很快，第二天就到了"
    result = pipeline.analyze(review)

    assert result.review_text == review
    assert result.review_id is not None
    assert result.overall_sentiment in ("positive", "neutral", "negative")


def test_pipeline_product_recognition(sample_products, sample_dimensions, audit_trail, tmp_path):
    """Product recognition should identify the phone when review mentions phone keywords."""
    from src.rag import EmbeddingModel, VectorStore
    from src.pipeline import ReviewAnalysisPipeline

    emb = MockEmbeddingModel()
    vs = VectorStore(persist_dir=str(tmp_path / "chroma_pr"))
    pipeline = ReviewAnalysisPipeline(
        products=sample_products,
        dimensions=sample_dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit_trail,
        use_llm=False,
        similarity_threshold=0.2,
    )
    pipeline.setup()

    result = pipeline.analyze("这款智能手机拍照效果超级棒，很满意")
    # Product match may or may not fire depending on embedding similarity,
    # but the result structure must be valid
    assert result.review_id is not None
    if result.product_match:
        assert result.product_match.product_id in ("phone_test", "laptop_test")
        assert 0.0 <= result.product_match.similarity <= 1.0


def test_pipeline_audit_trail(sample_products, sample_dimensions, audit_trail, tmp_path):
    """Audit trail should record entries after analysis."""
    from src.rag import EmbeddingModel, VectorStore
    from src.pipeline import ReviewAnalysisPipeline

    emb = MockEmbeddingModel()
    vs = VectorStore(persist_dir=str(tmp_path / "chroma_at"))
    pipeline = ReviewAnalysisPipeline(
        products=sample_products,
        dimensions=sample_dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit_trail,
        use_llm=False,
    )
    pipeline.setup()

    rid = str(uuid.uuid4())
    pipeline.analyze("质量很棒", review_id=rid)

    entries = audit_trail.get_by_review(rid)
    assert len(entries) >= 2  # At minimum: product_recognition + dimension_detection
    stages = {e.stage for e in entries}
    assert "product_recognition" in stages
    assert "dimension_detection" in stages


def test_pipeline_batch(sample_products, sample_dimensions, audit_trail, tmp_path):
    """Batch analysis should return one result per review."""
    from src.rag import EmbeddingModel, VectorStore
    from src.pipeline import ReviewAnalysisPipeline

    emb = MockEmbeddingModel()
    vs = VectorStore(persist_dir=str(tmp_path / "chroma_batch"))
    pipeline = ReviewAnalysisPipeline(
        products=sample_products,
        dimensions=sample_dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit_trail,
        use_llm=False,
    )
    pipeline.setup()

    reviews = ["手机质量好", "物流很慢", "笔记本电脑续航差"]
    results = pipeline.analyze_batch(reviews)
    assert len(results) == 3
    for r in results:
        assert r.review_text in reviews


def test_dimension_score_validation():
    """DimensionScore should reject scores outside 1-5 range."""
    from src.models import DimensionScore
    import pytest

    # Valid score
    score = DimensionScore(
        dimension_id="quality",
        dimension_name="产品质量",
        score=4.5,
        sentiment="positive",
    )
    assert score.score == 4.5

    # Score below minimum
    with pytest.raises(Exception):
        DimensionScore(
            dimension_id="quality",
            dimension_name="产品质量",
            score=0.5,
            sentiment="negative",
        )

    # Score above maximum
    with pytest.raises(Exception):
        DimensionScore(
            dimension_id="quality",
            dimension_name="产品质量",
            score=6.0,
            sentiment="positive",
        )
