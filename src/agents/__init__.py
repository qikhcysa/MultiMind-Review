"""Agent package: multi-agent pipeline for review analysis."""
from src.agents.product_agent import ProductRecognitionAgent
from src.agents.dimension_agent import DimensionDetectionAgent
from src.agents.evidence_agent import EvidenceRetrievalAgent
from src.agents.scoring_agent import SentimentScoringAgent

__all__ = [
    "ProductRecognitionAgent",
    "DimensionDetectionAgent",
    "EvidenceRetrievalAgent",
    "SentimentScoringAgent",
]
