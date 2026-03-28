"""Sentiment Scoring Agent: produces multi-dimensional sentiment scores."""
from __future__ import annotations

import json
import uuid

from src.models import Dimension, DimensionScore, AuditEntry
from src.agents.llm_client import chat_complete


SYSTEM_PROMPT = """你是一个商品评论情感分析专家。你的任务是根据提供的评论证据，对指定的评价维度进行情感评分。

评分规则：
- 评分范围 1-5 分（1=非常负面，2=偏负面，3=中立，4=偏正面，5=非常正面）
- 情感类别：positive（正面），neutral（中立），negative（负面）
- 必须基于提供的证据进行评分，不得臆造

请以 JSON 格式返回：
{{
  "score": <1-5的浮点数>,
  "sentiment": "<positive|neutral|negative>",
  "reasoning": "<详细说明评分依据>"
}}"""

USER_PROMPT = """评价维度：{dimension_name}（{dimension_description}）

评论证据：
{evidence_text}

请对该维度进行情感评分。"""


class SentimentScoringAgent:
    """
    Assigns a sentiment score (1-5) to each dimension based on the retrieved
    evidence sentences.  Uses the LLM to reason about the evidence.
    Falls back to a simple keyword heuristic if the LLM is unavailable.
    """

    def __init__(self, dimensions: list[Dimension], use_llm: bool = True) -> None:
        self.dimensions = {d.id: d for d in dimensions}
        self.use_llm = use_llm
        self.name = "SentimentScoringAgent"

    def score(
        self,
        review_id: str,
        evidence_map: dict[str, list[str]],
    ) -> tuple[list[DimensionScore], list[AuditEntry]]:
        """
        Score each dimension.

        Returns:
            (list[DimensionScore], list[AuditEntry])
        """
        scores: list[DimensionScore] = []
        audits: list[AuditEntry] = []

        for dim_id, evidence in evidence_map.items():
            if dim_id not in self.dimensions:
                continue
            dim = self.dimensions[dim_id]

            if self.use_llm:
                dim_score, audit = self._score_with_llm(review_id, dim, evidence)
            else:
                dim_score, audit = self._score_with_heuristic(review_id, dim, evidence)

            scores.append(dim_score)
            audits.append(audit)

        return scores, audits

    # ------------------------------------------------------------------
    # LLM-based scoring
    # ------------------------------------------------------------------

    def _score_with_llm(
        self, review_id: str, dim: Dimension, evidence: list[str]
    ) -> tuple[DimensionScore, AuditEntry]:
        evidence_text = "\n".join(f"- {e}" for e in evidence) if evidence else "（无具体证据）"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    dimension_name=dim.name,
                    dimension_description=dim.description,
                    evidence_text=evidence_text,
                ),
            },
        ]
        raw = chat_complete(messages)
        score_val, sentiment, reasoning = self._parse_llm_response(raw)

        dim_score = DimensionScore(
            dimension_id=dim.id,
            dimension_name=dim.name,
            score=score_val,
            sentiment=sentiment,
            evidence=evidence,
            reasoning=reasoning,
        )
        audit = self._build_audit(review_id, dim, dim_score, raw)
        return dim_score, audit

    def _parse_llm_response(self, raw: str) -> tuple[float, str, str]:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            score = float(data.get("score", 3.0))
            score = max(1.0, min(5.0, score))
            sentiment = data.get("sentiment", "neutral")
            if sentiment not in ("positive", "neutral", "negative"):
                sentiment = "neutral"
            reasoning = data.get("reasoning", "")
            return score, sentiment, reasoning
        except (json.JSONDecodeError, ValueError, AttributeError):
            return 3.0, "neutral", f"Failed to parse LLM response: {raw[:200]}"

    # ------------------------------------------------------------------
    # Keyword heuristic fallback
    # ------------------------------------------------------------------

    POSITIVE_WORDS = {
        "好", "棒", "赞", "满意", "优秀", "完美", "喜欢", "推荐", "超好", "非常好",
        "good", "great", "excellent", "perfect", "love", "awesome", "satisfied",
    }
    NEGATIVE_WORDS = {
        "差", "烂", "糟", "不好", "失望", "垃圾", "后悔", "坏", "破",
        "bad", "poor", "terrible", "awful", "disappointed", "horrible", "hate",
    }

    def _score_with_heuristic(
        self, review_id: str, dim: Dimension, evidence: list[str]
    ) -> tuple[DimensionScore, AuditEntry]:
        text = " ".join(evidence).lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in text)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in text)

        if pos > neg:
            score, sentiment = 4.0, "positive"
        elif neg > pos:
            score, sentiment = 2.0, "negative"
        else:
            score, sentiment = 3.0, "neutral"

        reasoning = f"Heuristic: positive_words={pos}, negative_words={neg}"
        dim_score = DimensionScore(
            dimension_id=dim.id,
            dimension_name=dim.name,
            score=score,
            sentiment=sentiment,
            evidence=evidence,
            reasoning=reasoning,
        )
        audit = self._build_audit(review_id, dim, dim_score, "")
        return dim_score, audit

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_audit(
        self, review_id: str, dim: Dimension, dim_score: DimensionScore, raw_llm: str
    ) -> AuditEntry:
        return AuditEntry(
            entry_id=str(uuid.uuid4()),
            review_id=review_id,
            stage="sentiment_scoring",
            agent_name=self.name,
            input_data={
                "dimension_id": dim.id,
                "dimension_name": dim.name,
                "evidence": dim_score.evidence,
            },
            output_data={
                "score": dim_score.score,
                "sentiment": dim_score.sentiment,
            },
            retrieved_neighbors=[],
            similarities=[],
            reasoning=dim_score.reasoning,
        )
