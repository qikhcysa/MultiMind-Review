"""Dimension Detection Agent: identifies which sentiment dimensions are present in a review."""
from __future__ import annotations

import json
import uuid

from src.models import Dimension, AuditEntry
from src.agents.llm_client import chat_complete


SYSTEM_PROMPT = """你是一个商品评论分析专家。你的任务是从用户评论中识别涉及的评价维度。

请从以下维度列表中，识别在评论中被明确提及或强烈暗示的维度：
{dimension_list}

规则：
1. 只选择评论中有实际内容涉及的维度
2. 返回 JSON 格式：{{"detected_dimensions": ["id1", "id2", ...], "reasoning": "..."}}
3. 如果某个维度在评论中没有提及，不要包含它
4. reasoning 字段说明你的判断依据
"""

USER_PROMPT = """用户评论：
{review_text}

请识别上述评论涉及的评价维度。"""


class DimensionDetectionAgent:
    """
    Detects which sentiment dimensions are mentioned in a review using an LLM.
    Falls back to keyword-matching when the LLM is unavailable.
    """

    def __init__(self, dimensions: list[Dimension], use_llm: bool = True) -> None:
        self.dimensions = {d.id: d for d in dimensions}
        self.use_llm = use_llm
        self.name = "DimensionDetectionAgent"

    def detect(self, review_text: str, review_id: str) -> tuple[list[str], AuditEntry]:
        """Return (detected_dimension_ids, AuditEntry)."""
        if self.use_llm:
            return self._detect_with_llm(review_text, review_id)
        return self._detect_with_keywords(review_text, review_id)

    # ------------------------------------------------------------------
    # LLM-based detection
    # ------------------------------------------------------------------

    def _detect_with_llm(self, review_text: str, review_id: str) -> tuple[list[str], AuditEntry]:
        dim_list = "\n".join(
            f"- {d.id}: {d.name}（{d.description}）"
            for d in self.dimensions.values()
        )
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(dimension_list=dim_list),
            },
            {"role": "user", "content": USER_PROMPT.format(review_text=review_text)},
        ]
        raw_response = chat_complete(messages)
        detected_ids, reasoning = self._parse_llm_response(raw_response)

        audit = self._build_audit(review_id, review_text, detected_ids, reasoning, raw_response)
        return detected_ids, audit

    def _parse_llm_response(self, raw: str) -> tuple[list[str], str]:
        """Parse JSON response from LLM, fallback to empty on error."""
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            ids = [i for i in data.get("detected_dimensions", []) if i in self.dimensions]
            reasoning = data.get("reasoning", "")
            return ids, reasoning
        except (json.JSONDecodeError, AttributeError):
            return [], f"Failed to parse LLM response: {raw[:200]}"

    # ------------------------------------------------------------------
    # Keyword-based fallback
    # ------------------------------------------------------------------

    def _detect_with_keywords(self, review_text: str, review_id: str) -> tuple[list[str], AuditEntry]:
        text_lower = review_text.lower()
        detected_ids = []
        matched_info = {}
        for dim_id, dim in self.dimensions.items():
            hits = [kw for kw in dim.keywords if kw.lower() in text_lower]
            if hits:
                detected_ids.append(dim_id)
                matched_info[dim_id] = hits
        reasoning = (
            "Keyword matching: "
            + "; ".join(f"{k}: {v}" for k, v in matched_info.items())
            if matched_info
            else "No dimension keywords found in review."
        )
        audit = self._build_audit(review_id, review_text, detected_ids, reasoning, "")
        return detected_ids, audit

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_audit(
        self,
        review_id: str,
        review_text: str,
        detected_ids: list[str],
        reasoning: str,
        raw_llm_output: str,
    ) -> AuditEntry:
        return AuditEntry(
            entry_id=str(uuid.uuid4()),
            review_id=review_id,
            stage="dimension_detection",
            agent_name=self.name,
            input_data={"review_text": review_text},
            output_data={
                "detected_dimensions": detected_ids,
                "dimension_names": [self.dimensions[i].name for i in detected_ids],
            },
            retrieved_neighbors=[],
            similarities=[],
            reasoning=reasoning,
        )
