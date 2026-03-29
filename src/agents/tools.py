"""Tool definitions and executor for the orchestrator agent.

Each of the four specialist agents is exposed as an OpenAI function-calling
tool so that the LLM orchestrator can choose *which* tool to invoke and
*in what order*, rather than following a hard-coded pipeline sequence.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline

# ---------------------------------------------------------------------------
# OpenAI function-calling tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recognize_product",
            "description": (
                "识别评论中提及的产品，通过向量相似度匹配知识库中的商品。"
                "返回匹配到的商品名称、品牌、分类和相似度分数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review_text": {
                        "type": "string",
                        "description": "用户评论原文",
                    }
                },
                "required": ["review_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_dimensions",
            "description": (
                "检测评论中涉及的评价维度（如产品质量、物流配送、售后服务等）。"
                "返回检测到的维度列表及推理说明。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review_text": {
                        "type": "string",
                        "description": "用户评论原文",
                    }
                },
                "required": ["review_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_evidence",
            "description": (
                "为指定的评价维度从评论原文中检索最相关的证据句子。"
                "必须在 detect_dimensions 之后调用，使用其返回的维度ID列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review_text": {
                        "type": "string",
                        "description": "用户评论原文",
                    },
                    "dimension_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要检索证据的维度ID列表，来自 detect_dimensions 的结果",
                    },
                },
                "required": ["review_text", "dimension_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_sentiment",
            "description": (
                "根据各维度的证据句子进行情感评分（1-5分），并给出情感类别（positive/neutral/negative）。"
                "必须在 retrieve_evidence 之后调用，使用其返回的 evidence_map。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "review_id": {
                        "type": "string",
                        "description": "评论唯一标识符",
                    },
                    "evidence_map": {
                        "type": "object",
                        "description": (
                            "维度ID到证据列表的映射，格式：{dimension_id: [句子1, 句子2, ...]}，"
                            "来自 retrieve_evidence 的结果"
                        ),
                    },
                },
                "required": ["review_id", "evidence_map"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    Executes tool calls by delegating to the underlying specialist agents.

    Holds a reference to the pipeline to reuse already-indexed agents and
    the shared vector store / embedding model.
    """

    def __init__(self, pipeline: "ReviewAnalysisPipeline") -> None:
        self._pipeline = pipeline
        self._call_counter = 0

    def execute(self, tool_name: str, arguments_json: str) -> dict[str, Any]:
        """
        Parse *arguments_json* and dispatch to the correct agent method.

        Returns a JSON-serialisable dict that is fed back to the LLM as a
        tool result message.
        """
        try:
            args: dict[str, Any] = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON arguments: {exc}"}

        dispatch = {
            "recognize_product": self._recognize_product,
            "detect_dimensions": self._detect_dimensions,
            "retrieve_evidence": self._retrieve_evidence,
            "score_sentiment": self._score_sentiment,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name!r}"}
        try:
            return handler(args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Individual tool handlers
    # ------------------------------------------------------------------

    def _recognize_product(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_indexed()
        review_text: str = args["review_text"]
        review_id = self._new_call_id()
        match, _ = self._pipeline.product_agent.recognize(review_text, review_id)
        if match:
            return {
                "matched": True,
                "product_id": match.product_id,
                "product_name": match.product_name,
                "brand": match.brand,
                "category": match.category,
                "similarity": round(match.similarity, 4),
                "matched_keywords": match.matched_keywords,
            }
        return {"matched": False, "message": "未在知识库中识别到具体商品"}

    def _detect_dimensions(self, args: dict[str, Any]) -> dict[str, Any]:
        review_text: str = args["review_text"]
        review_id = self._new_call_id()
        dim_ids, audit = self._pipeline.dimension_agent.detect(review_text, review_id)
        dims = [
            {"id": d.id, "name": d.name, "description": d.description}
            for d in self._pipeline.dimensions
            if d.id in dim_ids
        ]
        return {
            "detected_dimensions": dims,
            "count": len(dims),
            "reasoning": audit.reasoning,
        }

    def _retrieve_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        review_text: str = args["review_text"]
        dimension_ids: list[str] = args["dimension_ids"]
        review_id = self._new_call_id()
        dims = [d for d in self._pipeline.dimensions if d.id in dimension_ids]
        if not dims:
            return {
                "evidence_map": {},
                "message": f"未找到指定维度：{dimension_ids}",
            }
        evidence_map, _ = self._pipeline.evidence_agent.retrieve(
            review_text, review_id, dims
        )
        return {"evidence_map": evidence_map}

    def _score_sentiment(self, args: dict[str, Any]) -> dict[str, Any]:
        review_id: str = args["review_id"]
        evidence_map: dict[str, list[str]] = args["evidence_map"]
        scores, _ = self._pipeline.scoring_agent.score(review_id, evidence_map)
        result = {
            "scores": [
                {
                    "dimension_id": s.dimension_id,
                    "dimension_name": s.dimension_name,
                    "score": s.score,
                    "sentiment": s.sentiment,
                    "reasoning": s.reasoning,
                }
                for s in scores
            ],
        }
        if scores:
            result["overall_score"] = round(
                sum(s.score for s in scores) / len(scores), 2
            )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        if not self._pipeline._products_indexed:
            self._pipeline.setup()

    def _new_call_id(self) -> str:
        self._call_counter += 1
        return f"tool_{self._call_counter}_{uuid.uuid4().hex[:8]}"
