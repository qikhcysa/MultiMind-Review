"""Dataset-level tool definitions and executor for the dataset analysis agent.

Exposes six OpenAI function-calling tools that operate on a *batch* of
pre-loaded reviews rather than on a single review.  The
:class:`DatasetToolExecutor` lazily runs
``pipeline.analyze_batch`` on first demand and caches the results for all
subsequent tool calls within the same session.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline
    from src.models import ReviewAnalysisResult

# ---------------------------------------------------------------------------
# OpenAI function-calling tool definitions (dataset-level)
# ---------------------------------------------------------------------------

DATASET_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "batch_analyze",
            "description": (
                "对数据集中的所有评论进行多维度情感分析，生成结构化结果。"
                "这是分析数据集的第一步，其他统计工具都依赖本工具的结果。"
                "只需调用一次，结果会被缓存，重复调用无额外开销。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_statistics",
            "description": (
                "获取数据集的整体统计摘要：评论总数、情感分布（正面/中性/负面占比）、"
                "整体平均评分、各维度平均分、商品分布。"
                "需要先调用 batch_analyze。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dimension_statistics",
            "description": (
                "获取指定维度或所有维度的详细统计：平均分、情感分布、评论数量、"
                "示例证据片段。需要先调用 batch_analyze。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension_id": {
                        "type": "string",
                        "description": "指定维度ID（可选，不填则返回所有维度的统计）",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_reviews",
            "description": (
                "按条件筛选数据集中的评论并返回样本。"
                "可按情感类别、商品名称、维度、评分范围过滤，支持排序。"
                "需要先调用 batch_analyze。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"],
                        "description": "按整体情感类别过滤",
                    },
                    "product_name": {
                        "type": "string",
                        "description": "按商品名称过滤（模糊匹配）",
                    },
                    "dimension_id": {
                        "type": "string",
                        "description": "仅返回包含该维度评分的评论",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "最低综合评分（1-5，包含）",
                    },
                    "max_score": {
                        "type": "number",
                        "description": "最高综合评分（1-5，包含）",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "最多返回多少条评论（默认5，最大20）",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["score_asc", "score_desc", "default"],
                        "description": (
                            "排序方式：score_asc=分数升序（最差在前）、"
                            "score_desc=分数降序（最好在前）、default=原始顺序"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_dimensions",
            "description": (
                "对所有评价维度按平均分排名，找出评分最高和最低的维度。"
                "需要先调用 batch_analyze。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": (
                            "排序方向：asc=升序（最差在前）、"
                            "desc=降序（最好在前），默认 desc"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "比较数据集中不同商品的评分和情感分布，"
                "找出哪些商品表现最好、哪些最差。"
                "需要先调用 batch_analyze。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dataset Tool Executor
# ---------------------------------------------------------------------------

class DatasetToolExecutor:
    """
    Executes dataset-level tool calls against a batch of reviews.

    Lazily runs ``pipeline.analyze_batch`` on first demand and caches the
    results for all subsequent tool calls within the same session.

    Args:
        pipeline: The initialised :class:`~src.pipeline.workflow.ReviewAnalysisPipeline`.
        reviews:  The list of raw review texts to analyse.
    """

    def __init__(
        self,
        pipeline: "ReviewAnalysisPipeline",
        reviews: list[str],
    ) -> None:
        self._pipeline = pipeline
        self._reviews = reviews
        self._results: list["ReviewAnalysisResult"] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments_json: str) -> dict[str, Any]:
        """Parse *arguments_json* and dispatch to the correct handler."""
        try:
            args: dict[str, Any] = (
                json.loads(arguments_json) if arguments_json.strip() else {}
            )
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON arguments: {exc}"}

        dispatch = {
            "batch_analyze": self._batch_analyze,
            "get_summary_statistics": self._get_summary_statistics,
            "get_dimension_statistics": self._get_dimension_statistics,
            "filter_reviews": self._filter_reviews,
            "rank_dimensions": self._rank_dimensions,
            "compare_products": self._compare_products,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name!r}"}
        try:
            return handler(args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @property
    def results(self) -> "list[ReviewAnalysisResult] | None":
        """Cached analysis results (``None`` if ``batch_analyze`` hasn't run yet)."""
        return self._results

    def update_reviews(self, reviews: list[str]) -> None:
        """Replace the review dataset and clear the cached results."""
        self._reviews = reviews
        self._results = None

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _batch_analyze(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Run the pipeline on all reviews and cache the results."""
        if self._results is not None:
            return {
                "already_analyzed": True,
                "total_analyzed": len(self._results),
                "message": (
                    f"数据集已分析（共 {len(self._results)} 条评论），"
                    "无需重复分析，可直接调用统计工具。"
                ),
            }
        review_ids = [str(uuid.uuid4()) for _ in self._reviews]
        self._results = self._pipeline.analyze_batch(self._reviews, review_ids)
        return {
            "total_analyzed": len(self._results),
            "message": (
                f"已完成 {len(self._results)} 条评论的分析，"
                "可以调用统计工具获取结果。"
            ),
        }

    def _require_results(self) -> "dict[str, Any] | None":
        """Return an error dict if results are not yet available, else ``None``."""
        if self._results is None:
            return {
                "error": "尚未完成数据分析，请先调用 batch_analyze 工具。"
            }
        return None

    def _get_summary_statistics(self, _args: dict[str, Any]) -> dict[str, Any]:
        if (err := self._require_results()):
            return err
        results = self._results
        total = len(results)

        sentiment_counts: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
        score_values: list[float] = []
        product_counts: dict[str, int] = defaultdict(int)
        dim_score_map: dict[str, dict[str, Any]] = {}

        for r in results:
            s = r.overall_sentiment if r.overall_sentiment in sentiment_counts else "neutral"
            sentiment_counts[s] += 1
            if r.overall_score is not None:
                score_values.append(r.overall_score)
            if r.product_match:
                product_counts[r.product_match.product_name] += 1
            for ds in r.dimension_scores:
                entry = dim_score_map.setdefault(
                    ds.dimension_id,
                    {"name": ds.dimension_name, "scores": []},
                )
                entry["scores"].append(ds.score)

        avg_score = (
            round(sum(score_values) / len(score_values), 2) if score_values else None
        )
        dim_averages = {
            v["name"]: round(sum(v["scores"]) / len(v["scores"]), 2)
            for v in dim_score_map.values()
        }

        return {
            "total_reviews": total,
            "sentiment_distribution": dict(sentiment_counts),
            "sentiment_percentages": {
                k: round(v / total * 100, 1) for k, v in sentiment_counts.items()
            },
            "average_overall_score": avg_score,
            "dimension_average_scores": dim_averages,
            "product_distribution": dict(product_counts),
        }

    def _get_dimension_statistics(self, args: dict[str, Any]) -> dict[str, Any]:
        if (err := self._require_results()):
            return err
        target_dim = args.get("dimension_id")

        dim_data: dict[str, dict[str, Any]] = {}
        for r in self._results:
            for ds in r.dimension_scores:
                if target_dim and ds.dimension_id != target_dim:
                    continue
                entry = dim_data.setdefault(
                    ds.dimension_id,
                    {
                        "dimension_id": ds.dimension_id,
                        "dimension_name": ds.dimension_name,
                        "scores": [],
                        "sentiment_counts": {
                            "positive": 0,
                            "neutral": 0,
                            "negative": 0,
                        },
                        "sample_evidence": [],
                    },
                )
                entry["scores"].append(ds.score)
                sent = (
                    ds.sentiment
                    if ds.sentiment in entry["sentiment_counts"]
                    else "neutral"
                )
                entry["sentiment_counts"][sent] += 1
                if len(entry["sample_evidence"]) < 3 and ds.evidence:
                    entry["sample_evidence"].append(ds.evidence[0])

        if not dim_data:
            msg = (
                f"未找到维度 {target_dim!r} 的数据"
                if target_dim
                else "没有维度分析数据"
            )
            return {"error": msg}

        stats = []
        for entry in dim_data.values():
            scores = entry["scores"]
            stats.append(
                {
                    "dimension_id": entry["dimension_id"],
                    "dimension_name": entry["dimension_name"],
                    "review_count": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 2),
                    "sentiment_distribution": entry["sentiment_counts"],
                    "sample_evidence": entry["sample_evidence"],
                }
            )
        return {"dimension_statistics": stats}

    def _filter_reviews(self, args: dict[str, Any]) -> dict[str, Any]:
        if (err := self._require_results()):
            return err

        sentiment_filter: str | None = args.get("sentiment")
        product_filter: str | None = args.get("product_name")
        dim_filter: str | None = args.get("dimension_id")
        min_score: float | None = args.get("min_score")
        max_score: float | None = args.get("max_score")
        top_n: int = min(int(args.get("top_n", 5)), 20)
        sort_by: str = args.get("sort_by", "default")

        filtered = []
        for r in self._results:
            if sentiment_filter and r.overall_sentiment != sentiment_filter:
                continue
            if product_filter:
                pname = r.product_match.product_name if r.product_match else ""
                if product_filter.lower() not in pname.lower():
                    continue
            if dim_filter:
                dim_ids = {ds.dimension_id for ds in r.dimension_scores}
                if dim_filter not in dim_ids:
                    continue
            if min_score is not None and (
                r.overall_score is None or r.overall_score < min_score
            ):
                continue
            if max_score is not None and (
                r.overall_score is None or r.overall_score > max_score
            ):
                continue
            filtered.append(r)

        if sort_by == "score_asc":
            filtered.sort(key=lambda r: r.overall_score or 0)
        elif sort_by == "score_desc":
            filtered.sort(key=lambda r: r.overall_score or 0, reverse=True)

        selected = filtered[:top_n]
        items = []
        for r in selected:
            item: dict[str, Any] = {
                "review_id": r.review_id[:8],
                "review_preview": (
                    r.review_text[:100] + ("…" if len(r.review_text) > 100 else "")
                ),
                "overall_sentiment": r.overall_sentiment,
                "overall_score": r.overall_score,
                "product": (
                    r.product_match.product_name if r.product_match else "未识别"
                ),
            }
            if dim_filter:
                for ds in r.dimension_scores:
                    if ds.dimension_id == dim_filter:
                        item["dimension_score"] = ds.score
                        item["dimension_sentiment"] = ds.sentiment
                        item["dimension_reasoning"] = ds.reasoning
                        break
            items.append(item)

        return {
            "total_matched": len(filtered),
            "returned": len(selected),
            "reviews": items,
        }

    def _rank_dimensions(self, args: dict[str, Any]) -> dict[str, Any]:
        if (err := self._require_results()):
            return err
        order = args.get("order", "desc")
        stats_result = self._get_dimension_statistics({})
        dim_stats = stats_result.get("dimension_statistics", [])
        if not dim_stats:
            return {"error": "没有可排名的维度数据"}

        ranked = sorted(
            dim_stats,
            key=lambda x: x["avg_score"],
            reverse=(order == "desc"),
        )
        return {
            "order": order,
            "ranking": [
                {
                    "rank": i + 1,
                    "dimension_name": item["dimension_name"],
                    "avg_score": item["avg_score"],
                    "review_count": item["review_count"],
                }
                for i, item in enumerate(ranked)
            ],
        }

    def _compare_products(self, _args: dict[str, Any]) -> dict[str, Any]:
        if (err := self._require_results()):
            return err

        product_data: dict[str, dict[str, Any]] = {}
        for r in self._results:
            pname = (
                r.product_match.product_name if r.product_match else "未识别商品"
            )
            entry = product_data.setdefault(
                pname,
                {
                    "product_name": pname,
                    "review_count": 0,
                    "scores": [],
                    "sentiment_counts": {
                        "positive": 0,
                        "neutral": 0,
                        "negative": 0,
                    },
                },
            )
            entry["review_count"] += 1
            if r.overall_score is not None:
                entry["scores"].append(r.overall_score)
            sent = (
                r.overall_sentiment
                if r.overall_sentiment in entry["sentiment_counts"]
                else "neutral"
            )
            entry["sentiment_counts"][sent] += 1

        comparison = []
        for entry in product_data.values():
            scores = entry["scores"]
            comparison.append(
                {
                    "product_name": entry["product_name"],
                    "review_count": entry["review_count"],
                    "avg_score": (
                        round(sum(scores) / len(scores), 2) if scores else None
                    ),
                    "sentiment_distribution": entry["sentiment_counts"],
                }
            )
        comparison.sort(key=lambda x: x["avg_score"] or 0, reverse=True)
        return {"product_comparison": comparison}
