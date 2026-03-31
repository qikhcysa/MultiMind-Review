"""DatasetOrchestratorAgent: ReAct agent for multi-turn dataset analysis.

Unlike :class:`~src.agents.orchestrator_agent.OrchestratorAgent` which analyses
one review at a time, this agent operates on a full batch of reviews and
exposes dataset-level statistics tools so that users can ask natural-language
questions such as:

* "这个数据集的整体情感分布如何？"
* "哪个维度评分最低？"
* "帮我找出所有负面评论"
* "不同商品的评分对比如何？"
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from src.agents.llm_client import get_llm_client
from src.agents.dataset_tools import DATASET_TOOLS, DatasetToolExecutor

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline
    from src.models import ReviewAnalysisResult


def _build_system_prompt(review_count: int) -> str:
    return f"""\
你是 MultiMind Review 数据集分析助手，专门对批量商品评论数据集进行深度分析。
当前数据集共包含 **{review_count}** 条用户评论。

你拥有以下工具，请根据用户需求灵活调用：

• batch_analyze          – 对所有评论进行多维度情感分析（分析前必须先调用一次）
• get_summary_statistics – 获取数据集整体统计摘要
• get_dimension_statistics – 获取各评价维度的详细统计
• filter_reviews         – 按条件筛选评论（情感、商品、维度、评分范围）
• rank_dimensions        – 对评价维度按平均分排名
• compare_products       – 比较不同商品的评分和情感表现

**行为规范**：
1. 首次收到分析请求时，先调用 batch_analyze 完成全量分析，再调用相关统计工具。
2. 如果 batch_analyze 返回 already_analyzed=true，说明已有缓存，直接调用统计工具即可。
3. 数字统计要精确，引用具体数字时务必与工具返回值保持一致。
4. 回复使用**中文**，格式清晰，关键数字用粗体标出。
5. 对于追问（如"哪个维度最差"、"找出负面评论"），直接调用合适工具，无需重新全量分析。
6. 在给出统计数据后，简短提供可行的改进建议或业务洞察。
"""


class DatasetOrchestratorAgent:
    """
    A ReAct-style agent for multi-turn natural-language conversations about
    a loaded review dataset.

    Unlike :class:`~src.agents.orchestrator_agent.OrchestratorAgent` which
    analyses one review at a time, this agent operates on a full batch of
    reviews and exposes dataset-level statistics tools.

    Usage::

        agent = DatasetOrchestratorAgent(pipeline, reviews=["评论1", "评论2", ...])

        # Ask about the whole dataset
        reply = agent.chat("这个数据集的整体情感分布如何？")
        print(reply)

        # Follow-up – no re-analysis needed
        reply = agent.chat("哪个维度评分最低？")
        print(reply)

        # Load a new dataset
        agent.update_dataset(new_reviews)

        # Clear conversation history (keeps analysis cache)
        agent.reset()
    """

    MAX_ITERATIONS: int = 10

    def __init__(
        self,
        pipeline: "ReviewAnalysisPipeline",
        reviews: list[str],
        use_llm: bool = True,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._pipeline = pipeline
        self.use_llm = use_llm
        self.max_iterations = max_iterations
        self._reviews = list(reviews)
        self._executor = DatasetToolExecutor(pipeline, self._reviews)
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process *user_message* about the dataset and return the agent's reply.

        Conversation history is maintained across calls so users can ask
        follow-up questions without re-running the analysis.

        Args:
            user_message: The user's natural-language question or request.

        Returns:
            The agent's Chinese natural-language reply (may include structured
            summaries derived from tool call results).
        """
        self._history.append({"role": "user", "content": user_message})

        if self.use_llm:
            reply = self._react_loop()
        else:
            reply = (
                "⚠️ LLM 未配置，请设置 OPENAI_API_KEY 环境变量后使用数据集分析功能。"
            )

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history (analysis results cache is preserved)."""
        self._history = []

    def update_dataset(self, reviews: list[str]) -> None:
        """
        Replace the loaded dataset and reset conversation history.

        The previous analysis cache is also cleared so that the next call
        to :meth:`chat` will trigger a fresh ``batch_analyze``.
        """
        self._reviews = list(reviews)
        self._executor.update_reviews(self._reviews)
        self._history = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history (excluding system prompt)."""
        return list(self._history)

    @property
    def is_analyzed(self) -> bool:
        """``True`` if batch analysis has been completed for the current dataset."""
        return self._executor.results is not None

    @property
    def analysis_results(self) -> "list[ReviewAnalysisResult] | None":
        """Cached :class:`~src.models.ReviewAnalysisResult` list, or ``None``."""
        return self._executor.results

    @property
    def review_count(self) -> int:
        """Number of reviews in the current dataset."""
        return len(self._reviews)

    # ------------------------------------------------------------------
    # ReAct loop (LLM + tool calling)
    # ------------------------------------------------------------------

    def _react_loop(self) -> str:
        """
        Run the Reason-Act-Observe loop until the LLM produces a final answer
        or ``max_iterations`` is reached.
        """
        client = get_llm_client()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt(len(self._reviews))},
            *self._history,
        ]

        for _ in range(self.max_iterations):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=DATASET_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )

            msg = response.choices[0].message

            # No tool calls → LLM produced a final answer
            if not msg.tool_calls:
                return msg.content or ""

            # Append assistant message with tool_calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute each tool call and append results
            for tool_call in msg.tool_calls:
                result = self._executor.execute(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return "已达到最大分析迭代次数，请尝试更简洁的问题或重新开始对话。"
