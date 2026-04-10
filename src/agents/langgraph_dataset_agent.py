"""LangGraph-based DatasetOrchestratorAgent for batch review analysis.

Replaces the hand-rolled ReAct loop in
:class:`~src.agents.dataset_agent.DatasetOrchestratorAgent` with LangGraph's
``create_react_agent`` pre-built graph.

The six dataset-level tool *handlers* are unchanged (they still live in
:mod:`src.agents.dataset_tools` via ``DatasetToolExecutor``).  Only the
orchestration layer — the manual message-list management and LLM call loop —
is replaced by LangGraph.

Public API is identical to :class:`~src.agents.dataset_agent.DatasetOrchestratorAgent`:
``chat()``, ``reset()``, ``update_dataset()``, ``history``,
``is_analyzed``, ``analysis_results``, ``review_count``.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.dataset_tools import DatasetToolExecutor

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline
    from src.models import ReviewAnalysisResult


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class LangGraphDatasetAgent:
    """
    LangGraph-based ReAct agent for multi-turn natural-language conversations
    about a loaded review dataset.

    Drop-in replacement for
    :class:`~src.agents.dataset_agent.DatasetOrchestratorAgent` with the same
    public API.  The six dataset-level tool handlers are reused via
    :class:`~src.agents.dataset_tools.DatasetToolExecutor`; only the
    orchestration loop is replaced by LangGraph ``create_react_agent``.

    Usage::

        agent = LangGraphDatasetAgent(pipeline, reviews=[...])

        reply = agent.chat("这个数据集的整体情感分布如何？")
        reply = agent.chat("哪个维度评分最低？")   # follow-up; reuses cache

        agent.update_dataset(new_reviews)
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

        # Conversation thread ID (MemorySaver key)
        self._thread_id = str(uuid.uuid4())

        if use_llm:
            self._checkpointer, self._graph = self._build_graph()
        else:
            self._checkpointer = None
            self._graph = None

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build the create_react_agent graph with dataset tools."""
        import os

        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY") or "placeholder",
            temperature=0.1,
            max_tokens=2048,
        )
        checkpointer = MemorySaver()
        graph = create_react_agent(
            model,
            tools=self._build_tools(),
            checkpointer=checkpointer,
            state_modifier=_build_system_prompt(len(self._reviews)),
        )
        return checkpointer, graph

    def _build_tools(self) -> list:
        """Return LangChain ``@tool``-decorated wrappers backed by DatasetToolExecutor."""
        from langchain_core.tools import tool as lc_tool

        executor = self._executor

        @lc_tool
        def batch_analyze() -> str:
            """对数据集中的所有评论进行多维度情感分析，生成结构化结果。
            这是分析数据集的第一步，其他统计工具都依赖本工具的结果。
            只需调用一次，结果会被缓存，重复调用无额外开销。"""
            return json.dumps(executor.execute("batch_analyze", "{}"), ensure_ascii=False)

        @lc_tool
        def get_summary_statistics() -> str:
            """获取数据集的整体统计摘要：评论总数、情感分布、整体平均评分、
            各维度平均分、商品分布。需要先调用 batch_analyze。"""
            return json.dumps(
                executor.execute("get_summary_statistics", "{}"), ensure_ascii=False
            )

        @lc_tool
        def get_dimension_statistics(dimension_id: str = "") -> str:
            """获取指定维度或所有维度的详细统计：平均分、情感分布、评论数量、
            示例证据片段。dimension_id 为空时返回所有维度。需要先调用 batch_analyze。"""
            args = json.dumps({"dimension_id": dimension_id} if dimension_id else {})
            return json.dumps(
                executor.execute("get_dimension_statistics", args), ensure_ascii=False
            )

        @lc_tool
        def filter_reviews(
            sentiment: str = "",
            product_name: str = "",
            dimension_id: str = "",
            min_score: float = 0.0,
            max_score: float = 0.0,
            top_n: int = 5,
            sort_by: str = "default",
        ) -> str:
            """按条件筛选数据集中的评论并返回样本。
            sentiment: positive/neutral/negative（空=不过滤）；
            product_name: 按商品名模糊匹配（空=不过滤）；
            dimension_id: 仅返回含该维度的评论（空=不过滤）；
            min_score/max_score: 评分区间过滤（0=不过滤）；
            top_n: 最多返回条数（默认5，最大20）；
            sort_by: score_asc/score_desc/default。
            需要先调用 batch_analyze。"""
            args: dict[str, Any] = {"top_n": top_n, "sort_by": sort_by}
            if sentiment:
                args["sentiment"] = sentiment
            if product_name:
                args["product_name"] = product_name
            if dimension_id:
                args["dimension_id"] = dimension_id
            if min_score > 0:
                args["min_score"] = min_score
            if max_score > 0:
                args["max_score"] = max_score
            return json.dumps(
                executor.execute("filter_reviews", json.dumps(args)), ensure_ascii=False
            )

        @lc_tool
        def rank_dimensions(order: str = "desc") -> str:
            """对所有评价维度按平均分排名，找出评分最高和最低的维度。
            order: desc=降序（最好在前），asc=升序（最差在前）。
            需要先调用 batch_analyze。"""
            return json.dumps(
                executor.execute("rank_dimensions", json.dumps({"order": order})),
                ensure_ascii=False,
            )

        @lc_tool
        def compare_products() -> str:
            """比较数据集中不同商品的评分和情感分布，找出表现最好/最差的商品。
            需要先调用 batch_analyze。"""
            return json.dumps(
                executor.execute("compare_products", "{}"), ensure_ascii=False
            )

        return [
            batch_analyze,
            get_summary_statistics,
            get_dimension_statistics,
            filter_reviews,
            rank_dimensions,
            compare_products,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process *user_message* about the dataset and return the agent's reply.

        Conversation history is maintained across calls via LangGraph
        ``MemorySaver``.

        Args:
            user_message: The user's natural-language question or request.

        Returns:
            The agent's Chinese natural-language reply.
        """
        if not self.use_llm or self._graph is None:
            return "⚠️ LLM 未配置，请设置 OPENAI_API_KEY 环境变量后使用数据集分析功能。"
        return self._llm_chat(user_message)

    def reset(self) -> None:
        """Clear conversation history (analysis results cache is preserved)."""
        self._thread_id = str(uuid.uuid4())

    def update_dataset(self, reviews: list[str]) -> None:
        """Replace the loaded dataset and reset conversation history.

        The previous analysis cache is cleared so the next ``chat()`` call
        will trigger a fresh ``batch_analyze``.
        """
        self._reviews = list(reviews)
        self._executor.update_reviews(self._reviews)
        self._thread_id = str(uuid.uuid4())

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history (user + assistant turns only)."""
        if not self.use_llm or self._graph is None:
            return []
        config = {"configurable": {"thread_id": self._thread_id}}
        try:
            state = self._graph.get_state(config)
            messages = state.values.get("messages", [])
        except Exception:  # noqa: BLE001
            return []
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content or ""})
        return result

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
    # LLM path
    # ------------------------------------------------------------------

    def _llm_chat(self, user_message: str) -> str:
        from langgraph.errors import GraphRecursionError

        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self.max_iterations * 3,
        }
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
            )
            return result["messages"][-1].content or ""
        except GraphRecursionError:
            return "已达到最大分析迭代次数，请尝试更简洁的问题或重新开始对话。"
