"""LangGraph-based OrchestratorAgent for single-review analysis.

Replaces the hand-rolled ReAct loop in
:class:`~src.agents.orchestrator_agent.OrchestratorAgent` with LangGraph's
``create_react_agent`` pre-built graph, providing:

* The same ``chat()`` / ``analyze()`` / ``reset()`` / ``history`` public API.
* Persistent multi-turn conversation state via ``MemorySaver`` (keyed by
  thread ID, reset by generating a new thread ID).
* LangChain ``@tool``-decorated specialist agent wrappers with proper JSON
  schemas, replacing the hand-crafted OpenAI function-calling dicts.
* Automatic ``GraphRecursionError`` handling as a safety valve.

The four specialist agents (product recognition, dimension detection, evidence
retrieval, sentiment scoring) remain unchanged — only the orchestration layer
is replaced by LangGraph.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from src.models import ReviewAnalysisResult
from src.agents.orchestrator_agent import _format_result  # reuse formatter

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是 MultiMind Review 智能评论分析助手，专门分析商品评论。

你拥有以下工具，可以按需调用——**不要硬编码固定顺序**，根据用户需求灵活决策：

• recognize_product    – 识别评论对应的商品
• detect_dimensions    – 检测评论涉及哪些评价维度
• retrieve_evidence    – 为各维度检索相关证据片段
• score_sentiment      – 对各维度证据进行情感评分（1-5分）

**行为规范**：
1. 当用户提供评论时，先调用 recognize_product 识别商品，再调用 detect_dimensions，
   再调用 retrieve_evidence，最后调用 score_sentiment，最终给出中文自然语言总结。
2. 如果某工具返回空结果（如未识别到商品），继续分析其他方面，并在总结中说明。
3. 如果评论文本模糊不清，可以向用户提问要求澄清，再继续分析。
4. 对于用户的追问（如"哪个维度评分最低？""为什么物流评分这么低？"），
   优先从已有的工具调用结果回答，无需重新分析。
5. 最终回复必须用**中文**，清晰列出：商品识别结果、各维度评分（含评分理由）、整体评价。
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class LangGraphOrchestratorAgent:
    """
    LangGraph-based ReAct orchestrator for single-review analysis.

    Drop-in replacement for :class:`~src.agents.orchestrator_agent.OrchestratorAgent`
    with the same public API.  The internal ``_react_loop`` is replaced by
    LangGraph's ``create_react_agent``, which manages the Reason→Act→Observe
    cycle, tool-call serialisation, and multi-turn memory automatically.

    Usage::

        agent = LangGraphOrchestratorAgent(pipeline)

        reply = agent.chat("这款手机质量非常好，物流也很快")
        reply = agent.chat("哪个维度评分最低？")   # follow-up; no re-analysis

        agent.reset()                              # new conversation thread
    """

    MAX_ITERATIONS: int = 10

    def __init__(
        self,
        pipeline: "ReviewAnalysisPipeline",
        use_llm: bool = True,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._pipeline = pipeline
        self.use_llm = use_llm
        self.max_iterations = max_iterations

        # Each thread_id identifies one conversation session in MemorySaver
        self._thread_id = str(uuid.uuid4())

        # Manual history list for the use_llm=False path
        self._heuristic_history: list[dict[str, Any]] = []

        if use_llm:
            self._checkpointer, self._graph = self._build_graph()
        else:
            self._checkpointer = None
            self._graph = None

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build the create_react_agent graph with MemorySaver checkpointer."""
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
            state_modifier=_SYSTEM_PROMPT,
        )
        return checkpointer, graph

    def _build_tools(self) -> list:
        """Return LangChain ``@tool``-decorated wrappers for the 4 specialist agents."""
        from langchain_core.tools import tool as lc_tool

        pipeline = self._pipeline

        @lc_tool
        def recognize_product(review_text: str) -> str:
            """识别评论中提及的产品，通过向量相似度匹配知识库中的商品。
            返回商品名称、品牌、分类和相似度分数。"""
            if not pipeline._products_indexed:
                pipeline.setup()
            rid = f"lg_{uuid.uuid4().hex[:8]}"
            match, _ = pipeline.product_agent.recognize(review_text, rid)
            if match:
                return json.dumps(
                    {
                        "matched": True,
                        "product_id": match.product_id,
                        "product_name": match.product_name,
                        "brand": match.brand,
                        "category": match.category,
                        "similarity": round(match.similarity, 4),
                        "matched_keywords": match.matched_keywords,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {"matched": False, "message": "未在知识库中识别到具体商品"},
                ensure_ascii=False,
            )

        @lc_tool
        def detect_dimensions(review_text: str) -> str:
            """检测评论中涉及的评价维度（如产品质量、物流配送、售后服务等）。
            返回检测到的维度列表及推理说明。"""
            rid = f"lg_{uuid.uuid4().hex[:8]}"
            dim_ids, audit = pipeline.dimension_agent.detect(review_text, rid)
            dims = [
                {"id": d.id, "name": d.name, "description": d.description}
                for d in pipeline.dimensions
                if d.id in dim_ids
            ]
            return json.dumps(
                {
                    "detected_dimensions": dims,
                    "count": len(dims),
                    "reasoning": audit.reasoning,
                },
                ensure_ascii=False,
            )

        @lc_tool
        def retrieve_evidence(review_text: str, dimension_ids: list[str]) -> str:
            """为指定的评价维度从评论原文中检索最相关的证据句子。
            必须在 detect_dimensions 之后调用，使用其返回的维度ID列表。"""
            rid = f"lg_{uuid.uuid4().hex[:8]}"
            dims = [d for d in pipeline.dimensions if d.id in dimension_ids]
            if not dims:
                return json.dumps(
                    {
                        "evidence_map": {},
                        "message": f"未找到指定维度：{dimension_ids}",
                    },
                    ensure_ascii=False,
                )
            evidence_map, _ = pipeline.evidence_agent.retrieve(review_text, rid, dims)
            return json.dumps({"evidence_map": evidence_map}, ensure_ascii=False)

        @lc_tool
        def score_sentiment(review_id: str, evidence_map: dict) -> str:
            """根据各维度的证据句子进行情感评分（1-5分），并给出情感类别。
            必须在 retrieve_evidence 之后调用，使用其返回的 evidence_map。"""
            scores, _ = pipeline.scoring_agent.score(review_id, evidence_map)
            result: dict[str, Any] = {
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
            return json.dumps(result, ensure_ascii=False)

        return [recognize_product, detect_dimensions, retrieve_evidence, score_sentiment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process *user_message* and return the agent's Chinese reply.

        Conversation history is maintained across calls via LangGraph
        ``MemorySaver`` (LLM mode) or an in-memory list (heuristic mode).

        Args:
            user_message: The user's text (a review or a follow-up question).

        Returns:
            The agent's natural-language reply.
        """
        if self.use_llm and self._graph is not None:
            return self._llm_chat(user_message)
        return self._heuristic_chat(user_message)

    def analyze(
        self, review_text: str, review_id: str | None = None
    ) -> ReviewAnalysisResult:
        """Run a structured analysis and return a :class:`ReviewAnalysisResult`.

        Delegates to the underlying pipeline for structured output.
        Does not affect the conversation history.
        """
        return self._pipeline.analyze(review_text, review_id=review_id)

    def reset(self) -> None:
        """Start a new conversation session (clears history)."""
        self._thread_id = str(uuid.uuid4())
        self._heuristic_history = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history (user + assistant turns only)."""
        if not self.use_llm or self._graph is None:
            return list(self._heuristic_history)
        return self._get_llm_history()

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
            return "已达到最大分析迭代次数，请尝试提供更具体的评论内容或重新开始对话。"

    def _get_llm_history(self) -> list[dict[str, Any]]:
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

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_chat(self, user_message: str) -> str:
        self._heuristic_history.append({"role": "user", "content": user_message})
        result = self._pipeline.analyze(user_message, review_id=str(uuid.uuid4()))
        reply = _format_result(result)
        self._heuristic_history.append({"role": "assistant", "content": reply})
        return reply
