"""Orchestrator Agent: the ReAct-style central brain of the system.

This agent turns the project from a fixed pipeline into a *real* agent by:

1. **Tool calling** – the LLM chooses *which* of the four specialist agents
   to invoke and *in what order*, based on the current conversation state.

2. **Multi-turn conversation** – a persistent ``conversation_history`` lets
   users ask follow-up questions (e.g. "which dimension scored lowest?")
   without re-running the full analysis.

3. **Self-correction** – if a tool returns an empty or ambiguous result the
   LLM can decide to retry with different parameters, call a different tool,
   or ask the user for clarification rather than silently producing wrong
   output.

4. **ReAct loop** – the agent iterates Reason → Act → Observe until it is
   confident it has enough information to give a final answer, bounded by
   ``max_iterations`` to prevent runaway calls.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, Any

from src.agents.llm_client import get_llm_client
from src.agents.tools import TOOLS, ToolExecutor
from src.models import ReviewAnalysisResult

if TYPE_CHECKING:
    from src.pipeline.workflow import ReviewAnalysisPipeline


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


class OrchestratorAgent:
    """
    A ReAct-style orchestrating agent for review analysis.

    Unlike :class:`~src.pipeline.workflow.ReviewAnalysisPipeline` which runs
    the four stages in a fixed order, this agent lets the LLM decide which
    tools to call, enabling dynamic planning, self-correction, and multi-turn
    conversation.

    Usage::

        agent = OrchestratorAgent(pipeline)

        # Single analysis
        reply = agent.chat("这款手机质量非常好，物流也很快")
        print(reply)

        # Follow-up question – no re-analysis needed
        reply = agent.chat("哪个维度评分最低？")
        print(reply)

        # Start a new conversation
        agent.reset()
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
        self._tool_executor = ToolExecutor(pipeline)

        # Persistent conversation history (system prompt excluded; prepended on each call)
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process *user_message* and return the agent's reply.

        Conversation history is maintained across calls so users can ask
        follow-up questions without re-running the analysis.

        Args:
            user_message: The user's text (e.g. a review or a follow-up question).

        Returns:
            The agent's natural-language reply.
        """
        self._history.append({"role": "user", "content": user_message})

        if self.use_llm:
            reply = self._react_loop()
        else:
            reply = self._heuristic_reply(user_message)

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def analyze(
        self, review_text: str, review_id: str | None = None
    ) -> ReviewAnalysisResult:
        """
        Run a structured analysis and return a :class:`ReviewAnalysisResult`.

        This method delegates to the underlying pipeline for structured output.
        Use :meth:`chat` for the conversational, tool-calling interface.

        Args:
            review_text: The raw review text.
            review_id:   Optional identifier; auto-generated when omitted.

        Returns:
            A :class:`ReviewAnalysisResult` with full structured analysis.
        """
        return self._pipeline.analyze(review_text, review_id=review_id)

    def reset(self) -> None:
        """Clear conversation history to start a new session."""
        self._history = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history (excluding system prompt)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # ReAct loop (LLM + tool calling)
    # ------------------------------------------------------------------

    def _react_loop(self) -> str:
        """
        Run the Reason-Act-Observe loop until the LLM produces a final answer
        or ``max_iterations`` is reached.

        Each iteration:
          1. Call the LLM with the full message context + tool definitions.
          2. If the LLM requests tool calls → execute them, append results.
          3. If the LLM produces a plain-text answer → return it.
        """
        client = get_llm_client()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Build the message list: system prompt + conversation history
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
        ]

        for iteration in range(self.max_iterations):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )

            msg = response.choices[0].message

            # ---- No tool calls: LLM produced a final answer ---------------
            if not msg.tool_calls:
                return msg.content or ""

            # ---- Append the assistant message with tool_calls -------------
            # We serialise only the fields the API expects back
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            messages.append(assistant_msg)

            # ---- Execute each tool call and append results ----------------
            for tool_call in msg.tool_calls:
                result = self._tool_executor.execute(
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

        # Safety valve: return a notice if the loop limit is hit
        return (
            "已达到最大分析迭代次数，请尝试提供更具体的评论内容或重新开始对话。"
        )

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_reply(self, user_message: str) -> str:
        """
        Non-LLM fallback: run the pipeline with keyword heuristics and format
        the result as a readable string.
        """
        result = self._pipeline.analyze(user_message, review_id=str(uuid.uuid4()))
        return _format_result(result)


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _format_result(result: ReviewAnalysisResult) -> str:
    """Render a :class:`ReviewAnalysisResult` as a readable Chinese summary."""
    lines: list[str] = [f"**分析结果（评论ID: {result.review_id[:8]}…）**\n"]

    # Product recognition
    if result.product_match:
        pm = result.product_match
        lines.append(
            f"• **识别商品**：{pm.product_name}（{pm.brand}，{pm.category}，"
            f"相似度 {pm.similarity:.2%}）"
        )
    else:
        lines.append("• **识别商品**：未识别到具体商品")

    # Per-dimension scores
    if result.dimension_scores:
        lines.append("\n**各维度评分：**")
        for ds in result.dimension_scores:
            emoji = (
                "😊" if ds.sentiment == "positive"
                else ("😞" if ds.sentiment == "negative" else "😐")
            )
            lines.append(
                f"  {emoji} **{ds.dimension_name}**：{ds.score:.1f}/5 ({ds.sentiment})"
            )
            if ds.reasoning:
                lines.append(f"     ↳ {ds.reasoning}")
    else:
        lines.append("\n未检测到可分析的评价维度。")

    # Overall
    overall = f"\n**整体评价：{result.overall_sentiment}"
    if result.overall_score is not None:
        overall += f"（{result.overall_score:.1f}/5）"
    overall += "**"
    lines.append(overall)

    return "\n".join(lines)
