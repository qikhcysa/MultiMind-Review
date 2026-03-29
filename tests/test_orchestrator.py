"""Tests for OrchestratorAgent: ReAct loop, multi-turn conversation, tool calling."""
from __future__ import annotations

import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.models import ProductInfo, Dimension
from src.audit import AuditTrail
from tests.conftest import MockEmbeddingModel


# ---------------------------------------------------------------------------
# Fixtures (reuse pipeline from test_pipeline.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_products():
    return [
        ProductInfo(
            id="phone_test",
            name="测试智能手机",
            brand="TestBrand",
            category="电子/手机",
            description="一款测试用的智能手机",
            features=["6寸屏幕"],
            keywords=["手机", "智能手机"],
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
            keywords=["质量", "做工", "耐用", "好", "差"],
        ),
        Dimension(
            id="delivery",
            name="物流配送",
            name_en="Delivery",
            description="发货速度、包装、物流",
            keywords=["物流", "快递", "发货", "到货"],
        ),
    ]


@pytest.fixture
def pipeline(sample_products, sample_dimensions, tmp_path):
    from src.rag import VectorStore
    from src.pipeline import ReviewAnalysisPipeline

    emb = MockEmbeddingModel()
    vs = VectorStore(persist_dir=str(tmp_path / "chroma_orch"))
    audit = AuditTrail(log_dir=str(tmp_path / "audit"), enabled=False)
    p = ReviewAnalysisPipeline(
        products=sample_products,
        dimensions=sample_dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit,
        use_llm=False,
        similarity_threshold=0.2,
    )
    p.setup()
    return p


# ---------------------------------------------------------------------------
# Helper: build a mock OpenAI response
# ---------------------------------------------------------------------------

def _mock_final_response(content: str):
    """Return a mock response object with no tool calls and the given text."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_tool_call_response(tool_name: str, arguments: dict):
    """Return a mock response that requests a single tool call."""
    tc = MagicMock()
    tc.id = f"call_{uuid.uuid4().hex[:8]}"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments, ensure_ascii=False)

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Tests: no-LLM heuristic mode
# ---------------------------------------------------------------------------

class TestOrchestratorHeuristic:
    """Orchestrator with use_llm=False falls back to the pipeline heuristic."""

    def test_chat_returns_string(self, pipeline):
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=False)
        reply = agent.chat("这款手机质量非常好，物流也很快")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_history_grows_with_each_turn(self, pipeline):
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=False)
        agent.chat("手机质量很好")
        assert len(agent.history) == 2  # user + assistant

        agent.chat("物流如何？")
        assert len(agent.history) == 4  # 2 more

    def test_reset_clears_history(self, pipeline):
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=False)
        agent.chat("手机质量很好")
        agent.reset()
        assert agent.history == []

    def test_analyze_returns_result_object(self, pipeline):
        from src.agents import OrchestratorAgent
        from src.models import ReviewAnalysisResult

        agent = OrchestratorAgent(pipeline, use_llm=False)
        result = agent.analyze("手机质量很棒，快递速度很快")
        assert isinstance(result, ReviewAnalysisResult)
        assert result.review_text == "手机质量很棒，快递速度很快"

    def test_analyze_does_not_pollute_history(self, pipeline):
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=False)
        agent.analyze("任意评论")
        assert agent.history == []


# ---------------------------------------------------------------------------
# Tests: LLM mode (mocked)
# ---------------------------------------------------------------------------

class TestOrchestratorLLMMode:
    """Orchestrator with use_llm=True; LLM is mocked via unittest.mock."""

    def test_react_loop_no_tool_calls(self, pipeline):
        """When LLM returns a final answer immediately (no tool calls), return it."""
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=True)
        final_reply = "这是最终答案。"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_final_response(final_reply)

        with patch("src.agents.orchestrator_agent.get_llm_client", return_value=mock_client):
            reply = agent.chat("这款手机质量非常好")

        assert reply == final_reply
        assert len(agent.history) == 2

    def test_react_loop_one_tool_call_then_final(self, pipeline):
        """LLM calls one tool, then gives a final answer."""
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=True)

        tool_response = _mock_tool_call_response(
            "detect_dimensions", {"review_text": "手机质量很好"}
        )
        final_response = _mock_final_response("分析完成，检测到产品质量维度，评价正面。")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [tool_response, final_response]

        with patch("src.agents.orchestrator_agent.get_llm_client", return_value=mock_client):
            reply = agent.chat("手机质量很好")

        assert "完成" in reply or "维度" in reply or isinstance(reply, str)
        # LLM was called twice (once for tool call, once for final answer)
        assert mock_client.chat.completions.create.call_count == 2

    def test_react_loop_max_iterations_guard(self, pipeline):
        """When the LLM keeps requesting tool calls, stop at max_iterations."""
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=True, max_iterations=3)

        # Always return a tool call — never a final answer
        infinite_tool = _mock_tool_call_response(
            "detect_dimensions", {"review_text": "测试"}
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = infinite_tool

        with patch("src.agents.orchestrator_agent.get_llm_client", return_value=mock_client):
            reply = agent.chat("测试无限循环")

        assert isinstance(reply, str)
        # Should not have looped more than max_iterations times
        assert mock_client.chat.completions.create.call_count <= 3

    def test_multi_turn_history_sent_to_llm(self, pipeline):
        """Each new chat() call includes the full conversation history."""
        from src.agents import OrchestratorAgent

        agent = OrchestratorAgent(pipeline, use_llm=True)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_final_response("好的")

        with patch("src.agents.orchestrator_agent.get_llm_client", return_value=mock_client):
            agent.chat("第一条消息")
            agent.chat("第二条消息")

        calls = mock_client.chat.completions.create.call_args_list
        # Extract the 'messages' kwarg from each call
        def _get_messages(call):
            return call.kwargs["messages"] if "messages" in call.kwargs else call.args[0]

        first_msgs = _get_messages(calls[0])
        second_msgs = _get_messages(calls[1])
        # Second call should include more messages than first (history grows)
        assert len(second_msgs) > len(first_msgs)


# ---------------------------------------------------------------------------
# Tests: ToolExecutor
# ---------------------------------------------------------------------------

class TestToolExecutor:
    """Unit tests for ToolExecutor dispatching."""

    def test_unknown_tool_returns_error(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute("nonexistent_tool", "{}")
        assert "error" in result

    def test_invalid_json_returns_error(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute("detect_dimensions", "not_json")
        assert "error" in result

    def test_detect_dimensions_returns_list(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute(
            "detect_dimensions",
            json.dumps({"review_text": "质量很好，快递也快"})
        )
        assert "detected_dimensions" in result
        assert isinstance(result["detected_dimensions"], list)

    def test_recognize_product_returns_dict(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute(
            "recognize_product",
            json.dumps({"review_text": "这款智能手机性能很强"})
        )
        assert "matched" in result

    def test_retrieve_evidence_unknown_dim_returns_message(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute(
            "retrieve_evidence",
            json.dumps({"review_text": "任意评论", "dimension_ids": ["nonexistent_dim"]})
        )
        assert "message" in result or "evidence_map" in result

    def test_retrieve_evidence_valid_dim(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        result = executor.execute(
            "retrieve_evidence",
            json.dumps({"review_text": "质量好，快递快", "dimension_ids": ["quality"]})
        )
        assert "evidence_map" in result
        assert isinstance(result["evidence_map"], dict)

    def test_score_sentiment_returns_scores(self, pipeline):
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        evidence_map = {"quality": ["质量非常好", "做工精细"]}
        result = executor.execute(
            "score_sentiment",
            json.dumps({"review_id": "test_review_001", "evidence_map": evidence_map})
        )
        assert "scores" in result
        assert isinstance(result["scores"], list)

    def test_full_tool_sequence(self, pipeline):
        """Simulate the full recommended tool calling sequence."""
        from src.agents.tools import ToolExecutor

        executor = ToolExecutor(pipeline)
        review = "这款手机质量非常好，物流也很快"

        # Step 1: product recognition
        prod_result = executor.execute(
            "recognize_product", json.dumps({"review_text": review})
        )
        assert "matched" in prod_result

        # Step 2: dimension detection
        dim_result = executor.execute(
            "detect_dimensions", json.dumps({"review_text": review})
        )
        assert "detected_dimensions" in dim_result
        dim_ids = [d["id"] for d in dim_result["detected_dimensions"]]

        # Step 3: evidence retrieval (only if dimensions detected)
        if dim_ids:
            ev_result = executor.execute(
                "retrieve_evidence",
                json.dumps({"review_text": review, "dimension_ids": dim_ids})
            )
            assert "evidence_map" in ev_result

            # Step 4: sentiment scoring
            score_result = executor.execute(
                "score_sentiment",
                json.dumps({
                    "review_id": "seq_test_001",
                    "evidence_map": ev_result["evidence_map"],
                })
            )
            assert "scores" in score_result
