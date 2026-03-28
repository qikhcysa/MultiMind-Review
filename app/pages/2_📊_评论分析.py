"""Review analysis page – submit a review and view multi-dimensional sentiment analysis."""
from __future__ import annotations

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="📊 评论分析", page_icon="📊", layout="wide")
st.title("📊 多维度评论分析")
st.caption("基于 Multi-Agent + RAG 的三阶段分析流程")


@st.cache_resource(show_spinner="正在初始化分析引擎…")
def get_pipeline():
    """Initialise and cache the analysis pipeline."""
    from src.config_loader import load_products, load_dimensions, load_settings
    from src.rag import EmbeddingModel, VectorStore
    from src.audit import AuditTrail
    from src.pipeline import ReviewAnalysisPipeline

    settings = load_settings()
    products = load_products()
    dimensions = load_dimensions()

    emb_model = EmbeddingModel()
    vs = VectorStore()
    audit = AuditTrail()

    use_llm = bool(os.getenv("OPENAI_API_KEY", "").strip())

    vs_cfg = settings.get("vector_store", {})
    pipeline_cfg = settings.get("pipeline", {})

    pipeline = ReviewAnalysisPipeline(
        products=products,
        dimensions=dimensions,
        embedding_model=emb_model,
        vector_store=vs,
        audit_trail=audit,
        use_llm=use_llm,
        top_k_products=vs_cfg.get("top_k", 5),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.5),
        evidence_top_k=pipeline_cfg.get("evidence_top_k", 3),
    )
    pipeline.setup()
    return pipeline


# ---- Input ----------------------------------------------------------------
st.markdown("### 📝 输入评论")

col_input, col_options = st.columns([3, 1])
with col_input:
    review_text = st.text_area(
        "评论内容",
        placeholder="请在此输入用户评论，例如：这款手机质量非常好，拍照效果超棒！物流很快，两天就到了。客服态度也不错。",
        height=150,
    )
with col_options:
    st.markdown("**分析选项**")
    use_llm_override = st.checkbox(
        "使用 LLM 分析",
        value=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        help="需要配置 OpenAI API Key",
    )
    review_id = st.text_input("Review ID（可选）", placeholder="留空自动生成")

# ---- Sample reviews -------------------------------------------------------
with st.expander("💡 示例评论"):
    samples = [
        "这款手机质量非常好，屏幕显示效果绝佳，拍照效果也很清晰。快递速度超快，第二天就收到了。价格稍贵但性价比还行。",
        "耳机降噪效果一般，佩戴不舒服，音质还可以。物流很慢，等了一个星期。客服回复很快，态度不错。",
        "背包做工精细，防水效果好，容量大。但价格有点贵，希望能打折。",
    ]
    for i, s in enumerate(samples):
        if st.button(f"使用示例 {i+1}", key=f"sample_{i}"):
            st.session_state["sample_review"] = s
            st.rerun()

if "sample_review" in st.session_state and not review_text:
    review_text = st.session_state["sample_review"]

# ---- Analysis trigger -----------------------------------------------------
if st.button("🚀 开始分析", type="primary", disabled=not review_text.strip()):
    with st.spinner("分析中，请稍候…"):
        try:
            pipeline = get_pipeline()
            # Override LLM setting based on checkbox
            pipeline.dimension_agent.use_llm = use_llm_override
            pipeline.scoring_agent.use_llm = use_llm_override

            rid = review_id.strip() or str(uuid.uuid4())
            result = pipeline.analyze(review_text, review_id=rid)
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"分析出错: {e}")

# ---- Results display ------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("---")
    st.markdown("### 🎯 分析结果")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if result.product_match:
            st.metric("📦 识别商品", result.product_match.product_name)
        else:
            st.metric("📦 识别商品", "未识别")
    with col2:
        st.metric("📊 分析维度", f"{len(result.detected_dimensions)} 个")
    with col3:
        sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(
            result.overall_sentiment, "😐"
        )
        st.metric("💬 整体情感", f"{sentiment_icon} {result.overall_sentiment}")
    with col4:
        if result.overall_score is not None:
            st.metric("⭐ 综合评分", f"{result.overall_score:.1f} / 5.0")
        else:
            st.metric("⭐ 综合评分", "N/A")

    # Product match detail
    if result.product_match:
        with st.expander("📦 商品识别详情", expanded=True):
            pm = result.product_match
            st.markdown(
                f"**商品**: {pm.product_name} | **品牌**: {pm.brand} | "
                f"**分类**: {pm.category} | **相似度**: {pm.similarity:.2%}"
            )

    # Dimension scores
    if result.dimension_scores:
        st.markdown("#### 📊 各维度情感评分")

        # Radar chart
        labels = [s.dimension_name for s in result.dimension_scores]
        values = [s.score for s in result.dimension_scores]
        fig = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(99, 110, 250, 0.3)",
                line=dict(color="rgb(99, 110, 250)"),
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20),
        )
        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            for score in result.dimension_scores:
                sentiment_color = {
                    "positive": "🟢", "neutral": "🟡", "negative": "🔴"
                }.get(score.sentiment, "🔵")
                with st.expander(
                    f"{sentiment_color} {score.dimension_name}  ★ {score.score:.1f}",
                    expanded=False,
                ):
                    st.write(f"**情感**: {score.sentiment}")
                    st.write(f"**推理**: {score.reasoning}")
                    if score.evidence:
                        st.write("**证据片段**:")
                        for ev in score.evidence:
                            st.caption(f"- {ev}")
    else:
        st.info("未检测到可分析的维度。")
