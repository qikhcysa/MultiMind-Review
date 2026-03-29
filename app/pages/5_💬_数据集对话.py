"""Dataset conversation page – select a review dataset and chat with the agent.

Users can upload a CSV file, paste reviews manually, or use the built-in
sample dataset.  The :class:`~src.agents.dataset_agent.DatasetOrchestratorAgent`
analyses the full dataset in one go (lazy, cached) and answers natural-language
questions such as:

* "这个数据集的整体情感分布如何？"
* "哪个维度评分最低？"
* "帮我找出所有负面评论"
* "不同商品的评分对比如何？"
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="💬 数据集对话", page_icon="💬", layout="wide")
st.title("💬 数据集智能对话")
st.caption("上传评论数据集，通过自然语言与 Agent 对话完成数据分析，获取结构化结果")

# ---------------------------------------------------------------------------
# Built-in sample dataset
# ---------------------------------------------------------------------------

_SAMPLE_REVIEWS: list[str] = [
    "这款手机质量非常好，屏幕显示效果绝佳，拍照效果也很清晰。快递速度超快，第二天就收到了。价格稍贵但性价比还行。",
    "耳机降噪效果一般，佩戴不舒服，音质还可以。物流很慢，等了一个星期。客服回复很快，态度不错。",
    "背包做工精细，防水效果好，容量大。但价格有点贵，希望能打折。",
    "手机信号很差，经常断网。外观漂亮，颜色很好看。充电速度比较慢，一个小时才充50%。",
    "耳机音质很好，低音浑厚，高音清晰。包装很精美，送礼首选。物流很快，超出预期。",
    "这款背包背起来不太舒服，肩带太硬了。但容量确实很大，能装很多东西。做工还不错，看起来很耐用。",
    "手机性能强劲，玩游戏不卡顿，散热也不错。就是电池续航有点差，一天要充两次电。",
    "客服态度很差，退换货流程复杂，等了很久才处理。产品本身质量还行。",
    "耳机质量很差，用了两周就坏了。售后服务也不好，不推荐购买。",
    "物流速度很快，包装完好，产品质量不错。整体满意，下次还会购买。",
    "手机外观设计很漂亮，运行速度也很快。就是拍照效果稍微差一点，夜间模式不够好。",
    "这款包包颜色很好看，做工也不错。收到货之后发现有个小划痕，有点遗憾。",
    "耳机连接很稳定，蓝牙不断连。音质表现中规中矩，适合日常使用。客服很耐心。",
    "价格实惠，商品质量超出预期，非常满意！物流也很给力，两天就到了。",
    "产品外包装破损严重，里面的商品还好。希望商家加强包装保护。",
]

_EXAMPLE_QUESTIONS: list[str] = [
    "这个数据集的整体情感分布如何？",
    "哪个评价维度得分最低？",
    "帮我找出所有负面评论",
    "不同商品的评分对比如何？",
    "物流维度的详细统计是什么？",
    "综合评分最低的 5 条评论是哪些？",
]


# ---------------------------------------------------------------------------
# Cached pipeline initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="正在初始化分析引擎…")
def _get_pipeline():
    from src.config_loader import load_products, load_dimensions, load_settings
    from src.rag import EmbeddingModel, VectorStore
    from src.audit import AuditTrail
    from src.pipeline import ReviewAnalysisPipeline

    settings = load_settings()
    products = load_products()
    dimensions = load_dimensions()

    emb = EmbeddingModel()
    vs = VectorStore()
    audit = AuditTrail()
    use_llm = bool(os.getenv("OPENAI_API_KEY", "").strip())

    vs_cfg = settings.get("vector_store", {})
    pipeline_cfg = settings.get("pipeline", {})

    pipeline = ReviewAnalysisPipeline(
        products=products,
        dimensions=dimensions,
        embedding_model=emb,
        vector_store=vs,
        audit_trail=audit,
        use_llm=use_llm,
        top_k_products=vs_cfg.get("top_k", 5),
        similarity_threshold=vs_cfg.get("similarity_threshold", 0.5),
        evidence_top_k=pipeline_cfg.get("evidence_top_k", 3),
    )
    pipeline.setup()
    return pipeline


# ---------------------------------------------------------------------------
# Helper: render structured results panel
# ---------------------------------------------------------------------------

def _render_results_panel(agent) -> None:
    """Render sentiment + dimension charts based on cached analysis results."""
    results = agent.analysis_results
    if not results:
        return

    with st.expander("📊 结构化分析结果（点击展开）", expanded=False):
        sentiments = [r.overall_sentiment for r in results]
        sent_counts = {
            "正面": sentiments.count("positive"),
            "中性": sentiments.count("neutral"),
            "负面": sentiments.count("negative"),
        }

        col_a, col_b = st.columns(2)

        with col_a:
            fig_pie = px.pie(
                values=list(sent_counts.values()),
                names=list(sent_counts.keys()),
                color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                title="情感分布",
            )
            fig_pie.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            dim_scores: dict[str, list[float]] = {}
            for r in results:
                for ds in r.dimension_scores:
                    dim_scores.setdefault(ds.dimension_name, []).append(ds.score)
            if dim_scores:
                dim_avgs = {
                    k: round(sum(v) / len(v), 2) for k, v in dim_scores.items()
                }
                fig_bar = px.bar(
                    x=list(dim_avgs.keys()),
                    y=list(dim_avgs.values()),
                    title="各维度平均评分",
                    color=list(dim_avgs.values()),
                    color_continuous_scale="RdYlGn",
                    range_y=[0, 5],
                    labels={"x": "维度", "y": "平均分"},
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(t=40, b=0),
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Summary metrics row
        total = len(results)
        scores = [r.overall_score for r in results if r.overall_score is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else None

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("📝 评论总数", total)
        mc2.metric("😊 正面", sent_counts["正面"])
        mc3.metric("😞 负面", sent_counts["负面"])
        mc4.metric("⭐ 平均分", f"{avg_score:.2f}" if avg_score else "N/A")


# ---------------------------------------------------------------------------
# Sidebar: dataset management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📂 数据集管理")

    data_source = st.radio(
        "数据来源",
        options=["使用内置示例数据", "上传 CSV 文件", "手动输入评论"],
        key="ds_source",
    )

    reviews: list[str] = []

    if data_source == "使用内置示例数据":
        reviews = _SAMPLE_REVIEWS
        st.success(f"✅ 已准备 {len(reviews)} 条示例评论")

    elif data_source == "上传 CSV 文件":
        uploaded = st.file_uploader(
            "上传包含评论的 CSV 文件",
            type=["csv"],
            help="CSV 需包含 'review'、'评论'、'text' 或 'content' 列",
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                candidates = [
                    c
                    for c in df.columns
                    if c.lower() in {"review", "评论", "text", "content", "评论内容"}
                ]
                if candidates:
                    review_col = candidates[0]
                    st.info(f"自动识别列：**{review_col}**")
                else:
                    review_col = st.selectbox("选择评论列", df.columns.tolist())
                if review_col:
                    reviews = df[review_col].dropna().astype(str).tolist()
                    st.success(f"✅ 已加载 {len(reviews)} 条评论")
                    with st.expander("预览数据", expanded=False):
                        st.dataframe(df[[review_col]].head(5), use_container_width=True)
            except Exception as exc:
                st.error(f"文件读取失败: {exc}")

    else:  # 手动输入
        raw = st.text_area(
            "每行输入一条评论",
            placeholder="这款手机质量很好…\n物流太慢了…\n客服态度不错…",
            height=200,
        )
        if raw.strip():
            reviews = [line.strip() for line in raw.splitlines() if line.strip()]
            st.success(f"✅ 已输入 {len(reviews)} 条评论")

    st.markdown("---")

    load_btn = st.button(
        "🚀 加载数据集并开始对话",
        type="primary",
        use_container_width=True,
        disabled=not reviews,
    )
    if load_btn:
        from src.agents.dataset_agent import DatasetOrchestratorAgent

        pipeline = _get_pipeline()
        use_llm = bool(os.getenv("OPENAI_API_KEY", "").strip())
        agent = DatasetOrchestratorAgent(pipeline, reviews, use_llm=use_llm)
        st.session_state["ds_agent"] = agent
        st.session_state["ds_messages"] = []
        st.rerun()

    if st.session_state.get("ds_agent") is not None:
        agent_loaded = st.session_state["ds_agent"]
        st.info(
            f"📊 当前数据集：**{agent_loaded.review_count}** 条评论  \n"
            f"🔬 分析状态：{'✅ 已分析' if agent_loaded.is_analyzed else '⏳ 待分析'}"
        )

    st.markdown("---")
    st.markdown("### 💡 示例问题")
    for q in _EXAMPLE_QUESTIONS:
        if st.button(q, key=f"eq_{hash(q)}", use_container_width=True):
            st.session_state["ds_preset"] = q
            st.rerun()


# ---------------------------------------------------------------------------
# Main area: welcome screen or chat interface
# ---------------------------------------------------------------------------

if st.session_state.get("ds_agent") is None:
    st.info("👈 请先在左侧选择数据来源，然后点击「加载数据集并开始对话」")
    st.markdown(
        """
        ### 功能介绍

        本页面让您通过**自然语言对话**分析一批商品评论，无需编写代码。

        | 步骤 | 操作 |
        |------|------|
        | 1️⃣ | 在左侧选择数据来源（示例数据 / 上传 CSV / 手动输入） |
        | 2️⃣ | 点击「加载数据集并开始对话」 |
        | 3️⃣ | 在对话框中用自然语言提问 |
        | 4️⃣ | Agent 自动分析并返回结构化结果 |

        ### 支持的问题类型

        | 类型 | 示例 |
        |------|------|
        | 整体统计 | "这个数据集有多少评论？情感分布如何？" |
        | 维度分析 | "哪个维度评分最低？物流评分如何？" |
        | 评论筛选 | "找出所有负面评论" "评分最低的5条是哪些？" |
        | 商品对比 | "哪个商品口碑最好？各商品评分对比？" |
        | 深度洞察 | "根据分析结果，有哪些改进建议？" |

        > 💡 首次提问时 Agent 会自动对全部评论进行分析（进度稍慢），
        > 后续追问直接复用缓存结果，响应迅速。
        """
    )

else:
    agent: "DatasetOrchestratorAgent" = st.session_state["ds_agent"]

    # Top status bar
    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        st.metric("📊 数据集大小", f"{agent.review_count} 条评论")
    with col2:
        status = "✅ 已分析" if agent.is_analyzed else "⏳ 待分析（首次提问时自动触发）"
        st.metric("🔬 分析状态", status)
    with col3:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state["ds_messages"] = []
            agent.reset()
            st.rerun()

    # Structured results panel (visible after analysis is complete)
    if agent.is_analyzed:
        _render_results_panel(agent)

    st.markdown("---")

    # Initialise message log
    if "ds_messages" not in st.session_state:
        st.session_state["ds_messages"] = []

    # Render conversation history
    for msg in st.session_state["ds_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Check for a preset question triggered by the sidebar buttons
    preset: str | None = st.session_state.pop("ds_preset", None)

    # Chat input (preset takes priority over manual typing)
    user_input: str | None = st.chat_input(
        "输入问题，例如：这个数据集的整体情感分布如何？",
        key="ds_chat_input",
    ) or preset

    if user_input:
        st.session_state["ds_messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent 分析中，请稍候…"):
                try:
                    reply = agent.chat(user_input)
                except Exception as exc:
                    reply = f"❌ 分析出错: {exc}"
            st.markdown(reply)

        st.session_state["ds_messages"].append(
            {"role": "assistant", "content": reply}
        )

        # Rerun to refresh the results panel after the first batch analysis
        st.rerun()
