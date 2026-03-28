"""Review clustering page – cluster reviews by dimension with LLM summaries."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.express as px
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="🔗 评论聚类", page_icon="🔗", layout="wide")
st.title("🔗 评论聚类总结")
st.caption("基于向量嵌入与密度聚类，自动归纳同维度评论，并利用大模型生成类簇总结词")


@st.cache_resource(show_spinner="初始化聚类引擎…")
def get_clusterer(eps: float, min_samples: int, use_llm: bool):
    from src.rag import EmbeddingModel
    from src.clustering import ReviewClusterer

    return ReviewClusterer(
        embedding_model=EmbeddingModel(),
        eps=eps,
        min_samples=min_samples,
        use_llm=use_llm,
    )


# ---- Load config -----------------------------------------------------------
try:
    from src.config_loader import load_dimensions, load_settings
    dimensions = load_dimensions()
    settings = load_settings()
    cluster_cfg = settings.get("clustering", {})
    default_eps = float(cluster_cfg.get("eps", 0.3))
    default_min_samples = int(cluster_cfg.get("min_samples", 2))
except Exception:
    dimensions = []
    default_eps = 0.3
    default_min_samples = 2

# ---- Sidebar params --------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 聚类参数")
    if dimensions:
        dim_options = {f"{d.name}（{d.id}）": d for d in dimensions}
        selected_dim_label = st.selectbox("选择分析维度", list(dim_options.keys()))
        selected_dim = dim_options[selected_dim_label]
    else:
        st.warning("请先在配置管理页面添加分析维度")
        selected_dim = None

    eps = st.slider("DBSCAN eps（聚类半径）", 0.05, 1.0, default_eps, 0.05)
    min_samples = st.number_input("DBSCAN min_samples", 1, 20, default_min_samples)
    use_llm = st.checkbox(
        "使用 LLM 生成总结",
        value=bool(os.getenv("OPENAI_API_KEY", "").strip()),
    )

# ---- Review input ----------------------------------------------------------
st.markdown("### 📝 输入评论（每行一条）")

sample_reviews = """这款手机质量非常好，做工精细，材质高档
手机质量差，用了一个月就坏了
做工很粗糙，屏幕容易碎
质量超好，用了两年依然如新
物流超快，第二天就到了
包装很结实，没有损坏
快递速度太慢了，等了一周
物流不错，但包装有点简陋
价格太贵了，性价比不高
价格实惠，物超所值
价格合理，性价比很高
"""

reviews_input = st.text_area(
    "评论内容（每行一条）",
    value=sample_reviews,
    height=250,
    help="每行输入一条评论，系统将自动进行聚类",
)

# ---- Clustering action -----------------------------------------------------
if st.button("🔗 开始聚类", type="primary"):
    reviews = [r.strip() for r in reviews_input.strip().split("\n") if r.strip()]
    if not reviews:
        st.error("请输入至少一条评论")
    elif selected_dim is None:
        st.error("请先选择分析维度")
    elif len(reviews) < 2:
        st.error("请输入至少 2 条评论才能进行聚类")
    else:
        with st.spinner(f"正在对 {len(reviews)} 条评论进行聚类…"):
            try:
                # Invalidate cache if params changed
                clusterer = get_clusterer.__wrapped__(eps, min_samples, use_llm)
                cluster_result = clusterer.cluster(
                    reviews=reviews,
                    dimension_id=selected_dim.id,
                    dimension_name=selected_dim.name,
                )
                st.session_state["cluster_result"] = cluster_result
                st.session_state["cluster_reviews"] = reviews
            except Exception as e:
                st.error(f"聚类出错: {e}")
                import traceback
                st.code(traceback.format_exc())

# ---- Results ---------------------------------------------------------------
if "cluster_result" in st.session_state:
    result = st.session_state["cluster_result"]
    reviews = st.session_state["cluster_reviews"]

    st.markdown("---")
    st.markdown("### 📊 聚类结果")

    col1, col2, col3 = st.columns(3)
    col1.metric("总评论数", result.total_reviews)
    col2.metric("有效类簇数", result.num_clusters)
    col3.metric("噪声点数", result.noise_count)

    if result.clusters:
        # Cluster summary cards
        st.markdown("#### 🗂️ 类簇详情")
        cols = st.columns(min(len(result.clusters), 3))
        for i, cluster in enumerate(result.clusters):
            with cols[i % 3]:
                avg_label = (
                    f"平均评分: ★{cluster.avg_sentiment_score:.1f}"
                    if cluster.avg_sentiment_score
                    else ""
                )
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:4px 0;">
                        <h4 style="margin:0; color:#4F8BF9;">类簇 {cluster.cluster_id + 1}</h4>
                        <p style="font-size:1.3em; font-weight:bold; margin:4px 0;">「{cluster.summary}」</p>
                        <p style="margin:0; color:#888;">📝 {cluster.size} 条评论 {avg_label}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("查看代表性评论"):
                    for rev in cluster.representative_reviews:
                        st.caption(f"• {rev}")

        # Bar chart
        chart_data = pd.DataFrame(
            {
                "类簇": [f"类簇{c.cluster_id + 1}：{c.summary}" for c in result.clusters],
                "评论数": [c.size for c in result.clusters],
            }
        )
        fig = px.bar(
            chart_data,
            x="类簇",
            y="评论数",
            title=f"{result.dimension_name} 维度评论分布",
            color="评论数",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("未能形成有效类簇，请尝试调整 eps 或 min_samples 参数，或增加评论数量。")

    if result.noise_count > 0:
        st.caption(f"ℹ️ {result.noise_count} 条评论被识别为噪声点（未归属任何类簇）")
