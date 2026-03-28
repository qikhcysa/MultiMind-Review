"""Audit trail page – view and explore all agent decision records."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="📋 审计追踪", page_icon="📋", layout="wide")
st.title("📋 审计追踪")
st.caption("端到端记录每条决策的检索邻居、相似度与智能体输出，保障结果可解释与持续优化")


@st.cache_resource(show_spinner=False)
def get_audit_trail():
    from src.audit import AuditTrail
    return AuditTrail()


audit_trail = get_audit_trail()

# ---- Sidebar filters -------------------------------------------------------
with st.sidebar:
    st.header("🔍 筛选条件")

    available_dates = audit_trail.available_dates()
    if available_dates:
        selected_date = st.selectbox("选择日期", ["（当前会话）"] + available_dates)
    else:
        selected_date = "（当前会话）"

    stage_filter = st.multiselect(
        "按阶段筛选",
        options=["product_recognition", "dimension_detection", "evidence_retrieval", "sentiment_scoring"],
        default=[],
        format_func=lambda x: {
            "product_recognition": "商品识别",
            "dimension_detection": "维度检测",
            "evidence_retrieval": "证据检索",
            "sentiment_scoring": "情感评分",
        }.get(x, x),
    )

    review_id_filter = st.text_input("按 Review ID 筛选", placeholder="输入完整或部分 ID")

# ---- Load entries ----------------------------------------------------------
if selected_date == "（当前会话）":
    entries = audit_trail.get_all()
else:
    entries = audit_trail.load_from_date(selected_date)

# Apply filters
if stage_filter:
    entries = [e for e in entries if e.stage in stage_filter]
if review_id_filter:
    entries = [e for e in entries if review_id_filter in e.review_id]

# ---- Summary metrics -------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📋 记录总数", len(entries))
review_ids = {e.review_id for e in entries}
col2.metric("📝 涉及评论", len(review_ids))
stages = {e.stage for e in entries}
col3.metric("🔄 覆盖阶段", len(stages))

if not entries:
    st.info("暂无审计记录。请先在「评论分析」页面进行分析，记录将在此展示。")
    st.stop()

# ---- Main table ------------------------------------------------------------
st.markdown("### 📊 审计记录总览")

df = audit_trail.to_dataframe(entries)
if not df.empty:
    stage_labels = {
        "product_recognition": "商品识别",
        "dimension_detection": "维度检测",
        "evidence_retrieval": "证据检索",
        "sentiment_scoring": "情感评分",
    }
    df["stage_label"] = df["stage"].map(stage_labels).fillna(df["stage"])

    display_cols = ["review_id", "stage_label", "agent_name", "top_similarity", "num_neighbors", "timestamp"]
    display_df = df[display_cols].copy()
    display_df.columns = ["Review ID", "阶段", "智能体", "最高相似度", "邻居数", "时间戳"]
    display_df["Review ID"] = display_df["Review ID"].str[:12] + "…"

    st.dataframe(display_df, use_container_width=True, height=300)

# ---- Detail view -----------------------------------------------------------
st.markdown("### 🔍 详细审计记录")

if review_ids:
    selected_review = st.selectbox(
        "选择 Review ID 查看详情",
        options=sorted(review_ids),
        format_func=lambda x: x[:20] + "…" if len(x) > 20 else x,
    )
    review_entries = [e for e in entries if e.review_id == selected_review]

    for entry in review_entries:
        stage_icon = {
            "product_recognition": "📦",
            "dimension_detection": "📊",
            "evidence_retrieval": "🔍",
            "sentiment_scoring": "⭐",
        }.get(entry.stage, "📋")
        stage_name = stage_labels.get(entry.stage, entry.stage)

        with st.expander(
            f"{stage_icon} [{stage_name}] {entry.agent_name} – {entry.timestamp.strftime('%H:%M:%S')}",
            expanded=False,
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📥 输入**")
                st.json(entry.input_data)
            with col2:
                st.markdown("**📤 输出**")
                st.json(entry.output_data)

            if entry.reasoning:
                st.markdown(f"**🧠 推理过程**: {entry.reasoning}")

            if entry.retrieved_neighbors:
                st.markdown("**🔗 检索邻居**")
                neighbor_data = []
                for j, n in enumerate(entry.retrieved_neighbors):
                    neighbor_data.append(
                        {
                            "排名": j + 1,
                            "ID": n.get("id", ""),
                            "文档片段": str(n.get("document", ""))[:80] + "…",
                            "相似度": f"{n.get('similarity', 0):.4f}",
                        }
                    )
                st.dataframe(pd.DataFrame(neighbor_data), use_container_width=True)

            st.caption(f"Entry ID: {entry.entry_id} | Review ID: {entry.review_id}")
