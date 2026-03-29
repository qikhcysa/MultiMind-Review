"""MultiMind-Review – Streamlit Main Application."""
from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="MultiMind Review 智能评论分析系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    st.title("🧠 MultiMind Review")
    st.subheader("智能商品评论分析系统")

    st.markdown(
        """
        欢迎使用 **MultiMind Review** – 基于 Multi-Agent + RAG 架构的智能评论分析平台。

        ### 核心功能

        | 功能模块 | 说明 |
        |---|---|
        | 📦 **商品识别** | 基于向量知识库与实体链接，精准匹配评论对应商品 |
        | 📊 **多维情感分析** | 多智能体协作完成维度识别→证据检索→评分的三阶段分析 |
        | 🔗 **评论聚类** | 基于密度聚类自动归纳同维度评论，LLM 生成类簇总结词 |
        | 💬 **数据集智能对话** | 上传评论数据集，通过自然语言与 Agent 对话完成批量数据分析 |
        | ⚙️ **无代码配置** | 图形化界面动态定义商品库、分类维度与检索参数 |
        | 📋 **审计追踪** | 端到端记录每条决策的检索邻居、相似度与智能体输出 |

        ### 使用说明

        1. 前往 **⚙️ 配置管理** 设置商品库、维度与 LLM 参数
        2. 前往 **📊 评论分析** 输入评论进行智能分析（支持单条分析和 Agent 对话）
        3. 前往 **🔗 评论聚类** 对批量评论进行聚类总结
        4. 前往 **💬 数据集对话** 上传数据集，用自然语言与 Agent 对话进行批量分析
        5. 前往 **📋 审计追踪** 查看所有决策记录

        > **注意**：首次使用需在 **⚙️ 配置管理** 页面配置 OpenAI API Key（或兼容接口）。
        """
    )

    # Quick system status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            from src.config_loader import load_products
            products = load_products()
            st.metric("📦 商品知识库", f"{len(products)} 个商品")
        except Exception:
            st.metric("📦 商品知识库", "未加载")

    with col2:
        try:
            from src.config_loader import load_dimensions
            dims = load_dimensions()
            st.metric("📊 分析维度", f"{len(dims)} 个维度")
        except Exception:
            st.metric("📊 分析维度", "未加载")

    with col3:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key != "your_openai_api_key_here":
            st.metric("🔑 LLM 状态", "✅ 已配置")
        else:
            st.metric("🔑 LLM 状态", "⚠️ 未配置")


if __name__ == "__main__":
    main()
