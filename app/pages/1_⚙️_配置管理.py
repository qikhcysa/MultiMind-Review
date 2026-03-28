"""Configuration management page – no-code interface for products, dimensions, and settings."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.config_loader import (
    load_dimensions,
    load_products,
    load_settings,
    save_dimensions,
    save_products,
    save_settings,
)
from src.models import Dimension, ProductInfo


st.set_page_config(page_title="⚙️ 配置管理", page_icon="⚙️", layout="wide")
st.title("⚙️ 配置管理")
st.caption("无需编程，通过图形界面动态配置商品库、分析维度与系统参数")

tab_dims, tab_products, tab_settings = st.tabs(["📊 分析维度", "📦 商品知识库", "🔧 系统设置"])

# ============================================================
# Tab 1: Dimensions
# ============================================================
with tab_dims:
    st.subheader("分析维度管理")
    st.info("在此定义情感分析的维度。智能体将按照这些维度对评论进行分析。")

    try:
        dimensions = load_dimensions()
    except Exception as e:
        st.error(f"加载维度配置失败: {e}")
        dimensions = []

    # Show existing dimensions
    for i, dim in enumerate(dimensions):
        with st.expander(f"📌 {dim.name}（{dim.id}）", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("维度名称", value=dim.name, key=f"dim_name_{i}")
                new_name_en = st.text_input("英文名", value=dim.name_en, key=f"dim_name_en_{i}")
            with col2:
                new_desc = st.text_area("描述", value=dim.description, key=f"dim_desc_{i}", height=100)
            new_kws = st.text_input(
                "关键词（逗号分隔）",
                value=", ".join(dim.keywords),
                key=f"dim_kws_{i}",
            )
            col_save, col_del = st.columns([1, 5])
            with col_save:
                if st.button("保存", key=f"dim_save_{i}"):
                    dimensions[i] = Dimension(
                        id=dim.id,
                        name=new_name,
                        name_en=new_name_en,
                        description=new_desc,
                        keywords=[k.strip() for k in new_kws.split(",") if k.strip()],
                    )
                    try:
                        save_dimensions(dimensions)
                        st.success("维度已保存！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败: {e}")
            with col_del:
                if st.button("🗑️ 删除", key=f"dim_del_{i}"):
                    dimensions.pop(i)
                    try:
                        save_dimensions(dimensions)
                        st.success("维度已删除！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

    st.markdown("---")
    st.subheader("➕ 添加新维度")
    with st.form("add_dimension_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.text_input("维度 ID（唯一标识，英文）", placeholder="e.g. packaging")
            new_name = st.text_input("维度名称", placeholder="e.g. 包装")
            new_name_en = st.text_input("英文名", placeholder="e.g. Packaging")
        with col2:
            new_desc = st.text_area("描述", placeholder="描述该维度涵盖的内容", height=100)
            new_kws = st.text_input("关键词（逗号分隔）", placeholder="包装, 盒子, 外包, package, box")
        submitted = st.form_submit_button("添加维度")
        if submitted:
            if not new_id or not new_name:
                st.error("维度 ID 和名称不能为空")
            elif any(d.id == new_id for d in dimensions):
                st.error(f"维度 ID '{new_id}' 已存在")
            else:
                dimensions.append(
                    Dimension(
                        id=new_id,
                        name=new_name,
                        name_en=new_name_en or new_name,
                        description=new_desc,
                        keywords=[k.strip() for k in new_kws.split(",") if k.strip()],
                    )
                )
                try:
                    save_dimensions(dimensions)
                    st.success(f"维度 '{new_name}' 已添加！")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")

# ============================================================
# Tab 2: Products
# ============================================================
with tab_products:
    st.subheader("商品知识库管理")
    st.info("在此维护商品知识库。系统将基于向量相似度将评论与对应商品匹配。")

    try:
        products = load_products()
    except Exception as e:
        st.error(f"加载商品库失败: {e}")
        products = []

    for i, prod in enumerate(products):
        with st.expander(f"📦 {prod.name}（{prod.id}）- {prod.brand}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("商品名称", value=prod.name, key=f"prod_name_{i}")
                new_brand = st.text_input("品牌", value=prod.brand, key=f"prod_brand_{i}")
                new_cat = st.text_input("分类", value=prod.category, key=f"prod_cat_{i}")
            with col2:
                new_desc = st.text_area("描述", value=prod.description, key=f"prod_desc_{i}", height=80)
                new_features = st.text_area(
                    "产品特性（每行一条）",
                    value="\n".join(prod.features),
                    key=f"prod_feats_{i}",
                    height=80,
                )
            new_kws = st.text_input(
                "关键词（逗号分隔）",
                value=", ".join(prod.keywords),
                key=f"prod_kws_{i}",
            )
            col_save, col_del = st.columns([1, 5])
            with col_save:
                if st.button("保存", key=f"prod_save_{i}"):
                    products[i] = ProductInfo(
                        id=prod.id,
                        name=new_name,
                        brand=new_brand,
                        category=new_cat,
                        description=new_desc,
                        features=[f.strip() for f in new_features.split("\n") if f.strip()],
                        keywords=[k.strip() for k in new_kws.split(",") if k.strip()],
                    )
                    try:
                        save_products(products)
                        st.success("商品已保存！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败: {e}")
            with col_del:
                if st.button("🗑️ 删除", key=f"prod_del_{i}"):
                    products.pop(i)
                    try:
                        save_products(products)
                        st.success("商品已删除！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

    st.markdown("---")
    st.subheader("➕ 添加新商品")
    with st.form("add_product_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.text_input("商品 ID（唯一标识，英文）", placeholder="e.g. tablet_001")
            new_name = st.text_input("商品名称", placeholder="e.g. 平板电脑 Pro")
            new_brand = st.text_input("品牌", placeholder="e.g. TechBrand")
            new_cat = st.text_input("分类", placeholder="e.g. 电子产品/平板")
        with col2:
            new_desc = st.text_area("描述", placeholder="商品的主要描述", height=80)
            new_features = st.text_area("产品特性（每行一条）", placeholder="10.5英寸屏幕\n256GB存储", height=80)
            new_kws = st.text_input("关键词（逗号分隔）", placeholder="平板, 平板电脑, ipad")
        submitted = st.form_submit_button("添加商品")
        if submitted:
            if not new_id or not new_name:
                st.error("商品 ID 和名称不能为空")
            elif any(p.id == new_id for p in products):
                st.error(f"商品 ID '{new_id}' 已存在")
            else:
                products.append(
                    ProductInfo(
                        id=new_id,
                        name=new_name,
                        brand=new_brand,
                        category=new_cat,
                        description=new_desc,
                        features=[f.strip() for f in new_features.split("\n") if f.strip()],
                        keywords=[k.strip() for k in new_kws.split(",") if k.strip()],
                    )
                )
                try:
                    save_products(products)
                    st.success(f"商品 '{new_name}' 已添加！")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")

# ============================================================
# Tab 3: Settings
# ============================================================
with tab_settings:
    st.subheader("系统设置")

    try:
        settings = load_settings()
    except Exception as e:
        st.error(f"加载设置失败: {e}")
        settings = {}

    st.markdown("#### 🔑 LLM 配置")
    col1, col2 = st.columns(2)
    with col1:
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="填写 OpenAI API Key 或兼容接口的密钥",
        )
        model_input = st.text_input(
            "模型名称",
            value=os.getenv("OPENAI_MODEL", settings.get("llm", {}).get("model", "gpt-4o-mini")),
        )
    with col2:
        base_url_input = st.text_input(
            "API Base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            help="如果使用兼容接口，请修改此 URL",
        )
        temperature_input = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(settings.get("llm", {}).get("temperature", 0.1)),
            step=0.1,
        )

    st.markdown("#### 🔍 检索参数")
    col1, col2, col3 = st.columns(3)
    vs_settings = settings.get("vector_store", {})
    with col1:
        top_k_input = st.number_input(
            "商品检索 Top-K",
            min_value=1, max_value=20,
            value=int(vs_settings.get("top_k", 5)),
        )
    with col2:
        threshold_input = st.slider(
            "相似度阈值",
            min_value=0.0, max_value=1.0,
            value=float(vs_settings.get("similarity_threshold", 0.5)),
            step=0.05,
        )
    with col3:
        evidence_k_input = st.number_input(
            "证据检索 Top-K",
            min_value=1, max_value=10,
            value=int(settings.get("pipeline", {}).get("evidence_top_k", 3)),
        )

    st.markdown("#### 🔗 聚类参数")
    cluster_settings = settings.get("clustering", {})
    col1, col2 = st.columns(2)
    with col1:
        eps_input = st.slider(
            "DBSCAN eps（聚类半径）",
            min_value=0.05, max_value=1.0,
            value=float(cluster_settings.get("eps", 0.3)),
            step=0.05,
        )
    with col2:
        min_samples_input = st.number_input(
            "DBSCAN min_samples",
            min_value=1, max_value=20,
            value=int(cluster_settings.get("min_samples", 2)),
        )

    if st.button("💾 保存设置", type="primary"):
        settings.setdefault("llm", {})
        settings["llm"]["model"] = model_input
        settings["llm"]["temperature"] = temperature_input
        settings.setdefault("vector_store", {})
        settings["vector_store"]["top_k"] = int(top_k_input)
        settings["vector_store"]["similarity_threshold"] = float(threshold_input)
        settings.setdefault("pipeline", {})
        settings["pipeline"]["evidence_top_k"] = int(evidence_k_input)
        settings.setdefault("clustering", {})
        settings["clustering"]["eps"] = float(eps_input)
        settings["clustering"]["min_samples"] = int(min_samples_input)
        try:
            save_settings(settings)
            # Persist API key to environment for this session
            if api_key_input:
                os.environ["OPENAI_API_KEY"] = api_key_input
            if model_input:
                os.environ["OPENAI_MODEL"] = model_input
            if base_url_input:
                os.environ["OPENAI_BASE_URL"] = base_url_input
            st.success("设置已保存！")
        except Exception as e:
            st.error(f"保存失败: {e}")
