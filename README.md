# MultiMind-Review

> 基于 **Multi-Agent + RAG** 架构的智能商品评论分析系统

---

## 项目概述

MultiMind-Review 针对人工处理用户评论效率低、维度单一、决策过程不透明的问题，构建了一套集**商品识别**、**多维度情感分析**与**评论聚类总结**于一体的智能分析系统。

### 核心功能

| 功能 | 说明 |
|---|---|
| 📦 **商品识别** | 基于向量知识库与实体链接，精准匹配评论对应商品 |
| 📊 **多维情感分析** | 三阶段 Multi-Agent 工作流：维度识别 → 证据检索 → 情感评分 |
| 🔗 **评论聚类** | DBSCAN 密度聚类 + LLM 生成类簇总结词 |
| ⚙️ **无代码配置** | Streamlit 图形界面，动态定义商品库、维度与检索参数 |
| 📋 **审计追踪** | 记录每条决策的检索邻居、相似度与智能体输出 |

---

## 系统架构

```
MultiMind-Review/
├── config/                  # YAML 配置文件
│   ├── dimensions.yaml      # 情感分析维度定义
│   ├── products.yaml        # 商品知识库
│   └── settings.yaml        # 系统参数（LLM、检索、聚类）
├── src/
│   ├── models/              # Pydantic 数据模型
│   ├── rag/                 # 向量嵌入 + ChromaDB 向量存储
│   ├── agents/              # 多智能体模块
│   │   ├── product_agent.py    # 商品识别智能体
│   │   ├── dimension_agent.py  # 维度检测智能体
│   │   ├── evidence_agent.py   # 证据检索智能体
│   │   └── scoring_agent.py    # 情感评分智能体
│   ├── pipeline/            # 三阶段工作流编排
│   ├── clustering/          # DBSCAN 聚类 + LLM 总结
│   ├── audit/               # 端到端审计追踪
│   └── config_loader.py     # YAML 配置读写工具
├── app/                     # Streamlit 前端
│   ├── main.py              # 首页
│   └── pages/
│       ├── 1_⚙️_配置管理.py    # 无代码配置界面
│       ├── 2_📊_评论分析.py    # 多维度分析页面
│       ├── 3_🔗_评论聚类.py    # 聚类总结页面
│       └── 4_📋_审计追踪.py    # 审计追踪页面
├── tests/                   # 单元测试
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY 等配置
```

### 3. 启动应用

```bash
streamlit run app/main.py
```

浏览器访问 `http://localhost:8501`

---

## 工作流程

### 三阶段分析流程

```
用户评论
   │
   ▼
[阶段 1] 商品识别
   商品知识库向量检索 → 余弦相似度匹配 → 商品实体链接
   │
   ▼
[阶段 2] 维度检测
   LLM 分析 / 关键词匹配 → 识别涉及的评价维度
   │
   ▼
[阶段 3a] 证据检索
   评论句子向量化 → 按维度检索最相关片段
   │
   ▼
[阶段 3b] 情感评分
   LLM 基于证据进行 1-5 分情感评分
   │
   ▼
审计追踪持久化 → 返回完整分析结果
```

### 评论聚类流程

```
批量评论 → 向量嵌入 → DBSCAN 密度聚类 → LLM 生成类簇总结词
```

---

## 无代码配置

通过 **⚙️ 配置管理** 页面，用户无需编程即可：

- **添加/编辑商品**：维护商品知识库（名称、品牌、描述、关键词）
- **自定义维度**：定义情感分析维度（质量、价格、物流、服务等）
- **调整参数**：LLM 模型选择、检索 Top-K、相似度阈值、DBSCAN 参数

---

## 审计追踪

每条分析记录包含：
- 检索的邻居文档及相似度分数
- 各智能体的输入/输出
- LLM 推理过程（reasoning）
- 时间戳与唯一标识

日志以 JSONL 格式按日期存储于 `audit_logs/` 目录。

---

## 运行测试

```bash
pytest tests/ -v
```

---

## 技术栈

- **LLM**: OpenAI API（兼容 GPT-4o、DeepSeek 等兼容接口）
- **嵌入模型**: sentence-transformers（多语言支持）
- **向量数据库**: ChromaDB
- **聚类算法**: scikit-learn DBSCAN
- **前端**: Streamlit
- **数据验证**: Pydantic v2
