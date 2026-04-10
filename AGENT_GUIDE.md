# MultiMind Review — Agent 使用说明文档

> 本文档面向开发者和使用者，完整介绍 MultiMind Review 的 Agent 体系：能做什么、各部分用了哪些框架和技术、以及如何通过代码调用。

---

## 目录

1. [系统总体架构](#1-系统总体架构)
2. [核心基础设施：RAG 层](#2-核心基础设施rag-层)
3. [四个专家 Agent](#3-四个专家-agent)
   - [3.1 ProductRecognitionAgent — 商品识别](#31-productrecognitionagent--商品识别)
   - [3.2 DimensionDetectionAgent — 维度检测](#32-dimensiondetectionagent--维度检测)
   - [3.3 EvidenceRetrievalAgent — 证据检索](#33-evidenceretrievalagent--证据检索)
   - [3.4 SentimentScoringAgent — 情感评分](#34-sentimentscoringagent--情感评分)
4. [分析 Pipeline（两种实现）](#4-分析-pipeline两种实现)
   - [4.1 ReviewAnalysisPipeline — 手写顺序流](#41-reviewanalysispipeline--手写顺序流)
   - [4.2 LangGraphPipeline — StateGraph 流图](#42-langgraphpipeline--stategraph-流图)
5. [对话 Orchestrator Agent（单条评论）](#5-对话-orchestrator-agent单条评论)
   - [5.1 OrchestratorAgent — 手写 ReAct 循环](#51-orchestratoragent--手写-react-循环)
   - [5.2 LangGraphOrchestratorAgent — LangGraph ReAct](#52-langgraphorchestrator-agent--langgraph-react)
6. [数据集对话 Agent（批量分析）](#6-数据集对话-agent批量分析)
   - [6.1 DatasetOrchestratorAgent — 手写 ReAct 循环](#61-datasetorchestratoragent--手写-react-循环)
   - [6.2 LangGraphDatasetAgent — LangGraph ReAct](#62-langgraphdatasetagent--langgraph-react)
7. [工具（Tools）详细说明](#7-工具tools详细说明)
   - [7.1 单条分析工具（4 个）](#71-单条分析工具4-个)
   - [7.2 数据集分析工具（6 个）](#72-数据集分析工具6-个)
8. [技术栈汇总](#8-技术栈汇总)
9. [完整使用场景示例](#9-完整使用场景示例)
   - [场景一：单条评论对话分析](#场景一单条评论对话分析)
   - [场景二：批量数据集自然语言分析](#场景二批量数据集自然语言分析)
   - [场景三：纯结构化 API 调用（无对话）](#场景三纯结构化-api-调用无对话)
10. [环境配置](#10-环境配置)

---

## 1. 系统总体架构

```
用户输入（自然语言）
        │
        ▼
┌───────────────────────────────────────┐
│        Orchestrator Agent             │  ← 对话入口，ReAct 循环
│  (LLM function calling / LangGraph)   │
└──────────────┬────────────────────────┘
               │ 调用工具（Tool Call）
               ▼
┌───────────────────────────────────────────────────────────┐
│                   Analysis Pipeline                        │
│                                                           │
│  ① ProductRecognitionAgent  ──  向量检索匹配商品知识库     │
│  ② DimensionDetectionAgent  ──  LLM 识别评价维度          │
│  ③ EvidenceRetrievalAgent   ──  RAG 检索维度证据句子      │
│  ④ SentimentScoringAgent    ──  LLM 对每维度 1-5 评分     │
└───────────────┬───────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
  向量数据库          LLM API
  (ChromaDB)      (OpenAI / 兼容接口)
        ▲
  句向量模型
  (sentence-transformers)
```

整个系统分为三层：
- **RAG 基础层**：ChromaDB 向量数据库 + sentence-transformers 语义嵌入
- **专家 Agent 层**：4 个专注单一任务的 Agent，LLM + 向量检索协同工作
- **Orchestrator 层**：ReAct 循环 Agent，通过自然语言理解用户意图，调用工具，多轮对话

---

## 2. 核心基础设施：RAG 层

### EmbeddingModel（`src/rag/embeddings.py`）

| 项目 | 内容 |
|---|---|
| 框架 | `sentence-transformers` |
| 默认模型 | `paraphrase-multilingual-MiniLM-L12-v2`（支持中英文） |
| 接口 | `encode(texts)` / `encode_single(text)` → `np.ndarray` |
| 降级策略 | 模型无法加载时自动切换为基于 SHA-256 的确定性哈希向量（用于离线测试） |

### VectorStore（`src/rag/vector_store.py`）

| 项目 | 内容 |
|---|---|
| 框架 | `ChromaDB`（持久化客户端） |
| 相似度度量 | 余弦相似度（`hnsw:space: cosine`） |
| 主要 Collection | `products`（商品知识库）、`review_<id>`（临时评论句子库） |
| 核心方法 | `upsert()` 写入、`query()` 检索 Top-K、`delete_collection()` 清理临时集合 |

---

## 3. 四个专家 Agent

### 3.1 ProductRecognitionAgent — 商品识别

**文件**：`src/agents/product_agent.py`

**做什么**：将评论文本与商品知识库进行语义匹配，识别评论针对的具体商品。

**实现方式**：
1. 将评论文本用 `EmbeddingModel` 编码为向量
2. 在 ChromaDB `products` 集合中执行 Top-K 近邻检索
3. 若最高相似度 ≥ `similarity_threshold`（默认 0.5），则认为匹配成功

**输出**：`ProductMatch`（包含商品 ID、名称、品牌、分类、相似度分数、匹配关键词）或 `None`。

```python
match, audit = pipeline.product_agent.recognize("这款华为 P60 手机质量很好", review_id)
# match.product_name → "华为 P60"
# match.similarity   → 0.87
```

**技术**：sentence-transformers 语义嵌入 + ChromaDB 向量检索

---

### 3.2 DimensionDetectionAgent — 维度检测

**文件**：`src/agents/dimension_agent.py`

**做什么**：识别评论中涉及的评价维度（如"产品质量"、"物流配送"、"售后服务"等），维度列表由 `config/dimensions.yaml` 配置。

**实现方式**：

| 模式 | 触发条件 | 方法 |
|---|---|---|
| LLM 模式（默认） | `use_llm=True` | 构造系统提示 + 用户提示，调用 LLM，解析 JSON 响应中的 `detected_dimensions` 字段 |
| 关键词回退模式 | `use_llm=False` 或 LLM 不可用 | 遍历每个维度的关键词列表，检查是否出现在评论中 |

**输出**：检测到的维度 ID 列表 + `AuditEntry`（含推理说明）。

```python
dim_ids, audit = pipeline.dimension_agent.detect("快递很快，包装也很好", review_id)
# dim_ids → ["logistics", "packaging"]
# audit.reasoning → "LLM检测：提到了物流速度和包装质量"
```

**技术**：OpenAI function calling / Chat API + JSON 解析 + 关键词匹配（降级）

---

### 3.3 EvidenceRetrievalAgent — 证据检索

**文件**：`src/agents/evidence_agent.py`

**做什么**：对每个检测到的维度，从评论原文中找出最相关的句子作为情感评分的依据。

**实现方式**（纯 RAG，无 LLM）：
1. 将评论文本按句号、问号、感叹号等分割为句子列表
2. 将句子编码并存入临时 ChromaDB Collection（`review_<review_id>`）
3. 以"维度名称 + 维度描述"为查询，检索 Top-K 最相关句子
4. 分析完成后删除临时 Collection，避免数据积累

**输出**：`evidence_map = {dimension_id: [句子1, 句子2, ...]}`

```python
evidence_map, audits = pipeline.evidence_agent.retrieve(
    review_text, review_id, detected_dims
)
# evidence_map["logistics"] → ["快递两天就到了", "配送很及时"]
```

**技术**：纯向量检索（ChromaDB + sentence-transformers），无 LLM 调用

---

### 3.4 SentimentScoringAgent — 情感评分

**文件**：`src/agents/scoring_agent.py`

**做什么**：对每个维度的证据句子进行情感分析，输出 1-5 分评分及情感类别。

**实现方式**：

| 模式 | 方法 |
|---|---|
| LLM 模式（默认） | 将证据句子格式化后调用 LLM，LLM 返回 `{"score": 4.5, "sentiment": "positive", "reasoning": "..."}` |
| 关键词回退模式 | 统计正面词（好/棒/满意…）和负面词（差/烂/失望…）数量，多者胜出，得分固定为 4.0/2.0/3.0 |

**评分规则**：
- 1-2：负面（negative，overall_score ≤ 2.5）
- 3：中立（neutral）
- 4-5：正面（positive，overall_score ≥ 4.0）

```python
dim_scores, audits = pipeline.scoring_agent.score(review_id, evidence_map)
# dim_scores[0].dimension_name → "物流配送"
# dim_scores[0].score          → 4.5
# dim_scores[0].sentiment      → "positive"
# dim_scores[0].reasoning      → "证据显示配送速度快..."
```

**技术**：OpenAI Chat API + JSON 解析 + 关键词启发式（降级）

---

## 4. 分析 Pipeline（两种实现）

两种 Pipeline 提供**完全相同的公共 API**（`analyze()`、`analyze_batch()`、`setup()`、`update_products()`、`update_dimensions()`），可互换使用。

### 4.1 ReviewAnalysisPipeline — 手写顺序流

**文件**：`src/pipeline/workflow.py`

顺序执行四阶段：

```
recognize_product → detect_dimensions → retrieve_evidence → score_sentiment → 汇总
```

中间结果通过 Python 变量传递，流程固定不可跳步。

```python
from src.pipeline.workflow import ReviewAnalysisPipeline

pipeline = ReviewAnalysisPipeline(products=products, dimensions=dimensions)
result = pipeline.analyze("这款手机屏幕很好，但电池续航差")
print(result.overall_score)   # 3.2
print(result.overall_sentiment)  # "neutral"
```

---

### 4.2 LangGraphPipeline — StateGraph 流图

**文件**：`src/pipeline/langgraph_pipeline.py`

**框架**：LangGraph `StateGraph`

相同的四个 Agent 被封装为 LangGraph **节点（Node）**，通过显式边（Edge）连接，支持条件跳转：

```
START
  │
  ▼
recognize_product
  │
  ▼
detect_dimensions
  │
  ├─[有维度]──► retrieve_evidence ──[有证据]──► score_sentiment ──► aggregate ──► END
  │
  └─[无维度]──────────────────────────────────────────────────────► aggregate ──► END
```

**状态类型**（`ReviewState` TypedDict）：

```python
class ReviewState(TypedDict):
    review_text: str
    review_id: str
    product_match: ProductMatch | None
    detected_dim_ids: list[str]
    detected_dims: list[Dimension]
    evidence_map: dict[str, list[str]]
    dim_scores: list[DimensionScore]
    audit_entries: Annotated[list[AuditEntry], operator.add]  # 累加器
    overall_score: float | None
    overall_sentiment: str
```

`audit_entries` 使用 `operator.add` 作为 reducer，每个节点追加而不是覆盖，实现全程审计追踪。

```python
from src.pipeline.langgraph_pipeline import LangGraphPipeline

pipeline = LangGraphPipeline(products=products, dimensions=dimensions)
result = pipeline.analyze("这款手机屏幕很好，但电池续航差")
```

**技术**：LangGraph `StateGraph` + `TypedDict` 状态 + 条件边（`add_conditional_edges`）

---

## 5. 对话 Orchestrator Agent（单条评论）

### 5.1 OrchestratorAgent — 手写 ReAct 循环

**文件**：`src/agents/orchestrator_agent.py`

**做什么**：接收用户自然语言输入（单条评论或追问），通过 **ReAct（Reason-Act-Observe）** 循环自主决定调用哪些工具、调用顺序，给出中文分析回复。支持多轮对话。

**实现方式**：

```
用户输入
  │
  ▼
构建消息列表（system prompt + 对话历史）
  │
  ▼
循环（最多 max_iterations=10 次）：
  ├─ 调用 LLM（附带 4 个工具定义）
  ├─ 若 LLM 选择工具 → 执行工具 → 追加结果 → 继续循环
  └─ 若 LLM 给出最终回答 → 返回给用户
```

**关键特性**：
- **多轮对话**：`_history` 列表跨轮次保留，追问无需重新分析
- **自纠正**：LLM 可重试工具、调整参数或向用户要求澄清
- **降级模式**：`use_llm=False` 时绕过 LLM，直接运行 Pipeline 并格式化输出

```python
from src.agents.orchestrator_agent import OrchestratorAgent

agent = OrchestratorAgent(pipeline)

# 第一轮：提供评论
reply = agent.chat("这款手机质量非常好，物流也很快，但售后态度差")
print(reply)  # → 商品识别 + 各维度评分 + 整体评价

# 第二轮：追问（无需重新分析）
reply = agent.chat("哪个维度评分最低？")
print(reply)  # → "售后服务评分最低，为 2.0/5"

# 重置会话
agent.reset()
```

**技术**：OpenAI function calling API + Python 手写消息循环

---

### 5.2 LangGraphOrchestratorAgent — LangGraph ReAct

**文件**：`src/agents/langgraph_orchestrator.py`

**做什么**：与 `OrchestratorAgent` 功能完全相同，但将手写的 ReAct 循环替换为 LangGraph 的 `create_react_agent` 预构建图。

**实现方式**：

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

graph = create_react_agent(
    model=ChatOpenAI(...),
    tools=[recognize_product, detect_dimensions, retrieve_evidence, score_sentiment],
    checkpointer=MemorySaver(),      # 多轮对话记忆
    state_modifier=_SYSTEM_PROMPT,  # 系统提示
)
```

工具通过 LangChain `@tool` 装饰器定义（自动从 Python 类型注解生成 JSON Schema），替代手写的 OpenAI function 定义 dict。

**多轮记忆**：使用 `MemorySaver` + `thread_id`，每次 `reset()` 生成新 `thread_id` 开始新会话。

```python
from src.agents.langgraph_orchestrator import LangGraphOrchestratorAgent

agent = LangGraphOrchestratorAgent(pipeline)
reply = agent.chat("这款手机质量非常好，物流也很快")
reply = agent.chat("哪个维度评分最低？")
agent.reset()
```

**技术**：LangGraph `create_react_agent` + `MemorySaver` + LangChain `@tool` + `ChatOpenAI`

---

## 6. 数据集对话 Agent（批量分析）

### 6.1 DatasetOrchestratorAgent — 手写 ReAct 循环

**文件**：`src/agents/dataset_agent.py`

**做什么**：接收一批评论（`list[str]`），通过自然语言对话实现批量数据集分析，包括整体统计、维度排名、条件筛选、商品对比等。

**实现方式**：与 `OrchestratorAgent` 架构相同，但工具换为 6 个数据集级别工具，底层调用 `pipeline.analyze_batch()`。第一次分析结果被 **懒加载缓存**（`DatasetToolExecutor._results`），后续工具调用直接使用缓存，不重复分析。

```python
from src.agents.dataset_agent import DatasetOrchestratorAgent

agent = DatasetOrchestratorAgent(pipeline, reviews=reviews_list)

reply = agent.chat("这个数据集的整体情感分布如何？")
# → 触发 batch_analyze（首次），然后 get_summary_statistics

reply = agent.chat("哪个维度评分最低？")
# → 直接调用 rank_dimensions（使用缓存），无需重新分析

reply = agent.chat("找出所有负面评论")
# → 调用 filter_reviews(sentiment="negative")

# 切换数据集
agent.update_dataset(new_reviews)
```

**技术**：OpenAI function calling API + 手写 ReAct 循环 + 懒加载批量分析缓存

---

### 6.2 LangGraphDatasetAgent — LangGraph ReAct

**文件**：`src/agents/langgraph_dataset_agent.py`

**做什么**：与 `DatasetOrchestratorAgent` 功能相同，用 LangGraph `create_react_agent` 替代手写循环。

工具处理逻辑（`DatasetToolExecutor` 中的 6 个方法）**完全复用**，仅 Orchestrator 层由 LangGraph 接管。

```python
from src.agents.langgraph_dataset_agent import LangGraphDatasetAgent

agent = LangGraphDatasetAgent(pipeline, reviews=reviews_list)
reply = agent.chat("各商品的评分如何对比？")
```

**技术**：LangGraph `create_react_agent` + `MemorySaver` + `DatasetToolExecutor`（复用）

---

## 7. 工具（Tools）详细说明

### 7.1 单条分析工具（4 个）

定义文件：`src/agents/tools.py`（手写版）/ `src/agents/langgraph_orchestrator.py`（LangGraph 版）

| 工具名 | 功能 | 必要前置 | 返回字段 |
|---|---|---|---|
| `recognize_product` | 通过向量检索识别评论对应商品 | 无 | `matched`, `product_name`, `brand`, `similarity` |
| `detect_dimensions` | LLM 检测评论涉及的评价维度 | 无 | `detected_dimensions[]`, `count`, `reasoning` |
| `retrieve_evidence` | RAG 为各维度检索证据句子 | `detect_dimensions` | `evidence_map` |
| `score_sentiment` | LLM 对各维度证据进行 1-5 评分 | `retrieve_evidence` | `scores[]`, `overall_score` |

**典型工具调用序列**：

```
recognize_product → detect_dimensions → retrieve_evidence → score_sentiment
```

LLM 自主决定调用顺序，但通常遵循上述序列。

---

### 7.2 数据集分析工具（6 个）

定义文件：`src/agents/dataset_tools.py`

| 工具名 | 功能 | 必要前置 | 关键参数 |
|---|---|---|---|
| `batch_analyze` | 对全部评论运行 Pipeline，结果缓存 | 无 | 无 |
| `get_summary_statistics` | 整体统计：情感分布、平均分、商品分布 | `batch_analyze` | 无 |
| `get_dimension_statistics` | 各维度详细统计：均分、情感分布、示例证据 | `batch_analyze` | `dimension_id`（可选） |
| `filter_reviews` | 按条件筛选评论样本 | `batch_analyze` | `sentiment`, `product_name`, `min_score`, `max_score`, `sort_by`, `top_n` |
| `rank_dimensions` | 各维度按平均分排名 | `batch_analyze` | `order`（asc/desc） |
| `compare_products` | 不同商品评分与情感对比 | `batch_analyze` | 无 |

**自然语言 → 工具映射示例**：

| 用户提问 | Agent 调用的工具 |
|---|---|
| "帮我分析这批评论" | `batch_analyze` → `get_summary_statistics` |
| "哪个维度最差？" | `rank_dimensions(order="asc")` |
| "找出所有差评" | `filter_reviews(sentiment="negative")` |
| "评分低于 2 分的评论有哪些？" | `filter_reviews(max_score=2)` |
| "A 和 B 产品哪个口碑更好？" | `compare_products` |
| "物流维度的详细情况" | `get_dimension_statistics(dimension_id="logistics")` |

---

## 8. 技术栈汇总

| 层次 | 技术/框架 | 版本要求 | 用途 |
|---|---|---|---|
| **LLM 接入** | OpenAI Python SDK | `>=1.0.0` | Chat Completions + Function Calling |
| **LLM 模型** | OpenAI GPT-4o-mini（默认）或任意兼容接口 | — | 维度检测、情感评分、Orchestrator |
| **Agent 框架** | LangGraph | `>=0.2.73` | `StateGraph` 流图、`create_react_agent`、`MemorySaver` |
| **LLM 工具层** | LangChain Core | `>=1.2.22` | `@tool` 装饰器、`HumanMessage`/`AIMessage` |
| **LLM 模型客户端** | LangChain OpenAI | `>=0.3.0` | `ChatOpenAI` |
| **向量数据库** | ChromaDB | `>=0.5.0` | 商品知识库 + 证据句子临时存储 |
| **语义嵌入模型** | sentence-transformers | `>=3.0.0` | 文本向量化（多语言 MiniLM） |
| **数值计算** | NumPy | `>=1.26.0` | 向量运算 |
| **数据验证** | Pydantic | `>=2.0.0` | 所有数据模型（`ProductMatch`, `ReviewAnalysisResult` 等） |
| **配置管理** | PyYAML | `>=6.0` | `products.yaml`, `dimensions.yaml`, `settings.yaml` |
| **Web UI** | Streamlit | `>=1.35.0` | 多页面交互界面 |
| **可视化** | Plotly | `>=5.20.0` | 图表展示 |
| **环境变量** | python-dotenv | `>=1.0.0` | `.env` 文件加载 |

---

## 9. 完整使用场景示例

### 场景一：单条评论对话分析

**使用类**：`LangGraphOrchestratorAgent`（推荐）或 `OrchestratorAgent`

```python
import os
from src.config_loader import load_products, load_dimensions
from src.pipeline.langgraph_pipeline import LangGraphPipeline
from src.agents.langgraph_orchestrator import LangGraphOrchestratorAgent

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

# 初始化
products = load_products()     # 从 config/products.yaml 加载
dimensions = load_dimensions() # 从 config/dimensions.yaml 加载

pipeline = LangGraphPipeline(products=products, dimensions=dimensions)
agent = LangGraphOrchestratorAgent(pipeline)

# 第一轮：提供评论
reply = agent.chat("这款华为手机屏幕显示效果很好，颜色鲜艳，但电池续航只有一天，物流三天到货，速度一般")
print(reply)
# 示例输出：
# 识别商品：华为（相似度 0.83）
# 屏幕显示：4.5/5 😊 正面（证据：屏幕显示效果很好，颜色鲜艳）
# 电池续航：2.0/5 😞 负面（证据：电池续航只有一天）
# 物流配送：3.0/5 😐 中立（证据：物流三天到货，速度一般）
# 整体评价：中立（3.2/5）

# 第二轮：追问（无需重新分析）
reply = agent.chat("电池续航差的原因可能是什么？")
print(reply)
# → LLM 基于已有证据直接回答，无额外工具调用

# 第三轮：不同评论
agent.reset()  # 开始新会话
reply = agent.chat("快递包装太差了，东西都碎了，售后没人接电话")
```

---

### 场景二：批量数据集自然语言分析

**使用类**：`LangGraphDatasetAgent`（推荐）或 `DatasetOrchestratorAgent`

```python
from src.agents.langgraph_dataset_agent import LangGraphDatasetAgent

# 准备数据集（支持数百条）
reviews = [
    "手机质量很好，屏幕清晰，值得购买",
    "物流太慢了，等了一周，包装也不行",
    "售后服务很好，有问题马上解决",
    "电池不耐用，一天充两次，很烦",
    # ...更多评论
]

agent = LangGraphDatasetAgent(pipeline, reviews=reviews)

# 第一轮：触发批量分析并获取总结
reply = agent.chat("帮我分析这批评论，给出整体统计")
# → 自动调用 batch_analyze + get_summary_statistics
# 示例输出：
# 共分析 100 条评论
# 情感分布：正面 62%，中立 25%，负面 13%
# 平均综合评分：3.8/5
# 最高分维度：屏幕显示（4.3）
# 最低分维度：电池续航（2.7）

# 第二轮：深入分析（复用缓存）
reply = agent.chat("哪个维度问题最多？")
# → 调用 rank_dimensions(order="asc")

# 第三轮：筛选
reply = agent.chat("给我看 5 条评分最低的差评")
# → 调用 filter_reviews(sentiment="negative", sort_by="score_asc", top_n=5)

# 第四轮：商品对比
reply = agent.chat("不同商品的口碑差异大吗？")
# → 调用 compare_products

# 切换数据集（清空缓存，保持对话继续）
agent.update_dataset(new_reviews)
reply = agent.chat("新数据集的整体情况怎么样？")
```

---

### 场景三：纯结构化 API 调用（无对话）

直接使用 Pipeline，获取结构化结果，不走 Agent 对话层：

```python
# 单条分析
result = pipeline.analyze("这款耳机音质很好，但夹耳")
print(result.review_id)
print(result.product_match.product_name if result.product_match else "未识别")
print(result.overall_score)
for ds in result.dimension_scores:
    print(f"  {ds.dimension_name}: {ds.score}/5 ({ds.sentiment})")
    print(f"    {ds.reasoning}")

# 批量分析
results = pipeline.analyze_batch(reviews_list)
avg_score = sum(r.overall_score for r in results if r.overall_score) / len(results)
print(f"平均评分：{avg_score:.2f}")
```

---

## 10. 环境配置

### 必需环境变量

```bash
# .env 文件（复制 .env.example 后修改）

# LLM 配置（必须）
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini          # 可替换为其他兼容模型

# 使用 API 代理或私有部署时（可选）
OPENAI_BASE_URL=https://api.openai.com/v1

# 向量数据库存储路径（可选，默认 ./chroma_db）
CHROMA_PERSIST_DIR=./chroma_db

# 语义嵌入模型（可选，默认多语言 MiniLM）
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 启动 Web UI

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

页面导航：
- **⚙️ 配置管理**：配置商品库、分析维度、LLM 参数
- **📊 评论分析**：单条评论分析 + Agent 对话
- **🔗 评论聚类**：批量评论聚类总结
- **💬 数据集对话**：上传数据集，自然语言批量分析
- **📋 审计追踪**：查看所有决策记录

### 选择 Pipeline 和 Agent 版本

| 场景 | 推荐组合 |
|---|---|
| 单条评论结构化分析 | `ReviewAnalysisPipeline` + `OrchestratorAgent` |
| 单条评论对话分析（生产推荐） | `LangGraphPipeline` + `LangGraphOrchestratorAgent` |
| 批量数据集自然语言分析 | `LangGraphPipeline` + `LangGraphDatasetAgent` |
| 无 LLM 离线测试 | 任意 Pipeline，`use_llm=False` |
