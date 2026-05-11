# 银行智能客服 RAG 系统 (Banking RAG CS)

基于多层检索增强生成（RAG）的银行智能客服系统，支持中文银行业务问答。

## 架构概览

```
用户问题 → 查询扩展 → 多路检索（稠密+稀疏+知识图谱）→ 召回融合去重
       → Rerank 精排 → LLM 关联度判断 → [不满足→深度检索] → 生成回答
```

### 检索管道

| 阶段 | 说明 |
|------|------|
| **稠密检索** | ChromaDB + BGE-large-zh 向量相似度 |
| **稀疏检索** | Whoosh BM25 + jieba 中文分词 |
| **知识图谱** | NetworkX 实体关系网络遍历 |
| **融合去重** | RRF (Reciprocal Rank Fusion) + 语义去重 |
| **Rerank** | BGE-reranker-v2-m3 精排 |
| **关联度判断** | LLM 批量打分，不满足触发深度检索 |
| **深度检索** | 降低阈值 + 扩展查询词 + 增加 KG 跳数 |

## 技术栈

- **后端**: FastAPI + Uvicorn (Python 3.12+)
- **LLM**: DeepSeek v4-pro (via Anthropic SDK compatible endpoint)
- **向量数据库**: ChromaDB
- **全文检索引擎**: Whoosh
- **知识图谱**: NetworkX
- **嵌入模型**: BAAI/bge-large-zh-v1.5
- **重排序模型**: BAAI/bge-reranker-v2-m3
- **前端**: 原生 HTML/CSS/JS

## 项目结构

```
banking-rag-cs/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── requirements.txt        # Python 依赖
│   ├── .env.example            # 环境变量模板
│   ├── config/
│   │   └── settings.py         # 全局配置（检索参数、模型参数等）
│   ├── models/
│   │   └── schemas.py          # Pydantic 数据模型
│   ├── routers/
│   │   ├── chat.py             # /api/chat — 对话接口
│   │   └── admin.py            # /api/admin — 文档注入、索引管理
│   ├── services/
│   │   ├── llm_client.py       # LLM 调用客户端
│   │   ├── embedding_service.py # 向量编码
│   │   ├── vector_store.py     # ChromaDB 操作
│   │   ├── fulltext_search.py  # Whoosh BM25 检索
│   │   ├── knowledge_graph.py  # 知识图谱构建与遍历
│   │   ├── chunking_service.py # 多策略文档切分
│   │   ├── retrieval_pipeline.py # Round 1 + Round 2 检索管道
│   │   ├── fusion.py           # RRF 融合 + 语义去重
│   │   ├── reranker.py         # 重排序
│   │   ├── query_expander.py   # 查询扩展
│   │   ├── relevance_judge.py  # LLM 关联度判断
│   │   ├── response_generator.py # 答案生成
│   │   └── document_processor.py # 文档预处理
│   ├── prompts/                # Prompt 模板
│   └── data/                   # 示例数据
└── frontend/
    ├── index.html              # 聊天界面
    ├── css/style.css           # 样式
    └── js/app.js               # 前端逻辑
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 设置数据目录

```bash
export BANKING_RAG_DATA_DIR=/path/to/your/data/dir
```

### 4. 注入知识库文档

```bash
curl -X POST http://127.0.0.1:8000/api/admin/ingest \
  -H "Content-Type: application/json" \
  -d @data/ingest_payload.json
```

### 5. 启动服务

```bash
python -m uvicorn main:app --port 8000
```

浏览器打开 `http://127.0.0.1:8000` 即可使用。

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 组件健康检查 |
| `/api/chat` | POST | 发送问题，返回答案+引用+检索信息 |
| `/api/chat/stream` | POST | SSE 流式对话 |
| `/api/admin/ingest` | POST | 注入文档 |
| `/api/admin/stats` | GET | 索引统计 |
| `/api/admin/rebuild` | POST | 重建全部索引 |

## 覆盖业务领域

- 信用卡（申办、额度、年费、还款、逾期）
- 贷款（个人贷款、住房贷款、利率）
- 账户管理（开户、冻结、转账限额）
- 投资理财（基金、理财、风险评估）
- 防范诈骗（识别、应对、报案）
- 其他综合业务

## License

MIT
