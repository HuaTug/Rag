# 企业级 RAG 系统 v2.0

这是一个完全重构的企业级检索增强生成（RAG）系统，采用清洁架构（Clean Architecture）和领域驱动设计（DDD）原则。

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│                   (FastAPI REST API)                        │
├─────────────────────────────────────────────────────────────┤
│                     Application Layer                        │
│              (Use Cases, DI Container)                       │
├─────────────────────────────────────────────────────────────┤
│                       Domain Layer                           │
│         (Entities, Domain Services, Ports)                   │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│    (Adapters: Milvus, DeepSeek, Google, etc.)               │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构

```
src/
├── domain/                 # 领域层 - 核心业务逻辑
│   ├── entities/          # 领域实体
│   │   ├── document.py    # 文档实体
│   │   ├── query.py       # 查询实体
│   │   └── embedding.py   # 嵌入实体
│   ├── ports/             # 端口（抽象接口）
│   │   ├── repositories.py # 仓储接口
│   │   └── services.py    # 服务接口
│   └── services/          # 领域服务
│       ├── rag_service.py     # RAG核心服务
│       ├── document_service.py # 文档服务
│       └── retrieval_service.py # 检索服务
│
├── infrastructure/        # 基础设施层 - 外部依赖实现
│   ├── embedding/         # 嵌入服务适配器
│   │   ├── sentence_transformer.py
│   │   └── openai_embedding.py
│   ├── llm/              # LLM服务适配器
│   │   ├── deepseek.py
│   │   └── openai_llm.py
│   ├── vector_store/     # 向量存储适配器
│   │   └── milvus_store.py
│   ├── search/           # 搜索服务适配器
│   │   └── google_search.py
│   └── chunking/         # 分块服务适配器
│       ├── semantic_chunker.py
│       └── fixed_chunker.py
│
├── application/          # 应用层
│   ├── api.py           # REST API
│   └── container.py     # 依赖注入容器
│
└── config/              # 配置层
    ├── settings.py      # 配置管理
    └── logging.py       # 日志配置

tests/                   # 测试
├── conftest.py         # 测试配置
├── test_domain_entities.py
├── test_domain_services.py
└── test_infrastructure.py
```

## ✨ 核心特性

### 1. 清洁架构
- **依赖反转**: 业务逻辑不依赖于具体实现
- **端口-适配器模式**: 轻松替换外部服务（LLM、向量数据库等）
- **领域隔离**: 核心业务逻辑与框架无关

### 2. 多种检索策略
- **密集检索** (Dense Retrieval): 基于向量相似度
- **稀疏检索** (Sparse/BM25): 基于关键词
- **混合检索** (Hybrid): 结合密集和稀疏检索
- **多查询检索** (Multi-Query): 生成多个查询变体
- **HyDE检索**: 假设文档嵌入

### 3. 智能分块
- **语义分块**: 基于语义边界分割文档
- **固定大小分块**: 按字符数分割
- **重叠分块**: 保留上下文连贯性

### 4. 可扩展的服务
- **多LLM支持**: DeepSeek、OpenAI、自定义
- **多向量数据库**: Milvus、后续支持Pinecone等
- **多搜索引擎**: Google、Bing等

### 5. 企业级特性
- **类型安全配置**: 使用Pydantic Settings
- **结构化日志**: 使用structlog
- **依赖注入**: 统一管理服务生命周期
- **完善测试**: 单元测试和集成测试

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd /path/to/Rag

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements-new.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
```

### 3. 启动服务

```bash
# 方式一：使用main.py
python main.py

# 方式二：直接使用uvicorn
uvicorn src.application.api:app --reload --port 8000
```

### 4. 访问API

- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 📖 API 使用

### 查询接口

```bash
curl -X POST "http://localhost:8000/api/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是RAG？",
    "top_k": 5,
    "enable_web_search": true
  }'
```

### 响应示例

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "什么是RAG？",
  "answer": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术...",
  "confidence": 0.85,
  "sources": [
    {
      "title": "RAG介绍",
      "url": "https://example.com/rag",
      "source": "google",
      "score": 0.92
    }
  ],
  "processing_time_ms": 1234.56,
  "context_count": 5
}
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_domain_entities.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## 🔧 扩展指南

### 添加新的LLM提供商

1. 在 `src/infrastructure/llm/` 创建新文件
2. 实现 `LLMService` 接口
3. 在 `Container` 中注册

```python
# src/infrastructure/llm/custom_llm.py
from src.domain.ports.services import LLMService

class CustomLLM(LLMService):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # 实现你的逻辑
        pass
```

### 添加新的向量数据库

1. 在 `src/infrastructure/vector_store/` 创建新文件
2. 实现 `VectorStoreService` 接口
3. 在 `Container` 中注册

## 📊 与旧版对比

| 特性 | 旧版 | 新版 v2.0 |
|------|------|-----------|
| 架构 | 单体混杂 | 清洁架构 + DDD |
| 测试 | 无 | 完善的单元测试 |
| 配置管理 | 混乱 | Pydantic Settings |
| 依赖管理 | 硬编码 | 依赖注入容器 |
| 日志 | print + logging混用 | structlog结构化日志 |
| 扩展性 | 修改核心代码 | 实现接口即可 |
| 检索策略 | 单一向量检索 | 多种高级策略 |
| 分块策略 | 简单分块 | 语义分块 |
| 类型安全 | 弱 | 强（Pydantic + TypeHints） |

## 📄 许可证

MIT License
