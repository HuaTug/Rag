# RAG智能问答系统

一个基于多通道处理（MCP）架构的智能问答系统，结合了实时搜索和向量检索技术。

## 🎯 系统功能

### 核心特性
- **多通道搜索**: 支持Google搜索API实时获取最新信息
- **向量检索**: 基于Milvus的文档存储和相似度搜索
- **智能问答**: 集成DeepSeek大语言模型进行回答生成
- **多种界面**: 支持命令行和Web界面
- **异步处理**: 高效的并发搜索和处理

### 技术架构
```
用户查询 → MCP框架 → [Google搜索 + 向量检索] → LLM生成回答 → 用户
```

## 🚀 快速开始

### 方法一：一键安装（推荐）

```bash
cd /Users/xuzhihua/Python/Rag/mcp
./install.sh
./set_env.sh
python3 simple_rag.py
```

### 方法二：手动安装

#### 1. 环境准备

确保已安装Python 3.8+:
```bash
python3 --version
```

#### 2. 安装依赖

必需依赖:
```bash
pip install requests aiohttp beautifulsoup4 openai python-dotenv streamlit
```

可选依赖（高级功能）:
```bash
pip install sentence-transformers pymilvus pydantic
```

#### 3. 配置API密钥

设置环境变量:
```bash
export GOOGLE_API_KEY='your_google_api_key'
export GOOGLE_SEARCH_ENGINE_ID='your_search_engine_id'
export DEEPSEEK_API_KEY='your_deepseek_api_key'
```

或编辑 `config.json` 文件:
```json
{
  "google_search": {
    "api_key": "your_google_api_key",
    "search_engine_id": "your_search_engine_id"
  },
  "deepseek": {
    "api_key": "your_deepseek_api_key"
  }
}
```

#### 4. 启动系统

**简化版（推荐新手）:**
```bash
python3 simple_rag.py
```

**完整版:**
```bash
python3 rag_system.py
```

**Web界面:**
```bash
streamlit run web_interface.py
```

### 快速测试

```bash
python3 quick_test.py
```

## 📋 API密钥获取指南

### Google Custom Search API
1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 启用 "Custom Search API"
4. 创建API密钥
5. 设置 [Custom Search Engine](https://cse.google.com/cse/)

### DeepSeek API
1. 访问 [DeepSeek平台](https://platform.deepseek.com/)
2. 注册账号并获取API密钥
3. 或使用腾讯云代理接口

## 🏗️ 系统架构

### 核心组件

#### 1. MCP框架 (`mcp_framework.py`)
- **BaseChannel**: 通道基类
- **MCPProcessor**: 多通道处理器
- **QueryAnalyzer**: 查询分析器

#### 2. 搜索通道 (`search_channels.py`)
- **GoogleSearchChannel**: Google搜索实现
- 支持实时网页搜索
- 自动内容提取和清理

#### 3. 向量存储 (`dynamic_vector_store.py`)
- **DynamicVectorStore**: 动态向量存储
- **VectorStoreManager**: 存储管理器
- 基于Milvus的高效检索

#### 4. RAG处理器 (`enhanced_rag_processor.py`)
- **EnhancedRAGProcessor**: 增强RAG处理器
- 结合搜索和向量结果
- 智能答案生成

#### 5. 工具模块
- **ask_llm.py**: LLM客户端和调用
- **encoder.py**: 文本嵌入和编码
- **milvus_utils.py**: Milvus数据库工具

## 💡 使用示例

### 基本问答
```python
from mcp.rag_system import RAGSystemManager, RAGSystemConfig

# 初始化系统
config = RAGSystemConfig()
manager = RAGSystemManager(config)
await manager.initialize()

# 提问
answer = await manager.process_query("人工智能的发展历史是什么？")
print(answer)
```

### 不同查询类型
```python
# 事实性查询
await manager.process_query("什么是机器学习？", "factual")

# 分析性查询
await manager.process_query("AI和ML的区别是什么？", "analytical")

# 创意性查询
await manager.process_query("写一个关于AI的故事", "creative")
```

## ⚙️ 配置说明

### config.json配置文件
```json
{
  "google_search": {
    "api_key": "Google API密钥",
    "search_engine_id": "搜索引擎ID", 
    "timeout": 10,
    "max_results": 10
  },
  "deepseek": {
    "api_key": "DeepSeek API密钥",
    "base_url": "API基础URL",
    "model": "使用的模型名称"
  },
  "milvus": {
    "uri": "Milvus数据库路径",
    "collection_name": "集合名称",
    "dimension": 384
  },
  "rag": {
    "similarity_threshold": 0.7,
    "max_context_length": 4000,
    "combine_search_and_vector": true
  }
}
```

## 🔧 开发和调试

### 测试单个组件
```bash
# 测试Google搜索
python3 search_channels.py

# 测试向量存储
python3 dynamic_vector_store.py

# 测试LLM客户端
python3 ask_llm.py
```

### 查看日志
```bash
tail -f rag_system.log
```

### 错误排查
1. **API连接失败**: 检查API密钥和网络连接
2. **模块导入错误**: 确保依赖包已正确安装
3. **向量存储错误**: 检查Milvus数据库文件权限

## 📊 性能优化

### 建议配置
- **内存**: 推荐8GB+
- **存储**: SSD硬盘
- **网络**: 稳定的互联网连接

### 优化设置
```python
# 减少搜索结果数量
"max_results": 5

# 提高相似度阈值
"similarity_threshold": 0.8

# 减少上下文长度
"max_context_length": 2000
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📝 更新日志

### v1.0.0 (2024-01-xx)
- ✅ 初始版本发布
- ✅ MCP架构实现
- ✅ Google搜索集成
- ✅ Milvus向量存储
- ✅ DeepSeek LLM集成
- ✅ Web界面支持

## 📄 许可证

MIT License - 详见LICENSE文件

## 📞 支持

如有问题，请提交Issue或联系开发团队。

---

🎉 **开始探索智能问答的无限可能！**
