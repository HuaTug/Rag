"""
Pytest配置和fixtures
"""

import pytest
import asyncio
from typing import Generator
from unittest.mock import MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embedding_service():
    """Mock嵌入服务"""
    service = MagicMock()
    service.embed_text = AsyncMock(return_value=MagicMock(
        vector=[0.1] * 384,
        model="test-model",
        dimension=384,
    ))
    service.embed_texts = AsyncMock(return_value=MagicMock(
        embeddings=[MagicMock(vector=[0.1] * 384) for _ in range(3)],
        texts=["text1", "text2", "text3"],
        model="test-model",
    ))
    service.embed_query = AsyncMock(return_value=MagicMock(
        vector=[0.1] * 384,
        model="test-model",
        dimension=384,
    ))
    service.get_dimension.return_value = 384
    service.get_model_name.return_value = "test-model"
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM服务"""
    service = MagicMock()
    service.generate = AsyncMock(return_value=MagicMock(
        content="这是测试答案",
        model="test-model",
        tokens_used=100,
        latency_ms=500,
    ))
    service.analyze_query = AsyncMock(return_value={
        "intent": "factual",
        "keywords": ["测试"],
        "confidence": 0.9,
    })
    service.rewrite_query = AsyncMock(return_value="优化后的查询")
    service.is_available = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_vector_store():
    """Mock向量存储"""
    service = MagicMock()
    service.upsert = AsyncMock(return_value=3)
    service.search = AsyncMock(return_value=[
        MagicMock(
            id="test-id-1",
            score=0.95,
            content="测试内容1",
            metadata={"source": "test"},
        ),
        MagicMock(
            id="test-id-2",
            score=0.85,
            content="测试内容2",
            metadata={"source": "test"},
        ),
    ])
    service.delete = AsyncMock(return_value=1)
    service.count = AsyncMock(return_value=100)
    service.is_available = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_search_service():
    """Mock搜索服务"""
    service = MagicMock()
    service.search = AsyncMock(return_value=[
        MagicMock(
            title="搜索结果1",
            content="搜索内容1",
            url="https://example.com/1",
            source="google",
            score=0.9,
            metadata={},
        ),
    ])
    service.is_available = AsyncMock(return_value=True)
    return service


@pytest.fixture
def sample_document():
    """示例文档"""
    from src.domain.entities.document import Document, DocumentMetadata
    
    return Document(
        title="测试文档",
        content="这是一个测试文档的内容。包含多个段落。\n\n第二段内容。",
        metadata=DocumentMetadata(source="test"),
    )


@pytest.fixture
def sample_query():
    """示例查询"""
    from src.domain.entities.query import Query, QueryConfig
    
    return Query(
        original_text="什么是RAG？",
        config=QueryConfig(top_k=5),
    )
