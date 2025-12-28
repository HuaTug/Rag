"""
Infrastructure Tests - 基础设施层测试
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSemanticChunker:
    """语义分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        from src.infrastructure.chunking import SemanticChunker
        from src.domain.ports.services import ChunkingConfig
        
        return SemanticChunker(config=ChunkingConfig(
            chunk_size=200,
            chunk_overlap=20,
        ))
    
    @pytest.mark.asyncio
    async def test_chunk_short_text(self, chunker):
        """测试短文本分块"""
        text = "这是一个短文本。"
        
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    @pytest.mark.asyncio
    async def test_chunk_long_text(self, chunker):
        """测试长文本分块"""
        # 创建一个足够长的文本
        text = "这是第一段内容。" * 20 + "\n\n" + "这是第二段内容。" * 20
        
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) > 1
        # 每个chunk都应该有内容
        for chunk in chunks:
            assert len(chunk.content) > 0
    
    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self, chunker):
        """测试带元数据的分块"""
        text = "测试文本内容。"
        metadata = {"source": "test", "author": "tester"}
        
        chunks = await chunker.chunk_text(text, metadata=metadata)
        
        assert chunks[0].metadata["source"] == "test"


class TestFixedSizeChunker:
    """固定大小分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        from src.infrastructure.chunking import FixedSizeChunker
        from src.domain.ports.services import ChunkingConfig
        
        return FixedSizeChunker(config=ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
        ))
    
    @pytest.mark.asyncio
    async def test_fixed_chunk_size(self, chunker):
        """测试固定大小分块"""
        text = "A" * 250  # 250个字符
        
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        # 第一个chunk应该接近chunk_size
        assert len(chunks[0].content) <= 100


class TestGoogleSearchService:
    """Google搜索服务测试"""
    
    @pytest.mark.asyncio
    async def test_search_without_credentials(self):
        """测试无凭据时的搜索"""
        from src.infrastructure.search import GoogleSearchService
        
        service = GoogleSearchService(
            api_key="",
            search_engine_id="",
        )
        
        results = await service.search("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_is_available(self):
        """测试可用性检查"""
        from src.infrastructure.search import GoogleSearchService
        
        service = GoogleSearchService(
            api_key="test_key",
            search_engine_id="test_id",
        )
        
        assert await service.is_available() is True
        
        service_no_key = GoogleSearchService(
            api_key="",
            search_engine_id="",
        )
        
        assert await service_no_key.is_available() is False


class TestMilvusVectorStore:
    """Milvus向量存储测试"""
    
    @pytest.mark.asyncio
    async def test_upsert_empty(self):
        """测试空数据插入"""
        from src.infrastructure.vector_store import MilvusVectorStore
        
        # 使用mock客户端
        with patch('src.infrastructure.vector_store.milvus_store.MilvusVectorStore._create_client'):
            store = MilvusVectorStore()
            store._collection_initialized = True
            
            result = await store.upsert([], [], [])
            
            assert result == 0


class TestContainerDI:
    """依赖注入容器测试"""
    
    def test_container_creation(self):
        """测试容器创建"""
        from src.application.container import Container, AppConfig
        
        config = AppConfig(
            embedding_provider="sentence_transformer",
            embedding_model="all-MiniLM-L12-v2",
        )
        
        container = Container(config)
        
        assert container.config.embedding_provider == "sentence_transformer"
    
    def test_container_singleton(self):
        """测试服务单例"""
        from src.application.container import Container, AppConfig
        
        config = AppConfig()
        container = Container(config)
        
        # 多次获取应该返回同一实例
        service1 = container.get_chunking_service()
        service2 = container.get_chunking_service()
        
        assert service1 is service2
    
    def test_container_reset(self):
        """测试容器重置"""
        from src.application.container import Container, AppConfig
        
        container = Container(AppConfig())
        container.get_chunking_service()
        
        assert len(container._instances) > 0
        
        container.reset()
        
        assert len(container._instances) == 0
