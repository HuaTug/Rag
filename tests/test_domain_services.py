"""
Domain Service Tests - 领域服务测试
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from uuid import UUID

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.entities.query import Query, QueryConfig, RetrievalContext
from src.domain.entities.document import Document, DocumentMetadata
from src.domain.services.rag_service import RAGDomainService, RAGConfig


class TestRAGDomainService:
    """RAG领域服务测试"""
    
    @pytest.fixture
    def rag_service(self, mock_embedding_service, mock_llm_service, mock_vector_store):
        """创建RAG服务实例"""
        return RAGDomainService(
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            vector_store=mock_vector_store,
            search_service=None,
            chunking_service=None,
            rerank_service=None,
            config=RAGConfig(
                similarity_threshold=0.5,
                enable_web_search=False,
                enable_rerank=False,
            ),
        )
    
    @pytest.mark.asyncio
    async def test_process_query_basic(self, rag_service, mock_llm_service, mock_vector_store):
        """测试基本查询处理"""
        # 设置mock返回值
        mock_llm_service.rewrite_query = AsyncMock(return_value="优化后的查询")
        
        query = Query(
            original_text="什么是RAG？",
            config=QueryConfig(top_k=5, enable_web_search=False),
        )
        
        result = await rag_service.process_query(query)
        
        assert result is not None
        assert result.query_id == query.id
        assert result.answer is not None
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_query_no_context(self, rag_service, mock_vector_store):
        """测试无上下文时的处理"""
        mock_vector_store.search = AsyncMock(return_value=[])
        
        query = Query(
            original_text="测试查询",
            config=QueryConfig(top_k=5),
        )
        
        result = await rag_service.process_query(query)
        
        assert result is not None
        # 即使没有上下文，也应该能生成答案
        assert result.answer is not None
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, rag_service):
        """测试置信度计算"""
        contexts = [
            RetrievalContext(
                chunk_id=UUID(int=1),
                document_id=UUID(int=1),
                content="测试1",
                score=0.9,
            ),
            RetrievalContext(
                chunk_id=UUID(int=2),
                document_id=UUID(int=1),
                content="测试2",
                score=0.8,
            ),
        ]
        
        confidence = rag_service._calculate_confidence(contexts)
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # 高质量上下文应该有较高置信度
    
    @pytest.mark.asyncio
    async def test_should_search_web(self, rag_service, mock_search_service):
        """测试网络搜索判断"""
        rag_service.search_service = mock_search_service
        rag_service.config.enable_web_search = True
        
        query = Query(
            original_text="测试",
            config=QueryConfig(enable_web_search=True),
        )
        
        # 无上下文时应该搜索
        assert rag_service._should_search_web(query, []) is True
        
        # 有高质量上下文时不应该搜索
        high_quality_contexts = [
            RetrievalContext(
                chunk_id=UUID(int=1),
                document_id=UUID(int=1),
                content="测试",
                score=0.9,
            ),
            RetrievalContext(
                chunk_id=UUID(int=2),
                document_id=UUID(int=1),
                content="测试",
                score=0.85,
            ),
        ]
        assert rag_service._should_search_web(query, high_quality_contexts) is False


class TestRetrievalService:
    """检索服务测试"""
    
    @pytest.mark.asyncio
    async def test_dense_retrieval(self, mock_embedding_service, mock_vector_store):
        """测试密集检索"""
        from src.domain.services.retrieval_service import RetrievalDomainService
        
        service = RetrievalDomainService(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )
        
        query = Query(
            original_text="测试查询",
            config=QueryConfig(top_k=5),
        )
        
        results = await service._dense_retrieval(query)
        
        assert len(results) > 0
        mock_vector_store.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, mock_embedding_service, mock_vector_store):
        """测试RRF融合"""
        from src.domain.services.retrieval_service import RetrievalDomainService
        
        service = RetrievalDomainService(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )
        
        list1 = [
            RetrievalContext(
                chunk_id=UUID(int=1),
                document_id=UUID(int=1),
                content="内容1",
                score=0.9,
            ),
        ]
        list2 = [
            RetrievalContext(
                chunk_id=UUID(int=2),
                document_id=UUID(int=1),
                content="内容2",
                score=0.8,
            ),
        ]
        
        fused = service._reciprocal_rank_fusion(list1, list2)
        
        # 融合后应该有两个结果（去重后）
        assert len(fused) >= 1
