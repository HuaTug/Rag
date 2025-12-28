"""
Domain Entity Tests - 领域实体测试
"""

import pytest
from uuid import UUID
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.entities.document import (
    Document, 
    DocumentChunk, 
    DocumentMetadata,
    DocumentType,
    DocumentStatus,
)
from src.domain.entities.query import (
    Query,
    QueryResult,
    RetrievalContext,
    QueryConfig,
    QueryIntent,
)
from src.domain.entities.embedding import Embedding, EmbeddingBatch


class TestDocument:
    """文档实体测试"""
    
    def test_create_document(self):
        """测试创建文档"""
        doc = Document(
            title="测试文档",
            content="这是测试内容",
        )
        
        assert doc.title == "测试文档"
        assert doc.content == "这是测试内容"
        assert doc.status == DocumentStatus.PENDING
        assert isinstance(doc.id, UUID)
    
    def test_document_add_chunk(self):
        """测试添加分块"""
        doc = Document(title="测试", content="内容")
        chunk = DocumentChunk(content="分块内容")
        
        doc.add_chunk(chunk)
        
        assert len(doc.chunks) == 1
        assert doc.chunk_count == 1
        assert doc.chunks[0].document_id == doc.id
    
    def test_document_status_transitions(self):
        """测试文档状态转换"""
        doc = Document(title="测试", content="内容")
        
        assert doc.status == DocumentStatus.PENDING
        
        doc.mark_as_indexed()
        assert doc.status == DocumentStatus.INDEXED
        assert doc.indexed_at is not None
        
    def test_document_mark_failed(self):
        """测试标记失败"""
        doc = Document(title="测试", content="内容")
        
        doc.mark_as_failed("处理错误")
        
        assert doc.status == DocumentStatus.FAILED
        assert "error" in doc.metadata.extra


class TestDocumentMetadata:
    """文档元数据测试"""
    
    def test_metadata_immutability(self):
        """测试元数据不可变性"""
        meta = DocumentMetadata(source="test", tags=("tag1",))
        
        new_meta = meta.with_tag("tag2")
        
        assert "tag2" not in meta.tags
        assert "tag2" in new_meta.tags
        assert len(meta.tags) == 1
        assert len(new_meta.tags) == 2


class TestQuery:
    """查询实体测试"""
    
    def test_create_query(self):
        """测试创建查询"""
        query = Query(original_text="测试查询")
        
        assert query.original_text == "测试查询"
        assert query.processed_text == "测试查询"
        assert isinstance(query.id, UUID)
    
    def test_query_with_embedding(self):
        """测试添加向量"""
        query = Query(original_text="测试")
        embedding = [0.1] * 384
        
        new_query = query.with_embedding(embedding)
        
        assert query.embedding is None
        assert new_query.embedding == embedding
        assert new_query.has_embedding is True
    
    def test_query_config_defaults(self):
        """测试查询配置默认值"""
        config = QueryConfig()
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.enable_rerank is True


class TestRetrievalContext:
    """检索上下文测试"""
    
    def test_final_score_with_rerank(self):
        """测试最终分数（有重排分数）"""
        ctx = RetrievalContext(
            chunk_id=UUID(int=1),
            document_id=UUID(int=1),
            content="测试",
            score=0.8,
            rerank_score=0.95,
        )
        
        assert ctx.final_score == 0.95
    
    def test_final_score_without_rerank(self):
        """测试最终分数（无重排分数）"""
        ctx = RetrievalContext(
            chunk_id=UUID(int=1),
            document_id=UUID(int=1),
            content="测试",
            score=0.8,
        )
        
        assert ctx.final_score == 0.8
    
    def test_to_prompt_context(self):
        """测试转换为prompt上下文"""
        ctx = RetrievalContext(
            chunk_id=UUID(int=1),
            document_id=UUID(int=1),
            content="测试内容",
            score=0.9,
            source="test",
            title="测试标题",
        )
        
        prompt_ctx = ctx.to_prompt_context()
        
        assert "测试内容" in prompt_ctx
        assert "test" in prompt_ctx


class TestEmbedding:
    """嵌入实体测试"""
    
    def test_create_embedding(self):
        """测试创建嵌入"""
        emb = Embedding(vector=[0.1, 0.2, 0.3], model="test")
        
        assert len(emb) == 3
        assert emb.dimension == 3
    
    def test_normalize_embedding(self):
        """测试归一化"""
        emb = Embedding(vector=[3.0, 4.0], model="test")
        
        normalized = emb.normalize()
        
        # 3-4-5三角形，归一化后应该是 0.6, 0.8
        assert abs(normalized.vector[0] - 0.6) < 0.01
        assert abs(normalized.vector[1] - 0.8) < 0.01
