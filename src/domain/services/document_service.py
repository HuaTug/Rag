"""
Document Domain Service - 文档领域服务

处理文档的索引、分块和管理
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from ..entities.document import Document, DocumentChunk, DocumentStatus
from ..ports.repositories import DocumentRepository, ChunkRepository
from ..ports.services import EmbeddingService, ChunkingService, VectorStoreService


logger = logging.getLogger(__name__)


@dataclass
class IndexingConfig:
    """索引配置"""
    batch_size: int = 32
    parallel_workers: int = 4


class DocumentDomainService:
    """
    文档领域服务
    
    负责文档的处理和索引
    """
    
    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService,
        vector_store: VectorStoreService,
        config: Optional[IndexingConfig] = None,
    ):
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service
        self.vector_store = vector_store
        self.config = config or IndexingConfig()
        
        logger.info("文档领域服务初始化完成")
    
    async def index_document(self, document: Document) -> Document:
        """
        索引单个文档
        
        完整流程:
        1. 文档分块
        2. 生成向量
        3. 存储到向量数据库
        4. 更新文档状态
        """
        try:
            logger.info(f"开始索引文档: {document.id}")
            
            # 1. 分块
            document.status = DocumentStatus.PROCESSING
            await self.document_repo.save(document)
            
            chunks = await self.chunking_service.chunk_document(document)
            
            if not chunks:
                document.mark_as_failed("文档分块失败：没有生成任何分块")
                await self.document_repo.save(document)
                return document
            
            logger.info(f"文档分块完成: {len(chunks)} 个分块")
            
            # 2. 生成向量
            texts = [chunk.content for chunk in chunks]
            embedding_batch = await self.embedding_service.embed_texts(texts)
            
            # 更新分块向量
            for chunk, embedding in zip(chunks, embedding_batch.embeddings):
                chunk.embedding = embedding.vector
                document.add_chunk(chunk)
            
            # 3. 存储到向量数据库
            await self._store_chunks_to_vector_db(chunks)
            
            # 4. 保存分块到仓储
            await self.chunk_repo.bulk_save(chunks)
            
            # 5. 更新文档状态
            document.mark_as_indexed()
            await self.document_repo.save(document)
            
            logger.info(f"文档索引完成: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"文档索引失败: {e}", exc_info=True)
            document.mark_as_failed(str(e))
            await self.document_repo.save(document)
            raise
    
    async def index_documents_batch(self, documents: List[Document]) -> List[Document]:
        """批量索引文档"""
        results = []
        
        for document in documents:
            try:
                indexed_doc = await self.index_document(document)
                results.append(indexed_doc)
            except Exception as e:
                logger.error(f"批量索引中文档 {document.id} 失败: {e}")
                results.append(document)
        
        return results
    
    async def delete_document(self, document_id: UUID) -> bool:
        """删除文档及其分块"""
        try:
            # 获取文档
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                logger.warning(f"文档不存在: {document_id}")
                return False
            
            # 获取分块ID
            chunks = await self.chunk_repo.find_by_document_id(document_id)
            chunk_ids = [str(chunk.id) for chunk in chunks]
            
            # 从向量数据库删除
            if chunk_ids:
                await self.vector_store.delete(chunk_ids)
            
            # 删除分块
            await self.chunk_repo.delete_by_document_id(document_id)
            
            # 删除文档
            await self.document_repo.delete(document_id)
            
            logger.info(f"文档删除完成: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"文档删除失败: {e}")
            raise
    
    async def update_document(self, document: Document) -> Document:
        """更新文档（重新索引）"""
        # 先删除旧数据
        await self.delete_document(document.id)
        
        # 重新索引
        document.status = DocumentStatus.PENDING
        document.clear_chunks()
        
        return await self.index_document(document)
    
    async def _store_chunks_to_vector_db(self, chunks: List[DocumentChunk]) -> int:
        """将分块存储到向量数据库"""
        if not chunks:
            return 0
        
        ids = [str(chunk.id) for chunk in chunks]
        vectors = [chunk.embedding for chunk in chunks if chunk.embedding]
        contents = [chunk.content for chunk in chunks]
        metadata = [
            {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata,
            }
            for chunk in chunks
        ]
        
        return await self.vector_store.upsert(
            ids=ids,
            vectors=vectors,
            contents=contents,
            metadata=metadata,
        )
    
    async def get_document_stats(self) -> dict:
        """获取文档统计信息"""
        try:
            total_docs = await self.document_repo.count()
            vector_count = await self.vector_store.count()
            
            return {
                "total_documents": total_docs,
                "total_vectors": vector_count,
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
