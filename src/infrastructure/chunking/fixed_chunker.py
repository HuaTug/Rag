"""
Fixed Size Chunker - 固定大小分块器

按固定字符/token数进行分块
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.domain.entities.document import Document, DocumentChunk
from src.domain.ports.services import ChunkingService, ChunkingConfig


logger = logging.getLogger(__name__)


class FixedSizeChunker(ChunkingService):
    """
    固定大小分块器
    
    简单按字符数分块，带重叠
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig(strategy="fixed")
        logger.info(f"初始化固定大小分块器: size={self.config.chunk_size}")
    
    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """对文档进行分块"""
        return await self.chunk_text(
            document.content,
            metadata={
                "document_id": str(document.id),
                "title": document.title,
            }
        )
    
    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """对文本进行分块"""
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text) < self.config.min_chunk_size:
                # 太短的chunk合并到上一个
                if chunks:
                    chunks[-1].content += chunk_text
                continue
            
            chunks.append(DocumentChunk(
                id=uuid4(),
                content=chunk_text,
                chunk_index=len(chunks),
                start_char=i,
                end_char=min(i + chunk_size, len(text)),
                metadata=metadata,
                token_count=len(chunk_text) // 4,
            ))
        
        logger.info(f"固定分块完成: {len(chunks)} 个分块")
        return chunks
    
    def get_config(self) -> ChunkingConfig:
        """获取分块配置"""
        return self.config
