"""
Semantic Chunker - 语义分块器

基于语义边界进行文档分块
"""

import logging
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.domain.entities.document import Document, DocumentChunk
from src.domain.ports.services import ChunkingService, ChunkingConfig


logger = logging.getLogger(__name__)


class SemanticChunker(ChunkingService):
    """
    语义分块器
    
    基于语义边界（段落、句子）进行分块，保持语义完整性
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig(strategy="semantic")
        
        # 句子分隔符（支持中英文）
        self.sentence_delimiters = r'[。！？.!?]+'
        
        # 段落分隔符
        self.paragraph_delimiters = r'\n\n+'
        
        logger.info(f"初始化语义分块器: chunk_size={self.config.chunk_size}")
    
    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """对文档进行分块"""
        return await self.chunk_text(
            document.content,
            metadata={
                "document_id": str(document.id),
                "title": document.title,
                "source": document.metadata.source,
            }
        )
    
    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """对文本进行语义分块"""
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # 1. 先按段落分割
        paragraphs = re.split(self.paragraph_delimiters, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        current_start = 0
        char_position = 0
        
        for paragraph in paragraphs:
            # 如果当前段落加入后超过chunk_size，需要处理
            if len(current_chunk) + len(paragraph) > self.config.chunk_size:
                # 如果当前chunk已有内容，先保存
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk,
                        start_char=current_start,
                        end_char=char_position,
                        chunk_index=len(chunks),
                        metadata=metadata,
                    ))
                    current_start = char_position
                    current_chunk = ""
                
                # 如果单个段落就超过chunk_size，需要按句子分割
                if len(paragraph) > self.config.chunk_size:
                    sentence_chunks = self._split_by_sentences(
                        paragraph,
                        char_position,
                        len(chunks),
                        metadata,
                    )
                    chunks.extend(sentence_chunks)
                    char_position += len(paragraph) + 2  # +2 for \n\n
                    current_start = char_position
                else:
                    current_chunk = paragraph
                    char_position += len(paragraph) + 2
            else:
                # 添加到当前chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                char_position += len(paragraph) + 2
        
        # 保存最后的chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                content=current_chunk,
                start_char=current_start,
                end_char=char_position,
                chunk_index=len(chunks),
                metadata=metadata,
            ))
        
        # 添加重叠
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        logger.info(f"语义分块完成: {len(chunks)} 个分块")
        return chunks
    
    def _split_by_sentences(
        self,
        text: str,
        start_position: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """按句子分割长段落"""
        sentences = re.split(self.sentence_delimiters, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = start_position
        char_position = start_position
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk,
                        start_char=current_start,
                        end_char=char_position,
                        chunk_index=start_index + len(chunks),
                        metadata=metadata,
                    ))
                    current_start = char_position
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            char_position += len(sentence) + 1
        
        if current_chunk:
            chunks.append(self._create_chunk(
                content=current_chunk,
                start_char=current_start,
                end_char=char_position,
                chunk_index=start_index + len(chunks),
                metadata=metadata,
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        """创建分块对象"""
        return DocumentChunk(
            id=uuid4(),
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
            token_count=len(content) // 4,  # 粗略估计
        )
    
    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """添加分块重叠"""
        if len(chunks) <= 1:
            return chunks
        
        overlap_size = self.config.chunk_overlap
        
        for i in range(1, len(chunks)):
            prev_content = chunks[i-1].content
            if len(prev_content) > overlap_size:
                # 从前一个chunk末尾取overlap内容
                overlap_text = prev_content[-overlap_size:]
                # 尝试在句子边界截断
                last_period = max(
                    overlap_text.rfind("。"),
                    overlap_text.rfind("."),
                    overlap_text.rfind("！"),
                    overlap_text.rfind("？"),
                )
                if last_period > 0:
                    overlap_text = overlap_text[last_period+1:].strip()
                
                if overlap_text:
                    chunks[i].content = overlap_text + " " + chunks[i].content
        
        return chunks
    
    def get_config(self) -> ChunkingConfig:
        """获取分块配置"""
        return self.config
