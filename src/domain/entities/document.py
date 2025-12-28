"""
Document Entity - 文档领域实体

定义文档相关的核心业务对象，遵循DDD原则
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class DocumentType(Enum):
    """文档类型枚举"""
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"


class DocumentStatus(Enum):
    """文档状态枚举"""
    PENDING = "pending"           # 待处理
    PROCESSING = "processing"     # 处理中
    INDEXED = "indexed"           # 已索引
    FAILED = "failed"            # 处理失败
    ARCHIVED = "archived"        # 已归档


@dataclass(frozen=True)
class DocumentMetadata:
    """
    文档元数据（值对象）
    
    使用frozen=True确保不可变性
    """
    source: str                              # 来源（如 "google", "local", "api"）
    source_url: Optional[str] = None         # 原始URL
    author: Optional[str] = None             # 作者
    created_at: Optional[datetime] = None    # 原始创建时间
    language: str = "zh"                     # 语言
    tags: tuple = field(default_factory=tuple)  # 标签（使用tuple保证不可变）
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展字段
    
    def with_tag(self, tag: str) -> "DocumentMetadata":
        """添加标签（返回新对象）"""
        return DocumentMetadata(
            source=self.source,
            source_url=self.source_url,
            author=self.author,
            created_at=self.created_at,
            language=self.language,
            tags=self.tags + (tag,),
            extra=self.extra,
        )


@dataclass
class DocumentChunk:
    """
    文档分块（实体）
    
    代表一个可索引的文档片段
    """
    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)  # 所属文档ID
    content: str = ""                                  # 分块内容
    chunk_index: int = 0                              # 分块索引
    start_char: int = 0                               # 起始字符位置
    end_char: int = 0                                 # 结束字符位置
    embedding: Optional[List[float]] = None           # 向量表示
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 分块质量指标
    token_count: int = 0                              # token数量
    semantic_density: float = 0.0                     # 语义密度
    
    @property
    def has_embedding(self) -> bool:
        """是否已有向量"""
        return self.embedding is not None and len(self.embedding) > 0
    
    def __len__(self) -> int:
        """返回内容长度"""
        return len(self.content)
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Document:
    """
    文档聚合根（Aggregate Root）
    
    管理文档的完整生命周期
    """
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    content: str = ""
    doc_type: DocumentType = DocumentType.TEXT
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata = field(default_factory=lambda: DocumentMetadata(source="unknown"))
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    
    # 处理统计
    chunk_count: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        self.chunk_count = len(self.chunks)
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """添加分块"""
        chunk.document_id = self.id
        chunk.chunk_index = len(self.chunks)
        self.chunks.append(chunk)
        self.chunk_count = len(self.chunks)
        self.updated_at = datetime.utcnow()
    
    def mark_as_indexed(self) -> None:
        """标记为已索引"""
        self.status = DocumentStatus.INDEXED
        self.indexed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def mark_as_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = DocumentStatus.FAILED
        self.metadata = DocumentMetadata(
            source=self.metadata.source,
            source_url=self.metadata.source_url,
            author=self.metadata.author,
            created_at=self.metadata.created_at,
            language=self.metadata.language,
            tags=self.metadata.tags,
            extra={**self.metadata.extra, "error": error}
        )
        self.updated_at = datetime.utcnow()
    
    def clear_chunks(self) -> None:
        """清空分块"""
        self.chunks = []
        self.chunk_count = 0
        self.updated_at = datetime.utcnow()
    
    def get_full_text(self) -> str:
        """获取完整文本"""
        if self.chunks:
            return "\n".join(chunk.content for chunk in self.chunks)
        return self.content
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return False
        return self.id == other.id
