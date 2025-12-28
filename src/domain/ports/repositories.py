"""
Repository Ports - 仓储接口定义

定义数据持久化的抽象接口，遵循端口-适配器架构
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from uuid import UUID

from ..entities.document import Document, DocumentChunk
from ..entities.query import Query, QueryResult

# 泛型类型变量
T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    仓储基类接口
    
    定义通用的CRUD操作
    """
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """保存实体"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[T]:
        """根据ID获取实体"""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """删除实体"""
        pass
    
    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """检查实体是否存在"""
        pass


class DocumentRepository(BaseRepository[Document]):
    """
    文档仓储接口
    
    负责文档的持久化操作
    """
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """保存文档"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[Document]:
        """根据ID获取文档"""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """检查文档是否存在"""
        pass
    
    @abstractmethod
    async def find_by_source(self, source: str, limit: int = 100) -> List[Document]:
        """根据来源查找文档"""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: str, limit: int = 100) -> List[Document]:
        """根据状态查找文档"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """统计文档数量"""
        pass
    
    @abstractmethod
    async def bulk_save(self, documents: List[Document]) -> List[Document]:
        """批量保存文档"""
        pass
    
    @abstractmethod
    async def bulk_delete(self, ids: List[UUID]) -> int:
        """批量删除文档"""
        pass


class ChunkRepository(BaseRepository[DocumentChunk]):
    """
    文档分块仓储接口
    
    负责分块的持久化操作
    """
    
    @abstractmethod
    async def save(self, chunk: DocumentChunk) -> DocumentChunk:
        """保存分块"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[DocumentChunk]:
        """根据ID获取分块"""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """删除分块"""
        pass
    
    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """检查分块是否存在"""
        pass
    
    @abstractmethod
    async def find_by_document_id(self, document_id: UUID) -> List[DocumentChunk]:
        """根据文档ID查找所有分块"""
        pass
    
    @abstractmethod
    async def bulk_save(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """批量保存分块"""
        pass
    
    @abstractmethod
    async def delete_by_document_id(self, document_id: UUID) -> int:
        """删除指定文档的所有分块"""
        pass


class QueryRepository(BaseRepository[Query]):
    """
    查询仓储接口
    
    负责查询记录的持久化（用于审计和分析）
    """
    
    @abstractmethod
    async def save(self, query: Query) -> Query:
        """保存查询"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[Query]:
        """根据ID获取查询"""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """删除查询"""
        pass
    
    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """检查查询是否存在"""
        pass
    
    @abstractmethod
    async def find_by_user_id(self, user_id: str, limit: int = 100) -> List[Query]:
        """根据用户ID查找查询历史"""
        pass
    
    @abstractmethod
    async def find_by_session_id(self, session_id: str) -> List[Query]:
        """根据会话ID查找查询"""
        pass
    
    @abstractmethod
    async def save_result(self, result: QueryResult) -> QueryResult:
        """保存查询结果"""
        pass
    
    @abstractmethod
    async def get_result_by_query_id(self, query_id: UUID) -> Optional[QueryResult]:
        """根据查询ID获取结果"""
        pass
