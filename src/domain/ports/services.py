"""
Service Ports - 服务接口定义

定义外部服务的抽象接口，实现依赖反转
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator
from uuid import UUID

from ..entities.document import Document, DocumentChunk
from ..entities.query import Query, RetrievalContext
from ..entities.embedding import Embedding, EmbeddingBatch


# ============================================================
# Embedding Service - 向量嵌入服务
# ============================================================

@dataclass
class EmbeddingConfig:
    """嵌入服务配置"""
    model_name: str = "all-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True


class EmbeddingService(ABC):
    """
    向量嵌入服务接口
    
    负责将文本转换为向量表示
    """
    
    @abstractmethod
    async def embed_text(self, text: str) -> Embedding:
        """嵌入单个文本"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingBatch:
        """批量嵌入文本"""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> Embedding:
        """嵌入查询（可能使用不同的策略）"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass


# ============================================================
# LLM Service - 大语言模型服务
# ============================================================

@dataclass
class LLMConfig:
    """LLM服务配置"""
    provider: str = "deepseek"
    model: str = "deepseek-v3-0324"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 60.0
    retry_count: int = 3


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    tokens_used: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0


class LLMService(ABC):
    """
    大语言模型服务接口
    
    负责文本生成和理解
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        pass
    
    @abstractmethod
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        pass
    
    @abstractmethod
    async def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """重写查询"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


# ============================================================
# Vector Store Service - 向量存储服务
# ============================================================

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    provider: str = "milvus"
    endpoint: str = "./milvus_rag.db"
    collection_name: str = "rag_knowledge"
    dimension: int = 384
    metric_type: str = "IP"  # 内积


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class VectorStoreService(ABC):
    """
    向量存储服务接口
    
    负责向量的存储和检索
    """
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """插入或更新向量"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> int:
        """删除向量"""
        pass
    
    @abstractmethod
    async def get_by_ids(self, ids: List[str]) -> List[VectorSearchResult]:
        """根据ID获取向量"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """统计向量数量"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    async def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


# ============================================================
# Search Service - 外部搜索服务
# ============================================================

@dataclass
class SearchConfig:
    """搜索服务配置"""
    provider: str = "google"
    api_key: str = ""
    timeout: float = 10.0
    max_results: int = 10


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    content: str
    url: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = None


class SearchService(ABC):
    """
    外部搜索服务接口
    
    负责网络搜索
    """
    
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


# ============================================================
# Chunking Service - 分块服务
# ============================================================

@dataclass
class ChunkingConfig:
    """分块服务配置"""
    strategy: str = "semantic"  # semantic, fixed, sentence
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000


class ChunkingService(ABC):
    """
    文本分块服务接口
    
    负责将文档分割成可检索的片段
    """
    
    @abstractmethod
    async def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """对文档进行分块"""
        pass
    
    @abstractmethod
    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """对文本进行分块"""
        pass
    
    @abstractmethod
    def get_config(self) -> ChunkingConfig:
        """获取分块配置"""
        pass


# ============================================================
# Rerank Service - 重排服务
# ============================================================

@dataclass
class RerankConfig:
    """重排服务配置"""
    model: str = "cross-encoder"
    top_n: int = 5
    batch_size: int = 32


@dataclass
class RerankResult:
    """重排结果"""
    index: int
    score: float
    content: str


class RerankService(ABC):
    """
    重排服务接口
    
    负责对检索结果进行精排
    """
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """对文档进行重排"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        pass
