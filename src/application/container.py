"""
Dependency Injection Container - 依赖注入容器

统一管理服务的创建和依赖关系
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from src.domain.ports.services import (
    EmbeddingService,
    LLMService,
    VectorStoreService,
    SearchService,
    ChunkingService,
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
    SearchConfig,
)
from src.domain.services import RAGDomainService, DocumentDomainService, RetrievalDomainService
from src.infrastructure.embedding import SentenceTransformerEmbedding, OpenAIEmbedding
from src.infrastructure.llm import DeepSeekLLM, OpenAILLM
from src.infrastructure.vector_store import MilvusVectorStore
from src.infrastructure.search import GoogleSearchService
from src.infrastructure.chunking import SemanticChunker, FixedSizeChunker


logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """应用配置"""
    # Embedding
    embedding_provider: str = "sentence_transformer"  # sentence_transformer, openai
    embedding_model: str = "all-MiniLM-L12-v2"
    embedding_dimension: int = 384
    
    # LLM
    llm_provider: str = "deepseek"  # deepseek, openai
    llm_model: str = "deepseek-v3-0324"
    llm_api_key: str = ""
    llm_base_url: str = ""
    
    # Vector Store
    vector_store_provider: str = "milvus"
    vector_store_endpoint: str = "./milvus_rag.db"
    vector_store_collection: str = "rag_knowledge"
    
    # Search
    search_provider: str = "google"
    google_api_key: str = ""
    google_search_engine_id: str = ""
    
    # Chunking
    chunking_strategy: str = "semantic"  # semantic, fixed
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # RAG
    similarity_threshold: float = 0.5
    enable_rerank: bool = False
    enable_web_search: bool = True
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置"""
        return cls(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformer"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            
            llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
            llm_model=os.getenv("LLM_MODEL", "deepseek-v3-0324"),
            llm_api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://api.lkeap.cloud.tencent.com/v1"),
            
            vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "milvus"),
            vector_store_endpoint=os.getenv("MILVUS_ENDPOINT", "./milvus_rag.db"),
            vector_store_collection=os.getenv("MILVUS_COLLECTION", "rag_knowledge"),
            
            search_provider=os.getenv("SEARCH_PROVIDER", "google"),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            google_search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID", ""),
            
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", "semantic"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
            enable_rerank=os.getenv("ENABLE_RERANK", "false").lower() == "true",
            enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true",
        )


class Container:
    """
    依赖注入容器
    
    负责创建和管理所有服务实例
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()
        self._instances = {}
        
        logger.info("初始化依赖注入容器")
    
    # ============================================================
    # Infrastructure Services
    # ============================================================
    
    def get_embedding_service(self) -> EmbeddingService:
        """获取嵌入服务"""
        if "embedding" not in self._instances:
            if self.config.embedding_provider == "openai":
                self._instances["embedding"] = OpenAIEmbedding(
                    api_key=self.config.llm_api_key,
                    model=self.config.embedding_model,
                )
            else:
                self._instances["embedding"] = SentenceTransformerEmbedding(
                    config=EmbeddingConfig(
                        model_name=self.config.embedding_model,
                        dimension=self.config.embedding_dimension,
                    )
                )
        return self._instances["embedding"]
    
    def get_llm_service(self) -> LLMService:
        """获取LLM服务"""
        if "llm" not in self._instances:
            if self.config.llm_provider == "openai":
                self._instances["llm"] = OpenAILLM(
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model,
                    base_url=self.config.llm_base_url if self.config.llm_base_url else None,
                )
            else:
                self._instances["llm"] = DeepSeekLLM(
                    api_key=self.config.llm_api_key,
                    base_url=self.config.llm_base_url,
                    config=LLMConfig(model=self.config.llm_model),
                )
        return self._instances["llm"]
    
    def get_vector_store(self) -> VectorStoreService:
        """获取向量存储服务"""
        if "vector_store" not in self._instances:
            self._instances["vector_store"] = MilvusVectorStore(
                config=VectorStoreConfig(
                    endpoint=self.config.vector_store_endpoint,
                    collection_name=self.config.vector_store_collection,
                    dimension=self.config.embedding_dimension,
                )
            )
        return self._instances["vector_store"]
    
    def get_search_service(self) -> Optional[SearchService]:
        """获取搜索服务"""
        if "search" not in self._instances:
            if self.config.google_api_key and self.config.google_search_engine_id:
                self._instances["search"] = GoogleSearchService(
                    api_key=self.config.google_api_key,
                    search_engine_id=self.config.google_search_engine_id,
                )
            else:
                self._instances["search"] = None
        return self._instances["search"]
    
    def get_chunking_service(self) -> ChunkingService:
        """获取分块服务"""
        if "chunking" not in self._instances:
            config = ChunkingConfig(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if self.config.chunking_strategy == "fixed":
                self._instances["chunking"] = FixedSizeChunker(config)
            else:
                self._instances["chunking"] = SemanticChunker(config)
        return self._instances["chunking"]
    
    # ============================================================
    # Domain Services
    # ============================================================
    
    def get_rag_service(self) -> RAGDomainService:
        """获取RAG领域服务"""
        if "rag_service" not in self._instances:
            from src.domain.services.rag_service import RAGConfig
            
            self._instances["rag_service"] = RAGDomainService(
                embedding_service=self.get_embedding_service(),
                llm_service=self.get_llm_service(),
                vector_store=self.get_vector_store(),
                search_service=self.get_search_service() if self.config.enable_web_search else None,
                chunking_service=self.get_chunking_service(),
                rerank_service=None,  # TODO: 添加rerank服务
                config=RAGConfig(
                    similarity_threshold=self.config.similarity_threshold,
                    enable_rerank=self.config.enable_rerank,
                    enable_web_search=self.config.enable_web_search,
                ),
            )
        return self._instances["rag_service"]
    
    def get_retrieval_service(self) -> RetrievalDomainService:
        """获取检索领域服务"""
        if "retrieval_service" not in self._instances:
            self._instances["retrieval_service"] = RetrievalDomainService(
                embedding_service=self.get_embedding_service(),
                vector_store=self.get_vector_store(),
                llm_service=self.get_llm_service(),
                rerank_service=None,
            )
        return self._instances["retrieval_service"]
    
    def reset(self):
        """重置所有实例"""
        self._instances.clear()
        logger.info("容器已重置")


# 全局容器实例
_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局容器实例"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container():
    """重置全局容器"""
    global _container
    if _container:
        _container.reset()
    _container = None
