"""
Settings - 配置管理

使用Pydantic Settings实现类型安全的配置管理
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """嵌入服务配置"""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")
    
    provider: str = Field(default="sentence_transformer", description="嵌入服务提供商")
    model: str = Field(default="all-MiniLM-L12-v2", description="嵌入模型名称")
    dimension: int = Field(default=384, description="向量维度")
    batch_size: int = Field(default=32, description="批处理大小")


class LLMSettings(BaseSettings):
    """LLM服务配置"""
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")
    
    provider: str = Field(default="deepseek", description="LLM提供商")
    model: str = Field(default="deepseek-v3-0324", description="模型名称")
    api_key: str = Field(default="", description="API密钥")
    base_url: str = Field(default="http://api.lkeap.cloud.tencent.com/v1", description="API基础URL")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: int = Field(default=2000, ge=1, description="最大token数")
    timeout: float = Field(default=60.0, description="超时时间")


class VectorStoreSettings(BaseSettings):
    """向量存储配置"""
    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", extra="ignore")
    
    provider: str = Field(default="milvus", description="向量存储提供商")
    endpoint: str = Field(default="./milvus_rag.db", description="连接端点")
    collection: str = Field(default="rag_knowledge", description="集合名称")
    metric_type: str = Field(default="IP", description="距离度量类型")


class SearchSettings(BaseSettings):
    """搜索服务配置"""
    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="ignore")
    
    provider: str = Field(default="google", description="搜索服务提供商")
    google_api_key: str = Field(default="", description="Google API密钥")
    google_search_engine_id: str = Field(default="", description="Google搜索引擎ID")
    timeout: float = Field(default=10.0, description="超时时间")
    max_results: int = Field(default=10, description="最大结果数")


class RAGSettings(BaseSettings):
    """RAG服务配置"""
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")
    
    similarity_threshold: float = Field(default=0.5, ge=0, le=1, description="相似度阈值")
    top_k: int = Field(default=10, ge=1, description="返回结果数")
    enable_rerank: bool = Field(default=True, description="是否启用重排")
    enable_web_search: bool = Field(default=True, description="是否启用网络搜索")
    enable_query_rewrite: bool = Field(default=True, description="是否启用查询改写")
    max_context_tokens: int = Field(default=3000, description="最大上下文token数")
    chunk_size: int = Field(default=512, description="分块大小")
    chunk_overlap: int = Field(default=50, description="分块重叠")
    chunking_strategy: str = Field(default="semantic", description="分块策略")


class ServerSettings(BaseSettings):
    """服务器配置"""
    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")
    
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")
    workers: int = Field(default=1, description="工作进程数")
    reload: bool = Field(default=False, description="是否自动重载")
    log_level: str = Field(default="INFO", description="日志级别")


class Settings(BaseSettings):
    """主配置类"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # 应用信息
    app_name: str = Field(default="Enterprise RAG", description="应用名称")
    app_version: str = Field(default="2.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    
    # 子配置
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


def reload_settings():
    """重新加载配置"""
    get_settings.cache_clear()
    return get_settings()
