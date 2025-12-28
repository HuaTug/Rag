"""
OpenAI Embedding Adapter - OpenAI嵌入模型适配器

实现EmbeddingService接口，使用OpenAI API
"""

import logging
from typing import List, Optional

from src.domain.ports.services import EmbeddingService, EmbeddingConfig
from src.domain.entities.embedding import Embedding, EmbeddingBatch


logger = logging.getLogger(__name__)


class OpenAIEmbedding(EmbeddingService):
    """
    OpenAI嵌入服务实现
    
    支持OpenAI和兼容的API（如Azure OpenAI）
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.config = config or EmbeddingConfig(
            model_name=model,
            dimension=1536 if "small" in model else 3072,
        )
        self._client = None
        
        logger.info(f"初始化OpenAI Embedding: {model}")
    
    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self):
        """创建OpenAI客户端"""
        try:
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    async def embed_text(self, text: str) -> Embedding:
        """嵌入单个文本"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            
            vector = response.data[0].embedding
            
            return Embedding(
                vector=vector,
                model=self.model,
                dimension=len(vector),
            )
        except Exception as e:
            logger.error(f"OpenAI嵌入失败: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingBatch:
        """批量嵌入文本"""
        if not texts:
            return EmbeddingBatch(embeddings=[], texts=[], model=self.model)
        
        try:
            # OpenAI支持批量请求
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            
            embeddings = [
                Embedding(
                    vector=item.embedding,
                    model=self.model,
                    dimension=len(item.embedding),
                )
                for item in response.data
            ]
            
            return EmbeddingBatch(
                embeddings=embeddings,
                texts=texts,
                model=self.model,
            )
        except Exception as e:
            logger.error(f"OpenAI批量嵌入失败: {e}")
            raise
    
    async def embed_query(self, query: str) -> Embedding:
        """嵌入查询"""
        return await self.embed_text(query)
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.config.dimension
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model
