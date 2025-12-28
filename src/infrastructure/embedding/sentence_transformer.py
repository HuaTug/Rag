"""
SentenceTransformer Embedding Adapter - 开源嵌入模型适配器

实现EmbeddingService接口，使用sentence-transformers库
"""

import logging
from functools import lru_cache
from typing import List, Optional

from src.domain.ports.services import EmbeddingService, EmbeddingConfig
from src.domain.entities.embedding import Embedding, EmbeddingBatch


logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingService):
    """
    SentenceTransformer嵌入服务实现
    
    使用开源的sentence-transformers模型
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._cache = {}
        
        logger.info(f"初始化SentenceTransformer: {self.config.model_name}")
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.model_name)
            logger.info(f"模型加载成功: {self.config.model_name}")
            return model
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"无法加载模型 {self.config.model_name}: {e}")
    
    async def embed_text(self, text: str) -> Embedding:
        """嵌入单个文本"""
        # 检查缓存
        if self.config.cache_enabled and text in self._cache:
            return self._cache[text]
        
        # 生成嵌入
        vector = self.model.encode(text, normalize_embeddings=self.config.normalize)
        
        embedding = Embedding(
            vector=vector.tolist(),
            model=self.config.model_name,
            dimension=len(vector),
        )
        
        # 缓存结果
        if self.config.cache_enabled:
            self._cache[text] = embedding
            # 限制缓存大小
            if len(self._cache) > 1000:
                # 删除最早的一半
                keys = list(self._cache.keys())[:500]
                for k in keys:
                    del self._cache[k]
        
        return embedding
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingBatch:
        """批量嵌入文本"""
        if not texts:
            return EmbeddingBatch(embeddings=[], texts=[], model=self.config.model_name)
        
        # 批量编码
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100,
        )
        
        embeddings = [
            Embedding(
                vector=vec.tolist(),
                model=self.config.model_name,
                dimension=len(vec),
            )
            for vec in vectors
        ]
        
        return EmbeddingBatch(
            embeddings=embeddings,
            texts=texts,
            model=self.config.model_name,
        )
    
    async def embed_query(self, query: str) -> Embedding:
        """嵌入查询（与embed_text相同，但某些模型可能有不同处理）"""
        return await self.embed_text(query)
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.config.dimension
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.config.model_name
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
