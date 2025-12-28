"""
Milvus Vector Store Adapter - Milvus向量存储适配器

实现VectorStoreService接口
"""

import logging
from typing import Any, Dict, List, Optional

from src.domain.ports.services import VectorStoreService, VectorStoreConfig, VectorSearchResult


logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStoreService):
    """
    Milvus向量存储服务实现
    
    支持Milvus和Milvus Lite
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._client = None
        self._collection_initialized = False
        
        logger.info(f"初始化Milvus存储: {self.config.endpoint}")
    
    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self):
        """创建Milvus客户端"""
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=self.config.endpoint,
            )
            logger.info("Milvus客户端创建成功")
            return client
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus")
        except Exception as e:
            logger.error(f"Milvus客户端创建失败: {e}")
            raise
    
    async def _ensure_collection(self):
        """确保集合存在"""
        if self._collection_initialized:
            return
        
        try:
            if not self.client.has_collection(self.config.collection_name):
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    dimension=self.config.dimension,
                    metric_type=self.config.metric_type,
                    auto_id=False,
                )
                logger.info(f"创建集合: {self.config.collection_name}")
            self._collection_initialized = True
        except Exception as e:
            logger.error(f"确保集合存在失败: {e}")
            raise
    
    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """插入或更新向量"""
        await self._ensure_collection()
        
        if not ids or not vectors:
            return 0
        
        # 构建数据
        data = []
        for i, (id_, vector, content) in enumerate(zip(ids, vectors, contents)):
            item = {
                "id": id_,
                "vector": vector,
                "content": content,
            }
            if metadata and i < len(metadata):
                item["metadata"] = str(metadata[i])  # Milvus需要字符串
            data.append(item)
        
        try:
            result = self.client.upsert(
                collection_name=self.config.collection_name,
                data=data,
            )
            logger.info(f"Upsert完成: {len(data)} 条记录")
            return result.get("upsert_count", len(data))
        except Exception as e:
            logger.error(f"Upsert失败: {e}")
            raise
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        await self._ensure_collection()
        
        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["content", "metadata"],
            )
            
            search_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    # 解析metadata
                    metadata = {}
                    if "metadata" in hit.get("entity", {}):
                        try:
                            import ast
                            metadata = ast.literal_eval(hit["entity"]["metadata"])
                        except:
                            metadata = {}
                    
                    search_results.append(VectorSearchResult(
                        id=str(hit.get("id", "")),
                        score=hit.get("distance", 0.0),
                        content=hit.get("entity", {}).get("content", ""),
                        metadata=metadata,
                    ))
            
            logger.debug(f"搜索完成: {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    async def delete(self, ids: List[str]) -> int:
        """删除向量"""
        await self._ensure_collection()
        
        if not ids:
            return 0
        
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                ids=ids,
            )
            logger.info(f"删除完成: {len(ids)} 条记录")
            return len(ids)
        except Exception as e:
            logger.error(f"删除失败: {e}")
            raise
    
    async def get_by_ids(self, ids: List[str]) -> List[VectorSearchResult]:
        """根据ID获取向量"""
        await self._ensure_collection()
        
        if not ids:
            return []
        
        try:
            results = self.client.get(
                collection_name=self.config.collection_name,
                ids=ids,
                output_fields=["content", "metadata"],
            )
            
            return [
                VectorSearchResult(
                    id=str(item.get("id", "")),
                    score=1.0,
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                )
                for item in results
            ]
        except Exception as e:
            logger.error(f"获取失败: {e}")
            return []
    
    async def count(self) -> int:
        """统计向量数量"""
        await self._ensure_collection()
        
        try:
            stats = self.client.get_collection_stats(self.config.collection_name)
            return stats.get("row_count", 0)
        except Exception as e:
            logger.error(f"统计失败: {e}")
            return 0
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建集合"""
        try:
            if not self.client.has_collection(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    metric_type=self.config.metric_type,
                )
                logger.info(f"创建集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    async def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logger.info(f"删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            self.client.list_collections()
            return True
        except:
            return False
