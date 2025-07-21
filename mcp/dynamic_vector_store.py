
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

from mcp_framework import SearchResult
from encoder import emb_text
from milvus_utils import get_milvus_client


@dataclass
class VectorDocument:
    """向量文档数据结构"""
    id: str
    content: str
    title: str
    url: str
    source: str
    timestamp: float
    vector: List[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class DynamicVectorStore:
    """动态向量存储管理器"""
    
    def __init__(self, 
                 milvus_endpoint: str,
                 milvus_token: Optional[str] = None,
                 collection_name: str = "dynamic_rag_collection",
                 vector_dim: int = 384):
        self.milvus_endpoint = milvus_endpoint
        self.milvus_token = milvus_token
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = get_milvus_client(
                uri=self.milvus_endpoint,
                token=self.milvus_token
            )
            self._ensure_collection_exists()
            self.logger.info("Milvus客户端初始化成功")
        except Exception as e:
            self.logger.error(f"Milvus客户端初始化失败: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """确保集合存在"""
        if not self.client.has_collection(self.collection_name):
            # 创建集合
            schema = self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_dim,
                metric_type="IP",
                consistency_level="Strong",
                auto_id=False,  # 使用自定义ID
            )
            self.logger.info(f"创建集合: {self.collection_name}")
        else:
            self.logger.info(f"集合已存在: {self.collection_name}")
    
    def _generate_doc_id(self, content: str, url: str) -> str:
        """生成文档ID"""
        unique_string = f"{url}_{content[:100]}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    async def store_search_results(self, search_results: List[SearchResult]) -> int:
        """存储搜索结果到向量数据库"""
        if not search_results:
            return 0
        
        documents = []
        stored_count = 0
        
        for result in search_results:
            try:
                # 生成向量
                vector = emb_text(result.content)
                
                # 生成文档ID
                doc_id = self._generate_doc_id(result.content, result.url)
                
                # 检查是否已存在
                if await self._document_exists(doc_id):
                    self.logger.debug(f"文档已存在，跳过: {doc_id}")
                    continue
                
                # 创建向量文档
                doc = VectorDocument(
                    id=doc_id,
                    content=result.content,
                    title=result.title,
                    url=result.url,
                    source=result.source,
                    timestamp=result.timestamp,
                    vector=vector,
                    metadata=result.metadata or {}
                )
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.error(f"处理搜索结果时出错: {e}")
                continue
        
        # 批量插入
        if documents:
            stored_count = await self._batch_insert(documents)
        
        self.logger.info(f"成功存储 {stored_count} 个文档")
        return stored_count
    
    async def _document_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在"""
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{doc_id}"',
                output_fields=["id"],
                limit=1
            )
            return len(result) > 0
        except Exception as e:
            self.logger.warning(f"检查文档存在性时出错: {e}")
            return False
    
    async def _batch_insert(self, documents: List[VectorDocument]) -> int:
        """批量插入文档"""
        try:
            # 准备插入数据
            data = []
            for doc in documents:
                data.append({
                    "id": doc.id,
                    "vector": doc.vector,
                    "content": doc.content,
                    "title": doc.title,
                    "url": doc.url,
                    "source": doc.source,
                    "timestamp": doc.timestamp,
                    "metadata": json.dumps(doc.metadata, ensure_ascii=False)
                })
            
            # 执行插入
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            return result["insert_count"]
            
        except Exception as e:
            self.logger.error(f"批量插入失败: {e}")
            return 0
    
    async def search_similar(self, 
                           query: str, 
                           limit: int = 10,
                           similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 向量化查询
            query_vector = emb_text(query)
            
            # 执行搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=limit,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["content", "title", "url", "source", "timestamp", "metadata"]
            )
            
            # 过滤和格式化结果
            filtered_results = []
            for result in search_results[0]:
                if result["distance"] > similarity_threshold:
                    doc_data = {
                        "id": result["id"],
                        "content": result["entity"]["content"],
                        "title": result["entity"]["title"],
                        "url": result["entity"]["url"],
                        "source": result["entity"]["source"],
                        "timestamp": result["entity"]["timestamp"],
                        "similarity_score": result["distance"],
                        "metadata": json.loads(result["entity"]["metadata"]) if result["entity"]["metadata"] else {}
                    }
                    filtered_results.append(doc_data)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"相似性搜索失败: {e}")
            return []
    
    async def cleanup_old_documents(self, max_age_hours: int = 24):
        """清理过期文档"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # 查询过期文档
            old_docs = self.client.query(
                collection_name=self.collection_name,
                filter=f"timestamp < {cutoff_time}",
                output_fields=["id"],
                limit=1000  # 批量处理
            )
            
            if old_docs:
                # 删除过期文档
                doc_ids = [doc["id"] for doc in old_docs]
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'id in {doc_ids}'
                )
                
                self.logger.info(f"清理了 {len(doc_ids)} 个过期文档")
                return len(doc_ids)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"清理过期文档失败: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "total_documents": stats.get("row_count", 0),
                "collection_name": self.collection_name,
                "vector_dimension": self.vector_dim
            }
        except Exception as e:
            self.logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}


class VectorStoreManager:
    """向量存储管理器 - 统一管理多个向量存储"""
    
    def __init__(self):
        self.stores: Dict[str, DynamicVectorStore] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_store(self, name: str, store: DynamicVectorStore):
        """添加向量存储"""
        self.stores[name] = store
        self.logger.info(f"添加向量存储: {name}")
    
    def get_store(self, name: str) -> Optional[DynamicVectorStore]:
        """获取向量存储"""
        return self.stores.get(name)
    
    async def search_all_stores(self, 
                               query: str, 
                               limit_per_store: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """在所有存储中搜索"""
        results = {}
        
        for name, store in self.stores.items():
            try:
                store_results = await store.search_similar(query, limit_per_store)
                results[name] = store_results
            except Exception as e:
                self.logger.error(f"在存储 {name} 中搜索失败: {e}")
                results[name] = []
        
        return results