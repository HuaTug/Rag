#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动态向量存储管理器

提供实时搜索结果的向量化存储和检索功能。
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from mcp_framework import SearchResult

# 添加项目根目录到Python路径以正确导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from encoder import emb_text
from milvus_utils import get_milvus_client

@dataclass
class VectorDocument:
    """向量文档数据结构"""
    id: int
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
        """
        初始化动态向量存储
        
        Args:
            milvus_endpoint: Milvus服务端点
            milvus_token: Milvus访问令牌
            collection_name: 集合名称
            vector_dim: 向量维度
        """
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
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_dim,
                metric_type="IP",
                consistency_level="Strong",
                auto_id=False,
            )
            self.logger.info(f"创建集合: {self.collection_name}")
        else:
            self.logger.info(f"集合已存在: {self.collection_name}")

    def _generate_doc_id(self, content: str, url: str,title:str) -> int:
        """
        生成文档ID - 转换为数字ID
        
        Args:
            content: 文档内容
            url: 文档URL
            
        Returns:
            数字类型的文档ID
        """
        # 增加时间戳满足生成唯一ID，避免冲突
        timestamp_str = str(int(time.time()*1000))
        unique_string = f"{url}_{title}_{content[:200]}_{timestamp_str}"
        hash_hex = hashlib.md5(unique_string.encode()).hexdigest()
        # 将16进制哈希转换为整数，取前15位避免溢出
        doc_id = int(hash_hex[:15], 16)
        return doc_id
    
    def _is_content_similar(self,content1:str,content2:str,threshold:float=0.9) -> bool:
        """
        检查两个内容是否相似
        
        Args:
            content1: 第一个内容
            content2: 第二个内容
            threshold: 相似度阈值
            
        Returns:
            是否相似
        """
        # 使用简单的相似度检查（可以替换为更复杂的算法）
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold

    async def store_search_results(self, search_results: List[SearchResult]) -> int:
        """
        存储搜索结果到向量数据库
        
        Args:
            search_results: 搜索结果列表
            
        Returns:
            成功存储的文档数量
        """
        if not search_results:
            return 0
        
        documents = []
        stored_count = 0
        processed_contents = [] # 用于检查内容相似性    
        
        for result in search_results:
            try:
                # 检查内容相似性
                is_similar_to_existing=any(
                    self._is_content_similar(result.content, existing_content)
                    for existing_content in processed_contents
                )
                if is_similar_to_existing:
                    self.logger.debug(f"内容过于相似，跳过: {result.title[:50]}...")
                    continue

                # 生成向量
                vector = emb_text(result.content)
                
                # 生成文档ID（数字类型）
                doc_id = self._generate_doc_id(result.content, result.url,result.title)
                
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
    
    async def _document_exists(self, doc_id: int) -> bool:
        """
        检查文档是否已存在
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档是否存在
        """
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == {doc_id}',
                output_fields=["id"],
                limit=1
            )
            return len(result) > 0
        except Exception as e:
            self.logger.warning(f"检查文档存在性时出错: {e}")
            return False
    
    async def _batch_insert(self, documents: List[VectorDocument]) -> int:
        """
        批量插入文档
        
        Args:
            documents: 文档列表
            
        Returns:
            成功插入的文档数量
        """
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
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            similarity_threshold: 相似度阈值
            
        Returns:
            相似文档列表
        """
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
                        "metadata": (json.loads(result["entity"]["metadata"]) 
                                   if result["entity"]["metadata"] else {})
                    }
                    filtered_results.append(doc_data)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"相似性搜索失败: {e}")
            return []
    
    async def cleanup_old_documents(self, max_age_hours: int = 24) -> int:
        """
        清理过期文档
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的文档数量
        """
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
                id_list_str = "[" + ",".join(map(str, doc_ids)) + "]"
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'id in {id_list_str}'
                )
                
                self.logger.info(f"清理了 {len(doc_ids)} 个过期文档")
                return len(doc_ids)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"清理过期文档失败: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息字典
        """
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
        """初始化向量存储管理器"""
        self.stores: Dict[str, DynamicVectorStore] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_store(self, name: str, store: DynamicVectorStore):
        """
        添加向量存储
        
        Args:
            name: 存储名称
            store: 向量存储实例
        """
        self.stores[name] = store
        self.logger.info(f"添加向量存储: {name}")
    
    def get_store(self, name: str) -> Optional[DynamicVectorStore]:
        """
        获取向量存储
        
        Args:
            name: 存储名称
            
        Returns:
            向量存储实例或None
        """
        return self.stores.get(name)
    
    async def search_all_stores(self, 
                               query: str, 
                               limit_per_store: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        在所有存储中搜索
        
        Args:
            query: 查询文本
            limit_per_store: 每个存储的结果数量限制
            
        Returns:
            各存储的搜索结果字典
        """
        results = {}
        
        for name, store in self.stores.items():
            try:
                store_results = await store.search_similar(query, limit_per_store)
                results[name] = store_results
            except Exception as e:
                self.logger.error(f"在存储 {name} 中搜索失败: {e}")
                results[name] = []
        
        return results
    
    async def cleanup_all_stores(self, max_age_hours: int = 24) -> int:
        """
        清理所有存储中的过期数据
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            总清理文档数量
        """
        total_cleaned = 0
        
        for name, store in self.stores.items():
            try:
                cleaned = await store.cleanup_old_documents(max_age_hours)
                total_cleaned += cleaned
                self.logger.info(f"存储 {name} 清理了 {cleaned} 个文档")
            except Exception as e:
                self.logger.error(f"清理存储 {name} 失败: {e}")
        
        return total_cleaned
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有存储的统计信息
        
        Returns:
            所有存储的统计信息字典
        """
        stats = {}
        
        for name, store in self.stores.items():
            try:
                stats[name] = store.get_collection_stats()
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats


# 使用示例和测试
async def test_dynamic_vector_store():
    """测试动态向量存储"""
    print("🧪 测试动态向量存储...")
    
    # 创建存储实例
    store = DynamicVectorStore(
        milvus_endpoint="./test_dynamic.db",
        collection_name="test_dynamic_collection"
    )
    
    # 创建测试搜索结果
    from mcp_framework import ChannelType, SearchResult
    
    test_results = [
        SearchResult(
            title="测试文档1",
            content="这是一个关于人工智能的测试文档内容",
            url="https://example.com/ai-doc1",
            source="test_source",
            timestamp=time.time(),
            relevance_score=0.9,
            channel_type=ChannelType.SEARCH_ENGINE,
            metadata={"category": "AI"}
        ),
        SearchResult(
            title="测试文档2", 
            content="这是另一个关于机器学习的测试文档内容",
            url="https://example.com/ml-doc2",
            source="test_source",
            timestamp=time.time(),
            relevance_score=0.8,
            channel_type=ChannelType.SEARCH_ENGINE,
            metadata={"category": "ML"}
        )
    ]
    
    # 存储测试结果
    stored_count = await store.store_search_results(test_results)
    print(f"✅ 存储了 {stored_count} 个文档")
    
    # 搜索测试
    search_results = await store.search_similar("人工智能", limit=5)
    print(f"✅ 搜索到 {len(search_results)} 个相似文档")
    
    for result in search_results:
        print(f"  - {result['title']}: {result['similarity_score']:.3f}")
    
    # 统计信息
    stats = store.get_collection_stats()
    print(f"✅ 集合统计: {stats}")
    
    print("🎉 测试完成!")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_dynamic_vector_store())
