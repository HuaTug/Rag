#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Milvus向量数据库工具函数
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union

# 尝试导入pymilvus
try:
    from pymilvus import MilvusClient, DataType
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    print("Warning: pymilvus not available, using mock client")

logger = logging.getLogger(__name__)

class MockMilvusClient:
    """模拟Milvus客户端，用于测试"""
    
    def __init__(self, uri: str, token: Optional[str] = None):
        self.uri = uri
        self.token = token
        self.collections = {}
        logger.info(f"创建模拟Milvus客户端: {uri}")
    
    def has_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return collection_name in self.collections
    
    def create_collection(self, collection_name: str, dimension: int, **kwargs):
        """创建集合"""
        self.collections[collection_name] = {
            "dimension": dimension,
            "data": [],
            "schema": kwargs
        }
        logger.info(f"创建模拟集合: {collection_name}, 维度: {dimension}")
        return {"collection_name": collection_name}
    
    def insert(self, collection_name: str, data: List[Dict[str, Any]]):
        """插入数据"""
        if collection_name not in self.collections:
            raise ValueError(f"集合 {collection_name} 不存在")
        
        self.collections[collection_name]["data"].extend(data)
        insert_count = len(data)
        logger.info(f"向集合 {collection_name} 插入 {insert_count} 条数据")
        
        return {"insert_count": insert_count}
    
    def search(self, collection_name: str, data: List[List[float]], limit: int = 10, **kwargs):
        """搜索相似向量"""
        if collection_name not in self.collections:
            return [[]]
        
        # 模拟搜索结果
        collection_data = self.collections[collection_name]["data"]
        results = []
        
        for i, item in enumerate(collection_data[:limit]):
            result = {
                "id": item.get("id", f"mock_id_{i}"),
                "distance": 0.9 - i * 0.1,  # 模拟相似度分数
                "entity": {
                    "content": item.get("content", "模拟内容"),
                    "title": item.get("title", "模拟标题"),
                    "url": item.get("url", "https://example.com"),
                    "source": item.get("source", "mock"),
                    "timestamp": item.get("timestamp", 0),
                    "metadata": item.get("metadata", "{}")
                }
            }
            results.append(result)
        
        return [results]  # 返回嵌套列表格式
    
    def query(self, collection_name: str, filter: str = "", output_fields: List[str] = None, limit: int = 10):
        """查询数据"""
        if collection_name not in self.collections:
            return []
        
        collection_data = self.collections[collection_name]["data"]
        
        # 简单的ID过滤
        if 'id ==' in filter:
            target_id = filter.split('"')[1]
            results = [item for item in collection_data if item.get("id") == target_id]
        elif 'timestamp <' in filter:
            threshold = float(filter.split('<')[1].strip())
            results = [item for item in collection_data if item.get("timestamp", 0) < threshold]
        else:
            results = collection_data[:limit]
        
        return results
    
    def delete(self, collection_name: str, filter: str):
        """删除数据"""
        if collection_name not in self.collections:
            return
        
        collection_data = self.collections[collection_name]["data"]
        original_count = len(collection_data)
        
        # 简单的删除逻辑
        if 'id in' in filter:
            # 提取ID列表（简化处理）
            self.collections[collection_name]["data"] = []
        
        deleted_count = original_count - len(self.collections[collection_name]["data"])
        logger.info(f"从集合 {collection_name} 删除 {deleted_count} 条数据")
    
    def get_collection_stats(self, collection_name: str):
        """获取集合统计信息"""
        if collection_name not in self.collections:
            return {"row_count": 0}
        
        return {"row_count": len(self.collections[collection_name]["data"])}

def get_milvus_client(uri: str, token: Optional[str] = None) -> Union[MilvusClient, MockMilvusClient]:
    """
    获取Milvus客户端实例
    
    Args:
        uri: Milvus服务器地址
        token: 认证令牌（可选）
        
    Returns:
        Milvus客户端实例（MilvusClient 或 MockMilvusClient）
    """
    try:
        if PYMILVUS_AVAILABLE:
            if token:
                client = MilvusClient(uri=uri, token=token)
            else:
                client = MilvusClient(uri=uri)
            logger.info(f"连接到Milvus: {uri}")
            return client
        else:
            logger.warning("PyMilvus不可用，使用模拟客户端")
            return MockMilvusClient(uri, token)
            
    except Exception as e:
        logger.error(f"连接Milvus失败: {e}")
        logger.info("回退到模拟客户端")
        return MockMilvusClient(uri, token)

def get_search_results(client, collection_name: str, query_vector: List[float], 
                      top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
    """
    执行向量搜索
    
    Args:
        client: Milvus客户端
        collection_name: 集合名称
        query_vector: 查询向量
        top_k: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    try:
        search_results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["content", "title", "url", "source", "timestamp"]
        )
        
        results = []
        for result in search_results[0]:
            doc_data = {
                "id": result["id"],
                "score": result["distance"],
                "content": result["entity"]["content"],
                "title": result["entity"]["title"],
                "url": result["entity"]["url"],
                "source": result["entity"]["source"],
                "timestamp": result["entity"]["timestamp"]
            }
            results.append(doc_data)
        
        return results
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return []

# 兼容性函数
def create_collection_if_not_exists(client, collection_name: str, dimension: int = 384):
    """创建集合（如果不存在）"""
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type="IP",
            consistency_level="Strong",
            auto_id=False
        )
        logger.info(f"创建集合: {collection_name}")
    else:
        logger.info(f"集合已存在: {collection_name}")

if __name__ == "__main__":
    # 测试Milvus工具
    print("测试Milvus工具...")
    
    # 测试客户端连接
    client = get_milvus_client("./test_milvus.db")
    print(f"客户端类型: {type(client).__name__}")
    
    # 测试集合操作
    collection_name = "test_collection"
    
    if not client.has_collection(collection_name):
        client.create_collection(collection_name, dimension=384)
    
    # 测试数据插入
    test_data = [
        {
            "id": "test_1",
            "vector": [0.1] * 384,
            "content": "这是测试内容1",
            "title": "测试标题1",
            "url": "https://example.com/1",
            "source": "test",
            "timestamp": 1642680000,
            "metadata": "{}"
        }
    ]
    
    result = client.insert(collection_name, test_data)
    print(f"插入结果: {result}")
    
    # 测试搜索
    query_vector = [0.1] * 384
    search_results = get_search_results(client, collection_name, query_vector, top_k=5)
    print(f"搜索结果: {len(search_results)} 条")
    
    # 测试统计信息
    stats = client.get_collection_stats(collection_name)
    print(f"集合统计: {stats}")
