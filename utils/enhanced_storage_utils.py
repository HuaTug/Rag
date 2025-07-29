#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强存储工具类

提供简化的接口来使用增强文本处理和向量存储功能
"""

import logging
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.enhanced_rag_processor import EnhancedRAGProcessor
from service.channel_framework import SearchResult

class EnhancedStorageManager:
    """增强存储管理器 - 简化的接口"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化增强存储管理器
        
        Args:
            config: 配置字典，包含文本处理和存储参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 默认配置
        default_config = {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "enable_chinese_segmentation": True,
            "enable_keyword_extraction": True,
            "preserve_code_blocks": True,
            "similarity_threshold": 0.7,
            "min_chunk_size": 100,
            "max_chunk_size": 1200
        }
        
        # 合并配置
        self.config = {**default_config, **self.config}
        
        # 初始化RAG处理器
        self.rag_processor = EnhancedRAGProcessor(config=self.config)
        
        self.logger.info("✅ 增强存储管理器初始化完成")
    
    async def store_search_results(self, search_results: List[SearchResult]) -> bool:
        """
        存储搜索结果（主要接口）
        
        Args:
            search_results: 搜索结果列表
            
        Returns:
            bool: 存储是否成功
        """
        return await self.rag_processor.store_search_results_with_enhanced_processing(search_results)
    
    async def store_raw_data(self, raw_data: List[Dict[str, Any]]) -> bool:
        """
        存储原始数据（自动转换为SearchResult格式）
        
        Args:
            raw_data: 原始数据列表，每个元素应包含title, content, url等字段
            
        Returns:
            bool: 存储是否成功
        """
        try:
            # 转换为SearchResult格式
            search_results = []
            for data in raw_data:
                result = SearchResult(
                    title=data.get('title', ''),
                    content=data.get('content', ''),
                    url=data.get('url', ''),
                    source=data.get('source', 'unknown'),
                    timestamp=data.get('timestamp', None),
                    relevance_score=data.get('relevance_score', 1.0),
                    metadata=data.get('metadata', {})
                )
                search_results.append(result)
            
            return await self.store_search_results(search_results)
            
        except Exception as e:
            self.logger.error(f"❌ 存储原始数据失败: {e}")
            return False
    
    async def store_text_documents(self, documents: List[Dict[str, str]]) -> bool:
        """
        存储文本文档（简化接口）
        
        Args:
            documents: 文档列表，每个文档包含title和content
            
        Returns:
            bool: 存储是否成功
        """
        try:
            raw_data = []
            for i, doc in enumerate(documents):
                raw_data.append({
                    'title': doc.get('title', f'Document {i+1}'),
                    'content': doc.get('content', ''),
                    'url': doc.get('url', f'doc://{i+1}'),
                    'source': 'document',
                    'relevance_score': 1.0
                })
            
            return await self.store_raw_data(raw_data)
            
        except Exception as e:
            self.logger.error(f"❌ 存储文本文档失败: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict: 包含配置和统计信息的字典
        """
        return {
            "config": self.config,
            "text_processor_type": type(self.rag_processor.text_processor).__name__,
            "features": {
                "chinese_segmentation": self.config.get("enable_chinese_segmentation", False),
                "keyword_extraction": self.config.get("enable_keyword_extraction", False),
                "code_preservation": self.config.get("preserve_code_blocks", False)
            }
        }

# 便捷函数
async def quick_store_search_results(search_results: List[SearchResult], config: Dict[str, Any] = None) -> bool:
    """
    快速存储搜索结果的便捷函数
    
    Args:
        search_results: 搜索结果列表
        config: 可选配置
        
    Returns:
        bool: 存储是否成功
    """
    manager = EnhancedStorageManager(config)
    return await manager.store_search_results(search_results)

async def quick_store_raw_data(raw_data: List[Dict[str, Any]], config: Dict[str, Any] = None) -> bool:
    """
    快速存储原始数据的便捷函数
    
    Args:
        raw_data: 原始数据列表
        config: 可选配置
        
    Returns:
        bool: 存储是否成功
    """
    manager = EnhancedStorageManager(config)
    return await manager.store_raw_data(raw_data)

async def quick_store_documents(documents: List[Dict[str, str]], config: Dict[str, Any] = None) -> bool:
    """
    快速存储文档的便捷函数
    
    Args:
        documents: 文档列表
        config: 可选配置
        
    Returns:
        bool: 存储是否成功
    """
    manager = EnhancedStorageManager(config)
    return await manager.store_text_documents(documents)

# 使用示例
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """演示如何使用增强存储工具"""
        print("🛠️ 增强存储工具演示")
        
        # 方式1: 使用管理器类
        manager = EnhancedStorageManager({
            "chunk_size": 600,
            "enable_chinese_segmentation": True
        })
        
        # 存储原始数据
        raw_data = [
            {
                "title": "Python编程基础",
                "content": "Python是一种高级编程语言，具有简洁的语法和强大的功能。Python is widely used in web development, data science, and AI.",
                "url": "https://example.com/python-basics"
            }
        ]
        
        success = await manager.store_raw_data(raw_data)
        print(f"管理器存储结果: {'✅ 成功' if success else '❌ 失败'}")
        
        # 方式2: 使用便捷函数
        documents = [
            {
                "title": "机器学习概述",
                "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。Machine learning algorithms can be supervised, unsupervised, or reinforcement learning."
            }
        ]
        
        success = await quick_store_documents(documents)
        print(f"便捷函数存储结果: {'✅ 成功' if success else '❌ 失败'}")
        
        # 获取统计信息
        stats = manager.get_processing_stats()
        print(f"处理器统计: {stats}")
    
    asyncio.run(demo())