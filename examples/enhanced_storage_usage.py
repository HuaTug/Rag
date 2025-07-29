#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强存储功能使用示例

展示如何在不同场景下使用 _store_search_results_to_vector 函数
"""

import asyncio
import sys
import os
from typing import List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) 

from service.enhanced_rag_processor import EnhancedRAGProcessor
from service.channel_framework import SearchResult

@dataclass
class MockSearchResult:
    """模拟搜索结果"""
    title: str
    content: str
    url: str
    source: str = "google"
    timestamp: float = None
    relevance_score: float = 1.0
    metadata: dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
        if self.metadata is None:
            self.metadata = {}

class EnhancedStorageDemo:
    """增强存储功能演示"""
    
    def __init__(self):
        # 初始化RAG处理器
        config = {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "enable_chinese_segmentation": True,
            "enable_keyword_extraction": True,
            "preserve_code_blocks": True,
            "similarity_threshold": 0.7,
            # 添加向量存储配置
            "milvus_endpoint": "./milvus_rag.db",  # 使用本地文件数据库
            "vector_dim": 384,  # 向量维度
            "enable_search_engine": True,  
            "log_level": "INFO"
        }
        self.rag_processor = EnhancedRAGProcessor(config=config)
    
    async def demo_basic_usage(self):
        """基本使用示例"""
        print("🚀 基本使用示例")
        print("=" * 50)
        
        # 创建模拟搜索结果
        search_results = [
            MockSearchResult(
                title="Python机器学习入门",
                content="Python是机器学习领域最受欢迎的编程语言之一。它提供了丰富的库，如scikit-learn、pandas和numpy，使得数据处理和模型训练变得简单高效。Machine learning with Python involves data preprocessing, model selection, and evaluation.",
                url="https://example.com/python-ml"
            ),
            MockSearchResult(
                title="深度学习框架对比",
                content="TensorFlow和PyTorch是目前最流行的深度学习框架。TensorFlow由Google开发，具有强大的生产部署能力。PyTorch由Facebook开发，以其动态计算图和易用性著称。Both frameworks support GPU acceleration and have extensive community support.",
                url="https://example.com/dl-frameworks"
            )
        ]
        
        # 使用增强存储功能
        success = await self.rag_processor.store_search_results_with_enhanced_processing(search_results)
        
        if success:
            print("✅ 搜索结果存储成功！")
        else:
            print("❌ 搜索结果存储失败！")
        
        print()
    
    async def demo_batch_processing(self):
        """批量处理示例"""
        print("📦 批量处理示例")
        print("=" * 50)
        
        # 创建大量模拟数据
        batch_results = []
        topics = [
            ("人工智能发展", "人工智能技术正在快速发展，从机器学习到深度学习，再到大语言模型，AI正在改变我们的生活方式。"),
            ("区块链技术", "区块链是一种分布式账本技术，具有去中心化、不可篡改的特点。Bitcoin and Ethereum are popular blockchain platforms."),
            ("云计算服务", "云计算提供按需访问的计算资源，包括IaaS、PaaS和SaaS三种服务模式。AWS, Azure, and GCP are leading cloud providers."),
            ("物联网应用", "物联网连接各种设备，实现智能家居、智慧城市等应用。IoT devices collect and transmit data for analysis."),
            ("网络安全防护", "网络安全包括防火墙、入侵检测、加密等技术。Cybersecurity is crucial for protecting digital assets.")
        ]
        
        for i, (title, content) in enumerate(topics):
            batch_results.append(MockSearchResult(
                title=f"{title} - 第{i+1}部分",
                content=content * 2,  # 增加内容长度
                url=f"https://example.com/topic-{i+1}",
                relevance_score=0.8 + i * 0.05
            ))
        
        # 批量存储
        success = await self.rag_processor.store_search_results_with_enhanced_processing(batch_results)
        
        if success:
            print(f"✅ 批量存储 {len(batch_results)} 个结果成功！")
        else:
            print("❌ 批量存储失败！")
        
        print()
    
    async def demo_error_handling(self):
        """错误处理示例"""
        print("⚠️ 错误处理示例")
        print("=" * 50)
        
        # 测试空结果
        empty_results = []
        success = await self.rag_processor.store_search_results_with_enhanced_processing(empty_results)
        print(f"空结果处理: {'✅ 正确处理' if not success else '❌ 处理异常'}")
        
        # 测试无效内容
        invalid_results = [
            MockSearchResult(
                title="",
                content="",
                url="invalid-url"
            )
        ]
        success = await self.rag_processor.store_search_results_with_enhanced_processing(invalid_results)
        print(f"无效内容处理: {'✅ 正确处理' if not success else '❌ 处理异常'}")
        
        print()
    
    async def demo_integration_example(self):
        """集成使用示例"""
        print("🔗 集成使用示例")
        print("=" * 50)
        
        # 模拟从外部API获取搜索结果
        def fetch_external_search_results(query: str) -> List[MockSearchResult]:
            """模拟外部搜索API"""
            return [
                MockSearchResult(
                    title=f"关于'{query}'的搜索结果1",
                    content=f"这是关于{query}的详细内容。包含了相关的技术细节和应用场景。This content provides comprehensive information about {query} with technical details and use cases.",
                    url=f"https://api.example.com/search?q={query}",
                    source="external_api",
                    relevance_score=0.9
                ),
                MockSearchResult(
                    title=f"'{query}'最新发展动态",
                    content=f"{query}领域的最新发展包括技术创新、市场趋势和未来展望。Recent developments in {query} include technological innovations and market trends.",
                    url=f"https://news.example.com/{query}",
                    source="news_api",
                    relevance_score=0.85
                )
            ]
        
        # 查询并存储
        query = "量子计算"
        external_results = fetch_external_search_results(query)
        
        print(f"从外部API获取到 {len(external_results)} 个关于'{query}'的结果")
        
        # 使用增强存储
        success = await self.rag_processor.store_search_results_with_enhanced_processing(external_results)
        
        if success:
            print("✅ 外部搜索结果集成存储成功！")
        else:
            print("❌ 外部搜索结果集成存储失败！")
        
        print()

async def main():
    """主函数"""
    print("🎯 增强存储功能使用示例")
    print("=" * 60)
    
    demo = EnhancedStorageDemo()
    
    try:
        # 运行各种示例
        await demo.demo_basic_usage()
        await demo.demo_batch_processing()
        await demo.demo_error_handling()
        await demo.demo_integration_example()
        
        print("🎉 所有示例运行完成！")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())