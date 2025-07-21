
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_rag_processor import EnhancedRAGProcessor


async def test_system():
    """测试系统功能"""
    print("🚀 启动增强RAG系统测试...")
    
    # 配置
    config = {
        "milvus_endpoint": os.getenv("MILVUS_ENDPOINT", "localhost:19530"),
        "milvus_token": os.getenv("MILVUS_TOKEN"),
        "enable_search_engine": True,
        "search_engine": "duckduckgo",
        "enable_local_knowledge": True,
        "enable_news": False,
        "search_timeout": 30.0
    }
    
    try:
        # 初始化处理器
        processor = EnhancedRAGProcessor(config)
        print("✅ RAG处理器初始化成功")
        
        # 测试查询
        test_queries = [
            "什么是人工智能？",
            "Python装饰器的使用方法",
            "如何优化机器学习模型？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 测试查询 {i}: {query}")
            
            response = await processor.process_query(query, max_results=5)
            
            print(f"⏱️ 处理时间: {response.processing_time:.2f}秒")
            print(f"🎯 置信度: {response.confidence_score:.2f}")
            print(f"📚 来源数量: {len(response.sources)}")
            print(f"🔍 答案预览: {response.answer[:100]}...")
            
            if response.sources:
                print("📋 主要来源:")
                for j, source in enumerate(response.sources[:3