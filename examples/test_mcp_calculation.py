#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试MCP计算工具是否真正被调用
"""

import asyncio
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from service.enhanced_rag_processor import EnhancedRAGProcessor
from service.channel_framework import QueryContext, QueryType

async def test_mcp_calculation():
    """专门测试MCP计算工具调用"""
    print("🧮 测试MCP计算工具调用")
    print("=" * 60)
    
    # 配置启用MCP工具
    config = {
        "enable_mcp_tools": True,
        "mcp_server_path": "/data/workspace/MCP/mcp-server",
        "log_level": "INFO"
    }
    
    # 创建RAG处理器
    rag_processor = EnhancedRAGProcessor(config=config)
    
    try:
        # 测试简单的加法运算
        query = "计算 15 + 25 的结果"
        print(f"📝 测试查询: {query}")
        
        # 创建查询上下文
        context = QueryContext(
            query=query,
            query_type=QueryType.FACTUAL,
            max_results=5,
            timeout=30
        )
        
        # 处理查询
        response = await rag_processor.process_query(context)
        
        # 分析结果
        print(f"\n📊 查询结果分析:")
        print(f"✅ 答案: {response.answer}")
        print(f"⏱️ 处理时间: {response.processing_time:.2f}s")
        print(f"🎯 置信度: {response.confidence_score:.2f}")
        
        # 检查是否使用了MCP工具
        metadata = response.metadata
        used_calculation = metadata.get("used_calculation", False)
        tools_used = metadata.get("tools_used", [])
        
        print(f"\n🔧 工具使用情况:")
        print(f"  - 使用了计算功能: {used_calculation}")
        print(f"  - 使用的工具: {tools_used}")
        
        # 检查计算结果的来源
        if "mcp_calculation" in response.answer.lower() or "mcp calculator" in response.answer.lower():
            print("✅ 确认使用了MCP计算器")
        elif "内置计算器" in response.answer or "fallback" in str(metadata):
            print("⚠️ 使用了内置计算器（MCP可能失败）")
        else:
            print("❓ 计算方式不明确")
            
        # 显示详细的元数据
        print(f"\n📋 详细元数据:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_mcp_call():
    """直接测试MCP工具调用"""
    print("\n🔧 直接测试MCP工具调用")
    print("=" * 60)
    
    config = {
        "enable_mcp_tools": True,
        "mcp_server_path": "/data/workspace/MCP/mcp-server"
    }
    
    rag_processor = EnhancedRAGProcessor(config=config)
    
    try:
        # 直接调用MCP计算器
        print("📞 直接调用MCP计算器...")
        result = await rag_processor.call_mcp_tool("calculator", {
            "operation": "add",
            "x": 15,
            "y": 25
        })
        
        print(f"🔧 MCP调用结果:")
        print(f"  成功: {result['success']}")
        print(f"  结果: {result['result']}")
        print(f"  错误: {result['error']}")
        print(f"  执行时间: {result.get('execution_time', 0)}")
        
        if result['success']:
            print("✅ MCP计算器直接调用成功！")
        else:
            print(f"❌ MCP计算器调用失败: {result['error']}")
            
    except Exception as e:
        print(f"❌ 直接调用测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_calculation())
    asyncio.run(test_direct_mcp_call())
