#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP工具集成测试示例

演示如何在RAG系统中使用MCP工具
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.enhanced_rag_processor import EnhancedRAGProcessor
from service.channel_framework import QueryContext, QueryType


async def test_mcp_integration():
    """测试MCP工具集成"""
    print("🚀 测试MCP工具集成")
    print("=" * 60)
    
    # 配置RAG处理器，启用MCP工具
    config = {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "enable_chinese_segmentation": True,
        "enable_keyword_extraction": True,
        "preserve_code_blocks": True,
        "similarity_threshold": 0.7,
        "milvus_endpoint": "./milvus_rag.db",
        "vector_dim": 384,
        "enable_search_engine": True,
        "log_level": "INFO",
        # 启用MCP工具集成
        "enable_mcp_tools": True,
        "mcp_server_path": "/data/workspace/MCP/mcp-server"
    }
    
    # 创建RAG处理器
    rag_processor = EnhancedRAGProcessor(config=config)
    
    try:
        # 初始化MCP工具
        if rag_processor.mcp_integration:
            print("✅ MCP工具集成已启用")
            
            # 获取可用工具
            available_tools = rag_processor.get_available_mcp_tools()
            print(f"📋 可用MCP工具: {list(available_tools.keys())}")
            
            # 测试查询建议
            test_queries = [
                "计算 15 + 25 的结果",
                "查询用户数据库中的活跃用户",
                "搜索Python编程相关信息",
                "统计用户表中的记录数量"
            ]
            
            for query in test_queries:
                print(f"\n📝 测试查询: {query}")
                suggested_tools = rag_processor.suggest_mcp_tools_for_query(query)
                print(f"🛠️ 建议工具: {suggested_tools}")
                
                # 创建查询上下文
                context = QueryContext(
                    query=query,
                    query_type=QueryType.FACTUAL,
                    max_results=5,
                    timeout=30
                )
                
                # 处理查询
                try:
                    response = await rag_processor.process_query(context)
                    print(f"✅ 查询结果: {response.answer[:200]}...")
                    print(f"⏱️ 处理时间: {response.processing_time:.2f}s")
                    print(f"🎯 置信度: {response.confidence_score:.2f}")
                    
                    # 显示使用的工具
                    tools_used = response.metadata.get("tools_used", [])
                    if tools_used:
                        print(f"🔧 使用的工具: {tools_used}")
                    
                except Exception as e:
                    print(f"❌ 查询处理失败: {e}")
                
                print("-" * 40)
        else:
            print("⚠️ MCP工具集成未启用")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_direct_mcp_tools():
    """直接测试MCP工具调用"""
    print("\n🔧 直接测试MCP工具调用")
    print("=" * 60)
    
    config = {
        "enable_mcp_tools": True,
        "mcp_server_path": "/data/workspace/MCP/mcp-server"
    }
    
    rag_processor = EnhancedRAGProcessor(config=config)
    
    if rag_processor.mcp_integration:
        # 测试计算器
        print("🧮 测试计算器工具...")
        calc_result = await rag_processor.call_mcp_tool("calculator", {
            "operation": "add",
            "x": 15,
            "y": 25
        })
        print(f"计算结果: {calc_result}")
        
        # 测试数据库查询
        print("\n🗄️ 测试数据库查询工具...")
        db_result = await rag_processor.call_mcp_tool("database_query", {
            "query": "select",
            "query_type": "structured",
            "table_name": "users",
            "limit": 3
        })
        print(f"数据库查询结果: {db_result}")
        
        # 测试网络搜索
        print("\n🔍 测试网络搜索工具...")
        search_result = await rag_processor.call_mcp_tool("web_search", {
            "query": "Python编程",
            "limit": 3
        })
        print(f"搜索结果: {search_result}")
    
    else:
        print("⚠️ MCP工具集成未启用")


async def main():
    """主函数"""
    print("🎯 MCP工具集成测试")
    print("=" * 80)
    
    # 检查MCP服务器是否存在
    mcp_server_path = "/data/workspace/MCP/mcp-server"
    if not os.path.exists(mcp_server_path):
        print(f"❌ MCP服务器不存在: {mcp_server_path}")
        print("请先编译MCP服务器:")
        print("cd /data/workspace/MCP && go build -o mcp-server main.go")
        return
    
    try:
        # 测试集成
        await test_mcp_integration()
        
        # 直接测试工具
        await test_direct_mcp_tools()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
