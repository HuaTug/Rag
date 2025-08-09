#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试导入是否正常工作
"""

import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
service_dir = os.path.join(current_dir, 'service')

if service_dir not in sys.path:
    sys.path.insert(0, service_dir)

print(" 测试导入...")
print(f"当前目录: {current_dir}")
print(f"服务目录: {service_dir}")
print(f"Python路径: {sys.path[:3]}")

try:
    # 测试导入
    from channel_framework import QueryContext, QueryType
    print(" channel_framework 导入成功")
    
    from smart_query_analyzer import SmartQueryAnalyzer
    print(" smart_query_analyzer 导入成功")
    
    from enhanced_rag_processor import EnhancedRAGProcessor
    print(" enhanced_rag_processor 导入成功")
    
    print(" 所有导入测试通过！")
    
except ImportError as e:
    print(f" 导入失败: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f" 其他错误: {e}")
    import traceback
    traceback.print_exc()