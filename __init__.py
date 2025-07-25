#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG搜索系统 - 基于Milvus的智能检索增强生成系统

提供智能查询分析、向量搜索、多渠道数据获取等功能。
"""

__version__ = "1.0.0"
__author__ = "RAG Team"
__description__ = "基于Milvus的智能检索增强生成系统"

# 导出主要组件
from .service import (
    RAGService,
    QueryRequest,
    QueryResponse,
    HealthResponse
)

__all__ = [
    "RAGService",
    "QueryRequest", 
    "QueryResponse",
    "HealthResponse"
]