#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG服务包

包含智能RAG服务的核心组件。
"""

# 导出主要类和函数 - 避免循环导入，只导入基础框架
from .channel_framework import (
    MProcessor, 
    QueryContext, 
    QueryType, 
    SearchResult,
    ChannelType,
    BaseChannel
)

__all__ = [
    "MProcessor",
    "QueryContext",
    "QueryType",
    "SearchResult",
    "ChannelType",
    "BaseChannel"
]