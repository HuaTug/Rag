#!/bin/bash

echo "🤖 RAG系统命令行启动器"
echo "=========================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python"
    exit 1
fi

# 切换到脚本目录
cd "$(dirname "$0")"

# 启动RAG系统
echo "🚀 启动RAG系统..."
python3 rag_system.py

echo "👋 RAG系统已退出"
