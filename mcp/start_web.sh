#!/bin/bash

echo "🌐 RAG系统Web界面启动器"
echo "=========================="

# 检查环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python"
    exit 1
fi

if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit 未安装，请运行: pip install streamlit"
    exit 1
fi

# 切换到脚本目录
cd "$(dirname "$0")"

echo "🚀 启动Web界面..."
echo "📱 浏览器将自动打开: http://localhost:8501"

streamlit run web_interface.py

echo "👋 Web界面已关闭"
