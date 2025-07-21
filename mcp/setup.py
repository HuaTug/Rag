#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统依赖包管理和安装脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """运行命令并显示结果"""
    print(f"🔧 {description}")
    print(f"   执行: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 失败: {e}")
        print(f"   输出: {e.stdout}")
        print(f"   错误: {e.stderr}")
        return False

def install_requirements():
    """安装必要的依赖包"""
    print("📦 开始安装RAG系统依赖包...")
    
    # 基础依赖包列表
    packages = [
        "requests",           # HTTP请求
        "aiohttp",           # 异步HTTP请求
        "beautifulsoup4",    # HTML解析
        "sentence-transformers",  # 文本嵌入
        "pymilvus",          # Milvus向量数据库
        "openai",            # OpenAI API
        "python-dotenv",     # 环境变量管理
        "streamlit",         # Web界面
        "asyncio",           # 异步编程（通常内置）
    ]
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  警告: 建议使用Python 3.8或更高版本")
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 安装每个包
    success_count = 0
    for package in packages:
        if run_command(f"pip install {package}", f"安装 {package}"):
            success_count += 1
        else:
            print(f"   ⚠️  可选择稍后手动安装: pip install {package}")
    
    print(f"\n📊 安装结果: {success_count}/{len(packages)} 个包安装成功")
    
    # 特殊处理一些包
    print("\n🔧 检查特殊依赖...")
    
    # 检查sentence-transformers模型
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers 可用")
        
        # 尝试下载默认模型
        print("🔽 下载嵌入模型 (首次使用可能需要一些时间)...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ 嵌入模型下载成功")
        except Exception as e:
            print(f"⚠️  模型下载失败: {e}")
            print("   系统将使用备用方案")
            
    except ImportError:
        print("❌ sentence-transformers 不可用")
    
    print("\n🎉 依赖安装完成！")

def check_environment():
    """检查环境变量配置"""
    print("🔍 检查环境变量配置...")
    
    required_vars = [
        "GOOGLE_API_KEY",
        "GOOGLE_SEARCH_ENGINE_ID", 
        "DEEPSEEK_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: 已设置")
    
    if missing_vars:
        print(f"\n⚠️  缺少以下环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        
        print(f"\n请设置环境变量，例如:")
        print(f"export GOOGLE_API_KEY='your_google_api_key'")
        print(f"export GOOGLE_SEARCH_ENGINE_ID='your_search_engine_id'")
        print(f"export DEEPSEEK_API_KEY='your_deepseek_api_key'")
        
        return False
    else:
        print("✅ 所有必要的环境变量都已设置")
        return True

def create_startup_script():
    """创建启动脚本"""
    print("📝 创建启动脚本...")
    
    # 命令行启动脚本
    cli_script = """#!/bin/bash

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
"""
    
    # Web界面启动脚本
    web_script = """#!/bin/bash

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
"""
    
    # 写入启动脚本
    with open("start_cli.sh", "w") as f:
        f.write(cli_script)
    os.chmod("start_cli.sh", 0o755)
    
    with open("start_web.sh", "w") as f:
        f.write(web_script)
    os.chmod("start_web.sh", 0o755)
    
    print("✅ 启动脚本创建完成:")
    print("   - start_cli.sh: 命令行界面")
    print("   - start_web.sh: Web界面")

def main():
    """主函数"""
    print("🔧 RAG系统安装和配置工具")
    print("=" * 50)
    
    # 安装依赖
    install_requirements()
    
    print("\n" + "=" * 50)
    
    # 检查环境变量
    env_ok = check_environment()
    
    print("\n" + "=" * 50)
    
    # 创建启动脚本
    create_startup_script()
    
    print("\n" + "=" * 50)
    print("📋 安装总结")
    print("=" * 50)
    
    if env_ok:
        print("✅ 系统安装完成，可以开始使用！")
        print("\n🚀 启动方式:")
        print("   1. 命令行: ./start_cli.sh 或 python3 rag_system.py")
        print("   2. Web界面: ./start_web.sh 或 streamlit run web_interface.py")
    else:
        print("⚠️  请先配置必要的环境变量，然后重新运行检查")
        print("   python3 setup.py")
    
    print("\n📚 文档和帮助:")
    print("   - 配置文件: config.json")
    print("   - 日志文件: rag_system.log")
    print("   - 示例配置: 运行后会自动生成")

if __name__ == "__main__":
    main()
