#!/bin/bash

echo "🛠️  RAG系统一键安装脚本"
echo "============================="

# 检查Python环境
echo "📋 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python 3.8+"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python版本: $python_version"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未找到，请先安装pip"
    exit 1
fi

echo "✅ pip3 可用"

# 安装基础依赖
echo ""
echo "📦 安装基础依赖包..."
pip3 install --upgrade pip

# 必需的包
required_packages=(
    "requests"
    "aiohttp" 
    "beautifulsoup4"
    "openai"
    "python-dotenv"
    "streamlit"
)

# 可选的包（用于高级功能）
optional_packages=(
    "sentence-transformers"
    "pymilvus"
    "pydantic"
)

echo "安装必需包..."
for package in "${required_packages[@]}"; do
    echo "安装 $package..."
    pip3 install "$package"
done

echo ""
echo "安装可选包（用于高级功能）..."
for package in "${optional_packages[@]}"; do
    echo "尝试安装 $package..."
    pip3 install "$package" || echo "⚠️  $package 安装失败，将使用备用方案"
done

echo ""
echo "✅ 依赖安装完成！"

# 创建配置文件示例
echo ""
echo "📝 创建配置文件..."

cat > config.json << 'EOF'
{
  "google_search": {
    "api_key": "",
    "search_engine_id": "",
    "timeout": 10,
    "max_results": 10
  },
  "deepseek": {
    "api_key": "",
    "base_url": "http://api.lkeap.cloud.tencent.com/v1",
    "model": "deepseek-v3-0324"
  },
  "milvus": {
    "uri": "./milvus_rag.db",
    "collection_name": "rag_documents",
    "dimension": 384
  },
  "rag": {
    "similarity_threshold": 0.7,
    "max_context_length": 4000,
    "combine_search_and_vector": true
  }
}
EOF

echo "✅ 配置文件已创建: config.json"

# 创建环境变量设置脚本
cat > set_env.sh << 'EOF'
#!/bin/bash
# RAG系统环境变量设置脚本

echo "请设置以下环境变量："
echo ""
echo "1. Google Custom Search API"
echo "   获取地址: https://console.cloud.google.com/"
read -p "请输入 Google API Key: " google_api_key
read -p "请输入 Google Search Engine ID: " google_search_id

echo ""
echo "2. DeepSeek API"
echo "   获取地址: https://platform.deepseek.com/"
read -p "请输入 DeepSeek API Key: " deepseek_api_key

echo ""
echo "设置环境变量..."
export GOOGLE_API_KEY="$google_api_key"
export GOOGLE_SEARCH_ENGINE_ID="$google_search_id"
export DEEPSEEK_API_KEY="$deepseek_api_key"

# 保存到.env文件
cat > .env << EOL
GOOGLE_API_KEY=$google_api_key
GOOGLE_SEARCH_ENGINE_ID=$google_search_id
DEEPSEEK_API_KEY=$deepseek_api_key
EOL

echo "✅ 环境变量已设置并保存到 .env 文件"
echo ""
echo "现在可以运行 RAG 系统了："
echo "  python3 simple_rag.py"
EOF

chmod +x set_env.sh

echo ""
echo "🎉 安装完成！"
echo ""
echo "📋 下一步操作："
echo "1. 设置API密钥:"
echo "   ./set_env.sh"
echo ""
echo "2. 启动系统:"
echo "   python3 simple_rag.py"
echo ""
echo "3. 或启动Web界面:"
echo "   streamlit run web_interface.py"
echo ""
echo "📚 API密钥获取指南:"
echo "- Google: https://console.cloud.google.com/"
echo "- DeepSeek: https://platform.deepseek.com/"
echo ""
echo "🔧 如有问题，请检查:"
echo "- 网络连接"
echo "- API密钥有效性"
echo "- Python版本 (需要3.8+)"
