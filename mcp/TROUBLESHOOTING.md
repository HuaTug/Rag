# 🔧 RAG系统故障排除指南

## 常见问题及解决方案

### 1. 模块导入错误

#### 问题：`ModuleNotFoundError: No module named 'xxx'`

**解决方案：**
```bash
# 检查已安装的包
pip3 list

# 安装缺失的包
pip3 install requests aiohttp beautifulsoup4 openai python-dotenv streamlit

# 可选高级包
pip3 install sentence-transformers pymilvus pydantic
```

#### 问题：`cannot import name 'LocalKnowledgeChannel'`

**解决方案：**
这是正常的，该类已被禁用。使用简化版本：
```bash
python3 simple_rag.py
```

### 2. API配置问题

#### 问题：缺少API密钥

**解决方案：**
1. 设置环境变量：
```bash
export GOOGLE_API_KEY='your_key'
export GOOGLE_SEARCH_ENGINE_ID='your_id'
export DEEPSEEK_API_KEY='your_key'
```

2. 或编辑config.json文件：
```json
{
  "google_search": {
    "api_key": "your_google_api_key",
    "search_engine_id": "your_search_engine_id"
  },
  "deepseek": {
    "api_key": "your_deepseek_api_key"
  }
}
```

### 3. 网络连接问题

#### 问题：API调用超时

**解决方案：**
1. 检查网络连接
2. 验证API密钥有效性
3. 尝试使用代理或VPN

### 4. Web界面问题

#### 问题：Streamlit无法启动

**解决方案：**
```bash
# 安装Streamlit
pip3 install streamlit

# 检查端口是否被占用
lsof -i :8501

# 使用不同端口启动
streamlit run web_interface.py --server.port 8502
```

### 5. 权限问题

#### 问题：文件权限不足

**解决方案：**
```bash
# 设置脚本执行权限
chmod +x install.sh set_env.sh start_cli.sh start_web.sh

# 检查目录权限
ls -la
```

### 6. Python版本问题

#### 问题：Python版本不兼容

**解决方案：**
```bash
# 检查Python版本
python3 --version

# 使用Python 3.8+
# 如果版本过低，请升级Python
```

### 7. 依赖冲突

#### 问题：包版本冲突

**解决方案：**
```bash
# 创建虚拟环境
python3 -m venv rag_env
source rag_env/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

## 🧪 调试方法

### 1. 逐步测试

```bash
# 测试基础功能
python3 quick_test.py

# 测试简化版本
python3 simple_rag.py

# 测试完整版本
python3 rag_system.py
```

### 2. 查看日志

```bash
# 查看系统日志
tail -f rag_system.log

# 查看详细错误信息
python3 -v simple_rag.py
```

### 3. 分别测试组件

```bash
# 测试Google搜索
python3 search_channels.py

# 测试DeepSeek API
python3 ask_llm.py

# 测试文本嵌入
python3 encoder.py
```

## 📞 获取帮助

如果以上方法都无法解决问题：

1. **检查错误日志**：记录完整的错误信息
2. **环境信息**：Python版本、操作系统、已安装包列表
3. **复现步骤**：详细描述出现问题的操作步骤
4. **配置信息**：API密钥是否正确设置（不要泄露实际密钥）

## 🔄 重置系统

如果系统出现严重问题，可以完全重置：

```bash
# 删除所有生成的文件
rm -f rag_system.log milvus_rag.db .env

# 重新运行安装
./install.sh
./set_env.sh
```

## ✅ 成功标志

系统正常工作时，您应该能看到：

1. ✅ 所有必要模块成功导入
2. ✅ API密钥验证通过
3. ✅ Google搜索返回结果
4. ✅ DeepSeek API正常响应
5. ✅ 系统能够生成合理的回答

## 🎯 性能优化建议

1. **减少搜索结果数量**：在config.json中设置较小的max_results
2. **使用本地缓存**：启用嵌入向量缓存
3. **网络优化**：使用稳定的网络连接
4. **内存管理**：定期清理向量数据库
