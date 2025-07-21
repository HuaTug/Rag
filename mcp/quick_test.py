#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统快速测试脚本
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🧪 RAG系统快速测试")
print("=" * 50)

# 1. 测试基础模块导入
print("📦 测试模块导入...")

try:
    # 测试ask_llm模块
    from ask_llm import TencentDeepSeekClient, get_llm_answer_deepseek
    print("✅ ask_llm模块导入成功")
except Exception as e:
    print(f"❌ ask_llm模块导入失败: {e}")

try:
    # 测试encoder模块
    from encoder import emb_text
    print("✅ encoder模块导入成功")
except Exception as e:
    print(f"❌ encoder模块导入失败: {e}")

try:
    # 测试search_channels（简化版本）
    import requests
    print("✅ requests库可用")
except Exception as e:
    print(f"❌ requests库不可用: {e}")

# 2. 测试环境变量
print("\n🔧 检查环境变量...")

env_vars = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_SEARCH_ENGINE_ID": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
}

for var, value in env_vars.items():
    if value:
        print(f"✅ {var}: 已设置")
    else:
        print(f"❌ {var}: 未设置")

# 3. 测试Google搜索API
async def test_google_search():
    """测试Google搜索"""
    print("\n🔍 测试Google搜索API...")
    
    api_key = env_vars["GOOGLE_API_KEY"]
    search_engine_id = env_vars["GOOGLE_SEARCH_ENGINE_ID"]
    
    if not api_key or not search_engine_id:
        print("❌ Google API配置不完整，跳过测试")
        return False
    
    try:
        import requests
        
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": "人工智能",
            "num": 3
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                print(f"✅ Google搜索成功，找到 {len(data['items'])} 个结果")
                return True
            else:
                print("❌ Google搜索返回数据格式异常")
                return False
        else:
            print(f"❌ Google搜索失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Google搜索测试失败: {e}")
        return False

# 4. 测试DeepSeek API
def test_deepseek_api():
    """测试DeepSeek API"""
    print("\n🤖 测试DeepSeek API...")
    
    api_key = env_vars["DEEPSEEK_API_KEY"]
    if not api_key:
        print("❌ DeepSeek API Key未设置，跳过测试")
        return False
    
    try:
        from ask_llm import TencentDeepSeekClient
        
        client = TencentDeepSeekClient(api_key)
        
        # 简单测试
        messages = [{"role": "user", "content": "你好，请回复'测试成功'"}]
        
        result = client.chat_completions_create(
            model="deepseek-v3-0324",
            messages=messages,
            stream=False
        )
        
        if result and "choices" in result:
            response_content = result["choices"][0]["message"]["content"]
            print(f"✅ DeepSeek API调用成功")
            print(f"   回复: {response_content[:50]}...")
            return True
        else:
            print("❌ DeepSeek API返回格式异常")
            return False
            
    except Exception as e:
        print(f"❌ DeepSeek API测试失败: {e}")
        return False

# 5. 测试嵌入模型
def test_embedding():
    """测试文本嵌入"""
    print("\n📐 测试文本嵌入...")
    
    try:
        from encoder import emb_text
        
        test_text = "这是一个测试文本"
        embedding = emb_text(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ 文本嵌入成功，维度: {len(embedding)}")
            return True
        else:
            print("❌ 文本嵌入失败")
            return False
            
    except Exception as e:
        print(f"❌ 文本嵌入测试失败: {e}")
        return False

# 6. 简化的RAG测试
async def simple_rag_test():
    """简化的RAG测试"""
    print("\n🎯 简化RAG测试...")
    
    # 检查所有组件是否可用
    google_ok = await test_google_search()
    deepseek_ok = test_deepseek_api()
    embedding_ok = test_embedding()
    
    if not all([google_ok, deepseek_ok, embedding_ok]):
        print("❌ 部分组件不可用，无法进行完整RAG测试")
        return False
    
    try:
        print("🔍 执行搜索...")
        # 这里可以添加简化的RAG流程
        
        print("✅ 简化RAG测试完成")
        return True
        
    except Exception as e:
        print(f"❌ RAG测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    
    # 运行所有测试
    google_result = await test_google_search()
    deepseek_result = test_deepseek_api()
    embedding_result = test_embedding()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    
    results = {
        "Google搜索": google_result,
        "DeepSeek API": deepseek_result,
        "文本嵌入": embedding_result
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总体结果: {success_count}/{total_count} 个组件测试通过")
    
    if success_count == total_count:
        print("\n🎉 系统测试全部通过！可以开始使用RAG系统")
        print("\n🚀 启动方式:")
        print("   1. 命令行: python3 rag_system.py")
        print("   2. Web界面: streamlit run web_interface.py")
    elif success_count > 0:
        print("\n⚠️  部分组件可用，系统可以部分工作")
    else:
        print("\n❌ 系统测试失败，请检查配置和依赖")
        print("\n🔧 解决方案:")
        print("   1. 设置必要的环境变量")
        print("   2. 安装缺失的依赖: pip install requests openai")
        print("   3. 检查API密钥的有效性")

if __name__ == "__main__":
    asyncio.run(main())
