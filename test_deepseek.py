#!/usr/bin/env python3
"""
测试腾讯云DeepSeek API连接
"""

import os
from ask_llm import TencentDeepSeekClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_deepseek_api():
    """测试DeepSeek API连接"""
    
    # 获取API密钥
    api_key = os.getenv("TENCENT_API_KEY", "sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl")
    
    print(f"🔑 使用API密钥: {api_key[:20]}...")
    
    try:
        # 创建客户端
        client = TencentDeepSeekClient(api_key=api_key)
        print("✅ 客户端创建成功")
        
        # 测试简单的聊天
        test_messages = [
            {"role": "user", "content": "你好，请简单介绍一下你自己"}
        ]
        
        print("🚀 发送测试请求...")
        response = client.chat_completions_create(
            model="deepseek-v3-0324",
            messages=test_messages,
            stream=False
        )
        
        print("✅ API调用成功!")
        print(f"📝 响应内容: {response['choices'][0]['message']['content']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")
        return False

def test_rag_function():
    """测试RAG功能"""
    
    api_key = os.getenv("TENCENT_API_KEY", "sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl")
    
    try:
        from ask_llm import get_llm_answer_deepseek
        
        client = TencentDeepSeekClient(api_key=api_key)
        
        # 模拟上下文和问题
        context = "Milvus是一个开源的向量数据库，专门用于存储和检索大规模向量数据。它支持多种向量索引算法，能够进行高效的相似性搜索。"
        question = "什么是Milvus？"
        
        print("🧠 测试RAG功能...")
        answer = get_llm_answer_deepseek(client, context, question)
        
        print("✅ RAG功能测试成功!")
        print(f"📝 问题: {question}")
        print(f"📝 回答: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG功能测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 开始测试腾讯云DeepSeek API...")
    print("=" * 50)
    
    # 测试基本API连接
    basic_test = test_deepseek_api()
    print("=" * 50)
    
    # 测试RAG功能
    if basic_test:
        rag_test = test_rag_function()
    else:
        print("⚠️ 跳过RAG测试，因为基本API连接失败")
        rag_test = False
    
    print("=" * 50)
    print("📊 测试结果汇总:")
    print(f"   基本API连接: {'✅ 成功' if basic_test else '❌ 失败'}")
    print(f"   RAG功能: {'✅ 成功' if rag_test else '❌ 失败'}")
    
    if basic_test and rag_test:
        print("🎉 所有测试通过! 可以正常使用DeepSeek模型了。")
    else:
        print("⚠️ 存在问题，请检查API密钥和网络连接。") 