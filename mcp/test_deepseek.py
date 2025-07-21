#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek API测试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_deepseek_api():
    """测试DeepSeek API调用"""
    print("🧪 测试DeepSeek API调用")
    print("=" * 40)
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY","sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY 环境变量未设置")
        print("请设置：export DEEPSEEK_API_KEY='your_api_key'")
        return False
    
    try:
        from ask_llm import TencentDeepSeekClient
        
        # 初始化客户端
        client = TencentDeepSeekClient(api_key)
        print(f"✅ DeepSeek客户端初始化成功")
        print(f"   API Key: {api_key[:10]}...")
        print(f"   Base URL: {client.base_url}")
        
        # 测试1：简单问答
        print("\n📝 测试1：简单问答")
        messages = [
            {"role": "user", "content": "你好，请回复'测试成功'"}
        ]
        
        response = client.chat_completions_create(
            model="deepseek-v3-0324",
            messages=messages,
            stream=False,
            enable_search=False
        )
        
        if response and "choices" in response:
            answer = response["choices"][0]["message"]["content"]
            print(f"✅ 回答: {answer}")
        else:
            print(f"❌ 响应格式异常: {response}")
            return False
        
        # 测试2：带搜索的问答（类似curl示例）
        print("\n🔍 测试2：带搜索功能的问答")
        search_messages = [
            {"role": "user", "content": "哪吒2票房"}
        ]
        
        search_response = client.chat_completions_create(
            model="deepseek-v3-0324",
            messages=search_messages,
            stream=False,  # 注意：curl示例中是stream=true，这里改为false便于测试
            enable_search=True
        )
        
        if search_response and "choices" in search_response:
            search_answer = search_response["choices"][0]["message"]["content"]
            print(f"✅ 搜索回答: {search_answer[:200]}...")
        else:
            print(f"❌ 搜索响应格式异常: {search_response}")
            return False
        
        # 测试3：上下文问答
        print("\n📚 测试3：基于上下文的问答")
        context_messages = [
            {
                "role": "system", 
                "content": "你是一个智能助手，请基于提供的上下文信息回答问题。"
            },
            {
                "role": "user", 
                "content": """
上下文：人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。

问题：什么是人工智能？
"""
            }
        ]
        
        context_response = client.chat_completions_create(
            model="deepseek-v3-0324",
            messages=context_messages,
            stream=False,
            enable_search=False
        )
        
        if context_response and "choices" in context_response:
            context_answer = context_response["choices"][0]["message"]["content"]
            print(f"✅ 上下文回答: {context_answer[:200]}...")
        else:
            print(f"❌ 上下文响应格式异常: {context_response}")
            return False
        
        print("\n🎉 所有DeepSeek API测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def test_curl_equivalent():
    """测试等效于curl示例的调用"""
    print("\n🌐 测试等效curl调用")
    print("=" * 40)
    
    api_key = os.getenv("DEEPSEEK_API_KEY","sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY 未设置")
        return False
    
    try:
        import requests
        import json
        
        url = "http://api.lkeap.cloud.tencent.com/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # 与curl示例完全一致的payload
        payload = {
            "model": "deepseek-v3-0324",
            "messages": [
                {
                    "role": "user",
                    "content": "哪吒2票房"
                }
            ],
            "stream": False,  # 改为false便于测试
            "extra_body": {
                "enable_search": True
            }
        }
        
        print(f"📡 发送请求到: {url}")
        print(f"🔑 使用API Key: {api_key[:10]}...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                answer = result["choices"][0]["message"]["content"]
                print(f"✅ curl等效调用成功")
                print(f"📝 回答: {answer[:200]}...")
                return True
            else:
                print(f"❌ 响应格式异常: {result}")
                return False
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ curl等效测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🤖 DeepSeek API完整测试")
    print("=" * 50)
    
    # 测试1：使用封装的客户端
    test1_result = test_deepseek_api()
    
    # 测试2：直接HTTP调用（等效curl）
    test2_result = test_curl_equivalent()
    
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    print(f"封装客户端测试: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"curl等效测试: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！DeepSeek API配置正确")
        print("\n现在可以使用RAG系统了:")
        print("  python3 simple_rag.py")
        print("  python3 rag_system.py")
    else:
        print("\n❌ 部分测试失败，请检查:")
        print("  1. API密钥是否正确")
        print("  2. 网络连接是否正常")
        print("  3. API接口是否可访问")

if __name__ == "__main__":
    main()
