#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版RAG系统启动器
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """简化的RAG系统"""
    
    def __init__(self):
        self.initialized = False
        self.google_api_key = None
        self.google_search_engine_id = None
        self.deepseek_api_key = None
        
    def load_config(self):
        """加载配置"""
        # 从环境变量加载
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 从配置文件加载
        config_file = "config.json"
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if not self.google_api_key:
                    self.google_api_key = config.get("google_search", {}).get("api_key")
                if not self.google_search_engine_id:
                    self.google_search_engine_id = config.get("google_search", {}).get("search_engine_id")
                if not self.deepseek_api_key:
                    self.deepseek_api_key = config.get("deepseek", {}).get("api_key")
                    
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")
    
    def validate_config(self):
        """验证配置"""
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.google_search_engine_id:
            missing.append("GOOGLE_SEARCH_ENGINE_ID")
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
            
        return missing
    
    async def search_google(self, query, max_results=5):
        """Google搜索"""
        try:
            import requests
            
            api_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_search_engine_id,
                "q": query,
                "num": max_results
            }
            
            response = requests.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", "")
                    })
                
                return results
            else:
                logger.error(f"Google搜索失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Google搜索异常: {e}")
            return []
    
    def ask_deepseek(self, question, context=""):
        """DeepSeek问答"""
        try:
            from ask_llm import TencentDeepSeekClient
            
            client = TencentDeepSeekClient(self.deepseek_api_key)
            
            messages = []
            if context:
                system_prompt = """
你是一个智能助手。请基于提供的上下文信息回答用户问题。
如果上下文信息充分，请优先使用上下文中的信息回答。
如果上下文信息不够充分，可以结合你的知识给出有帮助的回答。
请确保回答准确、有条理，并尽可能提供具体的信息。
"""
                messages.append({"role": "system", "content": system_prompt})
                
                user_content = f"""
基于以下上下文信息回答问题：

{context}

问题：{question}
"""
            else:
                user_content = question
            
            messages.append({"role": "user", "content": user_content})
            
            # 调用DeepSeek API，参数与curl示例一致
            result = client.chat_completions_create(
                model="deepseek-v3-0324",
                messages=messages,
                stream=False,  # 不使用流式输出
                enable_search=True  # 启用搜索功能
            )
            
            if result and "choices" in result:
                return result["choices"][0]["message"]["content"]
            else:
                return "抱歉，无法获取回答"
                
        except Exception as e:
            logger.error(f"DeepSeek调用失败: {e}")
            return f"处理问题时出现错误: {str(e)}"
    
    async def process_query(self, query):
        """处理查询"""
        logger.info(f"处理查询: {query}")
        
        # 1. 搜索相关信息
        print("🔍 正在搜索相关信息...")
        search_results = await self.search_google(query)
        
        # 2. 构建上下文
        context = ""
        if search_results:
            print(f"✅ 找到 {len(search_results)} 个搜索结果")
            context_parts = []
            for i, result in enumerate(search_results[:3], 1):
                context_parts.append(f"{i}. {result['title']}\n{result['snippet']}\n来源: {result['url']}")
            context = "\n\n".join(context_parts)
        else:
            print("❌ 未找到相关搜索结果")
        
        # 3. 生成回答
        print("🤖 正在生成回答...")
        answer = self.ask_deepseek(query, context)
        
        return answer
    
    async def initialize(self):
        """初始化系统"""
        print("🚀 初始化简化版RAG系统...")
        
        # 加载配置
        self.load_config()
        
        # 验证配置
        missing = self.validate_config()
        if missing:
            print(f"❌ 缺少配置: {', '.join(missing)}")
            print("\n请设置环境变量或编辑config.json文件:")
            for var in missing:
                print(f"  export {var}='your_key'")
            return False
        
        print("✅ 配置验证通过")
        self.initialized = True
        return True
    
    async def run_interactive(self):
        """运行交互式问答"""
        if not self.initialized:
            if not await self.initialize():
                return
        
        print("\n🎯 简化版RAG系统就绪！")
        print("输入问题开始对话（输入'quit'退出）:\n")
        
        while True:
            try:
                user_input = input("👤 您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    continue
                
                print("🤖 正在思考...")
                answer = await self.process_query(user_input)
                print(f"💡 回答: {answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 处理失败: {e}\n")
        
        print("👋 再见！")

async def main():
    """主函数"""
    print("🤖 简化版RAG智能问答系统")
    print("=" * 50)
    
    system = SimpleRAGSystem()
    await system.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())
