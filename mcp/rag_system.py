#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统启动脚本和配置管理

这个脚本将所有组件整合起来，提供统一的启动和配置管理。
专门用于命令行界面，不包含Streamlit相关代码。
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional



# 导入所有组件
try:
    # 从mcp目录导入
    from mcp_framework import MCPProcessor, QueryAnalyzer, QueryType, QueryContext
    from search_channels import GoogleSearchChannel, create_google_search_channel
    from dynamic_vector_store import DynamicVectorStore, VectorStoreManager
    
    parent_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0,parent_dir)
    from enhanced_rag_processor import EnhancedRAGProcessor
    from ask_llm import TencentDeepSeekClient, get_llm_answer_deepseek
    from encoder import emb_text
    from milvus_utils import get_milvus_client
    
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print(f"当前Python路径: {sys.path}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RAGSystemConfig:
    """RAG系统配置管理"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为config.json
        """
        self.config_file = config_file or "config.json"
        self.config = self._load_default_config()
        
        # 尝试加载配置文件
        if os.path.exists(self.config_file):
            self._load_config_file()
        
        # 从环境变量覆盖配置
        self._load_from_env()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
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
                "endpoint": "./milvus_rag.db",
                "token": None,
                "collection_name": "rag_documents",
                "vector_dim": 384
            },
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "cache_size": 1000
            },
            "rag": {
                "similarity_threshold": 0.5,
                "max_context_length": 4000,
                "combine_search_and_vector": True,
                "enable_smart_search": True,
                "min_vector_results": 3
            }
        }
    
    def _load_config_file(self):
        """从配置文件加载"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._deep_update(self.config, file_config)
            logger.info(f"配置文件加载成功: {self.config_file}")
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}")
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        env_mappings = {
            "GOOGLE_API_KEY": ["google_search", "api_key"],
            "GOOGLE_SEARCH_ENGINE_ID": ["google_search", "search_engine_id"],
            "DEEPSEEK_API_KEY": ["deepseek", "api_key"],
            "MILVUS_ENDPOINT": ["milvus", "endpoint"],
            "MILVUS_TOKEN": ["milvus", "token"]
        }
        
        for env_key, config_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                self._set_nested_config(config_path, env_value)
                logger.info(f"从环境变量加载: {env_key}")
    
    def _deep_update(self, target: dict, source: dict):
        """深度更新字典"""
        for key, value in source.items():
            if (key in target and 
                isinstance(target[key], dict) and 
                isinstance(value, dict)):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _set_nested_config(self, path: List[str], value: str):
        """设置嵌套配置"""
        current = self.config
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get(self, *path):
        """获取配置值"""
        current = self.config
        for key in path:
            current = current.get(key, {})
        return current
    
    def validate(self) -> bool:
        """验证必要的配置"""
        required_configs = [
            (["google_search", "api_key"], "Google API Key"),
            (["google_search", "search_engine_id"], "Google Search Engine ID"),
            (["deepseek", "api_key"], "DeepSeek API Key")
        ]
        
        missing = []
        for path, name in required_configs:
            if not self.get(*path):
                missing.append(name)
        
        if missing:
            logger.error(f"缺少必要配置: {', '.join(missing)}")
            return False
        
        return True


class RAGSystemManager:
    """RAG系统管理器"""
    
    def __init__(self, config: RAGSystemConfig):
        """
        初始化系统管理器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.mcp_processor = None
        self.rag_processor = None
        self.search_channel = None
        self.vector_store = None
        self.deepseek_client = None
        
    async def initialize(self):
        """初始化所有组件"""
        logger.info("🚀 开始初始化RAG系统...")
        
        # 验证配置
        if not self.config.validate():
            raise ValueError("配置验证失败，请检查必要的API密钥和配置")
        
        # 初始化DeepSeek客户端
        await self._init_deepseek_client()
        
        # 初始化Google搜索通道
        await self._init_search_channel()
        
        # 初始化向量存储
        await self._init_vector_store()
        
        # 初始化MCP处理器
        await self._init_mcp_processor()
        
        # 初始化增强RAG处理器
        await self._init_rag_processor()
        
        logger.info("🎉 RAG系统初始化完成！")
    
    async def _init_deepseek_client(self):
        """初始化DeepSeek客户端"""
        deepseek_config = {
            "api_key": self.config.get("deepseek", "api_key"),
            "base_url": self.config.get("deepseek", "base_url")
        }
        
        self.deepseek_client = TencentDeepSeekClient(
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config["base_url"]
        )
        
        # 测试DeepSeek连接
        try:
            test_messages = [{"role": "user", "content": "测试连接"}]
            test_response = self.deepseek_client.chat_completions_create(
                model=self.config.get("deepseek", "model"),
                messages=test_messages,
                stream=False,
                enable_search=False,
                temperature=0.1
            )
            if test_response and "choices" in test_response:
                logger.info("✅ DeepSeek API连接测试成功")
            else:
                logger.warning("⚠️ DeepSeek API连接测试返回异常格式")
        except Exception as e:
            logger.warning(f"⚠️ DeepSeek API连接测试失败: {e}")
        
        logger.info("✅ DeepSeek客户端初始化完成")
    
    async def _init_search_channel(self):
        """初始化Google搜索通道"""
        self.search_channel = create_google_search_channel(
            api_key=self.config.get("google_search", "api_key"),
            search_engine_id=self.config.get("google_search", "search_engine_id"),
            config={
                "timeout": self.config.get("google_search", "timeout"),
                "max_results": self.config.get("google_search", "max_results")
            }
        )
        logger.info("✅ Google搜索通道初始化完成")
    
    async def _init_vector_store(self):
        """初始化向量存储"""
        self.vector_store = DynamicVectorStore(
            milvus_endpoint=self.config.get("milvus", "endpoint"),
            milvus_token=self.config.get("milvus", "token"),
            collection_name=self.config.get("milvus", "collection_name"),
            vector_dim=self.config.get("milvus", "vector_dim")
        )
        logger.info("✅ 向量存储初始化完成")
    
    async def _init_mcp_processor(self):
        """初始化MCP处理器"""
        self.mcp_processor = MCPProcessor()
        
        # 注册搜索通道到MCP处理器
        if self.search_channel:
            self.mcp_processor.register_channel(self.search_channel)
        
        logger.info("✅ MCP处理器初始化完成")
    
    async def _init_rag_processor(self):
        """初始化增强RAG处理器"""
        channels = [self.search_channel] if self.search_channel else []
        
        # 构建EnhancedRAGProcessor期望的配置格式
        rag_config = {
            # Milvus配置
            "milvus_endpoint": self.config.get("milvus", "endpoint"),
            "endpoint": self.config.get("milvus", "endpoint"),
            "milvus_token": self.config.get("milvus", "token"),
            "token": self.config.get("milvus", "token"),
            "vector_dim": self.config.get("milvus", "vector_dim"),
            "dimension": self.config.get("milvus", "vector_dim"),
            
            # Google搜索配置
            "google_api_key": self.config.get("google_search", "api_key"),
            "google_search_engine_id": self.config.get("google_search", "search_engine_id"),
            "search_timeout": self.config.get("google_search", "timeout"),
            
            # RAG配置
            "similarity_threshold": self.config.get("rag", "similarity_threshold"),
            "max_context_length": self.config.get("rag", "max_context_length"),
            "combine_search_and_vector": self.config.get("rag", "combine_search_and_vector"),
            "enable_smart_search": self.config.get("rag", "enable_smart_search"),
            "min_vector_results": self.config.get("rag", "min_vector_results"),
            
            # 功能开关
            "enable_search_engine": True,
            "enable_local_knowledge": True,
            "enable_news": False
        }
        
        # 输出RAG配置用于调试
        logger.info(f"📋 RAG配置传递: similarity_threshold={rag_config.get('similarity_threshold')}, "
                   f"enable_smart_search={rag_config.get('enable_smart_search')}, "
                   f"min_vector_results={rag_config.get('min_vector_results')}")
        
        self.rag_processor = EnhancedRAGProcessor(
            vector_store=self.vector_store,
            search_channels=channels,
            llm_client=self.deepseek_client,
            config=rag_config
        )
        logger.info("✅ 增强RAG处理器初始化完成")
    
    async def process_query(self, query: str, query_type: str = "factual") -> str:
        """
        处理用户查询
        
        Args:
            query: 用户查询内容
            query_type: 查询类型 (factual, analytical, creative, conversational)
            
        Returns:
            str: 查询结果
        """
        if not self.rag_processor:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        # 转换查询类型
        query_type_map = {
            "factual": QueryType.FACTUAL,
            "analytical": QueryType.ANALYTICAL,
            "creative": QueryType.CREATIVE,
            "conversational": QueryType.CONVERSATIONAL
        }
        
        query_type_enum = query_type_map.get(query_type.lower(), QueryType.FACTUAL)
        
        # 创建查询上下文
        context = QueryContext(
            query=query,
            query_type=query_type_enum,
            max_results=self.config.get("google_search", "max_results")
        )
        
        # 处理查询
        response = await self.rag_processor.process_query(context)
        return response.answer
    
    async def process_query_stream(self, query: str, query_type: str = "factual") -> str:
        """
        处理用户查询 - 流式输出版本
        
        Args:
            query: 用户查询内容
            query_type: 查询类型
            
        Returns:
            str: 查询结果
        """
        if not self.rag_processor:
            raise RuntimeError("系统未初始化，请先调用initialize()")
        
        # 转换查询类型
        query_type_map = {
            "factual": QueryType.FACTUAL,
            "analytical": QueryType.ANALYTICAL,
            "creative": QueryType.CREATIVE,
            "conversational": QueryType.CONVERSATIONAL
        }
        
        query_type_enum = query_type_map.get(query_type.lower(), QueryType.FACTUAL)
        
        # 创建查询上下文
        context = QueryContext(
            query=query,
            query_type=query_type_enum,
            max_results=self.config.get("google_search", "max_results")
        )
        
        # 显示处理进度
        print("🤖 正在思考", end="", flush=True)
        for i in range(3):
            await asyncio.sleep(0.5)
            print(".", end="", flush=True)
        print(" 💭")
        
        # 处理查询
        response = await self.rag_processor.process_query(context)
        
        # 模拟流式输出效果
        answer = response.answer
        words = answer.split()
        
        print("💡 回答: ", end="", flush=True)
        for i, word in enumerate(words):
            print(word, end=" ", flush=True)
            if i % 5 == 4:  # 每5个词暂停一下
                await asyncio.sleep(0.1)
        
        print()  # 换行
        return response.answer
    
    async def test_system(self):
        """测试系统各个组件"""
        logger.info("🧪 开始系统测试...")
        
        test_queries = [
            ("人工智能的发展历史", "factual"),
            ("机器学习和深度学习的区别", "analytical"),
            ("请写一个关于AI的小故事", "creative")
        ]
        
        for query, query_type in test_queries:
            logger.info(f"测试查询: {query} ({query_type})")
            try:
                answer = await self.process_query(query, query_type)
                logger.info(f"✅ 回答: {answer[:100]}...")
            except Exception as e:
                logger.error(f"❌ 查询失败: {e}")
        
        logger.info("🎉 系统测试完成")


async def main():
    """主函数"""
    print("🤖 RAG系统启动器")
    print("=" * 50)
    
    # 加载配置
    config = RAGSystemConfig()
    
    # 检查配置
    if not config.validate():
        print("\n❌ 配置验证失败！")
        print("\n请设置以下环境变量或编辑config.json文件：")
        print("  export GOOGLE_API_KEY='your_google_api_key'")
        print("  export GOOGLE_SEARCH_ENGINE_ID='your_search_engine_id'")
        print("  export DEEPSEEK_API_KEY='your_deepseek_api_key'")
        print("\n或者创建config.json文件包含以上配置")
        
        # 创建示例配置文件
        if not os.path.exists("config.json"):
            config.save_config()
            print("\n已创建示例配置文件: config.json")
        
        return
    
    # 初始化系统管理器
    manager = RAGSystemManager(config)
    
    try:
        # 初始化系统
        await manager.initialize()
        
        # 测试系统
        await manager.test_system()
        
        # 交互式问答
        print("\n🎯 系统就绪！输入问题开始对话")
        print("💡 提示：输入 'stream:问题' 可以使用流式输出")
        print("输入'quit'退出:")
        
        while True:
            try:
                user_input = input("\n👤 您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    continue
                
                # 检查是否使用流式输出
                if user_input.startswith('stream:'):
                    query = user_input[7:].strip()
                    if query:
                        await manager.process_query_stream(query)
                    continue
                
                print("🤖 正在思考...")
                answer = await manager.process_query(user_input)
                print(f"💡 回答: {answer}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 处理失败: {e}")
        
        print("\n👋 再见！")
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"❌ 系统启动失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
