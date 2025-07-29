#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的RAG处理器

集成多通道搜索、向量存储和LLM生成，提供完整的RAG解决方案。
"""

import logging
import time
import sys
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from channel_framework import MProcessor, QueryContext, QueryAnalyzer, QueryType, SearchResult
from smart_query_analyzer import SmartQueryAnalyzer, QueryAnalysisResult
# 导入core模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from core.search_channels import GoogleSearchChannel
from core.dynamic_vector_store import DynamicVectorStore, VectorStoreManager
from core.ask_llm import get_llm_answer_with_prompt
from core.encoder import emb_text

load_dotenv()

@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    answer: str
    sources: List[Dict[str, Any]]
    search_results: List[SearchResult]
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]
    analysis_result: Optional[QueryAnalysisResult] = None  # 新增分析结果


class EnhancedRAGProcessor:
    """增强的RAG处理器"""
    
    def __init__(self, vector_store=None, search_channels=None, llm_client=None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 存储外部传入的组件
        self.vector_store = vector_store
        self.search_channels = search_channels or []
        self.llm_client = llm_client
        
        # 如果没有传入LLM客户端，自动创建一个
        if not self.llm_client:
            self._init_llm_client()
        
        # 初始化组件
        self.mcp_processor = MProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.query_analyzer = QueryAnalyzer()
        
        # 新增智能查询分析器
        self.smart_analyzer = SmartQueryAnalyzer(self.config)

        # 新增：初始化增强文本处理器
        from core.enhanced_text_processor import create_enhanced_text_processor
        text_processor_config = {
            "chunk_size": self.config.get("chunk_size", 800),
            "chunk_overlap": self.config.get("chunk_overlap", 100),
            "enable_chinese_segmentation": self.config.get("enable_chinese_segmentation", True),
            "enable_keyword_extraction": self.config.get("enable_keyword_extraction", True),
            "preserve_code_blocks": self.config.get("preserve_code_blocks", True)
        }

        self.text_processor = create_enhanced_text_processor(text_processor_config)
        self.logger.info(f"✅ 初始化增强文本处理器: {self.text_processor.__class__.__name__}")

        # 新增：初始化MCP工具集成
        self.mcp_integration = None
        if self.config.get("enable_mcp_tools", False):
            try:
                from core.mcp_tool_integration import MCPToolIntegration
                self.mcp_integration = MCPToolIntegration(self.config)
                self.logger.info("✅ MCP工具集成模块已加载")
            except ImportError as e:
                self.logger.warning(f"⚠️ MCP工具集成模块加载失败: {e}")
                self.mcp_integration = None

        
        # 智能查询策略配置 - 提高相似度阈值以过滤不相关内容
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)  # 提高相似度阈值到0.7
        self.min_similarity_for_answer = self.config.get("min_similarity_for_answer", 0.6)  # 生成答案的最低相似度
        self.min_vector_results = self.config.get("min_vector_results", 2)  # 减少最少向量结果数量
        self.enable_smart_search = self.config.get("enable_smart_search", True)  # 启用智能搜索
        self.enable_fallback_search = self.config.get("enable_fallback_search", True)  # 启用回退搜索
        
        # 输出配置信息用于调试
        self.logger.info(f"📊 智能搜索配置: similarity_threshold={self.similarity_threshold}, "
                        f"min_similarity_for_answer={self.min_similarity_for_answer}, "
                        f"min_vector_results={self.min_vector_results}, "
                        f"enable_smart_search={self.enable_smart_search}")
        
        # 初始化配置
        self._init_components()
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from core.ask_llm import TencentDeepSeekClient
            import os
            
            # 获取API密钥
            api_key = os.getenv("DEEPSEEK_API_KEY")
            
            if api_key:
                self.llm_client = TencentDeepSeekClient(api_key=api_key)
                self.logger.info("✅ 自动创建DeepSeek LLM客户端成功")
            else:
                self.logger.warning("⚠️ 未找到LLM API密钥，将使用智能回退模式")
                self.llm_client = None
                
        except Exception as e:
            self.logger.error(f"❌ 创建LLM客户端失败: {e}")
            self.llm_client = None
    
    def _init_components(self):
        """初始化各个组件"""
        try:
            # 1. 初始化向量存储
            self._init_vector_stores()
            
            # 2. 初始化搜索通道
            self._init_search_channels()
            
            # 3. 初始化MCP工具集成（异步初始化将在需要时进行）
            if self.mcp_integration:
                self.logger.info("🔧 MCP工具集成模块已准备就绪，将在首次使用时初始化")
            
            self.logger.info("增强RAG处理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def _init_vector_stores(self):
        """初始化向量存储"""
        try:
            # 获取Milvus配置，优先使用传入的配置
            milvus_endpoint = (
                self.config.get("milvus_endpoint") or 
                self.config.get("endpoint") or 
                "./milvus_rag.db"
            )
            milvus_token = (
                self.config.get("milvus_token") or 
                self.config.get("token")
            )
            vector_dim = (
                self.config.get("vector_dim") or 
                self.config.get("dimension") or 
                384
            )
            
            self.logger.info(f"🔧 初始化向量存储: endpoint={milvus_endpoint}, dim={vector_dim}")
            
            # 动态向量存储（用于实时搜索结果）
            try:
                dynamic_store = DynamicVectorStore(
                    milvus_endpoint=milvus_endpoint,
                    milvus_token=milvus_token,
                    collection_name="dynamic_search_results",
                    vector_dim=vector_dim
                )
                self.vector_store_manager.add_store("dynamic", dynamic_store)
                self.logger.info("✅ 动态向量存储初始化成功")
            except Exception as e:
                self.logger.error(f"❌ 动态向量存储初始化失败: {e}")
            
            # 本地知识库存储
            try:
                local_store = DynamicVectorStore(
                    milvus_endpoint=milvus_endpoint,
                    milvus_token=milvus_token,
                    collection_name="local_knowledge",
                    vector_dim=vector_dim
                )
                self.vector_store_manager.add_store("local", local_store)
                self.logger.info("✅ 本地知识库存储初始化成功")
            except Exception as e:
                self.logger.error(f"❌ 本地知识库存储初始化失败: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ 向量存储初始化失败: {e}")
            # 不抛出异常，允许系统继续运行，使用备用方法
    
    def _init_search_channels(self):
        """初始化搜索通道"""
        # 注册外部传入的搜索通道
        for channel in self.search_channels:
            self.mcp_processor.register_channel(channel)
        
        # Google搜索通道（如果配置中启用且没有外部传入）
        if (self.config.get("enable_search_engine", True) and 
            not any(hasattr(ch, 'channel_type') and 
                   str(ch.channel_type).endswith('SEARCH_ENGINE') 
                   for ch in self.search_channels)):
            
            search_config = {
                "api_key": self.config.get("google_api_key"),
                "search_engine_id": self.config.get("google_search_engine_id"),
                "timeout": self.config.get("search_timeout", 10),
                "priority": {"factual": 1, "analytical": 2, "creative": 5, "conversational": 3}
            }
            
            if search_config["api_key"] and search_config["search_engine_id"]:
                search_channel = GoogleSearchChannel(search_config)
                self.mcp_processor.register_channel(search_channel)
        
        # 本地知识库通道（暂时禁用，因为LocalKnowledgeChannel类不存在）
        if False:  # self.config.get("enable_local_knowledge", True):
            local_store = self.vector_store_manager.get_store("local")
            if local_store:
                local_config = {
                    "milvus_client": local_store.client,
                    "collection_name": "local_knowledge",
                    "encoder": emb_text,
                    "priority": {"factual": 2, "analytical": 1, "creative": 4, "conversational": 2}
                }
                # local_channel = LocalKnowledgeChannel(local_config)
                # self.mcp_processor.register_channel(local_channel)
        
        # 新闻通道（暂时禁用，因为NewsChannel类不存在）
        if False:  # self.config.get("enable_news", False):
            news_config = {
                "news_api_key": self.config.get("news_api_key"),
                "priority": {"factual": 3, "analytical": 3, "creative": 6, "conversational": 4}
            }
            # news_channel = NewsChannel(news_config)
            # self.mcp_processor.register_channel(news_channel)
    
    async def _ensure_mcp_initialized(self) -> bool:
        """确保MCP工具已初始化"""
        if self.mcp_integration and not hasattr(self.mcp_integration, '_is_initialized'):
            try:
                success = await self.mcp_integration.initialize()
                self.mcp_integration._is_initialized = success
                if success:
                    self.logger.info("✅ MCP工具集成延迟初始化成功")
                else:
                    self.logger.warning("⚠️ MCP工具集成延迟初始化失败")
                return success
            except Exception as e:
                self.logger.error(f"❌ MCP工具集成延迟初始化异常: {e}")
                self.mcp_integration._is_initialized = False
                return False
        elif self.mcp_integration:
            return getattr(self.mcp_integration, '_is_initialized', False)
        return False
    
    async def _init_mcp_tools(self):
        """初始化MCP工具集成"""
        return await self._ensure_mcp_initialized()
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具"""
        # 确保MCP已初始化
        if not await self._ensure_mcp_initialized():
            return {
                "success": False,
                "error": "MCP工具集成未初始化或初始化失败",
                "result": None
            }
        
        try:
            tool_call = await self.mcp_integration.call_tool_by_name(tool_name, arguments)
            return {
                "success": tool_call.success,
                "error": tool_call.error,
                "result": tool_call.result,
                "execution_time": tool_call.execution_time
            }
        except Exception as e:
            self.logger.error(f"MCP工具调用失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def get_available_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的MCP工具"""
        if self.mcp_integration:
            return self.mcp_integration.get_tool_definitions()
        return {}
    
    def suggest_mcp_tools_for_query(self, query: str) -> List[str]:
        """为查询建议合适的MCP工具"""
        if self.mcp_integration:
            return self.mcp_integration.suggest_tools_for_query(query)
        return []
    
    async def store_search_results_with_enhanced_processing(self, search_results: List[SearchResult]) -> bool:
        """
        公共方法：使用增强文本处理器存储搜索结果到向量数据库
        
        Args:
            search_results: 搜索结果列表
            
        Returns:
            bool: 存储是否成功
            
        Features:
            - 智能文本分块和清理
            - 中英文混合处理
            - 内容去重和质量评分
            - 关键词提取
            - 语言检测
        """
        return await self._store_search_results_to_vector(search_results)
    
    async def _store_search_results_to_vector(self, search_results: List[SearchResult]) -> bool:
        """将搜索结果存储到向量数据库（使用增强文本处理）"""
        try:
            if not search_results:
                self.logger.warning("⚠️ 没有搜索结果需要存储")
                return False
            
            # 转换搜索结果格式
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'title': result.title,
                    'content': result.content,
                    'url': result.url,
                    'source': result.source,
                    'timestamp': result.timestamp,
                    'relevance_score': result.relevance_score
                })
            
            # 使用增强文本处理器处理搜索结果
            text_chunks = self.text_processor.process_search_results(formatted_results)
            
            # 优化chunks用于embedding
            optimized_chunks = self.text_processor.optimize_for_embedding(text_chunks)
            
            if not optimized_chunks:
                self.logger.warning("⚠️ 没有生成有效的文本块")
                return False
            
            # 准备向量存储数据
            documents = []
            for chunk in optimized_chunks:
                documents.append({
                    'content': chunk.content,
                    'title': chunk.title,
                    'url': chunk.url,
                    'source': 'google_search',
                    'timestamp': time.time(),
                    'chunk_id': chunk.chunk_id,
                    'token_count': chunk.token_count,
                    'language': chunk.language,
                    'importance_score': chunk.importance_score,
                    'keywords': chunk.keywords,
                    'metadata': chunk.metadata
                })
            
            # 存储到向量数据库
            # 优先使用外部传入的向量存储，否则使用动态向量存储
            vector_store = self.vector_store or self.vector_store_manager.get_store("dynamic")
            
            if vector_store:
                # 对于DynamicVectorStore，直接使用原始搜索结果
                if hasattr(vector_store, 'store_search_results'):
                    stored_count = await vector_store.store_search_results(search_results)
                    if stored_count > 0:
                        self.logger.info(f"✅ 成功存储 {stored_count} 个搜索结果到向量数据库")
                        return True
                    else:
                        self.logger.error("❌ 向量存储失败")
                        return False
                # 对于其他类型的向量存储，尝试使用add_documents方法
                elif hasattr(vector_store, 'add_documents'):
                    success = await vector_store.add_documents(documents)
                    if success:
                        self.logger.info(f"✅ 成功存储 {len(optimized_chunks)} 个优化文本块到向量数据库")
                        return True
                    else:
                        self.logger.error("❌ 向量存储失败")
                        return False
                else:
                    self.logger.error("❌ 向量存储对象不支持存储操作")
                    return False
            else:
                self.logger.warning("⚠️ 向量存储未初始化")
                self.logger.warning("⚠️ 增强存储失败，尝试使用备用方法")
                # 尝试使用备用存储方法
            return await self._fallback_store_search_results(search_results)
                
        except Exception as e:
            self.logger.error(f"❌ 存储搜索结果到向量数据库失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.logger.warning("⚠️ 增强存储失败，尝试使用备用方法")
            # 尝试使用备用存储方法
            return await self._fallback_store_search_results(search_results)
    
    async def _fallback_store_search_results(self, search_results: List[SearchResult]) -> bool:
        """备用存储方法：使用原始的存储逻辑"""
        try:
            self.logger.info("🔄 使用备用存储方法...")
            await self._store_search_results(search_results)
            self.logger.info("✅ 备用存储方法执行成功")
            return True
        except Exception as e:
            self.logger.error(f"❌ 备用存储方法也失败了: {e}")
            return False

    async def process_query(self, context: QueryContext) -> RAGResponse:
        """
        智能处理查询请求 - 集成Go demo的分析能力
        
        流程：
        1. 智能分析查询意图
        2. 根据分析结果选择最优策略
        3. 执行相应的工具调用
        4. 生成综合回答
        """
        start_time = time.time()
        query = context.query
        
        try:
            self.logger.info(f"🤖 开始智能处理查询: {query}")
            
            # 1. 智能查询分析 - 核心改进
            analysis_result = await self.smart_analyzer.analyze_query_intent(query)
            self.logger.info(f"🧠 查询分析完成: {analysis_result.query_type} "
                           f"(置信度: {analysis_result.confidence:.2f})")
            
            # 1.5 MCP工具建议和增强
            if self.mcp_integration:
                suggested_tools = self.suggest_mcp_tools_for_query(query)
                if suggested_tools:
                    self.logger.info(f"🛠️ 建议使用MCP工具: {', '.join(suggested_tools)}")
                    
                    # 根据建议的工具调整分析结果
                    if "calculator" in suggested_tools and not analysis_result.needs_calculation:
                        # 检查是否应该启用计算
                        if re.search(r'\d+.*[+\-*/].*\d+', query):
                            analysis_result.needs_calculation = True
                            self.logger.info("🔧 根据MCP工具建议启用计算功能")
                    
                    if "database_query" in suggested_tools and not analysis_result.needs_database:
                        # 检查是否应该启用数据库查询
                        db_keywords = ["用户", "数据", "统计", "查询", "表"]
                        if any(keyword in query.lower() for keyword in db_keywords):
                            analysis_result.needs_database = True
                            # 构建简单的数据库查询参数
                            analysis_result.database_query = {
                                "query": "select",
                                "query_type": "structured", 
                                "table_name": "users",
                                "limit": 10
                            }
                            self.logger.info("🔧 根据MCP工具建议启用数据库查询功能")
            
            # 2. 根据分析结果执行相应策略
            search_results = []
            vector_results = []
            calculation_results = []
            database_results = []
            
            # 计算处理（如果需要）
            if analysis_result.needs_calculation:
                self.logger.info("🧮 执行数学计算...")
                
                # 优先尝试使用MCP计算器工具
                mcp_calc_success = False
                if self.mcp_integration:
                    # 尝试解析计算表达式
                    calc_args = analysis_result.calculation_args
                    if calc_args and isinstance(calc_args, dict):
                        operation = calc_args.get("operation")
                        
                        # 支持基本的二元运算
                        if operation in ["add", "subtract", "multiply", "divide"]:
                            # 直接使用x和y参数
                            x = calc_args.get("x")
                            y = calc_args.get("y")
                            
                            if x is not None and y is not None:
                                try:
                                    mcp_result = await self.call_mcp_tool("calculator", {
                                        "operation": operation,
                                        "x": float(x),
                                        "y": float(y)
                                    })
                                    
                                    if mcp_result["success"]:
                                        operation_symbols = {
                                            "add": "+",
                                            "subtract": "-",
                                            "multiply": "*",
                                            "divide": "/"
                                        }
                                        symbol = operation_symbols.get(operation, operation)
                                        calculation_results.append({
                                            "type": "mcp_calculation",
                                            "expression": f"{x} {symbol} {y}",
                                            "result": mcp_result["result"],
                                            "tool": "MCP Calculator",
                                            "execution_time": mcp_result.get("execution_time", 0)
                                        })
                                        mcp_calc_success = True
                                        self.logger.info(f"✅ MCP计算器执行成功: {x} {symbol} {y} = {mcp_result['result']}")
                                    else:
                                        self.logger.warning(f"⚠️ MCP计算器执行失败: {mcp_result.get('error')}")
                                        
                                except Exception as e:
                                    self.logger.error(f"❌ MCP计算器调用异常: {e}")
                        
                        elif operation == "get_current_date":
                            # 获取当前日期
                            try:
                                mcp_result = await self.call_mcp_tool("calculator", {
                                    "operation": "get_current_date"
                                })
                                
                                if mcp_result["success"]:
                                    calculation_results.append({
                                        "type": "mcp_calculation",
                                        "expression": "获取当前日期",
                                        "result": mcp_result["result"],
                                        "tool": "MCP Calculator",
                                        "execution_time": mcp_result.get("execution_time", 0)
                                    })
                                    mcp_calc_success = True
                                    self.logger.info(f"✅ MCP日期查询成功: {mcp_result['result']}")
                                else:
                                    self.logger.warning(f"⚠️ MCP日期查询失败: {mcp_result.get('error')}")
                                    
                            except Exception as e:
                                self.logger.error(f"❌ MCP日期查询异常: {e}")
                        
                        elif operation == "expression":
                            # 表达式计算
                            expression = calc_args.get("expression", "")
                            if expression:
                                try:
                                    mcp_result = await self.call_mcp_tool("calculator", {
                                        "operation": "expression",
                                        "expression": expression
                                    })
                                    
                                    if mcp_result["success"]:
                                        calculation_results.append({
                                            "type": "mcp_calculation",
                                            "expression": expression,
                                            "result": mcp_result["result"],
                                            "tool": "MCP Calculator",
                                            "execution_time": mcp_result.get("execution_time", 0)
                                        })
                                        mcp_calc_success = True
                                        self.logger.info(f"✅ MCP表达式计算成功: {expression} = {mcp_result['result']}")
                                    else:
                                        self.logger.warning(f"⚠️ MCP表达式计算失败: {mcp_result.get('error')}")
                                        
                                except Exception as e:
                                    self.logger.error(f"❌ MCP表达式计算异常: {e}")
    
            
            # 向量搜索（如果需要）
            if analysis_result.needs_vector_search:
                self.logger.info("🔍 执行向量搜索...")
                vector_results = await self._perform_vector_search(query, context.max_results)
                
                # 动态搜索策略：检查向量搜索结果质量
                if analysis_result.enable_dynamic_search and vector_results:
                    max_similarity = max((result.get("similarity_score", 0) for result in vector_results), default=0)
                    self.logger.info(f"📊 向量搜索最高相似度: {max_similarity:.3f}")
                    
                    if max_similarity < analysis_result.min_similarity_threshold:
                        self.logger.warning(f"⚠️ 向量搜索相似度过低 ({max_similarity:.3f} < {analysis_result.min_similarity_threshold})，启用网络搜索")
                        analysis_result.needs_web_search = True
                        analysis_result.web_search_query = query
                        analysis_result.reasoning += f" - 向量搜索相似度过低({max_similarity:.3f})，启用网络搜索"
                elif analysis_result.enable_dynamic_search and not vector_results:
                    self.logger.warning("⚠️ 向量搜索无结果，启用网络搜索")
                    analysis_result.needs_web_search = True
                    analysis_result.web_search_query = query
                    analysis_result.reasoning += " - 向量搜索无结果，启用网络搜索"

            # 网络搜索（如果需要）
            if analysis_result.needs_web_search:
                self.logger.info(f"🌐 执行网络搜索: {analysis_result.web_search_query}")
                search_context = QueryContext(
                    query=analysis_result.web_search_query,
                    query_type=context.query_type,
                    max_results=context.max_results,
                    timeout=context.timeout
                )
                search_results = await self._perform_search(search_context)
                
                # 使用增强文本处理器存储搜索结果到向量数据库
                if search_results:
                    success = await self._store_search_results_to_vector(search_results)
                    if success:
                        self.logger.info(f"💾 使用增强处理器成功存储了 {len(search_results)} 个搜索结果")
                    else:
                        self.logger.warning("⚠️ 增强存储失败，尝试使用备用方法")
                        await self._store_search_results(search_results)
            
            # 数据库查询（如果需要）
            if analysis_result.needs_database:
                self.logger.info("🗄️ 执行数据库查询...")
                
                # 优先尝试使用MCP数据库工具
                mcp_db_success = False
                if self.mcp_integration and analysis_result.database_query:
                    try:
                        db_query = analysis_result.database_query
                        mcp_db_args = {
                            "query": db_query.get("query", ""),
                            "query_type": db_query.get("query_type", "structured"),
                            "database": db_query.get("database", "default")
                        }
                        
                        # 添加可选参数
                        optional_params = ["table_name", "fields", "where_conditions", 
                                         "order_by", "limit", "offset", "group_by", "having"]
                        for param in optional_params:
                            if param in db_query:
                                mcp_db_args[param] = db_query[param]
                        
                        mcp_result = await self.call_mcp_tool("database_query", mcp_db_args)
                        
                        if mcp_result["success"]:
                            database_results.append({
                                "type": "mcp_database",
                                "query": mcp_db_args,
                                "result": mcp_result["result"],
                                "source": "MCP数据库工具",
                                "timestamp": time.time(),
                                "execution_time": mcp_result.get("execution_time", 0)
                            })
                            mcp_db_success = True
                            self.logger.info("✅ MCP数据库查询执行成功")
                        else:
                            self.logger.warning(f"⚠️ MCP数据库查询失败: {mcp_result.get('error')}")
                            
                    except Exception as e:
                        self.logger.error(f"❌ MCP数据库查询异常: {e}")
                
                # 如果MCP数据库查询失败，使用内置方法
                if not mcp_db_success:
                    self.logger.info("🔄 使用内置数据库查询方法...")
                    database_results = await self._perform_database_query(analysis_result.database_query)
            
            # 3. 融合所有结果
            all_results = self._merge_all_results(
                search_results, vector_results, calculation_results, database_results
            )
            
            # 4. 生成智能答案
            answer, confidence = await self._generate_smart_answer(
                query, all_results, analysis_result
            )
            
            # 5. 构建增强响应
            processing_time = time.time() - start_time
            response = RAGResponse(
                answer=answer,
                sources=self._extract_sources(all_results),
                search_results=search_results,
                processing_time=processing_time,
                confidence_score=confidence,
                analysis_result=analysis_result,  # 新增分析结果
                metadata={
                    "query_type": context.query_type.value,
                    "analysis_type": analysis_result.query_type,
                    "analysis_confidence": analysis_result.confidence,
                    "total_results": len(all_results),
                    "search_results_count": len(search_results),
                    "vector_results_count": len(vector_results),
                    "calculation_results_count": len(calculation_results),
                    "database_results_count": len(database_results),
                    "used_search_engine": analysis_result.needs_web_search,
                    "used_vector_search": analysis_result.needs_vector_search,
                    "used_calculation": analysis_result.needs_calculation,
                    "used_database": analysis_result.needs_database,
                    "analysis_reasoning": analysis_result.reasoning,
                    "tools_used": [tool.name for tool in analysis_result.tool_calls],
                    "strategy": self.smart_analyzer.get_search_strategy(analysis_result)
                }
            )
            
            self.logger.info(f"✅ 智能查询处理完成，耗时: {processing_time:.2f}s, "
                           f"策略: {response.metadata['strategy']}")
            return response
            
        except Exception as e:
            self.logger.error(f"智能查询处理失败: {e}")
            return RAGResponse(
                answer=f"抱歉，处理您的查询时出现错误: {str(e)}",
                sources=[],
                search_results=[],
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={"error": str(e), "fallback": True}
            )
    
    async def _perform_search(self, context: QueryContext) -> List[SearchResult]:
        """执行实时搜索"""
        try:
            return await self.mcp_processor.process_query(context)
        except Exception as e:
            self.logger.error(f"实时搜索失败: {e}")
            return []
    
    async def _perform_vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        try:
            all_results = await self.vector_store_manager.search_all_stores(query, limit)
            
            # 合并所有存储的结果
            merged_results = []
            for store_name, results in all_results.items():
                for result in results:
                    result["store_name"] = store_name
                    merged_results.append(result)
            
            # 按相似度排序
            merged_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return merged_results[:limit]
            
        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            return []
    
    async def _store_search_results(self, search_results: List[SearchResult]):
        """存储搜索结果到向量数据库"""
        try:
            dynamic_store = self.vector_store_manager.get_store("dynamic")
            if dynamic_store:
                stored_count = await dynamic_store.store_search_results(search_results)
                self.logger.info(f"存储了 {stored_count} 个搜索结果")
        except Exception as e:
            self.logger.error(f"存储搜索结果失败: {e}")
    
    def _merge_results(self, 
                      search_results: List[SearchResult], 
                      vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """融合搜索结果和向量结果"""
        merged = []
        
        # 添加实时搜索结果（优先级更高）
        for result in search_results:
            merged.append({
                "content": result.content,
                "title": result.title,
                "url": result.url,
                "source": result.source,
                "score": result.relevance_score + 0.1,  # 给新搜索结果加权
                "type": "search",
                "timestamp": result.timestamp
            })
        
        # 添加向量搜索结果
        for result in vector_results:
            merged.append({
                "content": result.get("content", ""),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "source": result.get("source", ""),
                "score": result.get("similarity_score", 0),
                "type": "vector",
                "store_name": result.get("store_name", ""),
                "timestamp": result.get("timestamp", 0)
            })
        
        # 去重和排序
        seen_urls = set()
        deduplicated = []
        
        for result in merged:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(result)
            elif not url:  # 没有URL的结果也保留
                deduplicated.append(result)
        
        # 按分数排序
        deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return deduplicated
    
    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取来源信息"""
        sources = []
        for result in results[:10]:  # 最多返回10个来源
            source = {
                "title": result.get("title", "未知标题"),
                "url": result.get("url", ""),
                "source": result.get("source", "未知来源"),
                "score": result.get("score", 0),
                "type": result.get("type", "unknown")
            }
            sources.append(source)
        
        return sources
    
    def update_smart_search_config(self, 
                                  similarity_threshold: float = None,
                                  min_vector_results: int = None,
                                  enable_smart_search: bool = None):
        """更新智能搜索配置"""
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            self.logger.info(f"更新相似度阈值: {similarity_threshold}")
        
        if min_vector_results is not None:
            self.min_vector_results = min_vector_results
            self.logger.info(f"更新最少向量结果数量: {min_vector_results}")
        
        if enable_smart_search is not None:
            self.enable_smart_search = enable_smart_search
            self.logger.info(f"智能搜索开关: {enable_smart_search}")
    
    async def _perform_database_query(self, query_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行数据库查询（模拟实现）"""
        try:
            # 这里应该集成实际的数据库查询功能
            # 目前提供模拟数据
            query_type = query_args.get("query_type", "select")
            
            if query_type == "count":
                return [{
                    "type": "database_result",
                    "query": f"统计查询: {query_args}",
                    "result": "活跃用户: 1250, 非活跃用户: 350",
                    "source": "用户数据库",
                    "timestamp": time.time()
                }]
            
            elif query_type == "select":
                return [{
                    "type": "database_result",
                    "query": f"查询: {query_args}",
                    "result": "返回了5条用户记录",
                    "source": "用户数据库",
                    "timestamp": time.time()
                }]
            
            return []
            
        except Exception as e:
            self.logger.error(f"数据库查询失败: {e}")
            return []


# 使用示例和测试
    def _merge_all_results(self, 
                          search_results: List[SearchResult], 
                          vector_results: List[Dict[str, Any]],
                          calculation_results: List[Dict[str, Any]],
                          database_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """融合所有类型的结果"""
        all_results = []
        
        # 搜索结果
        for result in search_results:
            all_results.append({
                "type": "search",
                "title": result.title,
                "content": result.content,
                "url": result.url,
                "source": result.source,
                "timestamp": result.timestamp,
                "relevance_score": result.relevance_score,
                "channel_type": str(result.channel_type)
            })
        
        # 向量搜索结果
        for result in vector_results:
            result["type"] = "vector"
            all_results.append(result)
        
        # 计算结果
        for result in calculation_results:
            result["type"] = "calculation"
            result["source"] = "计算器"
            result["timestamp"] = time.time()
            all_results.append(result)
        
        # 数据库结果
        for result in database_results:
            all_results.append(result)
        
        # 按相关性排序
        all_results.sort(key=lambda x: x.get("relevance_score", x.get("similarity_score", 0.5)), reverse=True)
        
        return all_results


# 使用示例和测试
    async def _generate_smart_answer(self, 
                                    query: str, 
                                    results: List[Dict[str, Any]],
                                    analysis: QueryAnalysisResult) -> Tuple[str, float]:
        """生成智能答案 - 基于查询分析结果"""
        
        if not results:
            return self._generate_fallback_answer(query, analysis), 0.3
        
        # 构建上下文
        context_parts = []
        confidence_factors = []
        
        for result in results[:8]:  # 限制上下文长度
            result_type = result.get("type", "unknown")
            
            if result_type == "calculation":
                if "result" in result:
                    context_parts.append(f"[计算结果] {result.get('expression', '')} = {result['result']}")
                    confidence_factors.append(0.9)  # 计算结果置信度高
                elif "error" in result:
                    context_parts.append(f"[计算错误] {result['error']}")
                    confidence_factors.append(0.2)
            
            elif result_type == "database_result":
                context_parts.append(f"[数据库] {result.get('result', '')}")
                confidence_factors.append(0.8)
            
            elif result_type == "search":
                content = result.get("content", "")[:300]  # 限制长度
                source = result.get("source", "")
                context_parts.append(f"[搜索] {content}\n来源: {source}")
                confidence_factors.append(result.get("relevance_score", 0.5))
            
            elif result_type == "vector":
                content = result.get("content", "")[:300]
                source = result.get("source", "知识库")
                similarity = result.get("similarity_score", 0.5)
                context_parts.append(f"[知识库] {content}\n来源: {source} (相似度: {similarity:.2f})")
                confidence_factors.append(similarity)
        
        context = "\n\n".join(context_parts)
        
        # 根据分析类型构建提示词
        prompt = self._build_answer_prompt(query, context, analysis)
        
        try:
            # 调用LLM生成答案
            if hasattr(self, 'llm_client') and self.llm_client:
                # 使用真实的LLM客户端 - 直接使用构建好的prompt
                answer = get_llm_answer_with_prompt(
                    client=self.llm_client,
                    prompt=prompt  # 使用构建好的prompt
                )
            else:
                # LLM不可用时，使用智能回退答案生成
                self.logger.warning("⚠️ LLM客户端不可用，使用智能回退模式")
                answer = self._synthesize_intelligent_answer(query, context, analysis)
            
            # 计算置信度
            avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            final_confidence = min(0.95, avg_confidence * analysis.confidence)
            
            # 格式化答案
            formatted_answer = self._format_answer(answer)
            
            return formatted_answer, final_confidence
            
        except Exception as e:
            self.logger.error(f"LLM答案生成失败: {e}")
            return self._synthesize_answer_fallback(query, results), 0.4
    
    def _build_answer_prompt(self, query: str, context: str, analysis: QueryAnalysisResult) -> str:
        """根据查询分析结果构建答案提示词"""
        
        if analysis.query_type == "time":
            return f"""用户询问时间相关问题：{query}

以下是获取的最新信息：
{context}

请基于这些最新信息准确回答用户的时间相关问题。如果信息中包含具体的时间数据，请直接提供。"""
        
        elif analysis.query_type == "calculation":
            return f"""用户询问数学计算问题：{query}

计算结果：
{context}

请基于计算结果为用户提供清晰的数学答案，并简要说明计算过程。"""
        
        elif analysis.query_type == "technical":
            return f"""用户询问技术问题：{query}

相关技术信息：
{context}

请基于这些技术资料提供详细、准确的技术解答。可以包含技术细节和实现方法。"""
        
        else:
            return f"""用户问题：{query}

相关信息：
{context}

请综合以上信息，为用户提供准确、有用的回答。如果信息来源于不同渠道，请适当整合。"""
    
    def _synthesize_answer_fallback(self, query: str, results: List[Dict[str, Any]]) -> str:
        """合成备用答案"""
        if not results:
            return f"关于您的问题「{query}」，我没有找到相关信息。"
        
        answer_parts = [f"关于您的问题「{query}」，我找到了以下信息：\n"]
        
        for i, result in enumerate(results[:3], 1):
            result_type = result.get("type", "unknown")
            if result_type == "calculation" and "result" in result:
                answer_parts.append(f"{i}. 计算结果：{result.get('expression', '')} = {result['result']}")
            elif result_type == "database_result":
                answer_parts.append(f"{i}. 数据库查询：{result.get('result', '')}")
            else:
                content = result.get("content", "")[:200]
                source = result.get("source", "")
                answer_parts.append(f"{i}. {content} (来源：{source})")
        
        answer_parts.append("\n以上信息供您参考。")
        return "\n\n".join(answer_parts)
    
    def _format_answer(self, answer: str) -> str:
        """格式化答案输出，美化Markdown内容"""
        if not answer:
            return answer
        
        try:
            # 移除转义字符，恢复正常的换行符
            formatted = answer.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            
            # 处理代码块格式
            
            # 确保代码块有正确的换行
            formatted = re.sub(r'```(\w*)\n?', r'```\1\n', formatted)
            formatted = re.sub(r'\n?```', r'\n```', formatted)
            
            # 处理标题格式，确保标题前后有适当的空行
            formatted = re.sub(r'\n(#{1,6}\s+[^\n]+)', r'\n\n\1', formatted)
            formatted = re.sub(r'(#{1,6}\s+[^\n]+)\n([^#\n])', r'\1\n\n\2', formatted)
            
            # 处理列表项，确保格式正确
            formatted = re.sub(r'\n-\s+', r'\n- ', formatted)
            formatted = re.sub(r'\n\*\s+', r'\n* ', formatted)
            formatted = re.sub(r'\n(\d+\.)\s+', r'\n\1 ', formatted)
            
            # 处理段落间距，确保段落之间有适当的空行
            formatted = re.sub(r'\n\n\n+', r'\n\n', formatted)
            
            # 处理粗体和斜体格式
            formatted = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', formatted)
            formatted = re.sub(r'\*([^*]+)\*', r'*\1*', formatted)
            
            # 清理开头和结尾的多余空行
            formatted = formatted.strip()
            
            # 确保内容有良好的结构
            if formatted:
                # 如果内容很长，添加一个简洁的开头
                if len(formatted) > 1000 and not formatted.startswith('## ') and not formatted.startswith('# '):
                    lines = formatted.split('\n')
                    if len(lines) > 3:
                        # 检查是否需要添加概述
                        first_line = lines[0].strip()
                        if len(first_line) > 50 and '##' not in first_line:
                            formatted = f"**概述：** {first_line}\n\n{formatted}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"格式化答案时出错: {e}")
            # 如果格式化失败，至少处理基本的转义字符
            return answer.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
    
    def _generate_fallback_answer(self, query: str, analysis: QueryAnalysisResult) -> str:
        """生成回退答案 - 当没有搜索结果时使用"""
        fallback_messages = {
            "time": f"抱歉，我无法获取当前的时间信息来回答「{query}」。请您查看系统时间或联网获取最新时间。",
            "calculation": f"抱歉，我无法计算「{query}」。请检查计算表达式是否正确，或使用计算器工具。",
            "technical": f"关于「{query}」这个技术问题，我暂时没有找到相关资料。建议您：\n\n1. 查阅官方文档\n2. 搜索技术论坛\n3. 咨询技术专家",
            "database": f"抱歉，我无法查询到「{query}」相关的数据库信息。请检查查询条件或联系管理员。",
            "general": f"关于您的问题「{query}」，我暂时没有找到相关信息。建议您：\n\n1. 尝试重新描述问题\n2. 使用更具体的关键词\n3. 或者联网搜索获取最新信息"
        }
        
        query_type = analysis.query_type if analysis else "general"
        message = fallback_messages.get(query_type, fallback_messages["general"])
        
        # 添加分析推理信息（如果有的话）
        if analysis and analysis.reasoning:
            message += f"\n\n**分析说明：** {analysis.reasoning}"
        
        return message
    
    def _synthesize_intelligent_answer(self, query: str, context: str, analysis: QueryAnalysisResult) -> str:
        """智能合成答案 - 当LLM不可用时的智能回退"""
        try:
            # 根据查询类型和上下文智能生成答案
            if analysis.query_type == "calculation":
                # 查找计算结果
                calc_pattern = r'\[计算结果\]\s*([^=]+)=\s*([^\n]+)'
                calc_match = re.search(calc_pattern, context)
                if calc_match:
                    expression = calc_match.group(1).strip()
                    result = calc_match.group(2).strip()
                    return f"根据计算，{expression} = **{result}**"
                else:
                    return f"关于计算问题「{query}」，我找到了相关信息：\n\n{context[:300]}..."
            
            elif analysis.query_type == "time":
                # 查找时间信息
                time_keywords = ["时间", "日期", "今天", "现在", "当前"]
                if any(keyword in context for keyword in time_keywords):
                    # 提取时间相关信息
                    lines = context.split('\n')
                    time_info = []
                    for line in lines:
                        if any(keyword in line for keyword in time_keywords):
                            time_info.append(line.strip())
                    
                    if time_info:
                        return f"根据最新信息：\n\n" + "\n".join(time_info[:3])
                
                return f"关于时间问题「{query}」，我找到了以下信息：\n\n{context[:300]}..."
            
            elif analysis.query_type == "database":
                # 查找数据库结果
                if "[数据库]" in context:
                    db_info = []
                    lines = context.split('\n')
                    for line in lines:
                        if "[数据库]" in line:
                            db_info.append(line.replace("[数据库]", "").strip())
                    
                    if db_info:
                        return f"数据库查询结果：\n\n" + "\n".join(db_info)
                
                return f"关于数据查询「{query}」，我找到了以下信息：\n\n{context[:300]}..."
            
            else:
                # 通用智能合成
                # 提取关键信息片段
                segments = context.split('\n\n')
                key_segments = []
                
                for segment in segments[:5]:  # 最多处理5个段落
                    segment = segment.strip()
                    if len(segment) > 50:  # 过滤太短的片段
                        # 移除标记符号
                        clean_segment = re.sub(r'\[.*?\]\s*', '', segment)
                        key_segments.append(clean_segment[:200])  # 限制长度
                
                if key_segments:
                    return f"关于「{query}」，我整理了以下相关信息：\n\n" + "\n\n".join(f"• {seg}" for seg in key_segments[:3])
                else:
                    return f"关于「{query}」，我找到了一些相关信息，但可能需要进一步查证：\n\n{context[:400]}..."
        
        except Exception as e:
            self.logger.error(f"智能答案合成失败: {e}")
            # 最基本的回退
            return f"关于「{query}」，我找到了一些信息：\n\n{context[:300]}...\n\n以上信息供您参考。"
    
    def _beautify_technical_content(self, content: str) -> str:
        """专门美化技术内容"""
        if not content:
            return content
            
        try:
            
            # 添加技术文档的标准格式
            formatted = content
            
            # 为代码示例添加语法高亮提示
            formatted = re.sub(r'```\n(public class|import|package)', r'```java\n\1', formatted)
            formatted = re.sub(r'```\n(def |import |from |class )', r'```python\n\1', formatted)
            formatted = re.sub(r'```\n(<\?xml|<html|<div)', r'```xml\n\1', formatted)
            
            # 美化技术术语
            tech_terms = {
                'JDK': '**JDK (Java开发工具包)**',
                'JRE': '**JRE (Java运行时环境)**', 
                'JVM': '**JVM (Java虚拟机)**',
                'API': '**API**',
                'IDE': '**IDE**',
                'SDK': '**SDK**'
            }
            
            for term, formatted_term in tech_terms.items():
                # 只替换单独出现的术语，避免替换URL或代码中的内容
                formatted = re.sub(rf'\b{term}\b(?![/:])', formatted_term, formatted)
            
            # 添加技术要点的视觉分隔
            if '## 主要特性' in formatted or '## 基本特性' in formatted:
                formatted = f"📋 **技术文档**\n\n{formatted}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"美化技术内容时出错: {e}")
            return content
