#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的RAG处理器

集成多通道搜索、向量存储和LLM生成，提供完整的RAG解决方案。
"""

import asyncio
import logging
import time
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# 添加上级目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MCP框架
from channel_framework import MCPProcessor, QueryContext, QueryAnalyzer, QueryType, SearchResult

# 导入core模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from search_channels import GoogleSearchChannel
from dynamic_vector_store import DynamicVectorStore, VectorStoreManager

# 导入MCP目录下的智能查询分析器
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp'))
from smart_query_analyzer import SmartQueryAnalyzer, QueryAnalysisResult, SimpleCalculator

try:
    from ask_llm import get_llm_answer_deepseek
    from ..core.encoder import emb_text
    from ..core.milvus_utils import get_milvus_client
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    # 定义mock函数，匹配真实函数签名
    def get_llm_answer_deepseek(client, context: str, question: str, model: str = "deepseek-v3-0324", min_distance_threshold: float = 0.5) -> str:
        return f"模拟LLM响应 - 问题: {question}"
    
    def emb_text(text: str):
        # 返回模拟向量
        import random
        return [random.random() for _ in range(384)]
    
    def get_milvus_client():
        return None


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
        
        # 初始化组件
        self.mcp_processor = MCPProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.query_analyzer = QueryAnalyzer()
        
        # 新增智能查询分析器
        self.smart_analyzer = SmartQueryAnalyzer(self.config)
        self.calculator = SimpleCalculator()
        
        # 智能查询策略配置
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)  # 相似度阈值
        self.min_vector_results = self.config.get("min_vector_results", 3)  # 最少向量结果数量
        self.enable_smart_search = self.config.get("enable_smart_search", True)  # 启用智能搜索
        
        # 输出配置信息用于调试
        self.logger.info(f"📊 智能搜索配置: similarity_threshold={self.similarity_threshold}, "
                        f"min_vector_results={self.min_vector_results}, "
                        f"enable_smart_search={self.enable_smart_search}")
        
        # 初始化配置
        self._init_components()
    
    def _init_components(self):
        """初始化各个组件"""
        try:
            # 1. 初始化向量存储
            self._init_vector_stores()
            
            # 2. 初始化搜索通道
            self._init_search_channels()
            
            self.logger.info("增强RAG处理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def _init_vector_stores(self):
        """初始化向量存储"""
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
        
        # 动态向量存储（用于实时搜索结果）
        dynamic_store = DynamicVectorStore(
            milvus_endpoint=milvus_endpoint,
            milvus_token=milvus_token,
            collection_name="dynamic_search_results",
            vector_dim=vector_dim
        )
        self.vector_store_manager.add_store("dynamic", dynamic_store)
        
        # 本地知识库存储
        local_store = DynamicVectorStore(
            milvus_endpoint=milvus_endpoint,
            milvus_token=milvus_token,
            collection_name="local_knowledge",
            vector_dim=vector_dim
        )
        self.vector_store_manager.add_store("local", local_store)
    
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
            
            # 2. 根据分析结果执行相应策略
            search_results = []
            vector_results = []
            calculation_results = []
            database_results = []
            
            # 计算处理（如果需要）
            if analysis_result.needs_calculation:
                self.logger.info("🧮 执行数学计算...")
                calc_result = self.calculator.calculate(analysis_result.calculation_args)
                calculation_results.append(calc_result)
            
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
                
                # 存储新的搜索结果到向量数据库
                if search_results:
                    await self._store_search_results(search_results)
                    self.logger.info(f"💾 存储了 {len(search_results)} 个新的搜索结果")
            
            # 数据库查询（如果需要）
            if analysis_result.needs_database:
                self.logger.info("🗄️ 执行数据库查询...")
                # 这里可以集成实际的数据库查询功能
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
    
    def _should_perform_search(self, vector_results: List[Dict[str, Any]], context: QueryContext) -> Tuple[bool, str]:
        """
        判断是否需要执行搜索引擎查询
        
        Returns:
            Tuple[bool, str]: (是否需要搜索, 原因说明)
        """
        if not self.enable_smart_search:
            return True, "智能搜索已禁用"
        
        if not vector_results:
            return True, "向量数据库中没有相关结果"
        
        # 检查结果数量
        if len(vector_results) < self.min_vector_results:
            return True, f"向量结果数量不足 ({len(vector_results)} < {self.min_vector_results})"
        
        # 检查最高相似度
        max_similarity = max(result.get("similarity_score", 0) for result in vector_results)
        if max_similarity < self.similarity_threshold:
            return True, f"最高相似度不足 ({max_similarity:.3f} < {self.similarity_threshold})"
        
        # 检查高质量结果数量
        high_quality_results = [
            r for r in vector_results 
            if r.get("similarity_score", 0) >= self.similarity_threshold
        ]
        
        if len(high_quality_results) < 2:
            return True, f"高质量结果数量不足 ({len(high_quality_results)} < 2)"
        
        # 检查内容新鲜度（可选）
        current_time = time.time()
        recent_results = [
            r for r in high_quality_results
            if current_time - r.get("timestamp", 0) < 7 * 24 * 3600  # 7天内
        ]
        
        if len(recent_results) == 0:
            return True, "没有足够新鲜的高质量结果"
        
        # 特殊查询类型处理
        if context.query_type == QueryType.CREATIVE:
            return True, "创造性查询需要实时搜索"
        
        # 检查查询中是否包含时间相关词汇
        time_keywords = ["今天", "最新", "现在", "当前", "最近", "今年", "2024", "2025"]
        if any(keyword in context.query for keyword in time_keywords):
            return True, "查询包含时间相关词汇，需要最新信息"
        
        return False, f"向量数据库有足够的高质量结果 ({len(high_quality_results)} 个，最高相似度: {max_similarity:.3f})"
    
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
    
    async def _generate_answer(self, 
                              query: str, 
                              results: List[Dict[str, Any]]) -> Tuple[str, float]:
        """生成答案"""
        try:
            if not results:
                return "抱歉，没有找到相关信息来回答您的问题。", 0.0
            
            # 构建上下文
            context_parts = []
            for i, result in enumerate(results[:5]):  # 使用前5个最相关的结果
                content = result.get("content", "").strip()
                if content:
                    source_info = f"来源: {result.get('title', '未知')} ({result.get('source', '未知')})"
                    context_parts.append(f"参考资料 {i+1}:\n{content}\n{source_info}")
            
            context = "\n\n".join(context_parts)
            
            # 调用DeepSeek LLM生成答案
            try:
                # 使用传入的LLM客户端
                if hasattr(self, 'llm_client') and self.llm_client:
                    # 构建消息格式
                    messages = []
                    
                    if context:
                        system_prompt = """
你是一个智能助手。请基于提供的上下文信息回答用户问题。
如果上下文信息充分，请优先使用上下文中的信息回答。
如果上下文信息不够充分，可以结合你的知识给出有帮助的回答。
请确保回答准确、有条理，并尽可能提供具体的信息。
请提供完整详细的回答，不要截断内容。
"""
                        messages.append({"role": "system", "content": system_prompt})
                        
                        user_content = f"""
基于以下上下文信息回答问题：

{context}

问题：{query}
"""
                    else:
                        user_content = query
                    
                    messages.append({"role": "user", "content": user_content})
                    
                    # 调用DeepSeek API - 添加重要参数
                    response = self.llm_client.chat_completions_create(
                        model="deepseek-v3-0324",
                        messages=messages,
                        stream=False,
                        enable_search=True,  # 启用DeepSeek的搜索功能
                        temperature=0.7,     # 设置创造性参数
                        top_p=0.9,          # 设置核采样参数
                        frequency_penalty=0.0,  # 频率惩罚
                        presence_penalty=0.0    # 存在惩罚
                    )
                    
                    if response and "choices" in response:
                        answer = response["choices"][0]["message"]["content"]
                    else:
                        answer = "抱歉，无法生成回答"
                        
                else:
                    # 降级到简单的上下文拼接
                    if context:
                        answer = f"基于搜索结果：\n\n{context}\n\n回答：请参考以上信息来回答关于'{query}'的问题。"
                    else:
                        answer = f"抱歉，没有找到关于'{query}'的相关信息。"
                        
            except Exception as e:
                self.logger.error(f"LLM调用失败: {e}")
                # 降级方案
                if context:
                    answer = f"基于搜索到的信息：\n\n{context}"
                else:
                    answer = f"抱歉，处理您的问题时出现了错误：{str(e)}"
            
            # 计算置信度（基于结果数量和相关性）
            confidence = min(1.0, len(results) * 0.1 + sum(r.get("score", 0) for r in results[:3]) / 3)
            
            return answer, confidence
            
        except Exception as e:
            self.logger.error(f"生成答案失败: {e}")
            return f"生成答案时出现错误: {str(e)}", 0.0
    
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
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """清理过期数据"""
        try:
            dynamic_store = self.vector_store_manager.get_store("dynamic")
            if dynamic_store:
                cleaned_count = await dynamic_store.cleanup_old_documents(max_age_hours)
                self.logger.info(f"清理了 {cleaned_count} 个过期文档")
                return cleaned_count
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")
        return 0
    
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
    
    def get_smart_search_stats(self) -> Dict[str, Any]:
        """获取智能搜索统计信息"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "min_vector_results": self.min_vector_results,
            "enable_smart_search": self.enable_smart_search,
            "vector_stores": list(self.vector_store_manager.stores.keys())
        }
    
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
                # 使用真实的LLM客户端
                answer = get_llm_answer_deepseek(
                    client=self.llm_client,
                    context=context,
                    question=query
                )
            else:
                # 使用mock函数
                answer = f"基于分析结果回答：{prompt[:200]}..."
            
            # 计算置信度
            avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            final_confidence = min(0.95, avg_confidence * analysis.confidence)
            
            return answer, final_confidence
            
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
    
    def _generate_fallback_answer(self, query: str, analysis: QueryAnalysisResult) -> str:
        """生成备用答案"""
        if analysis.query_type == "time":
            return "抱歉，我无法获取当前的时间信息。请检查网络连接或稍后再试。"
        elif analysis.query_type == "calculation":
            return f"抱歉，无法完成计算：{query}。请检查表达式是否正确。"
        elif analysis.query_type == "technical":
            return f"关于您询问的技术问题「{query}」，我暂时没有找到相关信息。建议您查阅官方文档或技术论坛。"
        else:
            return f"抱歉，我暂时无法回答您的问题「{query}」。请尝试重新表述或提供更多上下文。"
    
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
