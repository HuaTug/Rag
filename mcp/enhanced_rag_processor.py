#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的RAG处理器

集成多通道搜索、向量存储和LLM生成，提供完整的RAG解决方案。
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from mcp_framework import MCPProcessor, QueryContext, QueryAnalyzer,QueryType,SearchResult
from search_channels import GoogleSearchChannel
from dynamic_vector_store import DynamicVectorStore, VectorStoreManager
from ask_llm import get_llm_answer_deepseek
from encoder import emb_text
from milvus_utils import get_milvus_client


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    answer: str
    sources: List[Dict[str, Any]]
    search_results: List[SearchResult]
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]


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
        处理查询请求 - 实现智能查询策略
        
        策略：
        1. 首先从向量数据库查找相似内容
        2. 如果找到足够相似且数量充足的内容，直接使用
        3. 否则调用搜索引擎获取新内容
        4. 将新内容存储到向量数据库
        """
        start_time = time.time()
        query = context.query
        
        try:
            # 1. 查询分析
            query_type = context.query_type
            self.logger.info(f"🔍 查询类型: {query_type.value}")
            
            # 2. 智能查询策略：先检查向量数据库
            vector_results = await self._perform_vector_search(query, context.max_results)
            
            # 3. 判断是否需要调用搜索引擎
            need_search, reason = self._should_perform_search(vector_results, context)
            
            search_results = []
            if need_search:
                self.logger.info(f"🌐 需要搜索引擎查询: {reason}")
                search_results = await self._perform_search(context)
                
                # 存储新的搜索结果到向量数据库
                if search_results:
                    await self._store_search_results(search_results)
                    self.logger.info(f"💾 存储了 {len(search_results)} 个新的搜索结果")
            else:
                self.logger.info(f"✅ 使用向量数据库结果: {reason}")
            
            # 4. 融合和排序结果
            all_results = self._merge_results(search_results, vector_results)
            
            # 5. 生成答案
            answer, confidence = await self._generate_answer(query, all_results)
            
            # 6. 构建响应
            processing_time = time.time() - start_time
            response = RAGResponse(
                answer=answer,
                sources=self._extract_sources(all_results),
                search_results=search_results,
                processing_time=processing_time,
                confidence_score=confidence,
                metadata={
                    "query_type": query_type.value,
                    "total_results": len(all_results),
                    "search_results_count": len(search_results),
                    "vector_results_count": len(vector_results),
                    "used_search_engine": need_search,
                    "search_reason": reason,
                    "similarity_threshold": self.similarity_threshold
                }
            )
            
            self.logger.info(f"✅ 查询处理完成，耗时: {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            return RAGResponse(
                answer=f"抱歉，处理您的查询时出现错误: {str(e)}",
                sources=[],
                search_results=[],
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={"error": str(e)}
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


# 使用示例和测试
async def test_smart_rag():
    """测试智能RAG系统"""
    print("🧪 测试智能RAG系统...")
    
    # 这里可以添加测试代码
    pass


if __name__ == "__main__":
    asyncio.run(test_smart_rag())
