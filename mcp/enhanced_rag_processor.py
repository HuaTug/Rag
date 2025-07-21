
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from mcp_framework import MCPProcessor, QueryContext, QueryAnalyzer, SearchResult
from search_channels import GoogleSearchChannel
from dynamic_vector_store import DynamicVectorStore, VectorStoreManager
from ask_llm import get_llm_answer
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
        # 动态向量存储（用于实时搜索结果）
        dynamic_store = DynamicVectorStore(
            milvus_endpoint=self.config.get("milvus_endpoint", "localhost:19530"),
            milvus_token=self.config.get("milvus_token"),
            collection_name="dynamic_search_results",
            vector_dim=384
        )
        self.vector_store_manager.add_store("dynamic", dynamic_store)
        
        # 本地知识库存储
        local_store = DynamicVectorStore(
            milvus_endpoint=self.config.get("milvus_endpoint", "localhost:19530"),
            milvus_token=self.config.get("milvus_token"),
            collection_name="local_knowledge",
            vector_dim=384
        )
        self.vector_store_manager.add_store("local", local_store)
    
    def _init_search_channels(self):
        """初始化搜索通道"""
        # Google搜索通道
        if self.config.get("enable_search_engine", True):
            search_config = {
                "api_key": self.config.get("google_api_key"),
                "search_engine_id": self.config.get("google_search_engine_id"),
                "timeout": self.config.get("search_timeout", 10),
                "priority": {"factual": 1, "analytical": 2, "creative": 5, "conversational": 3}
            }
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
    
    async def process_query(self, 
                           query: str, 
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           max_results: int = 10) -> RAGResponse:
        """处理查询请求"""
        start_time = time.time()
        
        try:
            # 1. 查询分析
            query_type = self.query_analyzer.analyze_query(query)
            self.logger.info(f"查询类型: {query_type.value}")
            
            # 2. 创建查询上下文
            context = QueryContext(
                query=query,
                query_type=query_type,
                user_id=user_id,
                session_id=session_id,
                max_results=max_results,
                timeout=self.config.get("search_timeout", 30.0)
            )
            
            # 3. 并行执行搜索和向量检索
            search_task = asyncio.create_task(self._perform_search(context))
            vector_task = asyncio.create_task(self._perform_vector_search(query, max_results))
            
            search_results, vector_results = await asyncio.gather(search_task, vector_task)
            
            # 4. 存储新的搜索结果到向量数据库
            if search_results:
                await self._store_search_results(search_results)
            
            # 5. 融合和排序结果
            all_results = self._merge_results(search_results, vector_results)
            
            # 6. 生成答案
            answer, confidence = await self._generate_answer(query, all_results)
            
            # 7. 构建响应
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
                    "vector_results_count": len(vector_results)
                }
            )
            
            self.logger.info(f"查询处理完成，耗时: {processing_time:.2f}s")
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
        
        # 添加实时搜索结果
        for result in search_results:
            merged.append({
                "content": result.content,
                "title": result.title,
                "url": result.url,
                "source": result.source,
                "score": result.relevance_score,
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
                    
                    # 调用DeepSeek API
                    response = self.llm_client.chat_completions_create(
                        model="deepseek-v3-0324",
                        messages=messages,
                        stream=False,
                        enable_search=True  # 启用DeepSeek的搜索功能
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