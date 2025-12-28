"""
RAG Domain Service - RAG核心领域服务

实现RAG的核心业务逻辑，协调各个领域组件
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from ..entities.document import Document, DocumentChunk
from ..entities.query import Query, QueryResult, RetrievalContext, QueryConfig
from ..ports.services import (
    EmbeddingService,
    LLMService,
    VectorStoreService,
    SearchService,
    ChunkingService,
    RerankService,
)


logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG服务配置"""
    similarity_threshold: float = 0.5
    enable_rerank: bool = True
    enable_web_search: bool = True
    enable_query_rewrite: bool = True
    max_context_tokens: int = 3000
    fallback_to_web: bool = True


class RAGDomainService:
    """
    RAG领域服务
    
    实现检索增强生成的核心业务逻辑
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: VectorStoreService,
        search_service: Optional[SearchService] = None,
        chunking_service: Optional[ChunkingService] = None,
        rerank_service: Optional[RerankService] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.search_service = search_service
        self.chunking_service = chunking_service
        self.rerank_service = rerank_service
        self.config = config or RAGConfig()
        
        logger.info("RAG领域服务初始化完成")
    
    async def process_query(self, query: Query) -> QueryResult:
        """
        处理用户查询
        
        完整的RAG流程:
        1. 查询理解和改写
        2. 向量检索
        3. 可选的网络搜索
        4. 重排序
        5. 答案生成
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 查询分析和改写
            processed_query = await self._preprocess_query(query)
            
            # 2. 生成查询向量
            query_embedding = await self.embedding_service.embed_query(
                processed_query.processed_text
            )
            processed_query = processed_query.with_embedding(query_embedding.vector)
            
            # 3. 向量检索
            contexts = await self._retrieve_contexts(processed_query)
            
            # 4. 可选的网络搜索补充
            if self._should_search_web(query, contexts):
                web_contexts = await self._search_web(processed_query)
                contexts.extend(web_contexts)
            
            # 5. 重排序
            if self.rerank_service and self.config.enable_rerank and contexts:
                contexts = await self._rerank_contexts(processed_query, contexts)
            
            # 6. 生成答案
            answer, confidence = await self._generate_answer(processed_query, contexts)
            
            # 7. 构建结果
            processing_time = (time.time() - start_time) * 1000
            
            result = QueryResult(
                query_id=query.id,
                answer=answer,
                confidence=confidence,
                contexts=contexts,
                model_used=self.llm_service.get_model_name() if hasattr(self.llm_service, 'get_model_name') else "",
                processing_time_ms=processing_time,
                sources=self._extract_sources(contexts),
            )
            
            logger.info(
                f"查询处理完成 | query_id={query.id} | "
                f"contexts={len(contexts)} | time={processing_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}", exc_info=True)
            raise
    
    async def _preprocess_query(self, query: Query) -> Query:
        """预处理查询"""
        processed_text = query.original_text
        
        # 查询改写（如果启用）
        if self.config.enable_query_rewrite:
            try:
                processed_text = await self.llm_service.rewrite_query(
                    query.original_text,
                    context="\n".join(query.conversation_history[-3:]) if query.conversation_history else None
                )
                logger.debug(f"查询改写: {query.original_text} -> {processed_text}")
            except Exception as e:
                logger.warning(f"查询改写失败，使用原始查询: {e}")
        
        # 创建处理后的查询对象
        return Query(
            id=query.id,
            original_text=query.original_text,
            processed_text=processed_text,
            intent=query.intent,
            keywords=query.keywords,
            entities=query.entities,
            language=query.language,
            config=query.config,
            user_id=query.user_id,
            session_id=query.session_id,
            conversation_history=query.conversation_history,
            created_at=query.created_at,
        )
    
    async def _retrieve_contexts(self, query: Query) -> List[RetrievalContext]:
        """从向量存储检索上下文"""
        if not query.has_embedding:
            logger.warning("查询没有向量，跳过向量检索")
            return []
        
        try:
            results = await self.vector_store.search(
                query_vector=query.embedding,
                top_k=query.config.top_k,
                filter=None,  # 可以添加过滤条件
            )
            
            contexts = []
            for result in results:
                if result.score >= self.config.similarity_threshold:
                    contexts.append(RetrievalContext(
                        chunk_id=UUID(result.id) if self._is_valid_uuid(result.id) else UUID(int=0),
                        document_id=UUID(int=0),  # 从metadata获取
                        content=result.content,
                        score=result.score,
                        metadata=result.metadata,
                        source=result.metadata.get("source", "vector_store"),
                        source_url=result.metadata.get("url"),
                        title=result.metadata.get("title"),
                    ))
            
            logger.info(f"向量检索完成: {len(results)} 结果, {len(contexts)} 通过阈值")
            return contexts
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    async def _search_web(self, query: Query) -> List[RetrievalContext]:
        """网络搜索"""
        if not self.search_service:
            return []
        
        try:
            results = await self.search_service.search(
                query.processed_text,
                max_results=5,
            )
            
            contexts = []
            for result in results:
                contexts.append(RetrievalContext(
                    chunk_id=UUID(int=0),
                    document_id=UUID(int=0),
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata or {},
                    source="web_search",
                    source_url=result.url,
                    title=result.title,
                ))
            
            logger.info(f"网络搜索完成: {len(contexts)} 结果")
            return contexts
            
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return []
    
    async def _rerank_contexts(
        self,
        query: Query,
        contexts: List[RetrievalContext]
    ) -> List[RetrievalContext]:
        """重排序上下文"""
        if not contexts:
            return contexts
        
        try:
            documents = [ctx.content for ctx in contexts]
            rerank_results = await self.rerank_service.rerank(
                query.processed_text,
                documents,
                top_n=query.config.rerank_top_n,
            )
            
            # 更新重排分数
            for result in rerank_results:
                if result.index < len(contexts):
                    contexts[result.index].rerank_score = result.score
            
            # 按最终分数排序
            contexts.sort(key=lambda x: x.final_score, reverse=True)
            
            # 只保留top_n
            contexts = contexts[:query.config.rerank_top_n]
            
            logger.info(f"重排序完成: 保留 {len(contexts)} 个上下文")
            return contexts
            
        except Exception as e:
            logger.warning(f"重排序失败，使用原始排序: {e}")
            return contexts
    
    async def _generate_answer(
        self,
        query: Query,
        contexts: List[RetrievalContext]
    ) -> tuple[str, float]:
        """生成答案"""
        # 构建上下文
        context_text = self._build_context_text(contexts, query.config.max_tokens)
        
        # 构建提示词
        prompt = self._build_prompt(query, context_text)
        
        # 调用LLM
        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
            )
            
            answer = response.content
            
            # 计算置信度（基于上下文匹配度）
            confidence = self._calculate_confidence(contexts)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}", 0.0
    
    def _should_search_web(self, query: Query, contexts: List[RetrievalContext]) -> bool:
        """判断是否需要网络搜索"""
        if not self.search_service or not self.config.enable_web_search:
            return False
        
        if not query.config.enable_web_search:
            return False
        
        # 如果向量检索结果不足或质量不高，则进行网络搜索
        if len(contexts) < 2:
            return True
        
        avg_score = sum(ctx.score for ctx in contexts) / len(contexts)
        if avg_score < self.config.similarity_threshold:
            return True
        
        return False
    
    def _build_context_text(self, contexts: List[RetrievalContext], max_tokens: int) -> str:
        """构建上下文文本"""
        if not contexts:
            return ""
        
        parts = []
        total_length = 0
        max_chars = max_tokens * 4  # 粗略估计
        
        for i, ctx in enumerate(contexts):
            ctx_text = ctx.to_prompt_context()
            if total_length + len(ctx_text) > max_chars:
                break
            parts.append(f"[参考{i+1}] {ctx_text}")
            total_length += len(ctx_text)
        
        return "\n\n".join(parts)
    
    def _build_prompt(self, query: Query, context_text: str) -> str:
        """构建提示词"""
        if context_text:
            return f"""请基于以下参考信息回答用户问题。如果参考信息不足以回答问题，请说明并尝试给出合理的回答。

参考信息：
{context_text}

用户问题：{query.original_text}

请给出详细、准确的回答："""
        else:
            return f"""用户问题：{query.original_text}

请给出详细、准确的回答："""
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个智能问答助手。请基于提供的参考信息回答用户问题。
要求：
1. 回答准确、有条理
2. 如果参考信息不足，明确说明
3. 适当引用来源
4. 使用用户的语言回答"""
    
    def _calculate_confidence(self, contexts: List[RetrievalContext]) -> float:
        """计算置信度"""
        if not contexts:
            return 0.3
        
        # 基于上下文分数计算
        scores = [ctx.final_score for ctx in contexts]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # 综合考虑平均分和最高分
        confidence = 0.6 * avg_score + 0.4 * max_score
        return min(1.0, max(0.0, confidence))
    
    def _extract_sources(self, contexts: List[RetrievalContext]) -> List[dict]:
        """提取来源信息"""
        sources = []
        seen_urls = set()
        
        for ctx in contexts:
            url = ctx.source_url
            if url and url not in seen_urls:
                sources.append({
                    "title": ctx.title or "Unknown",
                    "url": url,
                    "source": ctx.source,
                    "score": ctx.final_score,
                })
                seen_urls.add(url)
        
        return sources
    
    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        """检查是否是有效的UUID"""
        try:
            UUID(val)
            return True
        except (ValueError, AttributeError):
            return False
