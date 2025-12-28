"""
Retrieval Domain Service - 检索领域服务

实现高级检索策略
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ..entities.query import Query, RetrievalContext, RetrievalStrategy
from ..ports.services import (
    EmbeddingService,
    VectorStoreService,
    RerankService,
    LLMService,
)


logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 10
    similarity_threshold: float = 0.5
    enable_rerank: bool = True
    rerank_top_n: int = 5
    enable_query_expansion: bool = True


class RetrievalDomainService:
    """
    检索领域服务
    
    实现多种检索策略：
    1. 密集检索 (Dense Retrieval)
    2. 稀疏检索 (Sparse Retrieval - BM25)
    3. 混合检索 (Hybrid Retrieval)
    4. 多查询检索 (Multi-Query Retrieval)
    5. HyDE (Hypothetical Document Embeddings)
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        llm_service: Optional[LLMService] = None,
        rerank_service: Optional[RerankService] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.rerank_service = rerank_service
        self.config = config or RetrievalConfig()
        
        logger.info(f"检索服务初始化完成，策略: {self.config.strategy.value}")
    
    async def retrieve(self, query: Query) -> List[RetrievalContext]:
        """
        执行检索
        
        根据配置的策略选择检索方法
        """
        strategy = query.config.retrieval_strategy or self.config.strategy
        
        if strategy == RetrievalStrategy.DENSE:
            contexts = await self._dense_retrieval(query)
        elif strategy == RetrievalStrategy.SPARSE:
            contexts = await self._sparse_retrieval(query)
        elif strategy == RetrievalStrategy.HYBRID:
            contexts = await self._hybrid_retrieval(query)
        elif strategy == RetrievalStrategy.MULTI_QUERY:
            contexts = await self._multi_query_retrieval(query)
        elif strategy == RetrievalStrategy.HYDE:
            contexts = await self._hyde_retrieval(query)
        else:
            contexts = await self._dense_retrieval(query)
        
        # 可选重排序
        if self.config.enable_rerank and self.rerank_service and contexts:
            contexts = await self._rerank(query, contexts)
        
        return contexts
    
    async def _dense_retrieval(self, query: Query) -> List[RetrievalContext]:
        """密集向量检索"""
        logger.debug("执行密集检索")
        
        # 生成查询向量
        if not query.has_embedding:
            embedding = await self.embedding_service.embed_query(query.processed_text)
            query = query.with_embedding(embedding.vector)
        
        # 向量搜索
        results = await self.vector_store.search(
            query_vector=query.embedding,
            top_k=self.config.top_k,
        )
        
        return self._convert_to_contexts(results)
    
    async def _sparse_retrieval(self, query: Query) -> List[RetrievalContext]:
        """稀疏检索 (BM25)"""
        logger.debug("执行稀疏检索")
        
        # 这里需要实现BM25或其他稀疏检索
        # 简化实现：使用关键词匹配
        # 实际生产环境应使用Elasticsearch或专门的BM25实现
        
        # 目前回退到密集检索
        logger.warning("稀疏检索未完全实现，回退到密集检索")
        return await self._dense_retrieval(query)
    
    async def _hybrid_retrieval(self, query: Query) -> List[RetrievalContext]:
        """混合检索"""
        logger.debug("执行混合检索")
        
        # 并行执行密集和稀疏检索
        import asyncio
        
        dense_task = self._dense_retrieval(query)
        sparse_task = self._sparse_retrieval(query)
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )
        
        if isinstance(dense_results, Exception):
            dense_results = []
        if isinstance(sparse_results, Exception):
            sparse_results = []
        
        # 融合结果 (Reciprocal Rank Fusion)
        return self._reciprocal_rank_fusion(
            dense_results, 
            sparse_results,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )
    
    async def _multi_query_retrieval(self, query: Query) -> List[RetrievalContext]:
        """多查询检索"""
        logger.debug("执行多查询检索")
        
        if not self.llm_service:
            logger.warning("LLM服务不可用，回退到密集检索")
            return await self._dense_retrieval(query)
        
        # 生成多个查询变体
        query_variants = await self._generate_query_variants(query.processed_text)
        
        # 对每个变体执行检索
        all_contexts = []
        for variant in query_variants:
            variant_query = Query(
                original_text=variant,
                processed_text=variant,
                config=query.config,
            )
            contexts = await self._dense_retrieval(variant_query)
            all_contexts.extend(contexts)
        
        # 去重并重新排序
        return self._deduplicate_and_rank(all_contexts)
    
    async def _hyde_retrieval(self, query: Query) -> List[RetrievalContext]:
        """HyDE检索 (Hypothetical Document Embeddings)"""
        logger.debug("执行HyDE检索")
        
        if not self.llm_service:
            logger.warning("LLM服务不可用，回退到密集检索")
            return await self._dense_retrieval(query)
        
        # 生成假设文档
        hypothetical_doc = await self._generate_hypothetical_document(query.processed_text)
        
        # 对假设文档进行嵌入
        embedding = await self.embedding_service.embed_text(hypothetical_doc)
        
        # 使用假设文档的嵌入进行检索
        results = await self.vector_store.search(
            query_vector=embedding.vector,
            top_k=self.config.top_k,
        )
        
        return self._convert_to_contexts(results)
    
    async def _generate_query_variants(self, query: str) -> List[str]:
        """生成查询变体"""
        prompt = f"""请生成3个与以下查询语义相近但表达不同的查询变体：

原始查询：{query}

请直接输出3个变体，每行一个："""
        
        try:
            response = await self.llm_service.generate(prompt)
            variants = [line.strip() for line in response.content.split('\n') if line.strip()]
            return [query] + variants[:3]
        except Exception as e:
            logger.warning(f"生成查询变体失败: {e}")
            return [query]
    
    async def _generate_hypothetical_document(self, query: str) -> str:
        """生成假设文档"""
        prompt = f"""请根据以下问题，写一段可能包含答案的文档内容：

问题：{query}

请直接输出文档内容："""
        
        try:
            response = await self.llm_service.generate(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"生成假设文档失败: {e}")
            return query
    
    async def _rerank(
        self, 
        query: Query, 
        contexts: List[RetrievalContext]
    ) -> List[RetrievalContext]:
        """重排序"""
        if not contexts:
            return contexts
        
        documents = [ctx.content for ctx in contexts]
        
        try:
            rerank_results = await self.rerank_service.rerank(
                query.processed_text,
                documents,
                top_n=self.config.rerank_top_n,
            )
            
            for result in rerank_results:
                if result.index < len(contexts):
                    contexts[result.index].rerank_score = result.score
            
            contexts.sort(key=lambda x: x.final_score, reverse=True)
            return contexts[:self.config.rerank_top_n]
            
        except Exception as e:
            logger.warning(f"重排序失败: {e}")
            return contexts
    
    def _reciprocal_rank_fusion(
        self,
        list1: List[RetrievalContext],
        list2: List[RetrievalContext],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        k: int = 60,
    ) -> List[RetrievalContext]:
        """Reciprocal Rank Fusion"""
        scores = {}
        
        for rank, ctx in enumerate(list1):
            key = ctx.content[:100]  # 使用内容前100字符作为key
            scores[key] = scores.get(key, 0) + dense_weight / (k + rank + 1)
            scores[f"_ctx_{key}"] = ctx
        
        for rank, ctx in enumerate(list2):
            key = ctx.content[:100]
            scores[key] = scores.get(key, 0) + sparse_weight / (k + rank + 1)
            if f"_ctx_{key}" not in scores:
                scores[f"_ctx_{key}"] = ctx
        
        # 排序并返回
        sorted_keys = sorted(
            [k for k in scores.keys() if not k.startswith("_ctx_")],
            key=lambda x: scores[x],
            reverse=True
        )
        
        results = []
        for key in sorted_keys[:self.config.top_k]:
            ctx = scores.get(f"_ctx_{key}")
            if ctx:
                ctx.score = scores[key]
                results.append(ctx)
        
        return results
    
    def _deduplicate_and_rank(
        self, 
        contexts: List[RetrievalContext]
    ) -> List[RetrievalContext]:
        """去重并排序"""
        seen = set()
        unique = []
        
        for ctx in sorted(contexts, key=lambda x: x.score, reverse=True):
            content_key = ctx.content[:100]
            if content_key not in seen:
                seen.add(content_key)
                unique.append(ctx)
        
        return unique[:self.config.top_k]
    
    def _convert_to_contexts(self, results) -> List[RetrievalContext]:
        """转换搜索结果为上下文"""
        from uuid import UUID
        
        contexts = []
        for result in results:
            try:
                chunk_id = UUID(result.id) if self._is_valid_uuid(result.id) else UUID(int=0)
            except:
                chunk_id = UUID(int=0)
            
            contexts.append(RetrievalContext(
                chunk_id=chunk_id,
                document_id=UUID(int=0),
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                source=result.metadata.get("source", "vector_store") if result.metadata else "vector_store",
                source_url=result.metadata.get("url") if result.metadata else None,
                title=result.metadata.get("title") if result.metadata else None,
            ))
        
        return contexts
    
    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        """检查是否是有效的UUID"""
        from uuid import UUID
        try:
            UUID(val)
            return True
        except (ValueError, AttributeError):
            return False
