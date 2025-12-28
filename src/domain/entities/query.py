"""
Query Entity - 查询领域实体

定义查询相关的核心业务对象
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class QueryIntent(Enum):
    """查询意图枚举"""
    FACTUAL = "factual"              # 事实性查询
    ANALYTICAL = "analytical"        # 分析性查询
    COMPARATIVE = "comparative"      # 比较性查询
    PROCEDURAL = "procedural"        # 流程性查询
    EXPLORATORY = "exploratory"      # 探索性查询
    CONVERSATIONAL = "conversational"  # 对话性查询


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    DENSE = "dense"                  # 纯密集检索
    SPARSE = "sparse"                # 纯稀疏检索（BM25）
    HYBRID = "hybrid"                # 混合检索
    MULTI_QUERY = "multi_query"      # 多查询检索
    HYDE = "hyde"                    # 假设文档嵌入


@dataclass(frozen=True)
class QueryConfig:
    """
    查询配置（值对象）
    
    控制查询行为的参数
    """
    top_k: int = 10                          # 返回结果数
    similarity_threshold: float = 0.5        # 相似度阈值
    enable_rerank: bool = True               # 是否启用重排
    enable_web_search: bool = False          # 是否启用网络搜索
    enable_cache: bool = True                # 是否启用缓存
    max_tokens: int = 2000                   # 最大token数
    timeout_seconds: float = 30.0            # 超时时间
    
    # 高级配置
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    dense_weight: float = 0.7                # 密集检索权重
    sparse_weight: float = 0.3               # 稀疏检索权重
    rerank_top_n: int = 5                    # 重排后保留数量


@dataclass
class RetrievalContext:
    """
    检索上下文
    
    包含检索到的相关文档片段
    """
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float                             # 相似度分数
    rerank_score: Optional[float] = None     # 重排分数
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 来源信息
    source: str = ""
    source_url: Optional[str] = None
    title: Optional[str] = None
    
    @property
    def final_score(self) -> float:
        """获取最终分数（优先使用重排分数）"""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def to_prompt_context(self) -> str:
        """转换为可用于prompt的上下文"""
        header = f"[来源: {self.source}]"
        if self.title:
            header += f" {self.title}"
        return f"{header}\n{self.content}"


@dataclass
class Query:
    """
    查询聚合根（Aggregate Root）
    
    代表一次用户查询的完整生命周期
    """
    id: UUID = field(default_factory=uuid4)
    original_text: str = ""                  # 原始查询文本
    processed_text: str = ""                 # 处理后的查询文本
    embedding: Optional[List[float]] = None  # 查询向量
    
    # 查询分析结果
    intent: QueryIntent = QueryIntent.FACTUAL
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    language: str = "zh"
    
    # 配置
    config: QueryConfig = field(default_factory=QueryConfig)
    
    # 用户上下文
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.processed_text:
            self.processed_text = self.original_text
    
    def with_embedding(self, embedding: List[float]) -> "Query":
        """设置向量（返回新对象保持不变性）"""
        return Query(
            id=self.id,
            original_text=self.original_text,
            processed_text=self.processed_text,
            embedding=embedding,
            intent=self.intent,
            keywords=self.keywords,
            entities=self.entities,
            language=self.language,
            config=self.config,
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_history=self.conversation_history,
            created_at=self.created_at,
        )
    
    @property
    def has_embedding(self) -> bool:
        """是否已有向量"""
        return self.embedding is not None and len(self.embedding) > 0


@dataclass
class QueryResult:
    """
    查询结果（实体）
    
    包含完整的查询响应
    """
    id: UUID = field(default_factory=uuid4)
    query_id: UUID = field(default_factory=uuid4)
    
    # 生成的答案
    answer: str = ""
    confidence: float = 0.0
    
    # 检索上下文
    contexts: List[RetrievalContext] = field(default_factory=list)
    
    # 元信息
    model_used: str = ""
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    
    # 来源追踪
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_answer(self) -> bool:
        """是否有有效答案"""
        return bool(self.answer and self.answer.strip())
    
    @property
    def context_count(self) -> int:
        """上下文数量"""
        return len(self.contexts)
    
    def get_combined_context(self, max_tokens: int = 2000) -> str:
        """获取合并的上下文"""
        combined = []
        total_length = 0
        
        for ctx in sorted(self.contexts, key=lambda x: x.final_score, reverse=True):
            ctx_text = ctx.to_prompt_context()
            if total_length + len(ctx_text) > max_tokens * 4:  # 粗略估计
                break
            combined.append(ctx_text)
            total_length += len(ctx_text)
        
        return "\n\n---\n\n".join(combined)
