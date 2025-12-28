"""
RAG API - FastAPI REST API

提供HTTP接口访问RAG服务
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .container import get_container, Container


logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Models
# ============================================================

class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="查询文本")
    top_k: int = Field(10, ge=1, le=50, description="返回结果数")
    enable_web_search: bool = Field(True, description="是否启用网络搜索")
    enable_rerank: bool = Field(True, description="是否启用重排")
    similarity_threshold: float = Field(0.5, ge=0, le=1, description="相似度阈值")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class SourceInfo(BaseModel):
    """来源信息"""
    title: str
    url: Optional[str]
    source: str
    score: float


class QueryResponse(BaseModel):
    """查询响应"""
    request_id: str
    query: str
    answer: str
    confidence: float
    sources: List[SourceInfo]
    processing_time_ms: float
    context_count: int


class DocumentRequest(BaseModel):
    """文档请求"""
    title: str
    content: str
    source: str = "api"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str
    title: str
    status: str
    chunk_count: int
    message: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]


# ============================================================
# Lifespan Management
# ============================================================

startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 RAG服务启动中...")
    
    # 预热服务
    container = get_container()
    try:
        # 预加载embedding模型
        embedding_service = container.get_embedding_service()
        logger.info("✅ Embedding服务就绪")
        
        # 检查向量存储
        vector_store = container.get_vector_store()
        logger.info("✅ 向量存储就绪")
        
        # 检查LLM服务
        llm_service = container.get_llm_service()
        logger.info("✅ LLM服务就绪")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
    
    logger.info("🎉 RAG服务启动完成")
    
    yield
    
    logger.info("👋 RAG服务关闭中...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Enterprise RAG API",
    description="企业级检索增强生成服务",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Dependencies
# ============================================================

def get_rag_service():
    """获取RAG服务依赖"""
    return get_container().get_rag_service()


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    container = get_container()
    
    components = {}
    
    try:
        embedding = container.get_embedding_service()
        components["embedding"] = "healthy"
    except:
        components["embedding"] = "unhealthy"
    
    try:
        vector_store = container.get_vector_store()
        components["vector_store"] = "healthy"
    except:
        components["vector_store"] = "unhealthy"
    
    try:
        llm = container.get_llm_service()
        components["llm"] = "healthy"
    except:
        components["llm"] = "unhealthy"
    
    all_healthy = all(v == "healthy" for v in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="2.0.0",
        uptime_seconds=time.time() - startup_time,
        components=components,
    )


@app.post("/api/v2/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    执行RAG查询
    
    完整的RAG流程：查询理解 -> 向量检索 -> 可选网络搜索 -> 重排序 -> 答案生成
    """
    import uuid
    from src.domain.entities.query import Query, QueryConfig, RetrievalStrategy
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 构建查询对象
        query_obj = Query(
            original_text=request.query,
            config=QueryConfig(
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                enable_rerank=request.enable_rerank,
                enable_web_search=request.enable_web_search,
            ),
            user_id=request.user_id,
            session_id=request.session_id,
        )
        
        # 执行RAG
        rag_service = get_rag_service()
        result = await rag_service.process_query(query_obj)
        
        # 构建响应
        sources = [
            SourceInfo(
                title=s.get("title", ""),
                url=s.get("url"),
                source=s.get("source", ""),
                score=s.get("score", 0),
            )
            for s in result.sources
        ]
        
        return QueryResponse(
            request_id=request_id,
            query=request.query,
            answer=result.answer,
            confidence=result.confidence,
            sources=sources,
            processing_time_ms=result.processing_time_ms,
            context_count=result.context_count,
        )
        
    except Exception as e:
        logger.error(f"查询处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/documents", response_model=DocumentResponse)
async def index_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
):
    """
    索引文档
    
    将文档分块、向量化并存储到向量数据库
    """
    import uuid
    from src.domain.entities.document import Document, DocumentMetadata
    
    try:
        # 创建文档对象
        doc = Document(
            title=request.title,
            content=request.content,
            metadata=DocumentMetadata(
                source=request.source,
                extra=request.metadata,
            ),
        )
        
        # TODO: 实现异步索引
        # 这里简单实现，实际应该使用后台任务
        
        return DocumentResponse(
            id=str(doc.id),
            title=doc.title,
            status="pending",
            chunk_count=0,
            message="文档已提交，正在处理中",
        )
        
    except Exception as e:
        logger.error(f"文档索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/stats")
async def get_stats():
    """获取系统统计信息"""
    container = get_container()
    
    try:
        vector_store = container.get_vector_store()
        vector_count = await vector_store.count()
    except:
        vector_count = 0
    
    return {
        "vector_count": vector_count,
        "uptime_seconds": time.time() - startup_time,
    }


# ============================================================
# 兼容旧版API
# ============================================================

@app.post("/api/query")
async def legacy_query(request: QueryRequest):
    """兼容旧版查询接口"""
    return await query(request)


# ============================================================
# 运行入口
# ============================================================

def create_app() -> FastAPI:
    """创建应用实例"""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.application.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
