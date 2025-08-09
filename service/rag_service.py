#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能RAG服务 - 生产级API服务

基于FastAPI的REST API服务，提供智能查询处理能力。
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import logging
import time
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

import sys
import os
from dotenv import load_dotenv

# 加载环境变量（需要在其他导入之前）
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地模块 - 修复相对导入问题
from smart_query_analyzer import SmartQueryAnalyzer, QueryAnalysisResult
from enhanced_rag_processor import EnhancedRAGProcessor, RAGResponse
from channel_framework import QueryContext, QueryType

# Pydantic模型定义
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询")
    max_results: int = Field(5, ge=1, le=20, description="最大结果数量")
    timeout: int = Field(30, ge=5, le=120, description="超时时间（秒）")
    enable_search: bool = Field(True, description="是否启用网络搜索")
    enable_calculation: bool = Field(True, description="是否启用计算功能")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class QueryResponse(BaseModel):
    """查询响应模型"""
    request_id: str = Field(..., description="请求ID")
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的答案")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    processing_time: float = Field(..., description="处理时间（秒）")
    timestamp: str = Field(..., description="响应时间戳")
    
    # 分析结果
    analysis: Dict[str, Any] = Field(..., description="查询分析结果")
    
    # 数据源信息
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="信息来源")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="处理元数据")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str
    request_id: Optional[str] = None
    timestamp: str


# 服务管理类
class RAGService:
    """智能RAG服务核心类"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.start_time = time.time()
        
        # 核心组件
        self.rag_processor: Optional[EnhancedRAGProcessor] = None
        self.analyzer: Optional[SmartQueryAnalyzer] = None
        
        # 服务状态
        self.is_ready = False
        self.health_status = {
            "rag_processor": "initializing",
            "analyzer": "initializing",
            "vector_store": "initializing",
            "llm_client": "initializing"
        }
        
        self.logger.info(" RAG服务初始化中...")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # 默认配置
        default_config = {
            "similarity_threshold": 0.5,
            "enable_smart_search": True,
            "enable_semantic_analysis": True,
            "vector_dim": 384,
            "max_concurrent_requests": 10,
            "log_level": "INFO",
            # Google搜索配置
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "google_search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            "enable_search_engine": True,
            "search_timeout": 10,
            "enable_mcp_tools":True,
        }
        
        # 合并配置
        default_config.update(config)
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/tmp/rag_service.log', encoding='utf-8')  # 移动到tmp目录避免热重载检测
            ]
        )
        return logging.getLogger(self.__class__.__name__)
    
    async def initialize(self):
        """初始化服务组件"""
        try:
            self.logger.info(" 初始化RAG处理器...")
            self.rag_processor = EnhancedRAGProcessor(config=self.config)
            self.health_status["rag_processor"] = "healthy"
            
            self.logger.info(" 初始化查询分析器...")
            self.analyzer = SmartQueryAnalyzer(self.config)
            self.health_status["analyzer"] = "healthy"
            
            # 可以添加更多组件初始化
            self.health_status["vector_store"] = "healthy"
            self.health_status["llm_client"] = "healthy"
            
            self.is_ready = True
            self.logger.info(" RAG服务初始化完成")
            
        except Exception as e:
            self.logger.error(f" 服务初始化失败: {e}")
            self.health_status = {k: "error" for k in self.health_status.keys()}
            raise
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """处理查询请求"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f" 处理查询 [{request_id}]: {request.query}")
            
            if not self.is_ready:
                raise HTTPException(status_code=503, detail="服务尚未就绪")
            
            # 创建查询上下文
            context = QueryContext(
                query=request.query,
                query_type=QueryType.FACTUAL,
                max_results=request.max_results,
                timeout=request.timeout
            )
            
            # 处理查询
            rag_response = await self.rag_processor.process_query(context)
            
            # 构建响应
            response = QueryResponse(
                request_id=request_id,
                query=request.query,
                answer=rag_response.answer,
                confidence=rag_response.confidence_score,
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                analysis={
                    "query_type": rag_response.analysis_result.query_type if rag_response.analysis_result else "unknown",
                    "confidence": rag_response.analysis_result.confidence if rag_response.analysis_result else 0.0,
                    "strategy": rag_response.metadata.get("strategy", "unknown"),
                    "tools_used": rag_response.metadata.get("tools_used", []),
                    "reasoning": rag_response.analysis_result.reasoning if rag_response.analysis_result else ""
                },
                sources=rag_response.sources,
                metadata=rag_response.metadata
            )
            
            self.logger.info(f" 查询处理完成 [{request_id}]: {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f" 查询处理失败 [{request_id}]: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"查询处理失败: {str(e)}"
            )
    
    def get_health_status(self) -> HealthResponse:
        """获取服务健康状态"""
        return HealthResponse(
            status="healthy" if self.is_ready else "unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - self.start_time,
            components=self.health_status
        )
    
    async def shutdown(self):
        """关闭服务"""
        self.logger.info("🔄 正在关闭RAG服务...")
        self.is_ready = False
        # 这里可以添加清理逻辑
        self.logger.info(" RAG服务已关闭")


# 全局服务实例
rag_service = RAGService()


# FastAPI应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    await rag_service.initialize()
    yield
    # 关闭时
    await rag_service.shutdown()


# 创建FastAPI应用
app = FastAPI(
    title="智能RAG服务",
    description="基于MCP的智能检索增强生成服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 依赖注入
async def get_rag_service() -> RAGService:
    """获取RAG服务实例"""
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return rag_service


# API路由定义
@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    健康检查端点
    
    返回服务的健康状态和各组件状态
    """
    return rag_service.get_health_status()


@app.post("/query", response_model=QueryResponse, summary="智能查询")
async def query_endpoint(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    智能查询端点
    
    处理用户查询并返回智能生成的答案
    
    - **query**: 用户查询文本
    - **max_results**: 最大结果数量 (1-20)
    - **timeout**: 超时时间秒数 (5-120)
    - **enable_search**: 是否启用网络搜索
    - **enable_calculation**: 是否启用计算功能
    """
    return await service.process_query(request)


@app.get("/", summary="根路径")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "智能RAG服务",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/stats", summary="服务统计")
async def get_stats(service: RAGService = Depends(get_rag_service)):
    """获取服务统计信息"""
    return {
        "uptime_seconds": time.time() - service.start_time,
        "status": "healthy" if service.is_ready else "unhealthy",
        "components": service.health_status,
        "timestamp": datetime.now().isoformat()
    }


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTP异常处理"""
    return {
        "error": "HTTP_ERROR",
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """通用异常处理"""
    logging.error(f"未处理的异常: {exc}")
    return {
        "error": "INTERNAL_ERROR",
        "message": "服务内部错误",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # 检查是否为生产环境
    is_production = os.getenv("ENVIRONMENT") == "production"
    
    # 开发环境运行
    uvicorn.run(
        "rag_service:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_production,  # 生产环境关闭热重载
        log_level="info",
        reload_excludes=["*.log", "*.db", "*.lock", "__pycache__/*", "*.pyc"] if not is_production else None
    )
