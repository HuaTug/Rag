#!/usr/bin/env python3
"""
Main Entry Point - 主入口

运行企业级RAG服务
"""

import uvicorn

from src.config.settings import get_settings
from src.config.logging import setup_logging


def main():
    """启动服务"""
    settings = get_settings()
    
    # 配置日志
    setup_logging(
        level=settings.server.log_level,
        json_format=not settings.debug,
    )
    
    # 启动服务器
    uvicorn.run(
        "src.application.api:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=settings.server.reload,
        log_level=settings.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
