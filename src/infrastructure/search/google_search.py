"""
Google Search Adapter - Google搜索服务适配器

实现SearchService接口
"""

import logging
from typing import List, Optional

import aiohttp
import requests

from src.domain.ports.services import SearchService, SearchConfig, SearchResult


logger = logging.getLogger(__name__)


class GoogleSearchService(SearchService):
    """
    Google搜索服务实现
    
    使用Google Custom Search API
    """
    
    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        config: Optional[SearchConfig] = None,
    ):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.config = config or SearchConfig()
        
        self.api_url = "https://www.googleapis.com/customsearch/v1"
        
        logger.info("初始化Google搜索服务")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """执行搜索"""
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google API配置不完整")
            return []
        
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(max_results, 10),  # API最多返回10个
        }
        
        try:
            # 使用同步请求（更稳定）
            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.config.timeout,
            )
            
            if response.status_code != 200:
                logger.error(f"Google API请求失败: {response.status_code}")
                return []
            
            data = response.json()
            return self._parse_results(data)
            
        except Exception as e:
            logger.error(f"Google搜索失败: {e}")
            return []
    
    def _parse_results(self, data: dict) -> List[SearchResult]:
        """解析搜索结果"""
        results = []
        
        items = data.get("items", [])
        for i, item in enumerate(items):
            try:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    content=item.get("snippet", ""),
                    url=item.get("link", ""),
                    source="google",
                    score=1.0 - (i * 0.1),  # 简单的排名分数
                    metadata={
                        "display_link": item.get("displayLink", ""),
                        "search_rank": i + 1,
                    }
                ))
            except Exception as e:
                logger.warning(f"解析结果失败: {e}")
                continue
        
        logger.info(f"Google搜索完成: {len(results)} 个结果")
        return results
    
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        return bool(self.api_key and self.search_engine_id)
