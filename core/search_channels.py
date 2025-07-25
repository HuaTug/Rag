
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google搜索引擎通道实现（使用官方API）

这个模块提供了基于Google Custom Search API的通道实现。
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

import aiohttp
import requests  # 添加同步请求库
from bs4 import BeautifulSoup

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.channel_framework import BaseChannel, ChannelType, SearchResult, QueryContext
from dotenv import load_dotenv


load_dotenv()

class GoogleSearchChannel(BaseChannel):
    """Google搜索引擎通道（使用官方API）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Google搜索通道
        
        Args:
            config: 配置字典，必须包含 api_key 和 search_engine_id
        """
        if config is None:
            config = {}
        
        # 设置默认配置
        default_config = {
            "priority": {
                "factual": 1,
                "analytical": 2, 
                "creative": 3,
                "conversational": 2
            },
            "timeout": 10,
            "max_content_length": 1000,
        }
        default_config.update(config)
        
        super().__init__(ChannelType.SEARCH_ENGINE, default_config)
        
        # Google API 配置
        self.api_key = self.config.get("api_key")
        self.search_engine_id = self.config.get("search_engine_id")  # Custom Search Engine ID
        self.timeout = self.config.get("timeout", 10)
        self.max_content_length = self.config.get("max_content_length", 1000)
        
        if not self.api_key:
            self.logger.error("Google API Key 未配置！")
        if not self.search_engine_id:
            self.logger.error("Google Custom Search Engine ID 未配置！")
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """
        执行Google搜索
        
        Args:
            context: 查询上下文
            
        Returns:
            搜索结果列表
        """
        return await self._google_api_search(context)
    
    def is_available(self) -> bool:
        """
        检查Google搜索是否可用
        
        Returns:
            True表示可用
        """
        return bool(self.api_key and self.search_engine_id)
    
    async def _google_api_search(self, context: QueryContext) -> List[SearchResult]:
        """
        使用Google Custom Search API进行搜索
        
        Args:
            context: 查询上下文
            
        Returns:
            搜索结果列表
        """
        if not self.is_available():
            self.logger.error("Google API 配置不完整")
            return []
        
        try:
            self.logger.info(f"开始Google API搜索: {context.query}")
            
            # Google Custom Search API URL
            api_url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": context.query,
                "num": min(context.max_results, 10),  # API最多返回10个结果
            }
            
            # 使用同步requests请求，和test.py保持一致
            response = requests.get(api_url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return await self._process_api_results(data)
            else:
                error_text = response.text
                self.logger.error(f"Google API 请求失败: {response.status_code} - {error_text}")
                return []
                        
        except Exception as e:
            self.logger.error(f"Google API搜索失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return []
    
    async def _process_api_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """
        处理API返回的搜索结果
        
        Args:
            data: API返回的JSON数据
            
        Returns:
            搜索结果列表
        """
        results = []
        
        if "items" not in data:
            self.logger.warning("API返回数据中没有搜索结果")
            return results
        
        for i, item in enumerate(data["items"]):
            try:
                title = item.get("title", "无标题")
                snippet = item.get("snippet", "")
                url = item.get("link", "")
                
                # 获取更详细的页面内容（可选）
                detailed_content = await self._fetch_page_content(url) if url else ""
                content = detailed_content if detailed_content else snippet
                
                results.append(SearchResult(
                    title=title,
                    content=content,
                    url=url,
                    source="google_api",
                    timestamp=time.time(),
                    relevance_score=1.0 - (i * 0.1),
                    channel_type=self.channel_type,
                    metadata={
                        "search_rank": i + 1,
                        "snippet": snippet,
                        "content_length": len(content)
                    }
                ))
                
            except Exception as e:
                self.logger.warning(f"处理搜索结果时出错: {e}")
                continue
        
        self.logger.info(f"Google API搜索完成，找到 {len(results)} 个结果")
        return results
    
    async def _fetch_page_content(self, url: str) -> str:
        """
        获取页面详细内容（可选功能）
        
        Args:
            url: 页面URL
            
        Returns:
            页面内容
        """
        try:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/91.0.4472.124 Safari/537.36'
                )
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),  # 较短超时
                headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._extract_content(html_content)
                        
        except Exception as e:
            self.logger.debug(f"获取页面内容失败 {url}: {e}")
        
        return ""
    
    def _extract_content(self, html_content: str) -> str:
        """
        提取页面主要内容
        
        Args:
            html_content: HTML内容
            
        Returns:
            提取的文本内容
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除不需要的标签
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # 优先提取主要内容区域
            main_content = None
            for selector in ["main", "article", ".content", "#content", ".main"]:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            # 提取文本
            text = main_content.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # 限制长度
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
            
            return text
            
        except Exception as e:
            self.logger.debug(f"提取内容失败: {e}")
            return ""


# 为了保持向后兼容性，提供一个别名
SearchEngineChannel = GoogleSearchChannel


def create_google_search_channel(api_key: str, search_engine_id: str, config: Dict[str, Any] = None) -> GoogleSearchChannel:
    """
    创建Google搜索通道的工厂函数
    
    Args:
        api_key: Google API Key
        search_engine_id: Google Custom Search Engine ID
        config: 额外配置字典
        
    Returns:
        GoogleSearchChannel实例
    """
    if config is None:
        config = {}
    
    config.update({
        "api_key": api_key,
        "search_engine_id": search_engine_id
    })
    
    return GoogleSearchChannel(config)


# 测试函数
async def test_google_search():
    """测试Google搜索功能"""
    print("🧪 开始测试Google API搜索通道")
    
    # 从环境变量获取配置
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not search_engine_id:
        print("❌ 请设置环境变量:")
        print("   export GOOGLE_API_KEY='your_api_key'")
        print("   export GOOGLE_SEARCH_ENGINE_ID='your_search_engine_id'")
        return
    
    # 创建搜索通道
    channel = create_google_search_channel(api_key, search_engine_id)
    
    if not channel.is_available():
        print("❌ Google搜索通道不可用")
        return
    
    print("✅ Google API搜索通道可用")
    
    # 创建测试查询
    from service.channel_framework import QueryContext, QueryType
    
    test_queries = [
        "九三阅兵",
        "人工智能最新发展",
        "机器学习算法"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        
        context = QueryContext(
            query=query,
            query_type=QueryType.FACTUAL,
            max_results=10
        )
        
        try:
            results = await channel.search(context)
            
            if results:
                print(f"✅ 找到 {len(results)} 个结果:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.title}")
                    print(f"     URL: {result.url}")
                    print(f"     相关性: {result.relevance_score:.2f}")
                    print(f"     内容预览: {result.content}...")
                    print()
            else:
                print("❌ 没有找到结果")
                
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
    
    print("🎉 测试完成")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    asyncio.run(test_google_search())