import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from pydantic import BaseModel


class ChannelType(Enum):
    """通道类型枚举"""
    SEARCH_ENGINE = "search_engine"
    LOCAL_KNOWLEDGE = "local_knowledge"
    REAL_TIME_WEB = "real_time_web"
    SOCIAL_MEDIA = "social_media"
    NEWS_FEED = "news_feed"


class QueryType(Enum):
    """查询类型枚举"""
    FACTUAL = "factual"          # 事实性查询
    ANALYTICAL = "analytical"    # 分析性查询
    CREATIVE = "creative"        # 创造性查询
    CONVERSATIONAL = "conversational"  # 对话性查询


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    title: str
    content: str
    url: str
    source: str
    timestamp: float
    relevance_score: float
    channel_type: ChannelType
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryContext:
    """查询上下文"""
    query: str
    query_type: QueryType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    language: str = "zh"
    max_results: int = 10
    timeout: float = 30.0
    filters: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class BaseChannel(ABC):
    """基础通道抽象类"""
    
    def __init__(self, channel_type: ChannelType, config: Dict[str, Any]):
        self.channel_type = channel_type
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查通道是否可用"""
        pass
    
    def get_priority(self, query_type: QueryType) -> int:
        """获取通道优先级（数字越小优先级越高）"""
        return self.config.get("priority", {}).get(query_type.value, 10)


class MockSearchChannel(BaseChannel):
    """模拟搜索通道（用于测试）"""
    
    def __init__(self):
        config = {
            "priority": {
                "factual": 1,
                "analytical": 2,
                "creative": 3,
                "conversational": 4
            }
        }
        super().__init__(ChannelType.SEARCH_ENGINE, config)
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """模拟搜索"""
        print(f"🔍 MockSearchChannel 正在搜索: {context.query}")
        
        # 模拟搜索延迟
        await asyncio.sleep(0.5)
        
        # 返回模拟结果
        results = [
            SearchResult(
                title=f"关于'{context.query}'的搜索结果1",
                content=f"这是关于{context.query}的详细内容1...",
                url="https://example.com/result1",
                source="mock_search",
                timestamp=time.time(),
                relevance_score=0.9,
                channel_type=self.channel_type
            ),
            SearchResult(
                title=f"关于'{context.query}'的搜索结果2",
                content=f"这是关于{context.query}的详细内容2...",
                url="https://example.com/result2",
                source="mock_search",
                timestamp=time.time(),
                relevance_score=0.8,
                channel_type=self.channel_type
            )
        ]
        
        print(f"✅ MockSearchChannel 找到 {len(results)} 个结果")
        return results
    
    def is_available(self) -> bool:
        """检查通道是否可用"""
        return True


class MockKnowledgeChannel(BaseChannel):
    """模拟知识库通道（用于测试）"""
    
    def __init__(self):
        config = {
            "priority": {
                "factual": 2,
                "analytical": 1,
                "creative": 4,
                "conversational": 3
            }
        }
        super().__init__(ChannelType.LOCAL_KNOWLEDGE, config)
    
    async def search(self, context: QueryContext) -> List[SearchResult]:
        """模拟知识库搜索"""
        print(f"📚 MockKnowledgeChannel 正在搜索: {context.query}")
        
        # 模拟搜索延迟
        await asyncio.sleep(0.3)
        
        # 返回模拟结果
        results = [
            SearchResult(
                title=f"知识库中关于'{context.query}'的条目",
                content=f"从本地知识库中找到的关于{context.query}的专业解释...",
                url="local://knowledge/item1",
                source="local_knowledge",
                timestamp=time.time(),
                relevance_score=0.95,
                channel_type=self.channel_type
            )
        ]
        
        print(f"✅ MockKnowledgeChannel 找到 {len(results)} 个结果")
        return results
    
    def is_available(self) -> bool:
        """检查通道是否可用"""
        return True


class MCPProcessor:
    """MCP多通道处理器"""
    
    def __init__(self):
        self.channels: Dict[ChannelType, BaseChannel] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_channel(self, channel: BaseChannel):
        """注册通道"""
        self.channels[channel.channel_type] = channel
        self.logger.info(f"注册通道: {channel.channel_type.value}")
        print(f"✅ 已注册通道: {channel.channel_type.value}")
    
    def unregister_channel(self, channel_type: ChannelType):
        """注销通道"""
        if channel_type in self.channels:
            del self.channels[channel_type]
            self.logger.info(f"注销通道: {channel_type.value}")
    
    async def process_query(self, context: QueryContext) -> List[SearchResult]:
        """处理查询请求"""
        print(f"\n🚀 开始处理查询: {context.query}")
        print(f"📊 查询类型: {context.query_type.value}")
        
        self.logger.info(f"处理查询: {context.query}")
        
        # 1. 查询分析和路由
        selected_channels = self._route_query(context)
        print(f"🎯 选择了 {len(selected_channels)} 个通道进行搜索")
        
        # 2. 并行执行搜索
        tasks = []
        for channel in selected_channels:
            if channel.is_available():
                task = asyncio.create_task(
                    self._safe_search(channel, context)
                )
                tasks.append(task)
        
        # 3. 收集结果
        all_results = []
        if tasks:
            print("⏳ 正在并行执行搜索...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"搜索异常: {result}")
                    print(f"❌ 搜索异常: {result}")
        
        # 4. 结果去重和排序
        deduplicated_results = self._deduplicate_results(all_results)
        sorted_results = self._sort_results(deduplicated_results, context)
        
        final_results = sorted_results[:context.max_results]
        print(f"📋 最终返回 {len(final_results)} 个结果")
        
        return final_results
    
    def _route_query(self, context: QueryContext) -> List[BaseChannel]:
        """查询路由 - 根据查询类型选择合适的通道"""
        available_channels = [
            channel for channel in self.channels.values() 
            if channel.is_available()
        ]
        
        # 根据查询类型和优先级排序
        available_channels.sort(
            key=lambda ch: ch.get_priority(context.query_type)
        )
        
        # 根据查询类型选择通道数量
        if context.query_type == QueryType.FACTUAL:
            return available_channels[:3]  # 事实性查询使用多个通道
        elif context.query_type == QueryType.ANALYTICAL:
            return available_channels[:2]  # 分析性查询使用较少通道
        else:
            return available_channels[:1]  # 其他类型使用单个通道
    
    async def _safe_search(self, channel: BaseChannel, context: QueryContext) -> List[SearchResult]:
        """安全搜索包装"""
        try:
            return await asyncio.wait_for(
                channel.search(context), 
                timeout=context.timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"通道 {channel.channel_type.value} 搜索超时")
            print(f"⏰ 通道 {channel.channel_type.value} 搜索超时")
            return []
        except Exception as e:
            self.logger.error(f"通道 {channel.channel_type.value} 搜索异常: {e}")
            print(f"❌ 通道 {channel.channel_type.value} 搜索异常: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """结果去重"""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated.append(result)
        
        return deduplicated
    
    def _sort_results(self, results: List[SearchResult], context: QueryContext) -> List[SearchResult]:
        """结果排序"""
        # 综合排序：相关性分数 + 时间新鲜度 + 来源权重
        def sort_key(result: SearchResult):
            time_factor = max(0, 1 - (time.time() - result.timestamp) / (24 * 3600))  # 24小时内的时间衰减
            source_weight = self._get_source_weight(result.source)
            return result.relevance_score * 0.6 + time_factor * 0.2 + source_weight * 0.2
        
        return sorted(results, key=sort_key, reverse=True)
    
    def _get_source_weight(self, source: str) -> float:
        """获取来源权重"""
        weights = {
            "wikipedia": 0.9,
            "stackoverflow": 0.8,
            "github": 0.8,
            "arxiv": 0.9,
            "news": 0.7,
            "blog": 0.5,
            "social": 0.3,
            "unknown": 0.5
        }
        return weights.get(source.lower(), 0.5)


class QueryAnalyzer:
    """查询分析器"""
    
    @staticmethod
    def analyze_query(query: str) -> QueryType:
        """分析查询类型"""
        query_lower = query.lower()
        
        # 事实性查询关键词
        factual_keywords = ["什么是", "谁是", "何时", "哪里", "定义", "解释"]
        if any(keyword in query_lower for keyword in factual_keywords):
            return QueryType.FACTUAL
        
        # 分析性查询关键词
        analytical_keywords = ["为什么", "如何", "比较", "分析", "原因", "影响"]
        if any(keyword in query_lower for keyword in analytical_keywords):
            return QueryType.ANALYTICAL
        
        # 创造性查询关键词
        creative_keywords = ["创建", "设计", "写", "生成", "创作"]
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryType.CREATIVE
        
        # 默认为对话性查询
        return QueryType.CONVERSATIONAL


# 使用示例
async def example_usage():
    """使用示例"""
    print("🎉 MCP框架测试开始")
    print("=" * 50)
    
    # 创建MCP处理器
    mcp = MCPProcessor()
    
    # 注册模拟通道
    print("\n📝 注册通道...")
    mcp.register_channel(MockSearchChannel())
    mcp.register_channel(MockKnowledgeChannel())
    
    # 测试查询列表
    test_queries = [
        "什么是人工智能？",
        "如何学习Python编程？",
        "创建一个网站需要什么？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} 测试 {i} {'='*20}")
        
        # 创建查询上下文
        context = QueryContext(
            query=query,
            query_type=QueryAnalyzer.analyze_query(query),
            max_results=5
        )
        
        # 处理查询
        results = await mcp.process_query(context)
        
        # 显示结果
        print(f"\n📊 查询结果 ({len(results)} 个):")
        print("-" * 40)
        
        if results:
            for j, result in enumerate(results, 1):
                print(f"{j}. 标题: {result.title}")
                print(f"   来源: {result.source}")
                print(f"   相关性: {result.relevance_score:.2f}")
                print(f"   通道: {result.channel_type.value}")
                print(f"   内容: {result.content[:50]}...")
                print()
        else:
            print("❌ 没有找到结果")
    
    print("\n🎉 测试完成!")


def main():
    """主函数"""
    print("🚀 启动MCP框架演示")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    asyncio.run(example_usage())


if __name__ == "__main__":
    main()
