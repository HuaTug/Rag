from search_channels import GoogleSearchChannel
from mcp_framework import QueryContext, QueryType
import asyncio

async def main():
    # 创建搜索通道
    channel = GoogleSearchChannel()

    # 创建查询上下文
    context = QueryContext(
        query="Python编程教程",
        query_type=QueryType.FACTUAL,
        max_results=5
    )

    # 执行搜索
    results = await channel.search(context)
    for result in results:
        print(result)

# 运行异步函数
asyncio.run(main())