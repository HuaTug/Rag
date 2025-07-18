#!/usr/bin/env python3
"""
RAG系统调试脚本
用于测试向量检索效果和诊断问题
"""

import os
from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from dotenv import load_dotenv



COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT", "./milvus_demo.db")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
load_dotenv()

# 设置使用开源嵌入模型
os.environ["USE_OPENSOURCE_EMBEDDING"] = "true"

def test_retrieval(query: str, top_k: int = 5):
    """测试检索效果"""
    print(f"\n🔍 测试查询: '{query}'")
    print("=" * 50)
    
    # 获取Milvus客户端
    milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
    
    # 生成查询向量
    query_vector = emb_text(query)
    
    # 检索
    search_res = get_search_results(milvus_client, COLLECTION_NAME, query_vector, ["text"])
    
    # 显示结果
    for i, res in enumerate(search_res[0], 1):
        text = res["entity"]["text"]
        distance = res["distance"]
        
        print(f"\n📄 结果 {i}:")
        print(f"   相似度: {distance:.4f}")
        print(f"   内容: {text[:100]}...")
        if len(text) > 100:
            print(f"   [总长度: {len(text)} 字符]")
    
    return search_res[0]

def analyze_collection():
    """分析集合中的数据"""
    print("\n📊 分析Milvus集合数据")
    print("=" * 50)
    
    milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
    
    # 检查集合是否存在
    if not milvus_client.has_collection(COLLECTION_NAME):
        print(f"❌ 集合 '{COLLECTION_NAME}' 不存在")
        return
    
    # 获取集合统计信息
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)
    print(f"✅ 集合 '{COLLECTION_NAME}' 统计信息:")
    print(f"   文档数量: {stats['row_count']}")
    
    # 随机查询几个结果看看数据质量
    query_vector = emb_text("测试")
    search_res = get_search_results(milvus_client, COLLECTION_NAME, query_vector, ["text"])
    
    print(f"\n📝 数据样本 (前3条):")
    for i, res in enumerate(search_res[0][:3], 1):
        text = res["entity"]["text"]
        print(f"   {i}. {text[:80]}...")

if __name__ == "__main__":
    print("🚀 RAG系统调试工具")
    
    # 分析集合
    analyze_collection()
    
    # 测试一些查询
    test_queries = [
        "票房",
        "哪吒票房多少",
        "票房表现如何",
        "电影赚了多少钱",
        "35亿",
        "全球票房",
        "制作团队",
        "导演是谁"
    ]
    
    print(f"\n🧪 开始测试 {len(test_queries)} 个查询...")
    
    for query in test_queries:
        results = test_retrieval(query)
        
        # 检查是否找到相关结果
        relevant_count = sum(1 for res in results if res["distance"] > 0.3)
        if relevant_count == 0:
            print(f"⚠️  查询 '{query}' 没有找到相关结果 (相似度阈值 > 0.3)")
    
    print(f"\n✅ 调试完成！") 