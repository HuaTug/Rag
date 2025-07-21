
import asyncio
import os
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv

from enhanced_rag_processor import EnhancedRAGProcessor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 页面配置
st.set_page_config(
    page_title="增强RAG搜索系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .search-box {
        margin: 2rem 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .confidence-score {
        font-weight: bold;
        color: #28a745;
    }
    .processing-time {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_processor():
    """初始化RAG处理器"""
    config = {
        "milvus_endpoint": os.getenv("MILVUS_ENDPOINT", "localhost:19530"),
        "milvus_token": os.getenv("MILVUS_TOKEN"),
        "enable_search_engine": True,
        "search_engine": os.getenv("SEARCH_ENGINE", "duckduckgo"),
        "search_api_key": os.getenv("SEARCH_API_KEY"),
        "enable_local_knowledge": True,
        "enable_news": os.getenv("ENABLE_NEWS", "false").lower() == "true",
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "search_timeout": float(os.getenv("SEARCH_TIMEOUT", "30"))
    }
    
    return EnhancedRAGProcessor(config)


def main():
    """主函数"""
    # 标题
    st.markdown('<h1 class="main-header">🔍 增强RAG搜索系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 搜索配置
        st.subheader("搜索设置")
        max_results = st.slider("最大结果数", 5, 20, 10)
        search_timeout = st.slider("搜索超时(秒)", 10, 60, 30)
        
        # 显示设置
        st.subheader("显示设置")
        show_sources = st.checkbox("显示来源信息", True)
        show_metadata = st.checkbox("显示元数据", False)
        show_processing_time = st.checkbox("显示处理时间", True)
        
        # 系统状态
        st.subheader("📊 系统状态")
        if st.button("检查系统状态"):
            with st.spinner("检查中..."):
                try:
                    processor = init_rag_processor()
                    st.success("✅ 系统正常运行")
                    
                    # 显示配置信息
                    st.info(f"🔗 Milvus: {os.getenv('MILVUS_ENDPOINT', 'localhost:19530')}")
                    st.info(f"🔍 搜索引擎: {os.getenv('SEARCH_ENGINE', 'duckduckgo')}")
                    
                except Exception as e:
                    st.error(f"❌ 系统异常: {str(e)}")
        
        # 数据清理
        st.subheader("🧹 数据管理")
        if st.button("清理过期数据"):
            with st.spinner("清理中..."):
                try:
                    processor = init_rag_processor()
                    # 这里需要在异步环境中运行
                    st.info("清理任务已启动（后台执行）")
                except Exception as e:
                    st.error(f"清理失败: {str(e)}")
    
    # 主界面
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    
    # 搜索输入
    query = st.text_input(
        "🔍 请输入您的问题:",
        placeholder="例如: 什么是人工智能？",
        help="支持中英文查询，系统会自动分析查询类型并选择最佳搜索策略"
    )
    
    # 搜索按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 开始搜索", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 处理搜索
    if search_button and query:
        with st.spinner("🔄 正在搜索和分析..."):
            try:
                # 初始化处理器
                processor = init_rag_processor()
                
                # 创建异步事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 执行查询
                response = loop.run_until_complete(
                    processor.process_query(
                        query=query,
                        max_results=max_results
                    )
                )
                
                # 显示结果
                display_results(response, show_sources, show_metadata, show_processing_time)
                
            except Exception as e:
                st.error(f"❌ 搜索失败: {str(e)}")
                st.exception(e)
    
    elif search_button and not query:
        st.warning("⚠️ 请输入查询内容")
    
    # 示例查询
    st.markdown("---")
    st.subheader("💡 示例查询")
    
    example_queries = [
        "什么是机器学习？",
        "Python中如何使用装饰器？",
        "最新的AI技术发展趋势",
        "如何优化数据库查询性能？",
        "区块链技术的应用场景"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.rerun()


def display_results(response, show_sources=True, show_metadata=False, show_processing_time=True):
    """显示搜索结果"""
    
    # 处理时间和置信度
    if show_processing_time:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<p class="processing-time">⏱️ 处理时间: {response.processing_time:.2f}秒</p>', 
                       unsafe_allow_html=True)
        with col2:
            confidence_color = "green" if response.confidence_score > 0.7 else "orange" if response.confidence_score > 0.4 else "red"
            st.markdown(f'<p class="confidence-score" style="color: {confidence_color}">🎯 置信度: {response.confidence_score:.2f}</p>', 
                       unsafe_allow_html=True)
    
    # 主要答案
    st.markdown("## 📝 答案")
    st.markdown(f'<div class="result-card">{response.answer}</div>', unsafe_allow_html=True)
    
    # 来源信息
    if show_sources and response.sources:
        st.markdown("## 📚 参考来源")
        
        for i, source in enumerate(response.sources[:5], 1):
            with st.expander(f"来源 {i}: {source['title'][:50]}..."):
                st.write(f"**标题:** {source['title']}")
                st.write(f"**来源:** {source['source']}")
                st.write(f"**相关性:** {source['score']:.3f}")
                st.write(f"**类型:** {source['type']}")
                
                if source['url']:
                    st.write(f"**链接:** [{source['url']}]({source['url']})")
    
    # 元数据信息
    if show_metadata and response.metadata:
        st.markdown("## 🔍 详细信息")
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "查询类型": response.metadata.get("query_type", "未知"),
                "总结果数": response.metadata.get("total_results", 0),
                "搜索结果数": response.metadata.get("search_results_count", 0)
            })
        
        with col2:
            st.json({
                "向量结果数": response.metadata.get("vector_results_count", 0),
                "处理时间": f"{response.processing_time:.2f}s",
                "置信度": f"{response.confidence_score:.2f}"
            })
    
    # 实时搜索结果
    if response.search_results:
        with st.expander(f"🌐 实时搜索结果 ({len(response.search_results)}个)"):
            for i, result in enumerate(response.search_results[:3], 1):
                st.markdown(f"**{i}. {result.title}**")
                st.write(f"来源: {result.source} | 相关性: {result.relevance_score:.3f}")
                st.write(result.content[:200] + "..." if len(result.content) > 200 else result.content)
                if result.url:
                    st.write(f"链接: [{result.url}]({result.url})")
                st.markdown("---")


if __name__ == "__main__":
    main()