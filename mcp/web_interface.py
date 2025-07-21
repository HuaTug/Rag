#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web界面启动器 - 使用Streamlit创建友好的Web界面
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_system import RAGSystemManager, RAGSystemConfig

# 页面配置
st.set_page_config(
    page_title="智能RAG问答系统", 
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统（缓存）"""
    config = RAGSystemConfig()
    return RAGSystemManager(config), config

async def async_process_query(manager, query, query_type):
    """异步处理查询"""
    return await manager.process_query(query, query_type)

def main():
    st.title("🤖 智能RAG问答系统")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # API配置检查
        st.subheader("📋 配置状态")
        
        # 创建配置实例检查
        config = RAGSystemConfig()
        
        google_api_ok = bool(config.get("google_search", "api_key"))
        search_engine_ok = bool(config.get("google_search", "search_engine_id"))
        deepseek_api_ok = bool(config.get("deepseek", "api_key"))
        
        st.write("Google API Key:", "✅" if google_api_ok else "❌")
        st.write("Search Engine ID:", "✅" if search_engine_ok else "❌")
        st.write("DeepSeek API Key:", "✅" if deepseek_api_ok else "❌")
        
        if not all([google_api_ok, search_engine_ok, deepseek_api_ok]):
            st.error("⚠️ 配置不完整！")
            st.markdown("""
            请设置环境变量：
            ```bash
            export GOOGLE_API_KEY='your_key'
            export GOOGLE_SEARCH_ENGINE_ID='your_id' 
            export DEEPSEEK_API_KEY='your_key'
            ```
            """)
            return
        
        st.success("✅ 配置完整")
        
        # 查询类型选择
        st.subheader("🎯 查询类型")
        query_type = st.selectbox(
            "选择查询类型：",
            ["factual", "analytical", "creative", "conversational"],
            format_func=lambda x: {
                "factual": "📊 事实查询",
                "analytical": "🔍 分析性查询", 
                "creative": "🎨 创意性查询",
                "conversational": "💬 对话式查询"
            }[x]
        )
        
        # 高级设置
        with st.expander("🔧 高级设置"):
            max_results = st.slider("最大搜索结果数", 5, 20, 10)
            similarity_threshold = st.slider("相似度阈值", 0.1, 1.0, 0.7)
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 提出您的问题")
        
        # 预设问题
        st.write("💡 **示例问题**：")
        example_questions = [
            "人工智能的发展历史是什么？",
            "机器学习和深度学习有什么区别？", 
            "请解释一下大语言模型的工作原理",
            "AI在医疗领域有哪些应用？"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📌 {question}", key=f"example_{i}"):
                st.session_state.user_input = question
        
        # 用户输入
        user_input = st.text_area(
            "在这里输入您的问题：",
            value=st.session_state.get('user_input', ''),
            height=100,
            placeholder="例如：什么是机器学习？"
        )
        
        col_submit, col_clear = st.columns([1, 1])
        
        with col_submit:
            submit_button = st.button("🚀 提交问题", type="primary")
        
        with col_clear:
            if st.button("🗑️ 清空"):
                st.session_state.user_input = ""
                st.rerun()
    
    with col2:
        st.subheader("📊 系统状态")
        
        # 初始化状态
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if not st.session_state.system_initialized:
            if st.button("⚡ 初始化系统"):
                with st.spinner("正在初始化RAG系统..."):
                    try:
                        manager, config = initialize_rag_system()
                        
                        # 运行异步初始化
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(manager.initialize())
                        loop.close()
                        
                        st.session_state.rag_manager = manager
                        st.session_state.system_initialized = True
                        st.success("✅ 系统初始化成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 初始化失败: {e}")
        else:
            st.success("✅ 系统已就绪")
            
            if st.button("🔄 重新初始化"):
                st.session_state.system_initialized = False
                st.rerun()
    
    # 处理用户问题
    if submit_button and user_input.strip():
        if not st.session_state.get('system_initialized', False):
            st.error("⚠️ 请先初始化系统！")
            return
        
        with st.spinner("🤖 AI正在思考中..."):
            try:
                manager = st.session_state.rag_manager
                
                # 运行异步查询
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                answer = loop.run_until_complete(
                    async_process_query(manager, user_input, query_type)
                )
                loop.close()
                
                # 显示结果
                st.markdown("---")
                st.subheader("🎯 AI回答")
                
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #0066cc;">
                    <h4>💡 回答：</h4>
                    <p style="font-size: 16px; line-height: 1.6;">{answer}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # 保存到历史记录
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    'question': user_input,
                    'answer': answer,
                    'query_type': query_type
                })
                
            except Exception as e:
                st.error(f"❌ 处理失败: {e}")
    
    # 聊天历史
    if st.session_state.get('chat_history'):
        st.markdown("---")
        st.subheader("📝 对话历史")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # 显示最近5条
            with st.expander(f"问题 {len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                st.write("🙋 **问题**:", chat['question'])
                st.write("🤖 **回答**:", chat['answer'])
                st.write("🏷️ **类型**:", chat['query_type'])
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
        <p>🔧 RAG智能问答系统 | 基于大语言模型和向量检索技术</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
