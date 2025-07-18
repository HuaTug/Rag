import os
import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import get_llm_answer, get_llm_answer_deepseek, OpenAI, TencentDeepSeekClient

from dotenv import load_dotenv


load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT", "./milvus_demo.db")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
# 添加腾讯云DeepSeek API配置
TENCENT_API_KEY = os.getenv("TENCENT_API_KEY", "sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl")
USE_DEEPSEEK = os.getenv("USE_DEEPSEEK", "true").lower() == "true"


# Logo
st.image("./pics/Milvus_Logo_Official.png", width=200)

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">RAG Demo</div>
    <div class="description">
        This chatbot is built with Milvus vector database, supported by open-source embedding model.<br>
        It supports conversation based on knowledge from the Milvus development guide document.<br>
        <strong>对话模型: {'腾讯云 DeepSeek-V3' if USE_DEEPSEEK else 'OpenAI GPT'}</strong><br>
        <strong>嵌入模型: Sentence-Transformers (开源)</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

# Get clients
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# 根据配置选择使用的LLM客户端
if USE_DEEPSEEK:
    llm_client = TencentDeepSeekClient(api_key=TENCENT_API_KEY)
    st.info("✅ 使用腾讯云 DeepSeek-V3 模型")
else:
    llm_client = OpenAI()
    st.info("✅ 使用 OpenAI GPT 模型")

# 设置使用开源嵌入模型
os.environ["USE_OPENSOURCE_EMBEDDING"] = "true"

retrieved_lines_with_distances = []

with st.form("my_form"):
    question = st.text_area("Enter your question:")
    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # Generate query embedding
        query_vector = emb_text(question)
        # Search in Milvus collection
        search_res = get_search_results(
            milvus_client, COLLECTION_NAME, query_vector, ["text"]
        )

        # Retrieve lines and distances
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]
        
        # 设置相似度阈值（距离越小越相似，对于IP距离，通常>0.7表示较高相似度）
        similarity_threshold = 0.3  # 可以根据实际效果调整
        
        # 过滤低相似度的结果
        relevant_results = [
            (text, distance) for text, distance in retrieved_lines_with_distances 
            if distance > similarity_threshold
        ]
        
        # Create context from retrieved lines
        if relevant_results:
            context = "\n\n".join([text for text, _ in relevant_results])
            st.info(f"✅ 找到 {len(relevant_results)} 条相关信息")
        else:
            context = ""
            st.warning("⚠️ 在知识库中未找到高度相关的信息，将基于通用知识回答")
        
        # 根据客户端类型调用相应的函数
        if USE_DEEPSEEK:
            answer = get_llm_answer_deepseek(llm_client, context, question)
        else:
            answer = get_llm_answer(llm_client, context, question)

        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)


# Display the retrieved lines in a more readable format
st.sidebar.subheader("Retrieved Lines with Distances:")
for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Result {idx}:**")
    st.sidebar.markdown(f"> {line}")
    st.sidebar.markdown(f"*Distance: {distance:.2f}*")
