import streamlit as st
from openai import OpenAI
import os


# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}


@st.cache_resource
def get_sentence_transformer_model():
    """获取sentence-transformers模型"""
    try:
        from sentence_transformers import SentenceTransformer
        # 使用多语言支持的嵌入模型
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级英文模型
        # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 多语言支持
        return model
    except ImportError:
        st.error("请安装 sentence-transformers: pip install sentence-transformers")
        return None


embedding_cache = get_embedding_cache()


def emb_text_opensource(text: str, model_name: str = "all-MiniLM-L6-v2"):
    """使用开源模型进行文本嵌入"""
    if text in embedding_cache:
        return embedding_cache[text]
    
    model = get_sentence_transformer_model()
    if model is None:
        raise RuntimeError("无法加载sentence-transformers模型")
    
    # 生成嵌入向量
    embedding = model.encode(text).tolist()
    embedding_cache[text] = embedding
    return embedding


def emb_text_openai(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
    """使用OpenAI进行文本嵌入（保留作为备选）"""
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = client.embeddings.create(input=text, model=model).data[0].embedding
        embedding_cache[text] = embedding
        return embedding


def emb_text(client_or_text, text=None, model="auto"):
    """
    统一的嵌入接口，自动选择可用的嵌入方法
    
    Args:
        client_or_text: 如果是字符串则直接使用开源模型，如果是OpenAI客户端则使用OpenAI
        text: 要嵌入的文本（当第一个参数是客户端时使用）
        model: 模型选择，"auto"为自动选择
    """
    # 判断是否使用开源模型
    use_opensource = os.getenv("USE_OPENSOURCE_EMBEDDING", "true").lower() == "true"
    
    if use_opensource or isinstance(client_or_text, str):
        # 使用开源模型
        input_text = client_or_text if isinstance(client_or_text, str) else text
        return emb_text_opensource(input_text)
    else:
        # 使用OpenAI模型
        return emb_text_openai(client_or_text, text, model)
