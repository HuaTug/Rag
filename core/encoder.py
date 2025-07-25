#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本嵌入模块

提供统一的文本嵌入接口，支持开源模型和OpenAI模型。
移除了Streamlit依赖，使用标准的缓存机制。
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional, Union

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingCache:
    """嵌入向量缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
        """
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[List[float]]:
        """获取缓存的嵌入向量"""
        if key in self.cache:
            # 更新访问顺序
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: List[float]):
        """设置缓存的嵌入向量"""
        # 如果缓存已满，删除最久未访问的条目
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()


# 全局缓存实例
_embedding_cache = EmbeddingCache()


@lru_cache(maxsize=1)
def get_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """
    获取sentence-transformers模型（带缓存）
    
    Args:
        model_name: 模型名称
        
    Returns:
        SentenceTransformer模型实例或None
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
    
    try:
        # 使用多语言支持的嵌入模型
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(f"无法加载sentence-transformers模型 {model_name}: {e}")


def emb_text_opensource(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    使用开源模型进行文本嵌入
    
    Args:
        text: 要嵌入的文本
        model_name: 模型名称
        
    Returns:
        嵌入向量列表
        
    Raises:
        ImportError: 当sentence-transformers未安装时
        RuntimeError: 当模型加载失败时
    """
    # 检查缓存
    cache_key = f"{model_name}:{text}"
    cached_embedding = _embedding_cache.get(cache_key)
    if cached_embedding is not None:
        return cached_embedding
    
    # 获取模型
    model = get_sentence_transformer_model(model_name)
    if model is None:
        raise RuntimeError("无法加载sentence-transformers模型")
    
    # 生成嵌入向量
    embedding = model.encode(text).tolist()
    
    # 缓存结果
    _embedding_cache.set(cache_key, embedding)
    
    return embedding


def emb_text_openai(
    client: OpenAI, 
    text: str, 
    model: str = "text-embedding-3-small"
) -> List[float]:
    """
    使用OpenAI进行文本嵌入
    
    Args:
        client: OpenAI客户端实例
        text: 要嵌入的文本
        model: OpenAI嵌入模型名称
        
    Returns:
        嵌入向量列表
        
    Raises:
        ImportError: 当openai库未安装时
        Exception: 当API调用失败时
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("请安装 openai: pip install openai")
    
    # 检查缓存
    cache_key = f"openai:{model}:{text}"
    cached_embedding = _embedding_cache.get(cache_key)
    if cached_embedding is not None:
        return cached_embedding
    
    try:
        # 调用OpenAI API
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        
        # 缓存结果
        _embedding_cache.set(cache_key, embedding)
        
        return embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI嵌入API调用失败: {e}")


def emb_text(
    client_or_text: Union[OpenAI, str], 
    text: Optional[str] = None, 
    model: str = "auto"
) -> List[float]:
    """
    统一的嵌入接口，自动选择可用的嵌入方法
    
    Args:
        client_or_text: 如果是字符串则直接使用开源模型，如果是OpenAI客户端则使用OpenAI
        text: 要嵌入的文本（当第一个参数是客户端时使用）
        model: 模型选择，"auto"为自动选择
        
    Returns:
        嵌入向量列表
        
    Raises:
        ValueError: 当参数不正确时
        ImportError: 当所需库未安装时
        RuntimeError: 当嵌入生成失败时
    """
    # 判断是否使用开源模型
    use_opensource = os.getenv("USE_OPENSOURCE_EMBEDDING", "true").lower() == "true"
    
    if use_opensource or isinstance(client_or_text, str):
        # 使用开源模型
        input_text = client_or_text if isinstance(client_or_text, str) else text
        if input_text is None:
            raise ValueError("文本参数不能为空")
        
        # 选择模型
        if model == "auto":
            model_name = "all-MiniLM-L6-v2"  # 默认轻量级英文模型
        else:
            model_name = model
        
        return emb_text_opensource(input_text, model_name)
    else:
        # 使用OpenAI模型
        if not isinstance(client_or_text, OpenAI):
            raise ValueError("第一个参数必须是OpenAI客户端实例")
        if text is None:
            raise ValueError("文本参数不能为空")
        
        # 选择模型
        if model == "auto":
            openai_model = "text-embedding-3-small"  # 默认OpenAI模型
        else:
            openai_model = model
        
        return emb_text_openai(client_or_text, text, openai_model)


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """
    获取嵌入向量的维度
    
    Args:
        model_name: 模型名称
        
    Returns:
        嵌入向量维度
    """
    # 常见模型的维度映射
    dimension_map = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    return dimension_map.get(model_name, 384)  # 默认返回384


def clear_embedding_cache():
    """清空嵌入向量缓存"""
    _embedding_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """
    获取缓存统计信息
    
    Returns:
        包含缓存统计信息的字典
    """
    return {
        "cache_size": len(_embedding_cache.cache),
        "max_size": _embedding_cache.max_size,
        "hit_rate": len(_embedding_cache.access_order) / max(1, len(_embedding_cache.cache))
    }


# 便捷函数
def encode_text(text: str, use_openai: bool = False, **kwargs) -> List[float]:
    """
    便捷的文本编码函数
    
    Args:
        text: 要编码的文本
        use_openai: 是否使用OpenAI（需要设置相应的环境变量）
        **kwargs: 其他参数
        
    Returns:
        嵌入向量列表
    """
    if use_openai and OPENAI_AVAILABLE:
        # 尝试使用OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            return emb_text(client, text, kwargs.get("model", "auto"))
    
    # 使用开源模型
    return emb_text(text, model=kwargs.get("model", "auto"))


if __name__ == "__main__":
    # 测试代码
    test_text = "这是一个测试文本"
    
    print("🧪 测试文本嵌入功能...")
    
    try:
        # 测试开源模型
        print("测试开源模型...")
        embedding = emb_text(test_text)
        print(f"✅ 开源模型嵌入成功，维度: {len(embedding)}")
        
        # 测试缓存
        print("测试缓存功能...")
        embedding2 = emb_text(test_text)
        print(f"✅ 缓存测试成功，结果一致: {embedding == embedding2}")
        
        # 显示缓存统计
        stats = get_cache_stats()
        print(f"📊 缓存统计: {stats}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
