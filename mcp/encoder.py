#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本嵌入编码器
支持多种嵌入模型
"""

import os
import logging
from typing import List
import numpy as np

# 尝试导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using dummy encoder")

logger = logging.getLogger(__name__)

# 全局模型实例
_model = None

def get_embedding_model():
    """获取嵌入模型实例"""
    global _model
    
    if _model is None:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # 使用轻量级的中文嵌入模型
                model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
                _model = SentenceTransformer(model_name)
                logger.info(f"加载嵌入模型: {model_name}")
            except Exception as e:
                logger.warning(f"加载sentence_transformers模型失败: {e}")
                _model = DummyEncoder()
        else:
            _model = DummyEncoder()
    
    return _model

class DummyEncoder:
    """虚拟编码器，用于测试"""
    
    def encode(self, texts, **kwargs):
        """生成随机向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 生成384维的随机向量
        vectors = []
        for text in texts:
            # 基于文本内容生成伪随机向量
            np.random.seed(hash(text) % (2**32))
            vector = np.random.normal(0, 1, 384).astype(np.float32)
            # 归一化
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        
        return np.array(vectors) if len(vectors) > 1 else vectors[0]

def emb_text(text: str) -> List[float]:
    """
    对文本进行向量化编码
    
    Args:
        text: 输入文本
        
    Returns:
        384维向量列表
    """
    if not text or not text.strip():
        # 返回零向量
        return [0.0] * 384
    
    try:
        model = get_embedding_model()
        
        if isinstance(model, DummyEncoder):
            vector = model.encode(text)
        else:
            # sentence_transformers模型
            vector = model.encode(text, convert_to_tensor=False)
        
        # 确保返回Python列表
        if hasattr(vector, 'tolist'):
            return vector.tolist()
        elif isinstance(vector, np.ndarray):
            return vector.tolist()
        else:
            return list(vector)
            
    except Exception as e:
        logger.error(f"文本编码失败: {e}")
        # 返回零向量作为fallback
        return [0.0] * 384

def batch_emb_text(texts: List[str]) -> List[List[float]]:
    """
    批量对文本进行向量化编码
    
    Args:
        texts: 文本列表
        
    Returns:
        向量列表
    """
    if not texts:
        return []
    
    try:
        model = get_embedding_model()
        
        if isinstance(model, DummyEncoder):
            vectors = model.encode(texts)
        else:
            vectors = model.encode(texts, convert_to_tensor=False)
        
        # 转换为Python列表
        result = []
        for vector in vectors:
            if hasattr(vector, 'tolist'):
                result.append(vector.tolist())
            elif isinstance(vector, np.ndarray):
                result.append(vector.tolist())
            else:
                result.append(list(vector))
        
        return result
        
    except Exception as e:
        logger.error(f"批量文本编码失败: {e}")
        # 返回零向量列表作为fallback
        return [[0.0] * 384 for _ in texts]

# 兼容性函数
def encode_text(text: str) -> List[float]:
    """兼容性函数"""
    return emb_text(text)

if __name__ == "__main__":
    # 测试编码器
    print("测试文本编码器...")
    
    test_texts = [
        "这是一个测试文本",
        "人工智能是未来的发展方向",
        "Python是一种优秀的编程语言"
    ]
    
    print(f"嵌入模型: {type(get_embedding_model()).__name__}")
    
    for text in test_texts:
        vector = emb_text(text)
        print(f"文本: {text}")
        print(f"向量维度: {len(vector)}")
        print(f"向量前5个值: {vector[:5]}")
        print("-" * 50)
    
    # 测试批量编码
    print("\n测试批量编码...")
    batch_vectors = batch_emb_text(test_texts)
    print(f"批量编码结果: {len(batch_vectors)} 个向量")
