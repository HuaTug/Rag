"""
Embedding Entity - 向量嵌入领域实体
"""

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass
class Embedding:
    """
    向量嵌入（值对象）
    """
    vector: List[float]
    model: str = ""
    dimension: int = 0
    
    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.vector)
    
    def __len__(self) -> int:
        return self.dimension
    
    def normalize(self) -> "Embedding":
        """L2归一化"""
        import math
        norm = math.sqrt(sum(x * x for x in self.vector))
        if norm > 0:
            normalized = [x / norm for x in self.vector]
            return Embedding(vector=normalized, model=self.model, dimension=self.dimension)
        return self


@dataclass
class EmbeddingBatch:
    """
    批量向量嵌入
    """
    id: UUID = field(default_factory=uuid4)
    embeddings: List[Embedding] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    model: str = ""
    
    @property
    def size(self) -> int:
        return len(self.embeddings)
    
    def get_vectors(self) -> List[List[float]]:
        """获取所有向量"""
        return [emb.vector for emb in self.embeddings]
