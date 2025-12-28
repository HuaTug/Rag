"""
OpenAI LLM Adapter - OpenAI大语言模型适配器

实现LLMService接口
"""

import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

from src.domain.ports.services import LLMService, LLMConfig, LLMResponse


logger = logging.getLogger(__name__)


class OpenAILLM(LLMService):
    """
    OpenAI LLM服务实现
    
    支持OpenAI API和兼容的API
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.config = config or LLMConfig(model=model)
        self._client = None
        
        logger.info(f"初始化OpenAI LLM: {model}")
    
    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """生成文本"""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            latency = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI生成失败: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        prompt = f"""分析以下查询的意图，返回JSON：

查询：{query}

返回格式：
{{
    "intent": "factual|analytical|comparative|procedural|exploratory",
    "keywords": ["关键词"],
    "entities": ["实体"],
    "needs_web_search": true/false,
    "confidence": 0.0-1.0
}}"""
        
        response = await self.generate(prompt)
        
        try:
            import json
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content)
        except:
            return {"intent": "factual", "keywords": [], "confidence": 0.5}
    
    async def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """重写查询"""
        prompt = f"优化查询：{query}"
        if context:
            prompt += f"\n上下文：{context}"
        prompt += "\n直接返回优化后的查询："
        
        response = await self.generate(prompt)
        return response.content.strip()
    
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            self.client.models.list()
            return True
        except:
            return False
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model
