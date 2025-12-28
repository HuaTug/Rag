"""
DeepSeek LLM Adapter - DeepSeek大语言模型适配器

实现LLMService接口
"""

import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp
import requests

from src.domain.ports.services import LLMService, LLMConfig, LLMResponse


logger = logging.getLogger(__name__)


class DeepSeekLLM(LLMService):
    """
    DeepSeek LLM服务实现
    
    支持DeepSeek API和腾讯云DeepSeek
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        config: Optional[LLMConfig] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.config = config or LLMConfig()
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        logger.info(f"初始化DeepSeek LLM: {self.config.model}")
    
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
        
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API请求失败: {response.status} - {error_text}")
                    
                    result = await response.json()
        except aiohttp.ClientError as e:
            # 回退到同步请求
            logger.warning(f"异步请求失败，尝试同步请求: {e}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout,
            )
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            result = response.json()
        
        latency = (time.time() - start_time) * 1000
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=result.get("model", self.config.model),
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=result["choices"][0].get("finish_reason", ""),
            latency_ms=latency,
        )
    
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
        
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except:
                            continue
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        prompt = f"""请分析以下用户查询的意图，返回JSON格式：

查询：{query}

请返回以下格式的JSON：
{{
    "intent": "factual|analytical|comparative|procedural|exploratory",
    "keywords": ["关键词1", "关键词2"],
    "entities": ["实体1", "实体2"],
    "needs_web_search": true/false,
    "needs_calculation": true/false,
    "confidence": 0.0-1.0
}}

只返回JSON，不要其他内容："""
        
        response = await self.generate(prompt)
        
        try:
            import json
            # 提取JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except:
            return {
                "intent": "factual",
                "keywords": [],
                "entities": [],
                "needs_web_search": False,
                "needs_calculation": False,
                "confidence": 0.5,
            }
    
    async def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """重写查询"""
        prompt = f"""请优化以下查询，使其更适合检索：

原始查询：{query}
"""
        if context:
            prompt += f"\n上下文：{context}\n"
        
        prompt += "\n请直接返回优化后的查询，不要其他内容："
        
        response = await self.generate(prompt)
        return response.content.strip()
    
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5,
            )
            return response.status_code == 200
        except:
            return False
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.config.model
