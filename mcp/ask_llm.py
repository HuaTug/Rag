#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM客户端 - 支持DeepSeek和OpenAI
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 尝试导入OpenAI客户端
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not available")

# 尝试导入requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests package not available")

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    model: str
    usage: Dict[str, Any] = None
    finish_reason: str = "stop"
    error: Optional[str] = None

class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """聊天补全接口"""
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, base_url, model)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAIClient")
        
        # 初始化OpenAI客户端
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        logger.info(f"初始化OpenAI客户端: {model}")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """OpenAI聊天补全"""
        try:
            # 设置默认参数
            params = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "stream": False
            }
            
            response = self.client.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump() if hasattr(response.usage, 'model_dump') else dict(response.usage),
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )

class DeepSeekClient(BaseLLMClient):
    """DeepSeek客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        super().__init__(api_key, base_url, model)
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required for DeepSeekClient")
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"初始化DeepSeek客户端: {model}")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """DeepSeek聊天补全"""
        try:
            # 构建请求数据
            payload = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", "stop")
                )
            else:
                error_msg = f"DeepSeek API错误: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return LLMResponse(
                    content="",
                    model=self.model,
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )

class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端，用于测试"""
    
    def __init__(self, model: str = "mock-model"):
        super().__init__("mock-key", "mock-url", model)
        logger.info(f"初始化模拟LLM客户端: {model}")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """模拟聊天补全"""
        # 基于用户消息生成模拟回复
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        # 简单的模拟逻辑
        if "天气" in user_content:
            mock_response = "今天天气不错，适合出门活动。"
        elif "时间" in user_content:
            mock_response = "现在是2024年，具体时间请查看系统时钟。"
        elif "搜索" in user_content or "查找" in user_content:
            mock_response = "根据搜索结果，我找到了以下相关信息：这是一个模拟的搜索结果，包含了您需要的基本信息。"
        else:
            mock_response = f"这是对您问题的模拟回复：'{user_content[:50]}...'。在实际环境中，这里会是AI模型的真实回复。"
        
        return LLMResponse(
            content=mock_response,
            model=self.model,
            usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            finish_reason="stop"
        )

def create_llm_client(provider: str = "mock", **kwargs) -> BaseLLMClient:
    """
    创建LLM客户端
    
    Args:
        provider: 提供商 ("openai", "deepseek", "mock")
        **kwargs: 客户端参数
        
    Returns:
        LLM客户端实例
    """
    provider = provider.lower()
    
    try:
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI客户端不可用，使用模拟客户端")
                return MockLLMClient()
            
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("未提供OpenAI API密钥，使用模拟客户端")
                return MockLLMClient()
            
            return OpenAIClient(
                api_key=api_key,
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model", "gpt-3.5-turbo")
            )
            
        elif provider == "deepseek":
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests库不可用，使用模拟客户端")
                return MockLLMClient()
            
            api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.warning("未提供DeepSeek API密钥，使用模拟客户端")
                return MockLLMClient()
            
            return DeepSeekClient(
                api_key=api_key,
                base_url=kwargs.get("base_url", "https://api.deepseek.com/v1"),
                model=kwargs.get("model", "deepseek-chat")
            )
            
        else:
            logger.info("使用模拟LLM客户端")
            return MockLLMClient(model=kwargs.get("model", "mock-model"))
            
    except Exception as e:
        logger.error(f"创建LLM客户端失败: {e}")
        logger.info("回退到模拟客户端")
        return MockLLMClient()

def ask_llm(question: str, context: str = "", provider: str = "mock", **kwargs) -> str:
    """
    向LLM提问的便捷函数
    
    Args:
        question: 用户问题
        context: 上下文信息
        provider: LLM提供商
        **kwargs: 额外参数
        
    Returns:
        LLM回复内容
    """
    try:
        client = create_llm_client(provider, **kwargs)
        
        # 构建消息
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"请基于以下上下文信息回答用户问题：\n\n{context}"
            })
        
        messages.append({
            "role": "user", 
            "content": question
        })
        
        # 调用LLM
        response = client.chat_completion(messages, **kwargs)
        
        if response.error:
            logger.error(f"LLM调用出错: {response.error}")
            return f"抱歉，处理您的问题时出现错误：{response.error}"
        
        return response.content
        
    except Exception as e:
        logger.error(f"ask_llm函数执行失败: {e}")
        return f"抱歉，无法处理您的问题：{str(e)}"

if __name__ == "__main__":
    # 测试LLM客户端
    print("测试LLM客户端...")
    
    # 测试模拟客户端
    mock_client = create_llm_client("mock")
    print(f"模拟客户端: {type(mock_client).__name__}")
    
    # 测试对话
    test_messages = [
        {"role": "user", "content": "今天天气怎么样？"}
    ]
    
    response = mock_client.chat_completion(test_messages)
    print(f"模拟回复: {response.content}")
    
    # 测试便捷函数
    answer = ask_llm("什么是人工智能？", provider="mock")
    print(f"便捷函数回复: {answer}")
    
    # 测试带上下文的问答
    context = "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。"
    answer_with_context = ask_llm("AI有什么应用？", context=context, provider="mock")
    print(f"带上下文回复: {answer_with_context}")
