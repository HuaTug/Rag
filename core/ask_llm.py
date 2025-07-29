import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TencentDeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "http://api.lkeap.cloud.tencent.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    
    def chat_completions_create(self, model: str, messages: list, stream: bool = False, enable_search: bool = True):
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "extra_body": {
                "enable_search": enable_search
            }
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            if stream:
                return response.iter_lines()
            else:
                result = response.json()
                return result
        else:
            raise Exception(f"API调用失败: {response.status_code}, {response.text}")


def get_llm_answer_with_prompt(client: TencentDeepSeekClient, prompt: str, model: str = "deepseek-v3-0324"):
    """
    使用预构建的prompt直接调用LLM
    
    Args:
        client: DeepSeek客户端
        prompt: 预构建的完整prompt
        model: 模型名称
    
    Returns:
        str: LLM生成的答案
    """
    try:
        response = client.chat_completions_create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个智能助手，请根据用户提供的信息和要求准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            enable_search=True
        )
        
        if response and "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return "抱歉，无法生成回答"
            
    except Exception as e:
        return f"调用LLM时出现错误: {str(e)}"