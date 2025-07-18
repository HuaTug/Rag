import requests
import json
from openai import OpenAI


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


def get_llm_answer_deepseek(client: TencentDeepSeekClient, context: str, question: str, model: str = "deepseek-v3-0324", min_distance_threshold: float = 0.5):
    # Define system and user prompts
    SYSTEM_PROMPT = """
    你是一个智能助手。你可以根据提供的上下文信息回答问题，如果上下文中没有相关信息，你也可以基于你的知识给出有帮助的回答。
    请优先使用上下文信息，如果上下文不包含答案，请明确说明并基于你的知识提供帮助。
    """
    
    # 检查是否有有效的上下文
    if not context.strip():
        USER_PROMPT = f"""
        上下文信息为空，请基于你的知识回答以下问题：
        <question>
        {question}
        </question>
        """
    else:
        USER_PROMPT = f"""
        请使用以下上下文信息回答问题。如果上下文中没有相关信息，请说明并基于你的知识提供帮助。
        
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

    response = client.chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        stream=False
    )

    answer = response['choices'][0]['message']['content']
    return answer


def get_llm_answer(client: OpenAI, context: str, question: str, model: str = "gpt-4o", min_distance_threshold: float = 0.5):
    # Define system and user prompts
    SYSTEM_PROMPT = """
    你是一个智能助手。你可以根据提供的上下文信息回答问题，如果上下文中没有相关信息，你也可以基于你的知识给出有帮助的回答。
    请优先使用上下文信息，如果上下文不包含答案，请明确说明并基于你的知识提供帮助。
    """
    
    # 检查是否有有效的上下文
    if not context.strip():
        USER_PROMPT = f"""
        上下文信息为空，请基于你的知识回答以下问题：
        <question>
        {question}
        </question>
        """
    else:
        USER_PROMPT = f"""
        请使用以下上下文信息回答问题。如果上下文中没有相关信息，请说明并基于你的知识提供帮助。
        
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    answer = response.choices[0].message.content
    return answer
