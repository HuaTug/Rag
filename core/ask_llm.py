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
    # Define enhanced system and user prompts
    SYSTEM_PROMPT = """
    你是一个专业的智能研究助手，具备以下能力和特点：

    **核心职责：**
    1. 基于提供的上下文信息进行准确、详细的回答
    2. 当上下文信息不足时，基于你的专业知识提供有价值的补充
    3. 保持客观、准确、有条理的回答风格

    **回答要求：**
    - 优先使用上下文信息，引用具体内容时请标注来源
    - 区分已知信息和推测内容，明确标识信息来源
    - 提供结构化、逻辑清晰的回答
    - 对于复杂问题，请分步骤或分要点解答
    - 如果信息不确定，请如实说明并提供可能的解释

    **特殊情况处理：**
    - 上下文为空：基于知识库回答，并说明信息来源
    - 上下文不完整：结合上下文和知识提供综合回答
    - 上下文矛盾：指出矛盾并提供平衡的观点
    - 敏感话题：保持中立，提供事实性信息

    请始终保持专业、准确、有帮助的回答风格。
    """
    
    # 检查是否有有效的上下文
    if not context.strip():
        USER_PROMPT = f"""
        **查询情况：** 当前没有相关的上下文信息

        **任务要求：**
        1. 请基于你的知识库回答以下问题
        2. 请在回答中明确说明信息来源于AI知识库
        3. 如果涉及实时信息，请提醒用户验证最新数据
        4. 提供结构化的详细回答

        **用户问题：**
        <question>
        {question}
        </question>

        **回答格式要求：**
        - 如果是事实性问题，请提供准确信息并标注知识截止时间
        - 如果是分析性问题，请提供多角度分析
        - 如果需要最新信息，请明确提醒用户
        """
    else:
        USER_PROMPT = f"""
        **查询情况：** 已找到相关的上下文信息

        **任务要求：**
        1. 仔细分析提供的上下文信息
        2. 优先基于上下文回答问题
        3. 如果上下文信息不完整，可以适当补充相关知识
        4. 在回答中明确区分哪些来自上下文，哪些来自知识库

        **上下文信息：**
        <context>
        {context}
        </context>

        **用户问题：**
        <question>
        {question}
        </question>

        **回答格式要求：**
        - 开始时简要说明找到的相关信息
        - 基于上下文提供详细回答
        - 如需补充信息，请明确标注
        - 如果上下文信息与问题不完全匹配，请说明
        - 最后可以提供相关的延伸信息或建议

        **信息质量评估：**
        请在回答时评估上下文信息的相关性和完整性，并在必要时说明。
        """

    # 调用API时添加更多参数控制
    response = client.chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        stream=False,
        enable_search=False  # 禁用模型自带搜索，避免与RAG系统冲突
    )

    answer = response['choices'][0]['message']['content']
    return answer