import os
import requests

# 从环境变量获取API密钥和搜索引擎ID，若无则使用提供的默认值
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCJffa8kg0c1_Ef7zl18QUMZVvqGwBVtrM")
search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "e6676dbfd052c4ecf")

def test_google_search_api():
    """测试Google Custom Search API的可用性"""
    base_url = "https://www.googleapis.com/customsearch/v1"
    
    # 定义要询问的多个问题
    questions = [
        "美国总统是谁",
        "中国的首都是哪里",
        "世界上最高的山峰是什么",
        "新冠疫情开始于哪一年",
        "Python编程语言是谁发明的",
        "2024年奥运会将在哪里举行"
    ]
    
    for idx, question in enumerate(questions, 1):
        print(f"\n==== 问题 {idx}: {question} ====")
        
        # 搜索参数 - 修改num为6以获取6个结果
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": question,
            "num": 6  # 请求6个结果
        }
        
        try:
            # 发送请求
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # 检查响应状态码
            if response.status_code == 200:
                # 成功响应
                if "items" in data and len(data["items"]) > 0:
                    print(f"✅ API密钥有效，成功获取 {len(data['items'])} 个搜索结果！")
                    for i, item in enumerate(data["items"], 1):
                        print(f"\n回答 {i}:")
                        print(f"  标题: {item['title']}")
                        print(f"  来源: {item['link']}")
                        if "snippet" in item:
                            print(f"  摘要: {item['snippet'][:100]}...")  # 限制摘要长度
                else:
                    print("⚠️ API密钥有效，但搜索结果为空（可能是搜索词或搜索引擎配置问题）。")
                    print(f"响应摘要: {data.get('searchInformation', {})}")
            else:
                # 错误响应
                error_msg = data.get("error", {}).get("message", "未知错误")
                print(f"❌ API请求失败，状态码: {response.status_code}")
                print(f"错误信息: {error_msg}")
                
                # 常见错误分析
                if response.status_code == 403:
                    print("可能原因: API密钥无效、未启用API、配额已用完或IP被封锁。")
                    break  # 如果认证失败，停止后续请求
                elif response.status_code == 400:
                    print("可能原因: 搜索引擎ID无效或参数格式错误。")
                    break  # 如果参数错误，停止后续请求
                    
        except Exception as e:
            print(f"发生异常: {str(e)}")
            break  # 发生异常时停止后续请求

if __name__ == "__main__":
    test_google_search_api()    