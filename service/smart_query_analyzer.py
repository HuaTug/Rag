#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能查询分析器

集成Go demo的智能分析能力，提供语义理解和工具选择功能。
"""

import asyncio
import json
import logging
import re
import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import aiohttp
import time

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入原始函数和客户端
try:
    from ..core.ask_llm import get_llm_answer_deepseek as _original_llm_function, TencentDeepSeekClient
    _llm_available = True
except ImportError:
    _original_llm_function = None
    TencentDeepSeekClient = None
    _llm_available = False

# 创建适配函数
async def get_llm_answer_deepseek(query: str, search_flag: bool = False, timeout: int = 15) -> str:
    """
    适配函数，兼容原始LLM函数的调用方式
    """
    if not _llm_available:
        return f"模拟LLM响应: {query}"
    
    try:
        # 尝试从环境变量获取API密钥
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY","sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl") or os.getenv("TENCENT_API_KEY")
        
        if not api_key:
            # 如果没有API密钥，返回模拟响应
            return f"LLM分析: 需要设置DEEPSEEK_API_KEY或TENCENT_API_KEY环境变量。查询: {query[:100]}..."
        
        # 创建客户端并调用原始函数
        client = TencentDeepSeekClient(api_key=api_key)
        
        # 调用原始函数，使用适当的参数
        response = _original_llm_function(
            client=client,
            context="",  # 空上下文，让LLM基于知识库回答
            question=query,
            model="deepseek-v3-0324"
        )
        
        return response
        
    except Exception as e:
        # 如果调用失败，返回错误信息和模拟响应
        return f"LLM调用失败 ({str(e)})，使用默认分析: {query[:100]}..."


@dataclass
class ToolCall:
    """工具调用结构"""
    name: str
    args: Dict[str, Any]
    reasoning: str = ""


@dataclass
class QueryAnalysisResult:
    """查询分析结果"""
    needs_web_search: bool = False
    web_search_query: str = ""
    needs_database: bool = False
    database_query: Dict[str, Any] = None
    needs_calculation: bool = False
    calculation_args: Dict[str, Any] = None
    needs_vector_search: bool = True  # 默认启用向量搜索
    query_type: str = "general"
    confidence: float = 0.0
    reasoning: str = ""
    tool_calls: List[ToolCall] = None
    # 新增：动态搜索策略
    enable_dynamic_search: bool = True  # 启用动态搜索策略
    min_similarity_threshold: float = 0.8  # 最小相似度阈值
    
    def __post_init__(self):
        if self.database_query is None:
            self.database_query = {}
        if self.calculation_args is None:
            self.calculation_args = {}
        if self.tool_calls is None:
            self.tool_calls = []


class SmartQueryAnalyzer:
    """智能查询分析器 - 集成Go demo的分析能力"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 分析配置 - 优先使用语义分析
        self.enable_semantic_analysis = self.config.get("enable_semantic_analysis", True)
        self.fallback_to_keywords = self.config.get("fallback_to_keywords", True)
        self.analysis_timeout = self.config.get("analysis_timeout", 15)
        
        self.logger.info(f"🧠 智能查询分析器初始化完成")
        self.logger.info(f"   🎯 语义分析(LLM): {'✅ 启用' if self.enable_semantic_analysis else '❌ 禁用'}")
        self.logger.info(f"   🔄 关键词回退: {'✅ 启用' if self.fallback_to_keywords else '❌ 禁用'}")
        
        # 关键词配置（仅用于回退）
        self._init_keyword_patterns()
        
        self.logger.info(f"🧠 智能查询分析器初始化完成 - 语义分析: {self.enable_semantic_analysis}")
    
    def _init_keyword_patterns(self):
        """初始化关键词匹配模式（仅用于回退分析）"""
        # 简化的关键词配置，仅用于回退情况
        self.time_keywords = ["今天", "现在", "几号", "几月", "几点", "当前", "日期", "时间"]
        self.calculation_keywords = ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"]
        
        self.logger.info("📋 关键词模式已初始化（仅用于回退分析）")
    
    async def analyze_query_intent(self, query: str) -> QueryAnalysisResult:
        """分析查询意图 - 主入口方法"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 开始分析查询: {query}")
            
            # 优先使用语义分析（大模型分析）
            if self.enable_semantic_analysis:
                try:
                    result = await self._semantic_analysis(query)
                    analysis_time = time.time() - start_time
                    self.logger.info(f"✅ 语义分析完成 - 耗时: {analysis_time:.2f}s")
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ 语义分析失败: {e}")
                    # 如果不允许回退到关键词，直接抛出异常
                    if not self.fallback_to_keywords:
                        raise
            
            # 只有在语义分析失败且允许回退时才使用关键词匹配
            self.logger.info("🔄 回退到简化的关键词分析...")
            result = await self._fallback_analysis(query)
            analysis_time = time.time() - start_time
            self.logger.info(f"✅ 回退分析完成 - 耗时: {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 查询分析失败: {e}")
            # 返回默认分析结果
            return QueryAnalysisResult(
                needs_vector_search=True,
                query_type="general",
                reasoning=f"分析失败，使用默认策略: {str(e)}",
                confidence=0.3,
                enable_dynamic_search=True
            )
    
    async def _semantic_analysis(self, query: str) -> QueryAnalysisResult:
        """使用LLM进行语义分析"""
        prompt = self._build_analysis_prompt(query)
        
        try:
            self.logger.info(f"🤖 调用LLM进行语义分析，查询: {query[:50]}...")
            
            # 调用LLM进行分析
            response = await get_llm_answer_deepseek(
                query=prompt,
                search_flag=False,  # 禁用搜索避免循环调用
                timeout=self.analysis_timeout
            )
            
            self.logger.info(f"📝 LLM响应长度: {len(response)} 字符")
            self.logger.debug(f"📝 LLM原始响应: {response[:300]}...")
            
            # 解析分析结果
            analysis = self._parse_llm_response(response, query)
            
            # 智能填充工具参数
            self._fill_tool_parameters(analysis, query)
            
            return analysis
            
        except ValueError as e:
            # JSON解析错误，提供更详细的错误信息
            self.logger.error(f"❌ JSON解析错误: {e}")
            if "Extra data" in str(e):
                self.logger.error("💡 可能原因: LLM返回了多个JSON对象或JSON后有额外文本")
                self.logger.error("💡 建议: 检查LLM提示词，确保只返回单个JSON对象")
            raise
            
        except Exception as e:
            self.logger.error(f"❌ LLM语义分析失败: {e}")
            self.logger.error(f"❌ 错误类型: {type(e).__name__}")
            raise
    
    def _build_analysis_prompt(self, query: str) -> str:
        """构建LLM分析提示词 - 让大模型智能语义分析并生成JSON"""
        return f"""你是一个智能RAG系统的查询分析器。请通过语义分析理解用户查询的真实意图，并智能选择最适合的工具组合。

用户查询："{query}"

🔧 **可用工具说明：**

1. **web_search** - 网络搜索
   - 适用场景：时间日期查询、最新信息、实时数据、新闻
   - 示例："今天几号？"、"最新AI发展"、"现在几点？"
   
2. **vector_search** - 向量知识库检索
   - 适用场景：技术概念、定义解释、历史知识、教程
   - 示例："什么是Python？"、"如何学习编程？"
   
3. **calculation** - 数学计算
   - 适用场景：数学运算、数值计算、公式求解
   - 示例："100+200等于多少？"、"计算平方根"
   
4. **database_query** - 数据库查询
   - 适用场景：用户统计、数据分析、记录查询
   - 示例："统计用户数量"、"查询活跃用户"

🧠 **智能分析要求：**

请仔细分析查询语义，理解用户的真实需求，然后智能填充以下JSON参数：

📝 **计算查询示例分析：**
- 查询："100+100等于多少？"
- 语义分析：用户明确要求进行数学加法运算
- 应设置：needs_calculation=true, calculation_args={{"operation":"add","x":100,"y":100}}
- 推理：这是明确的数学计算问题，直接使用calculation工具，不需要搜索

⏰ **时间查询示例分析：**
- 查询："今天几号？"
- 语义分析：用户询问当前日期信息
- 应设置：needs_web_search=true, web_search_query="今天日期", needs_vector_search=false
- 推理：时间信息需要实时获取，使用web_search获取最新日期

🔍 **技术概念示例分析：**
- 查询："什么是Go语言？"
- 语义分析：用户询问技术概念定义
- 应设置：needs_vector_search=true, enable_dynamic_search=true
- 推理：技术概念优先从知识库检索，如果质量不佳则启用网络搜索

📊 **参数详细说明：**

calculation_args支持的运算类型：
- 加法：{{"operation":"add","x":数字1,"y":数字2}}
- 减法：{{"operation":"subtract","x":数字1,"y":数字2}}  
- 乘法：{{"operation":"multiply","x":数字1,"y":数字2}}
- 除法：{{"operation":"divide","x":数字1,"y":数字2}}
- 获取日期：{{"operation":"get_current_date"}}
- 表达式：{{"operation":"expression","expression":"表达式内容"}}

database_query支持的查询类型：
- 统计查询：{{"query_type":"count","table":"users","group_by":"status"}}
- 条件查询：{{"query_type":"select","table":"users","where":{{"status":"active"}},"limit":10}}

🎯 **JSON格式要求：**

请根据语义分析，智能填充所有参数，确保逻辑一致：

{{
  "needs_web_search": 布尔值,
  "web_search_query": "如果需要网络搜索，填入搜索关键词",
  "needs_vector_search": 布尔值,
  "needs_database": 布尔值,
  "database_query": {{具体的数据库查询参数}},
  "needs_calculation": 布尔值,
  "calculation_args": {{具体的计算参数}},
  "query_type": "time/technical/calculation/database/general",
  "confidence": 0.0到1.0的置信度,
  "reasoning": "详细说明你的语义分析过程和参数设置理由",
  "enable_dynamic_search": 布尔值,
  "min_similarity_threshold": 0.8
}}

⚠️ **重要规则：**
1. 通过语义理解，不是关键词匹配
2. reasoning必须详细解释为什么这样设置参数
3. 如果是计算问题，必须正确解析数字和运算符
4. 如果是时间问题，web_search_query要具体
5. 置信度要反映分析的确定程度
6. 确保参数之间逻辑一致

现在请分析上述查询，只返回JSON结果："""
    
    def _parse_llm_response(self, response: str, query: str) -> QueryAnalysisResult:
        """解析LLM响应 - 直接使用大模型的语义分析结果"""
        try:
            self.logger.info(f"🔍 解析LLM响应，长度: {len(response)} 字符")
            
            # 清理响应文本
            response = response.strip()
            
            # 多种方式提取JSON
            json_str = None
            data = None
            
            # 方法1: 尝试直接解析整个响应
            try:
                data = json.loads(response)
                json_str = response
                self.logger.info("✅ 直接解析整个响应成功")
            except json.JSONDecodeError:
                pass
            
            # 方法2: 使用更精确的正则表达式匹配完整的JSON对象
            if data is None:
                # 匹配完整的JSON对象，考虑嵌套结构
                json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                json_matches = re.findall(json_pattern, response, re.DOTALL)
                
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        json_str = match
                        self.logger.info("✅ 正则表达式匹配JSON成功")
                        break
                    except json.JSONDecodeError:
                        continue
            
            # 方法3: 查找JSON代码块（```json ... ```）
            if data is None:
                json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
                if json_block_match:
                    try:
                        json_str = json_block_match.group(1).strip()
                        data = json.loads(json_str)
                        self.logger.info("✅ JSON代码块解析成功")
                    except json.JSONDecodeError:
                        pass
            
            # 方法4: 查找第一个完整的JSON对象
            if data is None:
                brace_count = 0
                start_idx = -1
                
                for i, char in enumerate(response):
                    if char == '{':
                        if start_idx == -1:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            try:
                                json_str = response[start_idx:i+1]
                                data = json.loads(json_str)
                                self.logger.info("✅ 手动匹配JSON对象成功")
                                break
                            except json.JSONDecodeError:
                                start_idx = -1
                                continue
            
            # 如果所有方法都失败
            if data is None:
                self.logger.error(f"LLM响应中无有效JSON格式: {response[:200]}...")
                raise ValueError("无法在响应中找到有效的JSON格式")
            
            # 验证必要字段
            if not isinstance(data, dict):
                raise ValueError("JSON解析结果不是字典类型")
            
            # 🧠 直接使用LLM的语义分析结果，信任其智能判断
            result = QueryAnalysisResult(
                needs_web_search=data.get("needs_web_search", False),
                web_search_query=data.get("web_search_query", ""),
                needs_vector_search=data.get("needs_vector_search", True),
                needs_database=data.get("needs_database", False),
                database_query=data.get("database_query", {}),
                needs_calculation=data.get("needs_calculation", False),
                calculation_args=data.get("calculation_args", {}),
                query_type=data.get("query_type", "general"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "LLM语义分析结果"),
                enable_dynamic_search=data.get("enable_dynamic_search", True),
                min_similarity_threshold=float(data.get("min_similarity_threshold", 0.8))
            )
            
            # 📝 记录分析结果
            self.logger.info(f"✅ LLM语义分析成功:")
            self.logger.info(f"   🎯 查询类型: {result.query_type}")
            self.logger.info(f"   🌐 网络搜索: {result.needs_web_search}")
            self.logger.info(f"   🔍 向量搜索: {result.needs_vector_search}")
            self.logger.info(f"   🧮 数学计算: {result.needs_calculation}")
            self.logger.info(f"   📊 数据库查询: {result.needs_database}")
            self.logger.info(f"   📈 置信度: {result.confidence:.2f}")
            self.logger.info(f"   💭 推理: {result.reasoning[:100]}...")
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"❌ 解析LLM响应失败: {e}")
            self.logger.error(f"原始响应: {response[:500]}...")
            self.logger.error(f"提取的JSON: {json_str[:200] if json_str else 'None'}...")
            # 回退到简化分析
            raise ValueError(f"LLM响应解析失败: {e}")
    
    async def _fallback_analysis(self, query: str) -> QueryAnalysisResult:
        """简化的回退分析（当LLM分析失败时使用）"""
        self.logger.info("📋 使用简化回退分析...")
        
        # 默认策略：使用向量搜索 + 动态搜索策略
        analysis = QueryAnalysisResult(
            needs_vector_search=True,
            query_type="general",
            confidence=0.6,
            reasoning="LLM分析失败，使用回退策略：向量搜索+动态搜索",
            enable_dynamic_search=True,
            min_similarity_threshold=0.8
        )
        
        # 简单的模式识别（仅作为基本保障）
        query_lower = query.lower()
        
        # 明显的时间查询
        if any(word in query_lower for word in ["今天", "现在", "几号", "几月", "几点", "当前时间", "日期"]):
            analysis.needs_web_search = True
            analysis.needs_vector_search = False
            analysis.web_search_query = "当前日期时间"
            analysis.query_type = "time"
            analysis.enable_dynamic_search = False
            analysis.reasoning = "回退分析：检测到时间相关查询"
            analysis.confidence = 0.8
        
        # 明显的计算查询
        elif any(word in query_lower for word in ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"]):
            analysis.needs_calculation = True
            analysis.calculation_args = self._parse_calculation(query)
            analysis.query_type = "calculation"
            analysis.reasoning = "回退分析：检测到计算查询"
            analysis.confidence = 0.7
        
        # 如果有日期查询，添加获取当前日期的功能
        if "几号" in query_lower or "几月" in query_lower or "日期" in query_lower:
            # 添加一个特殊的计算操作来获取当前日期
            if not analysis.needs_calculation:
                analysis.needs_calculation = True
                analysis.calculation_args = {"operation": "get_current_date"}
            analysis.reasoning += " + 获取当前日期"
        
        # 填充工具参数
        self._fill_tool_parameters(analysis, query)
        
        return analysis
    
    def _fill_tool_parameters(self, analysis: QueryAnalysisResult, query: str):
        """智能填充工具参数"""
        # 填充网络搜索参数
        if analysis.needs_web_search and not analysis.web_search_query:
            analysis.web_search_query = query
        
        # 填充数据库查询参数
        if analysis.needs_database and not analysis.database_query:
            analysis.database_query = self._build_database_query(query)
        
        # 填充计算参数
        if analysis.needs_calculation and not analysis.calculation_args:
            analysis.calculation_args = self._parse_calculation(query)
        
        # 构建工具调用列表
        tool_calls = []
        
        if analysis.needs_web_search:
            tool_calls.append(ToolCall(
                name="web_search",
                args={"query": analysis.web_search_query, "limit": 5},
                reasoning="需要获取最新信息"
            ))
        
        if analysis.needs_vector_search:
            tool_calls.append(ToolCall(
                name="vector_search",
                args={"query": query, "top_k": 5},
                reasoning="需要检索相关知识"
            ))
        
        if analysis.needs_database:
            tool_calls.append(ToolCall(
                name="database_query",
                args=analysis.database_query,
                reasoning="需要查询数据库信息"
            ))
        
        if analysis.needs_calculation:
            tool_calls.append(ToolCall(
                name="calculator",
                args=analysis.calculation_args,
                reasoning="需要进行数学计算"
            ))
        
        analysis.tool_calls = tool_calls
    
    def _build_database_query(self, query: str) -> Dict[str, Any]:
        """构建数据库查询参数"""
        query_lower = query.lower()
        
        if "统计" in query_lower or "数量" in query_lower or "count" in query_lower:
            return {
                "query_type": "count",
                "table": "users",
                "group_by": "status"
            }
        
        if "活跃" in query_lower:
            return {
                "query_type": "select",
                "table": "users",
                "where": {"status": "active"},
                "limit": 10
            }
        
        # 默认查询
        return {
            "query_type": "select",
            "table": "users",
            "limit": 5
        }
    
    def _parse_calculation(self, query: str) -> Dict[str, Any]:
        """解析数学计算表达式"""
        query_lower = query.lower()
        
        # 简单的数学表达式解析
        # 在实际应用中可以使用更复杂的表达式解析器
        
        if "加" in query_lower or "+" in query:
            # 尝试提取数字
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                return {
                    "operation": "add",
                    "x": float(numbers[0]),
                    "y": float(numbers[1])
                }
        
        if "减" in query_lower or "-" in query:
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                return {
                    "operation": "subtract",
                    "x": float(numbers[0]),
                    "y": float(numbers[1])
                }
        
        if "乘" in query_lower or "*" in query or "×" in query:
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                return {
                    "operation": "multiply",
                    "x": float(numbers[0]),
                    "y": float(numbers[1])
                }
        
        if "除" in query_lower or "/" in query or "÷" in query:
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                return {
                    "operation": "divide",
                    "x": float(numbers[0]),
                    "y": float(numbers[1])
                }
        
        # 默认计算
        return {
            "operation": "expression",
            "expression": query
        }
    
    def should_use_search_engine(self, analysis: QueryAnalysisResult) -> bool:
        """判断是否应该使用搜索引擎"""
        return analysis.needs_web_search or analysis.query_type in ["time", "news", "search"]
    
    def should_use_vector_store(self, analysis: QueryAnalysisResult) -> bool:
        """判断是否应该使用向量存储"""
        return analysis.needs_vector_search or analysis.query_type in ["technical", "general"]
    
    def get_search_strategy(self, analysis: QueryAnalysisResult) -> str:
        """获取搜索策略"""
        if analysis.needs_web_search and analysis.needs_vector_search:
            return "hybrid"  # 混合策略
        elif analysis.needs_web_search:
            return "search_only"  # 仅搜索
        elif analysis.needs_vector_search:
            return "vector_only"  # 仅向量
        else:
            return "direct_llm"  # 直接LLM


# 简单的数学计算器
class SimpleCalculator:
    """简单计算器 - 处理基本数学运算"""
    
    @staticmethod
    def calculate(args: Dict[str, Any]) -> Dict[str, Any]:
        """执行计算"""
        operation = args.get("operation", "")
        
        try:
            if operation == "get_current_date":
                # 获取当前日期
                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%Y年%m月%d日")
                weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
                return {
                    "result": f"今天是{date_str}，{weekday}",
                    "expression": f"当前日期: {date_str} ({weekday})"
                }
                
            elif operation == "add":
                result = float(args["x"]) + float(args["y"])
                return {"result": result, "expression": f"{args['x']} + {args['y']} = {result}"}
            
            elif operation == "subtract":
                result = float(args["x"]) - float(args["y"])
                return {"result": result, "expression": f"{args['x']} - {args['y']} = {result}"}
            
            elif operation == "multiply":
                result = float(args["x"]) * float(args["y"])
                return {"result": result, "expression": f"{args['x']} × {args['y']} = {result}"}
            
            elif operation == "divide":
                if float(args["y"]) == 0:
                    return {"error": "除数不能为零"}
                result = float(args["x"]) / float(args["y"])
                return {"result": result, "expression": f"{args['x']} ÷ {args['y']} = {result}"}
            
            elif operation == "expression":
                # 简单的表达式计算（安全起见，只支持基本运算）
                expression = args.get("expression", "")
                # 这里可以集成更复杂的表达式解析器
                return {"result": "表达式计算功能待实现", "expression": expression}
            
            else:
                return {"error": f"不支持的运算类型: {operation}"}
        
        except Exception as e:
            return {"error": f"计算错误: {str(e)}"}


if __name__ == "__main__":
    # 测试代码
    async def test_analyzer():
        analyzer = SmartQueryAnalyzer()
        
        test_queries = [
            "今天几号？",
            "什么是Milvus向量数据库？",
            "计算15.5加上24.3的结果",
            "查询活跃用户数量",
            "最新的AI技术发展如何？"
        ]
        
        for query in test_queries:
            print(f"\n📝 测试查询: {query}")
            print("-" * 40)
            
            result = await analyzer.analyze_query_intent(query)
            
            print(f"查询类型: {result.query_type}")
            print(f"需要网络搜索: {result.needs_web_search}")
            print(f"需要向量搜索: {result.needs_vector_search}")
            print(f"需要数据库查询: {result.needs_database}")
            print(f"需要计算: {result.needs_calculation}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"推理过程: {result.reasoning}")
            print(f"工具调用数: {len(result.tool_calls)}")
    
    # 运行测试
    asyncio.run(test_analyzer())
