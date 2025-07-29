#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP工具集成模块

为RAG系统提供MCP服务器工具调用能力
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import sys
import os

@dataclass
class MCPToolCall:
    """MCP工具调用结果"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


class MCPToolClient:
    """MCP工具客户端"""
    
    def __init__(self, mcp_server_path: str = None):
        """
        初始化MCP工具客户端
        
        Args:
            mcp_server_path: MCP服务器可执行文件路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mcp_server_path = mcp_server_path or "/data/workspace/MCP/mcp-server"
        self.process = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_running = False
        self._lock = threading.Lock()
        self.request_id = 0
        
        # 可用工具定义
        self.available_tools = {
            "calculator": {
                "description": "执行基本数学运算",
                "parameters": {
                    "operation": {"type": "string", "required": True, "enum": ["add", "subtract", "multiply", "divide"]},
                    "x": {"type": "number", "required": True},
                    "y": {"type": "number", "required": True}
                }
            },
            "database_query": {
                "description": "执行数据库查询，支持原始SQL和结构化查询",
                "parameters": {
                    "query_type": {"type": "string", "default": "raw", "enum": ["raw", "structured", "model"]},
                    "query": {"type": "string", "required": True},
                    "database": {"type": "string", "default": "default"},
                    "table_name": {"type": "string"},
                    "fields": {"type": "string", "default": "*"},
                    "where_conditions": {"type": "string"},
                    "order_by": {"type": "string"},
                    "limit": {"type": "number"},
                    "offset": {"type": "number"}
                }
            },
            "web_search": {
                "description": "网络搜索",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "number", "default": 10}
                }
            }
        }
        
        self.logger.info("MCP工具客户端初始化完成")
    
    async def start_server(self) -> bool:
        """启动MCP服务器"""
        try:
            if self.is_running:
                self.logger.warning("MCP服务器已在运行")
                return True
            
            if not os.path.exists(self.mcp_server_path):
                self.logger.error(f"MCP服务器文件不存在: {self.mcp_server_path}")
                return False
            
            # 启动MCP服务器进程
            self.process = subprocess.Popen(
                [self.mcp_server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.is_running = True
            
            # 启动IO处理线程
            threading.Thread(target=self._handle_output, daemon=True).start()
            threading.Thread(target=self._handle_input, daemon=True).start()
            
            self.logger.info("MCP服务器启动成功")
            
            # 等待服务器初始化
            await asyncio.sleep(1)
            
            # 发送初始化消息
            init_success = await self._initialize_connection()
            if not init_success:
                self.logger.error("MCP服务器初始化失败")
                await self.stop_server()
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动MCP服务器失败: {e}")
            return False
    
    async def stop_server(self):
        """停止MCP服务器"""
        try:
            self.is_running = False
            
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
            
            self.logger.info("MCP服务器已停止")
            
        except Exception as e:
            self.logger.error(f"停止MCP服务器失败: {e}")
    
    def _handle_output(self):
        """处理服务器输出"""
        while self.is_running and self.process:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self.output_queue.put(data)
                        except json.JSONDecodeError:
                            # 可能是日志或其他非JSON输出
                            self.logger.debug(f"非JSON输出: {line}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"处理输出时出错: {e}")
                break
    
    def _handle_input(self):
        """处理输入队列"""
        while self.is_running and self.process:
            try:
                if not self.input_queue.empty():
                    message = self.input_queue.get(timeout=1)
                    if self.process and self.process.stdin:
                        json_str = json.dumps(message)
                        self.process.stdin.write(json_str + '\n')
                        self.process.stdin.flush()
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"处理输入时出错: {e}")
                break
    
    async def _initialize_connection(self) -> bool:
        """初始化MCP连接"""
        try:
            # 发送初始化请求
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "RAG-MCP-Client",
                        "version": "1.0.0"
                    }
                }
            }
            
            self.input_queue.put(init_request)
            
            # 等待响应
            for _ in range(50):  # 最多等待5秒
                try:
                    response = self.output_queue.get(timeout=0.1)
                    if response.get("id") == init_request["id"]:
                        if "result" in response:
                            self.logger.info("MCP连接初始化成功")
                            return True
                        else:
                            self.logger.error(f"初始化失败: {response}")
                            return False
                except queue.Empty:
                    continue
            
            self.logger.error("初始化超时")
            return False
            
        except Exception as e:
            self.logger.error(f"初始化连接失败: {e}")
            return False
    
    def _get_request_id(self) -> int:
        """获取请求ID"""
        with self._lock:
            self.request_id += 1
            return self.request_id
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolCall:
        """调用MCP工具"""
        start_time = time.time()
        
        try:
            if not self.is_running:
                return MCPToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    success=False,
                    error="MCP服务器未运行",
                    execution_time=time.time() - start_time
                )
            
            # 验证工具是否存在
            if tool_name not in self.available_tools:
                return MCPToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    success=False,
                    error=f"未知工具: {tool_name}",
                    execution_time=time.time() - start_time
                )
            
            # 构建工具调用请求
            request_id = self._get_request_id()
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            self.logger.info(f"调用MCP工具: {tool_name} with {arguments}")
            
            # 发送请求
            self.input_queue.put(request)
            
            # 等待响应
            for _ in range(100):  # 最多等待10秒
                try:
                    response = self.output_queue.get(timeout=0.1)
                    if response.get("id") == request_id:
                        execution_time = time.time() - start_time
                        
                        if "result" in response:
                            result = response["result"]
                            return MCPToolCall(
                                tool_name=tool_name,
                                arguments=arguments,
                                result=result,
                                success=True,
                                execution_time=execution_time
                            )
                        elif "error" in response:
                            error = response["error"]
                            return MCPToolCall(
                                tool_name=tool_name,
                                arguments=arguments,
                                result=None,
                                success=False,
                                error=str(error),
                                execution_time=execution_time
                            )
                except queue.Empty:
                    continue
            
            # 超时
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                error="调用超时",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取可用工具列表"""
        return self.available_tools.copy()
    
    def is_tool_available(self, tool_name: str) -> bool:
        """检查工具是否可用"""
        return tool_name in self.available_tools


class MCPToolIntegration:
    """MCP工具集成管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化MCP工具集成
        
        Args:
            config: 配置信息
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = MCPToolClient(
            mcp_server_path=self.config.get("mcp_server_path")
        )
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化MCP工具集成"""
        try:
            success = await self.client.start_server()
            if success:
                self.is_initialized = True
                self.logger.info("MCP工具集成初始化成功")
                return True
            else:
                self.logger.error("MCP工具集成初始化失败")
                return False
        except Exception as e:
            self.logger.error(f"MCP工具集成初始化异常: {e}")
            return False
    
    async def shutdown(self):
        """关闭MCP工具集成"""
        try:
            await self.client.stop_server()
            self.is_initialized = False
            self.logger.info("MCP工具集成已关闭")
        except Exception as e:
            self.logger.error(f"关闭MCP工具集成失败: {e}")
    
    async def execute_calculation(self, operation: str, x: float, y: float) -> MCPToolCall:
        """执行数学计算"""
        return await self.client.call_tool("calculator", {
            "operation": operation,
            "x": x,
            "y": y
        })
    
    async def execute_database_query(self, 
                                   query: str,
                                   query_type: str = "raw",
                                   database: str = "default",
                                   **kwargs) -> MCPToolCall:
        """执行数据库查询"""
        arguments = {
            "query": query,
            "query_type": query_type,
            "database": database
        }
        arguments.update(kwargs)
        
        return await self.client.call_tool("database_query", arguments)
    
    async def execute_web_search(self, query: str, limit: int = 10) -> MCPToolCall:
        """执行网络搜索"""
        return await self.client.call_tool("web_search", {
            "query": query,
            "limit": limit
        })
    
    async def call_tool_by_name(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolCall:
        """通过工具名称调用工具"""
        if not self.is_initialized:
            return MCPToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                error="MCP工具集成未初始化"
            )
        
        return await self.client.call_tool(tool_name, arguments)
    
    def get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """获取工具定义"""
        return self.client.get_available_tools()
    
    def suggest_tools_for_query(self, query: str) -> List[str]:
        """根据查询建议合适的工具"""
        query_lower = query.lower()
        suggested_tools = []
        
        # 数学计算关键词
        math_keywords = ["计算", "加", "减", "乘", "除", "+", "-", "*", "/", "等于", "结果"]
        if any(keyword in query_lower for keyword in math_keywords):
            suggested_tools.append("calculator")
        
        # 数据库查询关键词
        db_keywords = ["查询", "数据库", "表", "用户", "统计", "count", "select", "数据"]
        if any(keyword in query_lower for keyword in db_keywords):
            suggested_tools.append("database_query")
        
        # 搜索关键词
        search_keywords = ["搜索", "查找", "最新", "新闻", "信息", "资料"]
        if any(keyword in query_lower for keyword in search_keywords):
            suggested_tools.append("web_search")
        
        return suggested_tools


# 使用示例
async def demo_mcp_integration():
    """演示MCP工具集成使用"""
    # 创建工具集成实例
    mcp_integration = MCPToolIntegration()
    
    try:
        # 初始化
        success = await mcp_integration.initialize()
        if not success:
            print("❌ MCP工具集成初始化失败")
            return
        
        print("✅ MCP工具集成初始化成功")
        
        # 测试计算器工具
        calc_result = await mcp_integration.execute_calculation("add", 10, 5)
        print(f"🧮 计算结果: {calc_result.result if calc_result.success else calc_result.error}")
        
        # 测试数据库查询
        db_result = await mcp_integration.execute_database_query(
            query="select",
            query_type="structured",
            table_name="users",
            limit=5
        )
        print(f"🗄️ 数据库查询: {db_result.result if db_result.success else db_result.error}")
        
        # 测试网络搜索
        search_result = await mcp_integration.execute_web_search("Python编程", 3)
        print(f"🔍 搜索结果: {search_result.result if search_result.success else search_result.error}")
        
    finally:
        # 清理
        await mcp_integration.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_mcp_integration())