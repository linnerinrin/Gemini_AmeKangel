#"""
#架构作用：MCP客户端端口（McpClientPort）的默认实现与测试实现
#调用方：tool_bridge.build_mcp_tool_specs与未来接入适配层；测试注入FakeMcpClient
#数据流：接入适配层注入实例→ build_mcp_tool_specs列工具→闭包内call_tool执行
#"""
"""
agent_framework/mcp_client/client.py  MCP客户端骨架（端口先行）

本文件提供两个McpClientPort实现：
- NotConfiguredMcpClient：占位默认实现。真实stdio/HTTP传输尚未落地；
在传输落地前，任何list_tools / call_tool调用
都以McpClientError显式失败，绝不静默返回空结果或半成品
- FakeMcpClient：测试用内存实现。构造注入工具描述列表与可选的name→结果dict，
供上层联调与单测在无真实MCP server的情况下钉死桥接行为

依赖边界：只依赖contracts（McpClientPort / McpToolDescriptor / McpClientError）
与标准库，不import原项目任何模块
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent_framework.contracts import (
    McpClientError,
    McpClientPort,
    McpToolDescriptor,
)


#"""
#模块一：占位默认实现
#传输（stdio/HTTP）尚未落地，端口先行；所有操作显式抛McpClientError
#"""
class NotConfiguredMcpClient:
    """McpClientPort的占位默认实现：传输未配置，一切调用显式失败。

    真实stdio/HTTP传输尚未落地；
    接入适配层在真实客户端就绪前，可注入本实现保证「未配置」状态可见，
    而不是拿到一个静默返回空列表的假成功。
    """

    def list_tools(self) -> List[McpToolDescriptor]:
        """未配置传输：直接抛McpClientError。"""
        raise McpClientError("MCP 客户端尚未配置传输实现")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """未配置传输：直接抛McpClientError。"""
        raise McpClientError("MCP 客户端尚未配置传输实现")


#"""
#模块二：测试用内存实现
#构造注入工具列表与预设结果；list_tools返回快照副本，防止调用方篡改内部状态
#"""
class FakeMcpClient:
    """McpClientPort的测试实现：内存数据驱动，无网络、无子进程。

    tools：要暴露给list_tools的工具描述列表；
    results：可选的name → 预设返回文本；call_tool命中返回预设文本，
    未命中（或工具不在tools中）抛McpClientError，便于单测钉死失败路径。
    """

    def __init__(self, tools: List[McpToolDescriptor],
                 results: Optional[Dict[str, str]] = None) -> None:
        self._tools: List[McpToolDescriptor] = list(tools)
        self._results: Dict[str, str] = dict(results or {})

    def list_tools(self) -> List[McpToolDescriptor]:
        """返回工具描述的快照副本：调用方改返回值不影响本实例内部状态。"""
        return list(self._tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """命中预设结果则返回文本；未命中或工具不存在抛McpClientError。"""
        known = {tool.name for tool in self._tools}
        if name in self._results:
            return self._results[name]
        if name not in known:
            raise McpClientError(f"MCP 工具不存在：{name}")
        raise McpClientError(f"FakeMcpClient 未配置工具 {name} 的返回结果")
