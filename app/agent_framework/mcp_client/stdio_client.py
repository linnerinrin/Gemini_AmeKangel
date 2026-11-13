# """
#架构作用：McpClientPort 的 stdio 传输实现（F5拍板），基于官方 mcp SDK 的子进程客户端
#调用方：app/tools/registry.py 底部按 data/mcp_servers.json 构造；测试仍可注入FakeMcpClient
#数据流：list_tools/call_tool → 独立线程 asyncio.run → stdio_client 起子进程握手 → 结果归一
# """
"""
agent_framework/mcp_client/stdio_client.py  MCP stdio 客户端（McpClientPort 真实传输）

设计取舍：
- 端口形状是同步方法，而官方 mcp SDK 是 async；且调用方（graph 工具执行）可能已处于
  运行中的事件循环里，不能就地 asyncio.run。故每次操作在**独立线程**里跑一个临时
  事件循环完成「起子进程→initialize→操作→关闭」全周期；
- 每次操作新建连接（握手代价换零生命周期管理）：实例本身无状态、无线程安全风险，
  可以被进程级注册表与每会话 FuncCallManager 同时引用；
- 一切异常（子进程启动失败/握手失败/超时/协议错误）统一归一为 McpClientError，
  与 NotConfiguredMcpClient 的失败语义一致，不向 graph 抛非预期异常类型

依赖边界：只依赖 contracts 与官方 mcp SDK，不 import 原项目其他模块
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from app.agent_framework.contracts import (
    McpClientError,
    McpToolDescriptor,
)


class StdioMcpClient:
    """McpClientPort 的 stdio 实现：每次操作拉起一个 server 子进程完成一次 RPC。

    command/args/env：子进程启动参数（同 StdioServerParameters 语义）；
    timeout：单次操作（含握手）整体超时秒数，超时按 McpClientError 上报；
    server：来源标识，写进 McpToolDescriptor.server 供命名冲突消解使用。
    """

    def __init__(self, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, *,
                 timeout: float = 30.0, server: str = "") -> None:
        self._command = command
        self._args = list(args or [])
        self._env = dict(env) if env else None
        self._timeout = float(timeout)
        self._server = server

    # """
    #模块一：同步出口
    #独立线程跑临时事件循环；wait_for 兜底整体超时；异常全归一为 McpClientError
    # """
    def list_tools(self) -> List[McpToolDescriptor]:
        return self._run(self._list_tools_async, "列出 MCP 工具")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        return self._run(lambda: self._call_tool_async(name, arguments), f"调用 MCP 工具 {name}")

    def _run(self, operation: Callable[[], Any], action: str):
        """在独立线程里 asyncio.run 一次操作；任何失败/超时统一抛 McpClientError。"""
        box: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                box["value"] = asyncio.run(asyncio.wait_for(operation(), self._timeout))
            except BaseException as exc:  # noqa: BLE001 线程边界必须全捕获，交给主线程归一
                box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(self._timeout + 5)  # 兜底：wait_for 之外再给线程一点收尾时间
        if thread.is_alive():
            raise McpClientError(f"{action}超时：子进程未在限时内退出")
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, McpClientError):
                raise exc
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                raise McpClientError(f"{action}超时（{self._timeout}s）") from exc
            raise McpClientError(f"{action}失败：{type(exc).__name__} {exc}") from exc
        return box.get("value")

    # """
    #模块二：协议操作
    #每次操作完整经历 stdio_client 握手；SDK 细节（参数形状/结果结构）集中在本模块
    # """
    async def _session_run(self, operation: Callable[[Any], Any]):
        #函数内延迟导入：未配置 MCP 的部署不必为 import 期付出 SDK 加载成本
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    async def _list_tools_async(self) -> List[McpToolDescriptor]:
        async def _op(session):
            return await session.list_tools()

        result = await self._session_run(_op)
        descriptors: List[McpToolDescriptor] = []
        for tool in result.tools:
            annotations = getattr(tool, "annotations", None)
            read_only = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False
            descriptors.append(McpToolDescriptor(
                name=tool.name,
                description=tool.description or "",
                parameters=dict(tool.inputSchema or {}),
                read_only=read_only,
                server=self._server,
            ))
        return descriptors

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]) -> str:
        async def _op(session):
            return await session.call_tool(name, arguments)

        result = await self._op_guard(name, _op)
        return result

    async def _op_guard(self, name: str, operation: Callable[[Any], Any]) -> str:
        result = await self._session_run(operation)
        if getattr(result, "isError", False):
            raise McpClientError(f"MCP 工具 {name} 返回错误：{self._extract_text(result)}")
        return self._extract_text(result)

    @staticmethod
    def _extract_text(result: Any) -> str:
        """把 content 块列表拼成纯文本；非文本块按类型占位，绝不静默丢弃。"""
        parts: List[str] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else f"[{getattr(block, 'type', 'unknown')} 内容]")
        return "\n".join(parts)
