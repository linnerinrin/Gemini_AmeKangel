# """
#架构作用：把MCP工具桥接为原项目ToolSpec形状的规格列表，是mcp_client包接入注册表的桥
#调用方：接入适配层（负责把McpClientPort实例挂到mgr.mcp_client并合并规格）
#数据流：build_mcp_tool_specs(client) → client.list_tools() → list[ToolSpec] →追加进TOOL_REGISTRY末尾
# """
"""
agent_framework/mcp_client/tool_bridge.py  MCP工具桥接（与app/tools/registry.py完全兼容）

约定（对齐readonly/tool_specs.py与registry.py的实现约定）：
- ToolSpec / SCOPE_EXTERNAL函数内延迟导入：本文件在接入前不得触发对原项目的import；
  MCP工具属外部来源，scope一律用SCOPE_EXTERNAL；
- impl为_bind_*(mgr)闭包工厂：工厂只捕获mgr与工具名，真正调用在被调用时；
- mgr语义（【待确认挂载点】）：通过getattr(mgr, "mcp_client")取得McpClientPort实例——
这是接入点约定，接入时由适配层把实例挂到FuncCallManager或包装owner上；
-实现函数返回str：McpClientError转成可读错误字符串返回，不向graph抛异常；
-实现函数不写docstring：描述只放ToolSpec.description，由适配器写入__doc__

重要：本文件不注册进TOOL_REGISTRY，接入前需人工评估风险
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Set

from app.agent_framework.contracts import (
    McpClientError,
    McpClientPort,
    McpToolDescriptor,
)


# """
#模块一：工具实现闭包（impl工厂）
#每个工厂捕获工具名，返回的闭包在调用时经mgr.mcp_client取客户端并调call_tool；
#失败路径（未挂载/ McpClientError）一律转为可读错误字符串，不向graph抛异常
# """
def _bind_mcp_tool(tool_name: str) -> Callable[[Any], Callable[..., str]]:
    """生成单个 MCP 工具的 impl 工厂；tool_name 为调 client.call_tool 用的原名。"""

    def _factory(mgr: Any) -> Callable[..., str]:
        def call_mcp_tool(**kwargs: Any) -> str:
            client = getattr(mgr, "mcp_client", None)
            if client is None:
                return "操作失败：MCP 客户端未挂载到 mgr.mcp_client（接入点待确认）"
            try:
                return client.call_tool(tool_name, kwargs)
            except McpClientError as exc:
                return f"MCP 工具 {tool_name} 调用失败：{exc}"

        return call_mcp_tool

    return _factory


# """
#模块二：命名冲突消解
# name保留原名；与已占用名冲突时加server前缀（<server>__<name>），无server时按序加序号
# """
def _resolve_name(tool: McpToolDescriptor, used: Set[str]) -> str:
    """冲突消解规则：原名未占用直接用；占用则尝试 <server>__<name>；仍冲突加数字后缀。"""
    if tool.name not in used:
        return tool.name
    if tool.server:
        prefixed = f"{tool.server}__{tool.name}"
        if prefixed not in used:
            return prefixed
    index = 2
    while f"{tool.name}_{index}" in used:
        index += 1
    return f"{tool.name}_{index}"


# """
#模块三：规格构建入口
#延迟导入ToolSpec / SCOPE_EXTERNAL；默认只桥接read_only=True的工具；
# list_tools抛McpClientError时打印警告并返回空列表，不向上抛
# """
def build_mcp_tool_specs(client: McpClientPort, *, include_write: bool = False) -> list:
    """把 client 暴露的 MCP 工具桥接成 ToolSpec 列表（默认仅只读工具）。

    client：McpClientPort 实例（联调用真实实现，单测可用 FakeMcpClient）；
    include_write：默认 False，只桥接 read_only=True 的工具（对应只读改造方向）；
    置 True 时放行写工具——【风险】写工具可能修改外部系统状态，仅在人工确认
    目标 server 行为后才应开启，并需配套审计。
    """
    from app.tools.registry import SCOPE_EXTERNAL, ToolSpec  #延迟导入：见模块docstring

    try:
        tools = client.list_tools()
    except McpClientError as exc:
        print(f"[mcp_client] 列出 MCP 工具失败，已跳过桥接：{exc}")
        return []

    specs: List[ToolSpec] = []
    used: Set[str] = set()
    for tool in tools:
        if not include_write and not tool.read_only:
            continue  #只读改造方向：写工具默认不桥接
        name = _resolve_name(tool, used)
        used.add(name)
        specs.append(ToolSpec(
            name=name,
            description=tool.description,   #透传server侧描述
            parameters=tool.parameters,     #透传server侧JSON Schema
            impl=_bind_mcp_tool(tool.name),  #调用时仍用原名回传server
            scope=SCOPE_EXTERNAL,
        ))
    return specs
