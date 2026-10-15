"""
mcp/server.py 把工具注册表暴露为 MCP Server（P6b）

唯一事实来源仍然是 app/tools/registry.py：本文件不定义任何工具的名字、描述或参数 schema，
只把 registry.external_specs()（方案 R5 的 5 个无状态工具）转成 MCP 工具。

    registry.TOOL_REGISTRY（唯一来源，7 个）
      ├─ adapters/openai.py    → 模型可见的 OpenAI function 载荷
      ├─ adapters/langgraph.py → ToolNode 执行体
      └─ mcp/server.py         → MCP tools（本文件，只取 SCOPE_EXTERNAL 的 5 个）

## schema 保真：为什么不是「让 SDK 从函数签名推导」

MCPServer.add_tool() 会用 func_metadata() 从函数签名推导 input schema。若放任推导，注册表里的
显式 JSON Schema 与推导结果就成了两份定义，正是 P6a 刚消灭的漂移陷阱。实测 mcp 2.3.0 允许
「注册表 schema 直接喂给 MCP」，所以走这条路（探针结论见 docs/mcp-server.md）：

1. 用 Tool.from_function() 造 MCP 内部的注册模型（mcp.server.mcpserver.tools.Tool）；
2. 立刻用 `tool.parameters = deepcopy(spec.parameters)` 覆盖成注册表里的显式 schema；
   server.py 的 list_tools() 原样把它当作 `inputSchema` 发给客户端：
   `MCPTool(..., input_schema=info.parameters, ...)`；
3. 包装函数故意写成 `**kwargs`，SDK 看不到真实形参，改为用 inspect.Signature 按注册表的
   properties / required 生成 `__signature__`。于是「对外公布的 inputSchema」与「mcp 侧参数校验
   用的 arg_model」也来自同一份注册表，不会出现「公布了 A、实际按 B 校验」的分裂。

tests/unit/test_mcp_server.py 用 deep-equal 把「MCP 列出的 schema == 注册表 parameters」钉死。

## 管理器从哪来

external 工具里 retrieve_knowledge / retrieve_context / rewrite 需要进程内管理器（RAG、模型），
output_time / search_web 不需要。这里统一传 _LazyManager：只有工具真正访问 mgr.xxx 时才去解析
管理器，所以 build_server() 很轻，只连 output_time 的客户端不会顺带把 torch/模型/数据库拉起来。

## 接线（由 main.py 负责，本文件不碰 main.py）

    app = FastAPI()
    mount_mcp(app)                       # 在 /mcp 挂上 StreamableHTTP 入口
    app.router.lifespan_context = mcp_lifespan
                                         # 挂载的 Starlette 子应用不会自己跑 lifespan，
                                         # 必须由父 app 的 lifespan 驱动 session_manager
stdio 直连（本地客户端，如 Claude Desktop / Cursor）：

    python -m app.mcp_server.server
"""
import argparse
import copy
import inspect
import sys
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from starlette.routing import Route

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools import Tool
from mcp.server.transport_security import TransportSecuritySettings

from app.tools.registry import ToolSpec, external_specs

MCP_SERVER_NAME = "ame-kangel-tools"
MCP_SERVER_VERSION = "1.0.0"
MCP_HTTP_PATH = "/mcp"
"""HTTP 入口路径：与 main.py 里其他路由并列，客户端填 http://<host>:<port>/mcp。"""

INSTRUCTIONS = (
    "AmeKangel 角色扮演项目的无状态工具出口。"
    "检索类工具（retrieve_knowledge / retrieve_context）需要本机已初始化知识库与向量模型，"
    "rewrite 需要本机已配置模型厂商与 API Key。"
    "save_memory / change_status 依赖「当前存档」这类进程内状态，不对 MCP 客户端开放。"
)

# JSON Schema 类型 → Python 类型；未标注类型（或自定义类型）一律回退 Any。
# 只覆盖注册表现有工具用到的类型，新增类型时在这里补一行即可。
_JSON_SCHEMA_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class LazyManager:
    """延迟解析管理器的代理：只有真正访问属性时才去 import app.managers。

    registry 里的 impl 是 `spec.impl(mgr)` 的闭包，只把 mgr 存起来，访问 mgr.xxx 才用到它；
    因此 output_time / search_web 这类不碰管理器的工具可以完全离线工作。
    """

    def __init__(self, provider: Callable[[], Any]):
        self._provider = provider
        self._resolved: Any = None

    def _manager(self) -> Any:
        if self._resolved is None:
            self._resolved = self._provider()
        return self._resolved

    def __getattr__(self, name: str) -> Any:
        # 注意：_provider / _resolved 已经由 __init__ 写进实例字典，不会走到这里
        return getattr(self._manager(), name)


def default_manager() -> Any:
    """默认管理器来源：app.managers 里的全局 FunctionCallManager 单例。

    先复用已被 ChatSession.init() 装配过的实例（HTTP 挂载场景：Web 进程里已经初始化）。
    独立跑 stdio server 时没有 ChatSession，就地用同样的三个管理器补一次 init，
    这样 retrieve_* / rewrite 才能拿到 rag_mgr / model_mgr。
    """
    from app.managers import (  # 延迟导入：app.managers 会连带加载 torch/模型等重依赖
        functioncall_manager,
        memory_manager,
        model_manager,
        rag_manager,
    )

    if getattr(functioncall_manager, "rag_mgr", None) is None:
        functioncall_manager.init(rag_manager, memory_manager, model_manager)
    return functioncall_manager


def _property_annotation(prop: Dict[str, Any]) -> Any:
    """JSON Schema 单个属性 → Python 类型标注（未知类型回退 Any）。"""
    return _JSON_SCHEMA_TYPES.get(prop.get("type"), Any)


def signature_from_schema(schema: Dict[str, Any]) -> inspect.Signature:
    """按注册表的 JSON Schema 生成函数签名（只用于 MCP 侧的参数校验）。

    形参名、必填集合、类型全部来自注册表，所以校验用的 arg_model 与公布的 inputSchema
    同源，不会互相漂移。
    """
    required = set(schema.get("required") or ())
    parameters: List[inspect.Parameter] = []
    for name, prop in (schema.get("properties") or {}).items():
        annotation = _property_annotation(prop)
        if name in required:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
            )
        else:
            # 非必填属性：沿用 schema 里的 default，没有就退回 None（标注同时放宽为 Optional，
            # 否则显式传 null 会被 pydantic 拒绝，而 schema 并没有禁止）
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Optional[annotation],
                    default=prop.get("default", None),
                )
            )
    return inspect.Signature(parameters)


def _build_tool(spec: ToolSpec, manager: Any) -> Tool:
    """把单个 ToolSpec 转成 MCP 工具（名字/描述/schema 都取自注册表）。"""

    async def invoke(**kwargs: Any) -> Any:
        # 真实形参是 **kwargs（SDK 通过 __signature__ 看到注册表的形参），
        # 这里按注册表的签名拿到实现函数再调用；注册表里既有同步实现也有协程实现。
        impl = spec.impl(manager)
        result = impl(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    invoke.__name__ = spec.name
    invoke.__doc__ = spec.description
    invoke.__signature__ = signature_from_schema(spec.parameters)

    # structured_output=False：不要为 str 返回值自动生成 outputSchema，
    # 注册表没有输出 schema，对外形状必须只有注册表那一份。
    tool = Tool.from_function(
        invoke, name=spec.name, description=spec.description, structured_output=False
    )
    # 关键一步：用注册表里的显式 schema 覆盖签名推导结果（用副本，防止调用方污染唯一来源）
    tool.parameters = copy.deepcopy(spec.parameters)
    return tool


def build_server(manager: Any = None) -> MCPServer:
    """构建只注册 5 个外部工具的 MCP Server。

    manager 传 None 时使用 LazyManager(default_manager)，即首次真正需要时才加载
    app.managers；测试可以注入替身，从而完全不联网、不加载模型。
    """
    if manager is None:
        manager = LazyManager(default_manager)
    tools = [_build_tool(spec, manager) for spec in external_specs()]
    return MCPServer(
        name=MCP_SERVER_NAME,
        instructions=INSTRUCTIONS,
        version=MCP_SERVER_VERSION,
        tools=tools,
    )


_server: Optional[MCPServer] = None
_mounted_server: Optional[MCPServer] = None


def get_server() -> MCPServer:
    """进程内单例：mount_mcp / mcp_lifespan / main 共用同一个实例。"""
    global _server
    if _server is None:
        _server = build_server()
    return _server


def _transport_security(allowed_hosts: Optional[List[str]]) -> TransportSecuritySettings:
    """StreamableHTTP 的 Host 头校验设置。

    SDK 在 host 为 localhost 时会默认只放行 127.0.0.1/localhost/[::1] 的 Host，而本服务是给
    本机/局域网里其他模型接入的：客户端填 192.168.x.x:port 或经反向代理时 Host 不是 localhost，
    默认值会直接回 421 Invalid Host header。所以默认关闭该校验；需要收紧时传 allowed_hosts。
    """
    if allowed_hosts is None:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(allowed_hosts=list(allowed_hosts))


def mount_mcp(
    app: Any,
    path: str = MCP_HTTP_PATH,
    server: Optional[MCPServer] = None,
    allowed_hosts: Optional[List[str]] = None,
) -> MCPServer:
    """把 StreamableHTTP 子应用挂到传入的 FastAPI/Starlette app 的 `path` 上。

    这里用 `Route(path, endpoint=子应用)` 而不是 `app.mount(path, 子应用)`：Starlette 的 Mount
    正则要求前缀后有斜杠，`app.mount("/mcp", 子应用)` 会让 POST /mcp 先吃一个 307 跳到 /mcp/，
    而 MCP 客户端默认不跟随重定向。Route 把子应用当 ASGI app 直接调用（Starlette 只对函数型
    endpoint 做 request→response 包装），路径原样传入，子应用里注册的也是同一个 path，
    因此对外就是干净的 /mcp。

    注意：挂载的 Starlette 子应用不会自己跑 lifespan，必须同时把 mcp_lifespan 接到父 app 上，
    否则 session_manager 没启动，请求会失败。
    """
    global _mounted_server
    server = server or get_server()
    sub_app = server.streamable_http_app(
        streamable_http_path=path,
        transport_security=_transport_security(allowed_hosts),
    )
    # methods=None：MCP 的 StreamableHTTP 端点同时用 POST/GET/DELETE，不能被 Route 默认的 GET 限制住
    app.router.routes.append(Route(path, endpoint=sub_app))
    _mounted_server = server
    return server


@asynccontextmanager
async def mcp_lifespan(app: Any = None, server: Optional[MCPServer] = None):
    """父 app 的 lifespan 里驱动 MCP 的 session_manager。

    可直接赋值给 app.router.lifespan_context（Starlette/FastAPI 会以 app 为参数调用它）：

        mount_mcp(app)
        app.router.lifespan_context = mcp_lifespan

    已有自己的 lifespan 时也可以嵌套：

        @asynccontextmanager
        async def lifespan(app):
            async with mcp_lifespan(app):
                yield
    """
    server = server or _mounted_server or get_server()
    try:
        manager = server.session_manager
    except RuntimeError as exc:  # session_manager 要先调用过 streamable_http_app()
        raise RuntimeError(
            "mcp_lifespan 必须在 mount_mcp(app) 之后进入：挂载时才会创建 session_manager"
        ) from exc
    async with manager.run():
        yield


def main(argv: Optional[List[str]] = None) -> None:
    """命令行入口：默认 stdio（本地客户端直连），可选独立 HTTP 服务。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.mcp_server.server",
        description="AmeKangel MCP Server：把工具注册表的 5 个无状态工具暴露给外部模型",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio=本地客户端直连（默认）；streamable-http=独立起 HTTP 服务",
    )
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http 监听地址")
    parser.add_argument("--port", type=int, default=8931, help="streamable-http 监听端口")
    args = parser.parse_args(argv)

    server = get_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
        return
    server.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        streamable_http_path=MCP_HTTP_PATH,
        transport_security=_transport_security(None),
    )


if __name__ == "__main__":
    sys.exit(main())
