# """
#架构作用：将工具注册表的external工具转换为MCP Server，并提供HTTP挂载及stdio入口
#调用方：FastAPI lifespan/mount装配或本地MCP客户端启动；模型/检索管理器按需延迟加载
#数据流：注册表schema → MCP Tool/签名校验→客户端请求→对应实现闭包
# """
"""
mcp/server.py把工具注册表暴露为MCP Server（P6b）

唯一事实来源仍然是app/tools/registry.py：本文件不定义任何工具的名字、描述或参数schema，
只把registry.external_specs()（方案R5的5个无状态工具）转成MCP工具

    registry.TOOL_REGISTRY（唯一来源，11个：7个核心 + 4个只读）
      ├─ adapters/openai.py    →模型可见的OpenAI function载荷
      ├─ adapters/langgraph.py → ToolNode执行体
      └─ mcp/server.py         → MCP tools（本文件，只取SCOPE_EXTERNAL的5个）

## schema保真：为什么不是「让SDK从函数签名推导」

MCPServer.add_tool()会用func_metadata()从函数签名推导input schema。若放任推导，注册表里的
显式JSON Schema与推导结果就成了两份定义，正是P6a刚消灭的漂移陷阱。实测mcp 2.3.0允许
「注册表schema直接喂给MCP」，所以走这条路（探针结论见docs/mcp-server.md）：

1.不调用Tool.from_function()（它不接受显式schema，事后再改tool.parameters又依赖SDK内部
可变属性），改为直接调用Tool的pydantic构造器，把deepcopy(spec.parameters)作为声明字段
   parameters传入；server.py的list_tools()原样把它当作`inputSchema`发给客户端：
   `MCPTool(..., input_schema=info.parameters, ...)`；
2. fn_metadata（参数校验用的arg_model）用SDK自己的func_metadata()从包装函数构建，
校验路径与from_function()完全一致，只是schema来源换成注册表；
3.包装函数故意写成`**kwargs`，SDK看不到真实形参，改为用inspect.Signature按注册表的
   properties / required生成`__signature__`。于是「对外公布的inputSchema」与「mcp侧参数校验
用的arg_model」也来自同一份注册表，不会出现「公布了A、实际按B校验」的分裂

tests/unit/test_mcp_server.py用deep-equal把「MCP列出的schema ==注册表parameters」钉死

##管理器从哪来

external工具里retrieve_knowledge / retrieve_context / rewrite需要进程内管理器（RAG、模型），
output_time / search_web不需要。这里统一传_LazyManager：只有工具真正访问mgr.xxx时才去解析
管理器，所以build_server()很轻，只连output_time的客户端不会顺带把torch/模型/数据库拉起来

##接线（由main.py负责，本文件不碰main.py）

    app = FastAPI()
    mount_mcp(app)                       #在/mcp挂上StreamableHTTP入口
    app.router.lifespan_context = mcp_lifespan
                                         #挂载的Starlette子应用不会自己跑lifespan，
                                         #必须由父app的lifespan驱动session_manager
stdio直连（本地客户端，如Claude Desktop / Cursor）：

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
from mcp.server.mcpserver.utilities.func_metadata import func_metadata
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

# JSON Schema类型→ Python类型；未标注类型（或自定义类型）一律回退Any
#只覆盖注册表现有工具用到的类型，新增类型时在这里补一行即可
_JSON_SCHEMA_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


# """
#模块一：LazyManager ——延迟解析app.managers
#只在属性访问时才导入重型管理器，让不依赖管理器的工具（output_time/search_web）可离线工作；
#默认来源是app.managers的进程级FunctionCallManager，必要时就地补一次init
# """
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
        #注意：_provider / _resolved已经由__init__写进实例字典，不会走到这里
        return getattr(self._manager(), name)


def default_manager() -> Any:
    """默认管理器来源：app.managers 里的全局 FunctionCallManager 单例。

    先复用已被 ChatSession.init() 装配过的实例（HTTP 挂载场景：Web 进程里已经初始化）。
    独立跑 stdio server 时没有 ChatSession，就地用同样的三个管理器补一次 init，
    这样 retrieve_* / rewrite 才能拿到 rag_mgr / model_mgr。
    """
    from app.managers import (  #延迟导入：app.managers会连带加载torch/模型等重依赖
        functioncall_manager,
        memory_manager,
        model_manager,
        rag_manager,
    )

    if getattr(functioncall_manager, "rag_mgr", None) is None:
        functioncall_manager.init(rag_manager, memory_manager, model_manager)
    return functioncall_manager


# """
#模块二：schema与签名保真——把注册表的JSON Schema变成MCP可见的Tool
#公布的inputSchema与校验用的arg_model同源（都从ToolSpec.parameters派生），避免漂移；
# Tool经pydantic构造器一次成形，不再事后改写SDK内部属性
# """
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
            #非必填属性：沿用schema里的default，没有就退回None（标注同时放宽为Optional，
            #否则显式传null会被pydantic拒绝，而schema并没有禁止）
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
        #真实形参是**kwargs（SDK通过__signature__看到注册表的形参），
        #这里按注册表的签名拿到实现函数再调用；注册表里既有同步实现也有协程实现
        impl = spec.impl(manager)
        result = impl(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    invoke.__name__ = spec.name
    invoke.__doc__ = spec.description
    invoke.__signature__ = signature_from_schema(spec.parameters)

    # structured_output=False：不要为str返回值自动生成outputSchema，
    #注册表没有输出schema，对外形状必须只有注册表那一份；
    #不用Tool.from_function()（它不收显式schema，事后改写tool.parameters依赖SDK内部可变性），
    #直接走Tool的pydantic构造器，把注册表schema作为声明字段一次传入（用副本，防调用方污染唯一来源）
    return Tool(
        fn=invoke,
        name=spec.name,
        description=spec.description,
        parameters=copy.deepcopy(spec.parameters),
        fn_metadata=func_metadata(invoke, structured_output=False),
        is_async=True,
    )


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


# """
#模块三：传输、挂载与进程入口
#同一server既可mount到FastAPI（HTTP，/mcp），也可用stdio独立运行；
#传输安全默认开启DNS rebinding校验，非本机Host须显式配置allowed_hosts
# """
def _transport_security(allowed_hosts: Optional[List[str]]) -> TransportSecuritySettings:
    """StreamableHTTP 的 Host 头校验设置。

    默认开启DNS rebinding防护（SDK默认只放行127.0.0.1/localhost/[::1] 的Host）。
    本机/局域网里其他模型接入时，客户端填 192.168.x.x:port 或经反向代理的Host会被
    421 Invalid Host header拒绝，必须把允许的Host:port显式传给allowed_hosts；
    只有确认只在可信网络内提供服务时才可考虑显式关闭该校验
    """
    if allowed_hosts is None:
        return TransportSecuritySettings(enable_dns_rebinding_protection=True)
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
    # methods=None：MCP的StreamableHTTP端点同时用POST/GET/DELETE，不能被Route默认的GET限制住
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
    except RuntimeError as exc:  # session_manager要先调用过streamable_http_app()
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
