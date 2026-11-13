"""MCP Server：工具注册表的MCP出口。

锁死三件事：
1. **对外集合与schema**：恰好是registry.external_specs()的无状态工具，
   且MCP列出的inputSchema与注册表parameters**逐字段deep-equal**
   ——核心约束：绝不允许出现第二份schema定义；
2. **工具真能执行**：output_time走真实实现；检索/改写/联网类全部用替身打桩，
   不联网、不需要API key、不加载模型；
3. **HTTP挂载**：mount_mcp(app)后app上有/mcp路由，mcp_lifespan()能正常进出，
   并用httpx的ASGI transport在进程内真的打一次initialize + tools/list。

本文件会真的import mcp，用例结束后由autouse fixture清掉sys.modules里的mcp*，
避免污染tests/unit/test_tools_registry.py::test_no_mcp_dependency_in_p6a（该用例验证注册表不引入mcp依赖）。
"""
import asyncio
import datetime
import inspect
import json
import sys
import types

import httpx
import pytest
from fastapi import FastAPI

from app.tools import registry

EXPECTED_EXTERNAL = [
    "output_time",
    "retrieve_knowledge",
    "retrieve_context",
    "rewrite",
    "search_web",
]
INTERNAL_ONLY = ["save_memory", "change_status"]

#进程内initialize用最小参数；协议版本取SDK握手能接受的版本
INITIALIZE_PARAMS = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "pytest", "version": "0"},
}


@pytest.fixture(autouse=True)
def no_mcp_leak():
    """跑完就摘掉mcp*（连app.mcp_server一起），让其它测试文件里的「未引入mcp依赖」断言不受影响。

    app.mcp_server.server必须一起摘：它import期就持有MCPServer / ToolError这些mcp类的引用，
    只删mcp*会让下一个用例重新import出一份新的mcp，于是「测试里import的ToolError」
    与「server实际抛的ToolError」不是同一个类对象。
    """
    yield
    doomed = [
        name
        for name in sys.modules
        if name in ("mcp", "mcp_types", "app.mcp_server")
        or name.startswith("mcp.")
        or name.startswith("mcp_types.")
        or name.startswith("app.mcp_server.")
    ]
    for name in doomed:
        sys.modules.pop(name, None)


@pytest.fixture
def mcp_module():
    """延迟import：模块级import会在pytest收集时就把mcp拉进sys.modules。"""
    from app.mcp_server import server

    return server


class FakeRAG:
    async def retrieve(self, query, caller="knowledge"):
        return f"{caller}:{query}"


class FakeModel:
    async def generate(self, messages, paras, use_tools=False):
        self.messages = messages
        self.paras = paras
        self.use_tools = use_tools
        return "rewritten"


class FakeManager:
    """注册表里external工具用到的管理器替身（只碰rag_mgr / model_mgr）。"""

    def __init__(self):
        self.rag_mgr = FakeRAG()
        self.model_mgr = FakeModel()


class FakeDDGS:
    """ddgs替身：不联网，返回空结果。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results):
        return []


@pytest.fixture
def server(mcp_module):
    return mcp_module.build_server(manager=FakeManager())


@pytest.fixture
def stub_ddgs(monkeypatch):
    module = types.ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return module


def run(coro):
    return asyncio.run(coro)


def text_of(result) -> str:
    """CallToolResult → 第一个文本块的内容。"""
    return result.content[0].text


def jsonrpc_payload(response: httpx.Response) -> dict:
    """HTTP返回可能是application/json，也可能是SSE的data:行（Accept里带了text/event-stream）。"""
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                if "result" in payload or "error" in payload:
                    return payload
        raise AssertionError(f"SSE 响应里没有可解析的 data 行：{response.text!r}")
    return response.json()


def test_exposes_exactly_the_five_stateless_tools(server):
    listed = run(server.list_tools())
    names = [tool.name for tool in listed]

    assert names == EXPECTED_EXTERNAL
    assert names == [spec.name for spec in registry.external_specs()]
    #依赖「当前存档」进程内状态的工具不外放
    for name in INTERNAL_ONLY:
        assert name not in names
    #也不允许冒出注册表之外的工具
    assert set(names) <= set(registry.tool_names())
    assert len(names) == len(set(names))


def test_mcp_schemas_are_the_registry_schemas(server):
    """核心断言：MCP的inputSchema与注册表parameters逐字段deep-equal（键名按MCP规范）。"""
    specs = {spec.name: spec for spec in registry.external_specs()}
    listed = run(server.list_tools())

    assert listed, "MCP 必须至少暴露一个工具"
    for tool in listed:
        spec = specs[tool.name]
        assert tool.description == spec.description
        assert tool.input_schema == spec.parameters
        #必须是副本：别把注册表里的唯一来源交出去
        assert tool.input_schema is not spec.parameters
        #注册表没有输出schema，就不许凭空生成一个
        assert tool.output_schema is None


def test_schema_survives_the_wire_over_an_official_client(server):
    """用官方客户端SDK走一遍真实协议：列出来的schema仍与注册表deep-equal。"""
    from mcp import Client

    specs = {spec.name: spec for spec in registry.external_specs()}

    async def scenario():
        async with Client(server, mode="legacy") as client:
            return await client.list_tools()

    result = run(scenario())
    assert [tool.name for tool in result.tools] == EXPECTED_EXTERNAL
    for tool in result.tools:
        assert tool.input_schema == specs[tool.name].parameters
        assert tool.description == specs[tool.name].description


def test_signature_from_schema_follows_required_and_defaults(mcp_module):
    """schema → 签名 的推导规则：required决定必填，default决定默认值。"""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 3},
            "flag": {"type": "boolean"},
        },
        "required": ["query"],
    }

    signature = mcp_module.signature_from_schema(schema)

    assert list(signature.parameters) == ["query", "limit", "flag"]
    assert signature.parameters["query"].default is inspect.Parameter.empty
    assert signature.parameters["limit"].default == 3
    assert signature.parameters["flag"].default is None


def test_tools_execute_core_branches(server, stub_ddgs):
    """五个外部工具都能真正执行；检索/改写/联网全部是替身，不联网、不需要API key。"""
    time_result = run(server.call_tool("output_time", {}))
    assert time_result.is_error is False
    time_text = text_of(time_result)
    assert "现在时间是" in time_text
    assert str(datetime.datetime.now().year) in time_text

    knowledge = run(server.call_tool("retrieve_knowledge", {"query": "设定"}))
    assert knowledge.is_error is False
    assert "knowledge:设定" in text_of(knowledge)

    context = run(server.call_tool("retrieve_context", {"query": "上文"}))
    assert context.is_error is False
    assert "context:上文" in text_of(context)

    rewritten = run(server.call_tool("rewrite", {"query": "他怎么样了"}))
    assert rewritten.is_error is False
    assert "rewritten" in text_of(rewritten)

    searched = run(server.call_tool("search_web", {"query": "新闻"}))
    assert searched.is_error is False
    assert "未搜索到相关结果" in text_of(searched)


def test_output_time_over_the_official_client(server):
    """真实握手证据的断言版本：initialize + tools/call。"""
    from mcp import Client

    async def scenario():
        async with Client(server, mode="legacy") as client:
            return await client.call_tool("output_time", {})

    result = run(scenario())
    assert result.is_error is False
    assert "现在时间是" in result.content[0].text


def test_registry_signature_is_enforced(server):
    """参数校验用的arg_model也来自注册表：必填query缺失必须报错，而不是静默放行。"""
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError):
        run(server.call_tool("retrieve_knowledge", {}))

    assert run(server.call_tool("retrieve_knowledge", {"query": "ok"})).is_error is False


def test_internal_only_tools_are_not_callable(server):
    """save_memory / change_status不在MCP上，调用必须失败。"""
    from mcp.server.mcpserver.exceptions import ToolError

    for name in INTERNAL_ONLY:
        with pytest.raises(ToolError):
            run(server.call_tool(name, {"memory": "x", "reason": "x"}))


def test_build_server_is_lazy_about_managers(mcp_module, monkeypatch):
    """默认（不注入manager）时构建与output_time都不该碰app.managers：不想为了取时间加载模型。"""

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"output_time 不应该访问 app.managers.{name}")

    monkeypatch.setitem(sys.modules, "app.managers", Boom())

    lazy_server = mcp_module.build_server()
    assert [tool.name for tool in run(lazy_server.list_tools())] == EXPECTED_EXTERNAL
    assert "现在时间是" in text_of(run(lazy_server.call_tool("output_time", {})))


def test_mount_mcp_registers_the_mcp_route(mcp_module, server):
    app = FastAPI()
    assert mcp_module.mount_mcp(app, server=server) is server

    paths = [getattr(route, "path", None) for route in app.routes]
    assert mcp_module.MCP_HTTP_PATH == "/mcp"
    assert "/mcp" in paths
    #不能只挂成/mcp/：客户端填的是/mcp，Starlette Mount会在这里回307而MCP客户端不跟随重定向
    assert "/mcp/" not in paths


def test_mcp_lifespan_enters_and_exits(mcp_module, server):
    app = FastAPI()
    mcp_module.mount_mcp(app, server=server)

    async def scenario():
        async with mcp_module.mcp_lifespan(app, server=server):
            return "entered"

    assert run(scenario()) == "entered"


def test_mcp_lifespan_wires_as_app_lifespan_context(mcp_module, server):
    """正文里的接线姿势：mount_mcp(app)之后把mcp_lifespan直接赋给lifespan_context。"""
    app = FastAPI()
    mcp_module.mount_mcp(app, server=server)
    app.router.lifespan_context = mcp_module.mcp_lifespan

    async def scenario():
        async with app.router.lifespan_context(app):
            return "started"

    assert run(scenario()) == "started"


def test_http_initialize_and_list_tools(mcp_module, server):
    """HTTP路径证据：进程内用httpx ASGI transport打/mcp的initialize + tools/list + tools/call。"""
    app = FastAPI()
    #DNS rebinding防护默认开启，测试客户端的Host是test，须显式列入allowed_hosts
    mcp_module.mount_mcp(app, server=server, allowed_hosts=["test"])

    async def scenario():
        async with mcp_module.mcp_lifespan(app, server=server):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                }
                init = await client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": INITIALIZE_PARAMS},
                    headers=headers,
                )
                assert init.status_code == 200, init.text
                session_id = init.headers["mcp-session-id"]

                session_headers = dict(headers, **{"mcp-session-id": session_id})
                initialized = await client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=session_headers,
                )
                assert initialized.status_code in (200, 202), initialized.text

                listed = await client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    headers=session_headers,
                )
                called = await client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "output_time", "arguments": {}}},
                    headers=session_headers,
                )
                return jsonrpc_payload(listed), jsonrpc_payload(called)

    listed_payload, called_payload = run(scenario())

    tools = listed_payload["result"]["tools"]
    assert [tool["name"] for tool in tools] == EXPECTED_EXTERNAL
    specs = {spec.name: spec for spec in registry.external_specs()}
    for tool in tools:
        assert tool["inputSchema"] == specs[tool["name"]].parameters
        assert tool["description"] == specs[tool["name"]].description

    assert called_payload["result"]["isError"] is False
    assert "现在时间是" in called_payload["result"]["content"][0]["text"]
