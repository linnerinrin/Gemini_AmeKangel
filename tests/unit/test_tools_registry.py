"""工具唯一注册表（P6a）：注册表 = 模型可见载荷 = 执行侧工具，三者必须由同一处派生。

app/tools/registry.py 是唯一事实来源，本文件锁住它的对外形状：
1) 工具名与顺序（test_func_call_manager.py 按位置解包、ToolNode 按名派发）
2) OpenAI 载荷（config.TOOLS_SCHEMAS：发给模型 + 工具调用白名单）与注册表逐字段一致
3) 执行侧 LangChain 工具的名字/描述来自注册表；对外（MCP）作用域只含无状态工具
"""
import subprocess
import sys
import types
from pathlib import Path

from app.config.config import config
from app.tools import registry
from app.tools.adapters import langgraph as langgraph_adapter
from app.tools.adapters import openai as openai_adapter

# 顺序即契约：与 test_func_call_manager.py 的位置解包顺序一致，改这里必须同步改那边
EXPECTED_TOOL_ORDER = [
    "output_time",
    "retrieve_knowledge",
    "retrieve_context",
    "rewrite",
    "save_memory",
    "search_web",
    "change_status",
]

# 方案 R5：MCP 侧只暴露无状态工具
EXPECTED_EXTERNAL = [
    "output_time",
    "retrieve_knowledge",
    "retrieve_context",
    "rewrite",
    "search_web",
]


def test_registry_order_and_names_are_frozen():
    assert registry.tool_names() == EXPECTED_TOOL_ORDER


def test_registry_names_are_unique_and_resolvable():
    names = registry.tool_names()
    assert len(set(names)) == len(names)
    for name in names:
        assert registry.get_spec(name).name == name
    try:
        registry.get_spec("not_a_tool")
    except KeyError:
        pass
    else:
        raise AssertionError("未注册的工具名必须抛 KeyError")


def test_config_payload_is_generated_from_registry():
    """config.TOOLS_SCHEMAS 不再是第二份定义，而是注册表的 OpenAI 出口"""
    assert config.TOOLS_SCHEMAS == openai_adapter.to_openai_schemas()


def test_openai_payload_shape_matches_registry():
    payload = openai_adapter.to_openai_schemas()
    assert isinstance(payload, list)
    assert [entry["function"]["name"] for entry in payload] == registry.tool_names()
    for entry, spec in zip(payload, registry.TOOL_REGISTRY):
        assert entry["type"] == "function"
        assert set(entry) == {"type", "function"}
        assert entry["function"]["description"] == spec.description
        assert entry["function"]["parameters"] == spec.parameters
        assert spec.description.strip()
        assert spec.parameters["type"] == "object"
        assert isinstance(spec.parameters["properties"], dict)


def test_payload_is_a_copy_not_the_registry_object():
    """出口必须给副本：调用方改动不能污染唯一来源"""
    payload = openai_adapter.to_openai_schemas()
    payload[0]["function"]["parameters"]["properties"]["注入"] = {"type": "string"}
    assert "注入" not in registry.get_spec("output_time").parameters["properties"]
    assert "注入" not in openai_adapter.to_openai_schemas()[0]["function"]["parameters"]["properties"]


def test_external_scope_only_exposes_stateless_tools():
    assert [spec.name for spec in registry.external_specs()] == EXPECTED_EXTERNAL
    internal = [spec.name for spec in registry.TOOL_REGISTRY if spec.scope == registry.SCOPE_INTERNAL]
    assert internal == ["save_memory", "change_status"]


def test_langgraph_adapter_binds_registry_name_and_description(monkeypatch):
    """执行侧工具的名字/描述必须来自注册表，不允许各实现再写一份 docstring"""
    seen = []
    fake_tools_module = types.ModuleType("langchain_core.tools")

    def fake_tool(func):
        seen.append(func)
        return func

    fake_tools_module.tool = fake_tool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools_module)

    tools = langgraph_adapter.build_tools(object())

    assert tools == seen
    assert [fn.__name__ for fn in seen] == registry.tool_names()
    assert [fn.__doc__ for fn in seen] == [spec.description for spec in registry.TOOL_REGISTRY]


def test_built_tools_expose_registry_names():
    """真实 langchain 装饰器与测试替身下都要拿到同一批名字（替身回退到函数名）"""
    tools = langgraph_adapter.build_tools(object())
    assert [getattr(t, "name", None) or t.__name__ for t in tools] == registry.tool_names()
    assert [getattr(t, "description", None) or t.__doc__ for t in tools] == [
        spec.description for spec in registry.TOOL_REGISTRY
    ]


def test_registry_and_adapters_do_not_pull_mcp():
    """唯一注册表与它的两个适配器都不依赖 MCP（P6a 的边界）。

    必须用**子进程**验证，不能在同进程里断言"全局 sys.modules 没有 mcp"：
    app/main.py 在挂载 MCP 服务时会合法地导入 mcp，于是那种断言会随测试执行顺序
    时红时绿（本仓库把 tests/unit 排在某些会导入 app 的用例后面就会翻车）。
    子进程从干净解释器出发，只导入注册表与两个适配器，真正验证边界。
    """
    code = (
        "import sys\n"
        "import app.tools.registry, app.tools.adapters.openai, app.tools.adapters.langgraph\n"
        "bad = [name for name in ('mcp', 'fastmcp', 'langchain_mcp_adapters') if name in sys.modules]\n"
        "print(','.join(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, (
        "注册表/适配器不应拉入 MCP 依赖，实际拉入："
        f"{result.stdout.strip()}\n--- stderr ---\n{result.stderr}"
    )
