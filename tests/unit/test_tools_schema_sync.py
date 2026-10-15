"""工具 schema 与工厂实现必须同名。

config.TOOLS_SCHEMAS（由 app/tools/registry.py 经 adapters/openai.py 派生）是模型可见定义，
同时是 ModelManager._validate_tools 的白名单来源；FuncCallManager._create_tools
（由同一注册表经 adapters/langgraph.py 派生）是实际执行体。两者一旦不同步（例如
change_emotion -> change_status 只改了一处），模型调用会被判定为"工具未注册"而整轮失败。
"""
from conftest import load_source_module


def _created_tool_names():
    module = load_source_module("func_call_manager_schema_sync", "app/managers/FuncCallManager.py")
    mgr = module.FunctionCallManager()
    mgr.init(rag_mgr=object(), memory_mgr=object(), model_mgr=object())
    # 真实 langchain 工具暴露 .name；测试替身装饰器下退回函数名
    return sorted(getattr(tool, "name", None) or tool.__name__ for tool in mgr.tools)


def test_declared_schema_names_match_created_tools():
    from app.config.config import config

    declared = sorted(entry["function"]["name"] for entry in config.TOOLS_SCHEMAS)
    created = _created_tool_names()

    assert declared, "工具注册表未加载到任何工具定义"
    assert created == declared, (
        "模型可见工具定义与 FuncCallManager 的工具名不一致：\n"
        f"  仅声明: {sorted(set(declared) - set(created))}\n"
        f"  仅实现: {sorted(set(created) - set(declared))}"
    )


def test_status_tool_is_declared():
    """状态工具必须作为 change_status 出现，且三个方向参数齐全。"""
    from app.config.config import config

    entry = next(
        (e for e in config.TOOLS_SCHEMAS if e["function"]["name"] == "change_status"), None
    )
    assert entry is not None, "模型可见工具定义缺少 change_status"
    properties = entry["function"]["parameters"]["properties"]
    assert set(properties) == {"amount_affection", "amount_stress", "amount_mental", "reason"}
