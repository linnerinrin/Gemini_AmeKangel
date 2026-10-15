import pytest
import asyncio
import sys
import types

from conftest import load_source_module


class FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = FakeAsyncOpenAI
sys.modules["openai"] = openai_module

model_module = load_source_module("model_manager_under_test", "app/managers/ModelManager.py")


def test_generation_result_stringifies_content():
    result = model_module.GenerationResult(content="hello", thinking="think")

    assert str(result) == "hello"
    assert result.tool_calls == []


def test_init_sets_tools_from_config():
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    # tools should be set in __init__ from config.TOOLS_SCHEMAS
    assert not hasattr(mgr, 'bind_tools'), "bind_tools should not exist"



class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self.message, finish_reason=getattr(self, "finish_reason", "stop"))]
        )


class FakeClient:
    def __init__(self, message):
        self.chat = types.SimpleNamespace(
            completions=FakeCompletions(message)
        )


def make_tool_call(name, arguments, call_id="call-1"):
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def test_init_exposes_model_meta_attributes():
    """R12：厂商/模型属性任何时候都可读（还没生成过就是 None）"""
    mgr = model_module.ModelManager()

    assert mgr.last_provider is None
    assert mgr.last_model is None


def test_generate_records_resolved_provider_and_model(monkeypatch):
    """记录的是本次实际解析出的厂商/模型（resolve_profile 的结果）"""
    message = types.SimpleNamespace(content="answer", tool_calls=[])
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(message)
    mgr.tools = []
    monkeypatch.setattr(mgr, "_load_config", lambda: {
        "provider": "openai", "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1", "api_key": "sk-test",
    })

    asyncio.run(mgr.generate([], {}, use_tools=False))

    assert mgr.last_provider == "openai"
    assert mgr.last_model == "gpt-4o-mini"
    # 发给 API 的也是解析出来的这个模型，而不是 config.DEEPSEEK_MODEL
    assert mgr.client.chat.completions.calls[0]["model"] == "gpt-4o-mini"


def test_injected_client_records_no_provider_or_model():
    """替身客户端没有真实厂商可记录时如实写 None，不瞎猜默认模型"""
    message = types.SimpleNamespace(content="answer", tool_calls=[])
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(message)
    mgr.tools = []

    asyncio.run(mgr.generate([], {}, use_tools=False))

    assert mgr.last_provider is None
    assert mgr.last_model is None


def test_generate_parses_tool_calls_and_reasoning():
    message = types.SimpleNamespace(
        content="answer",
        reasoning_content="think",
        tool_calls=[make_tool_call("search", '{"query": "hello"}')],
    )
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(message)
    mgr.tools = [{"type": "function", "function": {"name": "search"}}]

    result = asyncio.run(
        mgr.generate([{"role": "user", "content": "hi"}], {"temperature": 0.1}, use_tools=True)
    )

    assert result.content == "answer"
    assert result.thinking == "think"
    assert result.tool_calls == [{"name": "search", "args": {"query": "hello"}, "id": "call-1"}]
    assert mgr.client.chat.completions.calls[0]["tools"] == [{"type": "function", "function": {"name": "search"}}]


@pytest.mark.parametrize("arguments", ["{bad json}", "", "null", "[]", "42"])
def test_generate_rejects_invalid_tool_arguments(arguments):
    message = types.SimpleNamespace(content=None, reasoning_content=None, tool_calls=[make_tool_call("broken", arguments)])
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(message)
    mgr.tools = [{"type": "function", "function": {"name": "broken"}}]
    with pytest.raises(model_module.InvalidToolCallError):
        asyncio.run(mgr.generate([], {}, use_tools=True))
    assert len(mgr.client.chat.completions.calls) == 1


def test_generate_can_disable_tools():
    message = types.SimpleNamespace(content="answer", tool_calls=[])
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(message)
    mgr.tools = []
    result = asyncio.run(mgr.generate([], {}, use_tools=False))
    assert result.finish_reason == "stop"
    assert mgr.client.chat.completions.calls[0]["tools"] is None


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter"])
def test_generate_preserves_finish_metadata(finish):
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(types.SimpleNamespace(content="partial", tool_calls=[]))
    mgr.client.chat.completions.finish_reason = finish
    mgr.tools = []
    result = asyncio.run(mgr.generate([], {}, use_tools=False))
    assert result.finish_reason == finish
    assert result.truncated == (finish == "length")


@pytest.mark.parametrize("name,arguments,finish,enabled", [
    ("search", "{}", "length", True),
    ("", "{}", "tool_calls", True),
    ("sea", "{}", "tool_calls", True),
    ("unknown", "{}", "tool_calls", True),
    ("search", "{}", "tool_calls", False),
])
def test_generate_rejects_unsafe_tools(name, arguments, finish, enabled):
    mgr = model_module.ModelManager.__new__(model_module.ModelManager)
    mgr.client = FakeClient(types.SimpleNamespace(content="", tool_calls=[make_tool_call(name, arguments)]))
    mgr.client.chat.completions.finish_reason = finish
    mgr.tools = [{"type": "function", "function": {"name": "search"}}]
    with pytest.raises(model_module.InvalidToolCallError):
        asyncio.run(mgr.generate([], {}, use_tools=enabled))
    assert len(mgr.client.chat.completions.calls) == 1


def test_valid_empty_tool_object_is_preserved():
    result = model_module.ModelManager._result_from_message(types.SimpleNamespace(content="", tool_calls=[make_tool_call("search", "{}")]), "tool_calls")
    assert result.tool_calls[0]["args"] == {}
