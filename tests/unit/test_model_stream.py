import asyncio
import pytest
from types import SimpleNamespace

from conftest import load_source_module

model_module = load_source_module("model_manager_stream_under_test", "app/managers/ModelManager.py")
ModelManager = model_module.ModelManager


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def iterate():
            for chunk in self.chunks:
                yield chunk
        return iterate()


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.chunks)


def test_stream_generate_emits_reasoning_and_tool_name_once():
    tool_deltas = [
        SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="sea", arguments='{"q":')),
        SimpleNamespace(index=0, id=None, function=SimpleNamespace(name="rch", arguments='"test"}')),
    ]
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="先思考", content=None, tool_calls=[]))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content=None, tool_calls=[tool_deltas[0]]))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content=None, tool_calls=[tool_deltas[1]]))]),
        SimpleNamespace(choices=[]),
    ]
    manager = ModelManager.__new__(ModelManager)
    completions = FakeCompletions(chunks)
    manager.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    manager.tools = [{"type": "function", "function": {"name": "search"}}]
    events = []

    async def record(event):
        events.append(event)

    result = asyncio.run(manager.stream_generate([], {}, on_event=record))

    assert result.thinking == "先思考"
    assert result.tool_calls == [{"name": "search", "args": {"q": "test"}, "id": "call-1"}]
    assert events.count({"type": "tool_call", "name": "search"}) == 1
    assert {"type": "thinking_delta", "text": "先思考"} in events
    assert completions.kwargs["stream"] is True


def test_stream_generate_records_resolved_provider_and_model(monkeypatch):
    """R12：流式路径同样记录本次实际解析出的厂商/模型"""
    chunks = [SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=[]))])]
    manager = ModelManager.__new__(ModelManager)
    manager.tools = []
    manager.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(chunks)))
    monkeypatch.setattr(manager, "_load_config", lambda: {
        "provider": "deepseek", "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com", "api_key": "sk-test",
    })

    asyncio.run(manager.stream_generate([], {}, use_tools=False))

    assert manager.last_provider == "deepseek"
    assert manager.last_model == "deepseek-v4-flash"
    assert manager.client.chat.completions.kwargs["model"] == "deepseek-v4-flash"


def test_truncated_and_empty_outputs_do_not_break_the_turn():
    """被截断只做标记，不再抛异常中断整轮对话（debug 13）"""
    manager = ModelManager.__new__(ModelManager)
    manager.tools = []
    chunks = [SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
        delta=SimpleNamespace(content="部分回答", reasoning_content="思考", tool_calls=[]))])]
    manager.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(chunks)))

    result = asyncio.run(manager.stream_generate([], {}, use_tools=False))

    assert result.content == "部分回答"
    assert result.truncated is True

    manager.tools = []
    manager.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions([
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            delta=SimpleNamespace(content="", reasoning_content="思考", tool_calls=[]))])])))
    empty = asyncio.run(manager.stream_generate([], {}, use_tools=False))
    assert empty.content == ""
    assert empty.truncated is False


def test_emoji_labels_normalize_to_codes():
    from app.config.emoji_labels import normalize_emoji_text

    assert normalize_emoji_text("<永远爱你>") == "jine_pien_loveforever"
    assert normalize_emoji_text("【永远爱你】") == "jine_pien_loveforever"
    assert normalize_emoji_text("  永远爱你  ") == "jine_pien_loveforever"
    assert normalize_emoji_text("<好>") == "jine_pien_ok"
    # 夹在文字里的标签只替换标签本身，且不误伤普通单字
    assert normalize_emoji_text("好呀<永远爱你>") == "好呀jine_pien_loveforever"
    assert normalize_emoji_text("好的") == "好的"
    assert normalize_emoji_text("jine_pien_ok") == "jine_pien_ok"


@pytest.mark.parametrize("arguments,finish,name", [
    ('{"q":', "tool_calls", "search"),
    ("{}", "length", "search"),
    ("null", "tool_calls", "search"),
    ("[]", "tool_calls", "search"),
    ("{}", "tool_calls", "sea"),
])
def test_stream_rejects_invalid_tools_before_ready_event(arguments, finish, name):
    manager = ModelManager.__new__(ModelManager)
    manager.tools = [{"type": "function", "function": {"name": "search"}}]
    tool = SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name=name, arguments=arguments))
    chunks = [SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, delta=SimpleNamespace(tool_calls=[tool]))])]
    manager.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(chunks)))
    events = []
    async def emit(event):
        events.append(event)
    with pytest.raises(model_module.InvalidToolCallError):
        asyncio.run(manager.stream_generate([], {}, on_event=emit))
    assert not any(e["type"] == "tool_call" for e in events)


@pytest.mark.parametrize("value", ["<抱歉]", "【抱歉>", "好>", "(好", "(抱歉)", "（抱歉）", "嗯(抱歉)", "好"])
def test_invalid_or_round_wrappers_remain_text(value):
    from app.config.emoji_labels import normalize_emoji_text
    assert normalize_emoji_text(value) == value


@pytest.mark.parametrize("value", ["<抱歉>", "【抱歉】", "[抱歉]", "抱歉"])
def test_supported_emoji_forms(value):
    from app.config.emoji_labels import normalize_emoji_text
    assert normalize_emoji_text(value) == "jine_pien_sorry"
