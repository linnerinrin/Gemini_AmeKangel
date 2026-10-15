import asyncio
import pytest
from collections import deque
import sys
import types

from conftest import load_source_module


graph_module = types.ModuleType("app.managers.GraphManager")


class ImportTimeGraphManager:
    pass


graph_module.GraphManager = ImportTimeGraphManager
graph_module.ReflectionError = type("ReflectionError", (ValueError,), {})
sys.modules["app.managers.GraphManager"] = graph_module
managers_module = types.ModuleType("app.managers")
for name in ("ModelManager", "FuncCallManager", "RAGManager", "MemoryManager"):
    setattr(managers_module, name, type(name, (), {}))
sys.modules["app.managers"] = managers_module

chat_module = load_source_module("chat_session_under_test", "app/services/chat_session.py")


class FakeApp:
    async def ainvoke(self, state):
        return {
            "reply": "reply",
            "think_text": ["think"],
            "reflection_finish": True,
        }


class FakeGraphManager:
    def init(self, session):
        self.session = session

    def build(self):
        return FakeApp()

    def load_mode(self, mode):
        self.mode = mode


class FakeRAG:
    def __init__(self):
        self.mode = None
        self.context = None

    def load_mode(self, mode):
        self.mode = mode

    def update_context(self, context):
        self.context = context


class FakeFCall:
    def begin_turn(self):
        self.turn_started = True

    def init(self, rag_mgr, memory_mgr, model_mgr):
        self.initialized = True

    def load_mode(self, mode):
        self.mode = mode


class FakeMemory:
    def __init__(self):
        self.short_memory = deque(maxlen=3)
        self.long_memory = ""
        self.status = None
        self.status_prompts = ""
        self.history = []
        self.task_mode = "chat"

    def init(self, model_mgr):
        self.model_mgr = model_mgr

    def bind_session_id(self, session_id):
        self.session_id = session_id

    def set_task_mode(self, task_mode):
        task_mode = str(task_mode)
        if task_mode not in ("chat", "work"):
            raise ValueError(f"未知对话模式: {task_mode}")
        self.task_mode = task_mode

    def load_mode(self, mode):
        self.mode = mode

    def refresh(self, session_id):
        self.session_id = session_id

    def update_short_memory(self, prompt, reply, thinking=None, model=None, provider=None):
        self.last_short = (prompt, reply, thinking)
        # 落库时用的是当时的任务轴，测试要能验证这一点
        self.last_short_task_mode = self.task_mode
        # 这一轮透传下来的实际模型（R12）：拿不到就是 (None, None)
        self.last_model_meta = (provider, model)
        self.history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ])

    def load_short_memory(self):
        self.short_memory.append({"role": "user", "content": "loaded"})


class FakeModel:
    pass


def make_session(monkeypatch):
    monkeypatch.setattr(chat_module, "GraphManager", FakeGraphManager)
    return chat_module.ChatSession(FakeModel(), FakeFCall(), FakeRAG(), FakeMemory(), FakeGraphManager())


def test_load_mode_updates_mode(monkeypatch):
    session = make_session(monkeypatch)
    session.load_mode("0")

    assert session.current_mode == "0"


def _order_of(prompt, *segments):
    """返回每段首次出现的下标，用来断言装配顺序（重复段取首次即可）"""
    return [prompt.index(segment) for segment in segments]


def test_initial_prompts_follow_base_role_task_order(monkeypatch):
    """三个节点属性在构造后就是按 (mode, task_mode) 装配好的"""
    session = make_session(monkeypatch)

    assert session.current_task_mode == "chat"
    assert session.call_prompt == chat_module.config.call_prompt_for("chat")
    assert session.reflection_prompt == chat_module.config.reflection_prompt_for("chat")
    base = chat_module.config.BASE_PROMPT
    role = chat_module.config.ROLE_PROMPT[session.current_mode]
    task = chat_module.config.TASK_PROMPT["chat"]
    assert session.sys_prompt == chat_module.config.join_prompt(base, role, task)
    # 顺序必须是 base -> role -> task
    positions = _order_of(session.sys_prompt, base, role, task)
    assert positions == sorted(positions), f"装配顺序不是 base->role->task: {positions}"


def test_switch_task_mode_updates_all_three_node_prompts(monkeypatch):
    """切换任务轴后 sys_prompt / call_prompt / reflection_prompt 都要随之变化"""
    session = make_session(monkeypatch)
    chat_call, chat_sys, chat_reflection = (
        session.call_prompt, session.sys_prompt, session.reflection_prompt)

    asyncio.run(session.single_chat("hello", task_mode="work"))

    assert session.current_task_mode == "work"
    assert session.call_prompt != chat_call
    assert session.sys_prompt != chat_sys
    assert session.reflection_prompt != chat_reflection
    assert session.call_prompt == chat_module.config.call_prompt_for("work")
    assert session.reflection_prompt == chat_module.config.reflection_prompt_for("work")
    base = chat_module.config.BASE_PROMPT
    role = chat_module.config.ROLE_PROMPT[session.current_mode]
    task = chat_module.config.TASK_PROMPT["work"]
    assert session.sys_prompt == chat_module.config.join_prompt(base, role, task)
    positions = _order_of(session.sys_prompt, base, role, task)
    assert positions == sorted(positions), f"装配顺序不是 base->role->task: {positions}"
    # 切回 chat 也要复原
    asyncio.run(session.single_chat("hello", task_mode="chat"))
    assert (session.call_prompt, session.sys_prompt, session.reflection_prompt) == (
        chat_call, chat_sys, chat_reflection)


def test_load_mode_rebuilds_role_segment_but_keeps_task_axis(monkeypatch):
    """场景轴变化只换 role 段，任务轴不受影响"""
    session = make_session(monkeypatch)
    asyncio.run(session.single_chat("hello", task_mode="work"))
    work_task = chat_module.config.TASK_PROMPT["work"]

    session.load_mode("2")

    assert session.current_mode == "2"
    assert session.current_task_mode == "work"
    assert chat_module.config.ROLE_PROMPT["2"] in session.sys_prompt
    assert chat_module.config.ROLE_PROMPT["0"] not in session.sys_prompt
    assert session.sys_prompt.endswith(work_task)
    assert session.call_prompt == chat_module.config.call_prompt_for("work")
    assert session.reflection_prompt == chat_module.config.reflection_prompt_for("work")


def test_refresh_delegates_to_memory(monkeypatch):
    session = make_session(monkeypatch)
    session.refresh("session-1")

    assert session.memory_mgr.session_id == "session-1"
    assert session.status is None


def test_refresh_syncs_task_mode_from_memory(monkeypatch):
    """任务轴以 MemoryManager 为准，refresh 之后不能两边不一致"""
    session = make_session(monkeypatch)
    session.memory_mgr.task_mode = "work"

    session.refresh("session-1")

    assert session.current_task_mode == "work"


def test_validate_task_mode_keeps_current_and_rejects_unknown(monkeypatch):
    session = make_session(monkeypatch)

    assert session.current_task_mode == "chat"
    assert session.validate_task_mode(None) == "chat"
    assert session.validate_task_mode("work") == "work"
    with pytest.raises(ValueError, match="未知对话模式"):
        session.validate_task_mode("sleep")
    assert session.current_task_mode == "chat"


def test_single_chat_sets_task_mode_before_turn(monkeypatch):
    """任务轴必须在 begin_turn 之前落定：load_mode 会按轴重载短期记忆"""
    session = make_session(monkeypatch)
    order = []
    original_set = session.memory_mgr.set_task_mode

    def record_set(task_mode):
        order.append(("task_mode", task_mode))
        original_set(task_mode)

    def record_begin():
        order.append(("begin_turn", None))

    session.memory_mgr.set_task_mode = record_set
    session.fcall_mgr.begin_turn = record_begin

    asyncio.run(session.single_chat("hello", task_mode="work"))

    assert order[0] == ("task_mode", "work")
    assert order[1] == ("begin_turn", None)
    assert session.current_task_mode == "work"
    assert session.memory_mgr.task_mode == "work"
    # 这一轮落库时带的任务轴是 work
    assert session.memory_mgr.last_short_task_mode == "work"
    assert session.memory_mgr.last_short == (
        "hello", "reply", [{"kind": "text", "text": "think"}],
    )


def test_single_chat_rejects_unknown_task_mode(monkeypatch):
    session = make_session(monkeypatch)
    saved = []
    session.update_memory = lambda *args, **kwargs: saved.append(args)

    with pytest.raises(ValueError, match="未知对话模式"):
        asyncio.run(session.single_chat("hello", task_mode="sleep"))

    assert saved == []
    assert session.current_task_mode == "chat"
    assert session.memory_mgr.task_mode == "chat"


def test_stream_chat_carries_task_mode(monkeypatch):
    """流式路径同样要把任务轴透传到记忆层（顺序与 single_chat 一致）"""
    session = make_session(monkeypatch)
    saved = []
    session.update_memory = lambda *args, **kwargs: saved.append(args)

    assert asyncio.run(session.stream_chat("q", task_mode="work")) == ("reply", "think")

    assert session.current_task_mode == "work"
    assert session.memory_mgr.task_mode == "work"


def test_stream_chat_rejects_unknown_task_mode(monkeypatch):
    session = make_session(monkeypatch)
    saved = []
    session.update_memory = lambda *args, **kwargs: saved.append(args)

    with pytest.raises(ValueError, match="未知对话模式"):
        asyncio.run(session.stream_chat("q", task_mode="unknown"))

    assert saved == []
    assert session.memory_mgr.task_mode == "chat"


def test_single_chat_updates_memory_and_history(monkeypatch):
    session = make_session(monkeypatch)

    reply, think = asyncio.run(session.single_chat("hello"))

    assert reply == "reply"
    assert think == "think"
    # 非流式路径没有事件，思考文本要包成文本项一起入库
    assert session.memory_mgr.last_short == (
        "hello", "reply", [{"kind": "text", "text": "think"}],
    )
    assert session.history[-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    assert session.rag_mgr.context


def test_single_chat_passes_actual_model_meta_to_memory(monkeypatch):
    """R12：每轮落库要带上 ModelManager 实际解析出的厂商/模型，而不是配置里的静态值"""
    session = make_session(monkeypatch)
    session.model_mgr.last_provider = "openai"
    session.model_mgr.last_model = "gpt-4o-mini"

    asyncio.run(session.single_chat("hello"))

    assert session.memory_mgr.last_model_meta == ("openai", "gpt-4o-mini")


def test_stream_chat_passes_actual_model_meta_to_memory(monkeypatch):
    session = make_session(monkeypatch)
    session.model_mgr.last_provider = "deepseek"
    session.model_mgr.last_model = "deepseek-v4-flash"

    asyncio.run(session.stream_chat("q"))

    assert session.memory_mgr.last_model_meta == ("deepseek", "deepseek-v4-flash")


def test_missing_model_meta_is_passed_as_none(monkeypatch):
    """拿不到模型信息时写 None，不能因此报错挡住落库"""
    session = make_session(monkeypatch)  # FakeModel 上没有任何 last_* 属性

    asyncio.run(session.single_chat("hello"))

    assert session.memory_mgr.last_model_meta == (None, None)
    assert session.memory_mgr.last_short == (
        "hello", "reply", [{"kind": "text", "text": "think"}],
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("outcome", ["raise", "missing", "rejected"])
def test_review_failure_never_saves_history(monkeypatch, streaming, outcome):
    session = make_session(monkeypatch)
    saved, events = [], []
    session.update_memory = lambda *args, **kwargs: saved.append(args)
    async def invoke(state):
        if outcome == "raise":
            raise chat_module.ReflectionError("审查失败")
        result = {"reply": "unapproved draft", "think_text": []}
        if outcome == "rejected":
            result["reflection_finish"] = False
        return result
    session.app.ainvoke = invoke
    async def emit(event):
        events.append(event)
    with pytest.raises(chat_module.ReflectionError):
        asyncio.run(session.stream_chat("q", on_event=emit) if streaming else session.single_chat("q"))
    assert saved == []
    assert session.history == []
    assert not any(e["type"] == "done" for e in events)


def test_stream_approved_reply_saved_once_then_done(monkeypatch):
    session = make_session(monkeypatch)
    saved, events = [], []
    session.update_memory = lambda *args, **kwargs: saved.append(args)
    async def emit(event):
        if event["type"] == "done":
            assert saved == [("q", "reply", [])]
        events.append(event)
    assert asyncio.run(session.stream_chat("q", on_event=emit)) == ("reply", "think")
    assert sum(e["type"] == "done" for e in events) == 1


def test_stream_chat_persists_structured_thinking_trace(monkeypatch):
    """流式对话要把思考事件累积成结构化 trace 一起入库，历史才能回放同样的折叠块"""
    session = make_session(monkeypatch)
    saved = []
    session.update_memory = lambda *args, **kwargs: saved.append(args)
    emitted = []

    async def emit(event):
        emitted.append(event["type"])

    async def invoke(state):
        stream_event = state["stream_event"]
        await stream_event({"type": "status", "stage": "thinking"})
        await stream_event({"type": "node", "name": "agent"})
        await stream_event({"type": "thinking_delta", "text": "先"})
        await stream_event({"type": "thinking_delta", "text": "想一下"})
        await stream_event({"type": "tool_call", "name": "search_web"})
        await stream_event({"type": "node", "name": "generate_reply"})
        await stream_event({"type": "answer_delta", "text": "回复"})
        return {"reply": "reply", "think_text": ["think"], "reflection_finish": True}

    session.app.ainvoke = invoke

    assert asyncio.run(session.stream_chat("q", on_event=emit)) == ("reply", "think")

    assert saved == [("q", "reply", [
        {"kind": "node", "name": "agent"},
        {"kind": "text", "text": "先想一下"},
        {"kind": "tool_call", "name": "search_web"},
        {"kind": "node", "name": "generate_reply"},
    ])]
    # trace 只收集思考类事件，正文/状态照旧转发给前端
    assert emitted == ["status", "status", "node", "thinking_delta", "thinking_delta",
                       "tool_call", "node", "answer_delta", "done"]


def test_stream_chat_without_events_still_captures_trace(monkeypatch):
    """on_event 为空（内部调用）时也要累积 trace，只是不往外发"""
    session = make_session(monkeypatch)
    saved = []
    session.update_memory = lambda *args, **kwargs: saved.append(args)

    async def invoke(state):
        await state["stream_event"]({"type": "node", "name": "agent"})
        return {"reply": "reply", "think_text": [], "reflection_finish": True}

    session.app.ainvoke = invoke

    assert asyncio.run(session.stream_chat("q")) == ("reply", "")

    assert saved == [("q", "reply", [{"kind": "node", "name": "agent"}])]
