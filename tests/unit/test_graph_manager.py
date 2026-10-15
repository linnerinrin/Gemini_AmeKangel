import pytest
import asyncio
import sys
import types

from conftest import load_source_module


class FakeToolMessage:
    pass


class FakeAIMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, router):
        self.edges.append((source, router))

    def compile(self):
        return self


class FakeToolNode:
    def __init__(self, tools, messages_key=None):
        self.tools = tools
        self.messages_key = messages_key


langchain_core = sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
messages_module = types.ModuleType("langchain_core.messages")
messages_module.ToolMessage = FakeToolMessage
messages_module.AIMessage = FakeAIMessage
langgraph_module = types.ModuleType("langgraph")
constants_module = types.ModuleType("langgraph.constants")
constants_module.START = "__start__"
constants_module.END = "__end__"
graph_module_fake = types.ModuleType("langgraph.graph")
graph_module_fake.StateGraph = FakeStateGraph
prebuilt_module = types.ModuleType("langgraph.prebuilt")
prebuilt_module.ToolNode = FakeToolNode
sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.messages"] = messages_module
sys.modules["langgraph"] = langgraph_module
sys.modules["langgraph.constants"] = constants_module
sys.modules["langgraph.graph"] = graph_module_fake
sys.modules["langgraph.prebuilt"] = prebuilt_module

graph_module = load_source_module("graph_manager_under_test", "app/managers/GraphManager.py")


def make_manager(session):
    mgr = graph_module.GraphManager()
    mgr.init(session)
    return mgr


def test_toolcall_router_routes_without_tool_calls():
    mgr = make_manager(object())

    assert mgr.toolcall_router({"tool_finish": True}) == "generate_reply"


def test_toolcall_router_limits_tool_turns(monkeypatch):
    mgr = make_manager(object())

    assert mgr.toolcall_router({"tool_finish": False, "tool_turns": graph_module.MAX_TOOL_TURNS}) == "generate_reply"
    assert mgr.toolcall_router({"tool_finish": False, "tool_turns": 0}) == "tools"


def test_create_nodes_contains_expected_nodes():
    mgr = make_manager(object())

    assert set(mgr.create_nodes()) == {"agent", "generate_reply", "reflection"}


def test_reflection_router_accepts_or_retries():
    mgr = make_manager(object())

    assert mgr.reflection_router({"reflection_finish": True}) == graph_module.END
    with pytest.raises(graph_module.ReflectionError):
        mgr.reflection_router({"reflection_finish": False, "reflection_turns": graph_module.MAX_REFLECTION_TURNS + 1})
    assert mgr.reflection_router({"reflection_finish": False, "reflection_turns": 0}) == "generate_reply"


def test_build_compiles_graph_with_expected_nodes():
    session = types.SimpleNamespace(fcall_mgr=types.SimpleNamespace(tools=[]))
    mgr = make_manager(session)

    app = mgr.build()

    assert app is mgr.app
    assert {"agent", "generate_reply", "tools", "reflection"} <= set(app.nodes)


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, messages, paras, use_tools=False):
        self.calls.append((messages, paras, use_tools))
        return self.outputs.pop(0)


class FakeMemory:
    status_prompts = ""

    def load_status(self):
        return self.status_prompts

    def load_long_memory(self):
        return "long memory"


def make_session(model_outputs):
    return types.SimpleNamespace(
        current_mode="0",
        call_prompt="call prompt",
        sys_prompt="system prompt",
        reflection_prompt="reflection prompt",
        short_memory=[],
        inference_para={"temperature": 0.1},
        config=types.SimpleNamespace(
            think_para={"temperature": 0.2},
            reflect_para={"temperature": 0.3},
        ),
        model_mgr=FakeModel(model_outputs),
        memory_mgr=FakeMemory(),
    )


def test_agent_node_records_tool_decision():
    session = make_session([
        types.SimpleNamespace(content="thinking", tool_calls=[{"name": "tool", "args": {}, "id": "call-1"}])
    ])
    mgr = make_manager(session)

    result = asyncio.run(mgr.agent_node({
        "agent_message": [],
        "think_text": [],
        "tool_text": [],
        "query": "hello",
        "tool_turns": 0,
    }))

    assert result["think_text"] == ["thinking"]
    assert result["tool_finish"] is False
    assert result["tool_turns"] == 1
    assert result["reflection_finish"] is False
    assert isinstance(result["agent_message"][0], FakeAIMessage)


def test_generate_reply_node_uses_memory_and_model():
    session = make_session([
        types.SimpleNamespace(content="reply", tool_calls=[])
    ])
    mgr = make_manager(session)

    result = asyncio.run(mgr.generate_reply_node({
        "query": "hello",
    }))

    assert result == {"reply": "reply"}
    messages, paras, use_tools = session.model_mgr.calls[0]
    assert messages[0]["content"] == "system prompt"
    assert paras == {"temperature": 0.1}
    assert use_tools is False


def test_reflection_node_accepts_or_returns_feedback():
    accept_session = make_session([
        types.SimpleNamespace(content="__ACCEPT__", tool_calls=[], finish_reason="stop")
    ])
    accept_mgr = make_manager(accept_session)

    accepted = asyncio.run(accept_mgr.reflection_node({
        "reply": "reply",
        "query": "hello",
    }))

    feedback_session = make_session([
        types.SimpleNamespace(content="需要修改：fix tone", tool_calls=[], finish_reason="stop")
    ])
    feedback_mgr = make_manager(feedback_session)
    feedback = asyncio.run(feedback_mgr.reflection_node({
        "reply": "reply",
        "query": "hello",
        "reflection_turns": 0,
    }))

    assert accepted == {"reflection_finish": True, "reflection_review_retries": 0}
    assert feedback == {"reflection_text": ["需要修改：fix tone"], "reflection_turns": 1, "reflection_finish": False, "reflection_review_retries": 0}


def test_reflection_verdict_is_pushed_to_the_thinking_chain():
    """reflection 的判定结果必须出现在前端思考链里（debug 15）"""
    session = make_session([types.SimpleNamespace(content="需要修改：fix tone", tool_calls=[], finish_reason="stop")])
    mgr = make_manager(session)
    events = []

    async def emit(event):
        events.append(event)

    result = asyncio.run(mgr.reflection_node({
        "reply": "reply", "query": "hello", "reflection_turns": 0, "stream_event": emit,
    }))

    assert result["reflection_turns"] == 1
    assert {"type": "node", "name": "reflection"} in events
    assert any("需要修改" in e.get("text", "") and "fix tone" in e.get("text", "") for e in events)


def test_empty_reflection_retries_once_then_fails():
    session = make_session([types.SimpleNamespace(content="", finish_reason="stop") for _ in range(2)])
    mgr = make_manager(session)
    events = []
    async def emit(event):
        events.append(event)
    with pytest.raises(graph_module.ReflectionError):
        asyncio.run(mgr.reflection_node({"reply": "reply", "query": "hello", "stream_event": emit}))
    assert len(session.model_mgr.calls) == 2
    assert not any("审查结果：通过" in e.get("text", "") for e in events)


def test_stream_decision_is_thinking_and_reply_retries_reset():
    session = make_session([])
    events = []
    async def emit(event):
        events.append(event)
    async def stream(messages, paras, use_tools, on_event):
        await on_event({'type': 'answer_delta', 'text': '简短决策' if use_tools else 'answer'})
        return types.SimpleNamespace(content='answer', thinking='', tool_calls=[])
    session.model_mgr.stream_generate = stream
    mgr = make_manager(session)
    async def run():
        await mgr.agent_node({'agent_message': [], 'query': 'hi', 'stream_event': emit})
        await mgr.generate_reply_node({'query': 'hi', 'stream_event': emit})
        await mgr.generate_reply_node({'query': 'hi', 'stream_event': emit, 'reflection_text': ['retry']})
    asyncio.run(run())
    assert events == [
        {'type': 'node', 'name': 'agent'},
        {'type': 'thinking_delta', 'text': '简短决策'},
        {'type': 'node', 'name': 'generate_reply'},
        {'type': 'answer_reset'}, {'type': 'answer_delta', 'text': 'answer'},
        {'type': 'node', 'name': 'generate_reply'},
        {'type': 'answer_reset'}, {'type': 'answer_delta', 'text': 'answer'},
    ]


@pytest.mark.parametrize("content,finish,truncated", [
    ("不能 __ACCEPT__，需要修改", "stop", False),
    ("__ACCEPT__ extra", "stop", False),
    ("__ACCEPT__", "length", True),
    ("__ACCEPT__", "content_filter", False),
    ("__ACCEPT__", None, False),
    ("需要修改：", "stop", False),
])
def test_invalid_review_retries_only_review(content, finish, truncated):
    session = make_session([
        types.SimpleNamespace(content=content, finish_reason=finish, truncated=truncated),
        types.SimpleNamespace(content="__ACCEPT__", finish_reason="stop"),
    ])
    result = asyncio.run(make_manager(session).reflection_node({"reply": "draft", "query": "q"}))
    assert result == {"reflection_finish": True, "reflection_review_retries": 1}
    assert len(session.model_mgr.calls) == 2
    assert all(not call[2] for call in session.model_mgr.calls)


def test_review_retry_budget_survives_answer_rewrite():
    session = make_session([
        types.SimpleNamespace(content="", finish_reason="stop"),
        types.SimpleNamespace(content="需要修改：语气", finish_reason="stop"),
        types.SimpleNamespace(content="", finish_reason="stop"),
    ])
    mgr = make_manager(session)
    state = {"reply": "draft", "query": "q"}
    state.update(asyncio.run(mgr.reflection_node(state)))
    assert state["reflection_review_retries"] == 1
    assert state["reflection_finish"] is False
    with pytest.raises(graph_module.ReflectionError):
        asyncio.run(mgr.reflection_node(state))
    assert len(session.model_mgr.calls) == 3


def test_rewrite_exhaustion_is_failure_not_success():
    session = make_session([types.SimpleNamespace(content="需要修改：语气", finish_reason="stop")])
    with pytest.raises(graph_module.ReflectionError):
        asyncio.run(make_manager(session).reflection_node({"reply": "draft", "query": "q", "reflection_turns": graph_module.MAX_REFLECTION_TURNS}))
    assert len(session.model_mgr.calls) == 1


def test_agent_rejects_truncated_tool_calls_before_routing():
    session = make_session([types.SimpleNamespace(content="", truncated=True, tool_calls=[{"name": "tool", "args": {}, "id": "x"}])])
    with pytest.raises(ValueError):
        asyncio.run(make_manager(session).agent_node({"agent_message": [], "query": "q"}))


def node_prompt_sessions(task_mode, sys_prompt):
    """两轴正交下的桩 session：三个节点提示词都带上自身的 task_mode 便于断言"""
    call_prompt = f"call-{task_mode}"
    reflection_prompt = f"reflection-{task_mode}"
    agent_session = types.SimpleNamespace(
        task_mode=task_mode,
        call_prompt=call_prompt,
        sys_prompt=sys_prompt,
        reflection_prompt=reflection_prompt,
        model_mgr=FakeModel([types.SimpleNamespace(content="think", tool_calls=[])]),
        config=types.SimpleNamespace(think_para={"temperature": 0.2}),
        memory_mgr=types.SimpleNamespace(tools=[]),
    )
    reply_session = types.SimpleNamespace(
        task_mode=task_mode,
        call_prompt=call_prompt,
        sys_prompt=sys_prompt,
        reflection_prompt=reflection_prompt,
        short_memory=[],
        inference_para={"temperature": 0.1},
        model_mgr=FakeModel([types.SimpleNamespace(content="reply", tool_calls=[])]),
        memory_mgr=FakeMemory(),
    )
    reflection_session = types.SimpleNamespace(
        task_mode=task_mode,
        call_prompt=call_prompt,
        sys_prompt=sys_prompt,
        reflection_prompt=reflection_prompt,
        model_mgr=FakeModel([types.SimpleNamespace(content="__ACCEPT__", finish_reason="stop")]),
        config=types.SimpleNamespace(reflect_para={"temperature": 0.3}),
    )
    return agent_session, reply_session, reflection_session


def test_three_nodes_read_their_prompt_from_the_session():
    """三个节点各读 session 上的一个属性，两轴切换只需换 session 上的值（GraphManager 不改）"""
    observed = {}

    async def run(task_mode, sys_prompt):
        agent_session, reply_session, reflection_session = node_prompt_sessions(task_mode, sys_prompt)
        await make_manager(agent_session).agent_node({"agent_message": [], "query": "q"})
        await make_manager(reply_session).generate_reply_node({"query": "q"})
        await make_manager(reflection_session).reflection_node({"reply": "draft", "query": "q"})
        observed[task_mode] = {
            "agent": agent_session.model_mgr.calls[0][0][0]["content"],
            "generate_reply": reply_session.model_mgr.calls[0][0][0]["content"],
            "reflection": reflection_session.model_mgr.calls[0][0][0]["content"],
        }

    asyncio.run(run("chat", "sys-chat"))
    asyncio.run(run("work", "sys-work"))

    assert observed["chat"] == {
        "agent": "call-chat", "generate_reply": "sys-chat", "reflection": "reflection-chat",
    }
    assert observed["work"] == {
        "agent": "call-work", "generate_reply": "sys-work", "reflection": "reflection-work",
    }
    for node_name in ("agent", "generate_reply", "reflection"):
        assert observed["chat"][node_name] != observed["work"][node_name], f"{node_name} 未随任务轴变化"
