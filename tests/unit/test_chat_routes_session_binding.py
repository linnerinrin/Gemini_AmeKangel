"""P1-3 回归：生成请求必须落在启动时绑定的会话实例上，不能被门面指针漂移串档。

对应审计报告 P1-3：ChatFacade._current_id 是可变指针，而 stream_chat 把实际调用
推迟到 create_task 之后；期间任何换绑都会让本轮 prompt/记忆/RAG 落进错误会话。
"""

import asyncio

import pytest


@pytest.fixture
def routes(fake_services):
    from app.api.chat import chat_routes
    return chat_routes


class StubSession:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def stream_chat(self, prompt, mode=None, on_event=None):
        self.calls.append((prompt, mode))
        if on_event:
            await on_event({"type": "answer_delta", "text": self.name})
            await on_event({"type": "done", "answer": self.name, "think": ""})
        return self.name, ""


class DriftFacade:
    """模拟真实门面：__getattr__ 在“调用时”才解析当前指针。"""

    def __init__(self, sessions, current="A"):
        self._sessions = sessions
        self._current = sessions[current]
        self.session_id = current

    def resolve_current(self):
        return self._current

    def lock_for(self, session_id):
        return None

    def bind_session_id(self, session_id):
        session_id = str(session_id)
        if session_id not in self._sessions:
            raise ValueError(session_id)
        self._current = self._sessions[session_id]
        self.session_id = session_id

    def __getattr__(self, name):
        return getattr(self._current, name)


def _run_stream(routes, request, on_started=None):
    """同一个事件循环里启动流式请求、可选地中途换绑、再消费完整个SSE。"""

    async def scenario():
        response = await routes.stream_chat(request)
        #此刻 run() 刚被 create_task 排队、还没拿到执行权：换绑窗口真实存在
        if on_started is not None:
            on_started()
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        return "".join(chunks)

    return asyncio.run(scenario())


def test_stream_uses_snapshot_taken_before_task_runs(routes, monkeypatch):
    session_a, session_b = StubSession("A"), StubSession("B")
    facade = DriftFacade({"A": session_a, "B": session_b}, current="A")
    monkeypatch.setattr(routes, "chat", facade)

    request = routes.SingleChatRequest(prompt="在吗", session_id="A", mode="2")
    body = _run_stream(routes, request, on_started=lambda: facade.bind_session_id("B"))

    assert session_a.calls == [("在吗", "2")]
    assert session_b.calls == []
    assert '"A"' in body


def test_stream_does_not_follow_facade_attr_resolution(routes, monkeypatch):
    """即使 __getattr__ 此刻会解析到别的会话，也只用启动时捕获的实例。"""
    session_a, decoy = StubSession("A"), StubSession("decoy")

    class FacadeWithDecoy(DriftFacade):
        @property
        def stream_chat(self):
            return decoy.stream_chat

    facade = FacadeWithDecoy({"A": session_a}, current="A")
    monkeypatch.setattr(routes, "chat", facade)

    request = routes.SingleChatRequest(prompt="你好", session_id="A", mode="2")
    body = _run_stream(routes, request)

    assert session_a.calls == [("你好", "2")]
    assert decoy.calls == []
    assert '"A"' in body
