"""P1-4 回归：未配置 API Key 时不再 500/通用错误，而是可读 422 / 带引导文案的流式 error。"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config.llm_import import load_llm_settings


@pytest.fixture
def client(fake_services):
    import importlib
    module = importlib.import_module("app.main")
    return TestClient(module.app)


def _missing_key_error():
    """用与运行时同一个模块对象构造 NotConfiguredError，保证 isinstance 命中。"""
    llm_settings = load_llm_settings()
    return llm_settings.NotConfiguredError(llm_settings.missing_key_message("deepseek"))


def _sse_events(text):
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_single_chat_without_key_returns_422(client, fake_services):
    async def boom(prompt, mode=None):
        raise _missing_key_error()

    fake_services.chat.single_chat = boom

    response = client.post(
        "/chat/single_chat",
        json={"prompt": "在吗", "session_id": "session-1", "mode": "2"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "API Key" in detail and "控制面板" in detail


def test_stream_without_key_emits_actionable_error(client, fake_services):
    async def boom(prompt, mode=None, on_event=None):
        raise _missing_key_error()

    fake_services.chat.stream_chat = boom

    response = client.post(
        "/chat/stream_chat",
        json={"prompt": "在吗", "session_id": "session-1", "mode": "2"},
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    errors = [event for event in events if event.get("type") == "error"]
    assert len(errors) == 1
    assert "API Key" in errors[0]["message"]
    assert "生成过程中出现错误" not in errors[0]["message"]
