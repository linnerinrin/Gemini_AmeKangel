import pytest


def test_start_session_binds_chat_session(client, fake_services):
    response = client.post(
        "/chat/start_session",
        json={"session_id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_services.chat.session_id == "session-1"


def test_chat_generation_uses_backend_schema(client):
    client.post("/chat/new_session", json={"session_num": 1})
    response = client.post(
        "/chat/single_chat",
        json={"prompt": "hello"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["think"], str)


def test_chat_generation_passes_task_mode(client, fake_services):
    """任务轴要跟 mode 一起透传到服务层：chat=闲聊 / work=工作"""
    client.post("/chat/new_session", json={"session_num": 1})
    response = client.post(
        "/chat/single_chat",
        json={"prompt": "hello", "mode": "0", "task_mode": "work"},
    )

    assert response.status_code == 200
    assert fake_services.chat.last_task_mode == "work"


def test_stream_chat_passes_task_mode(client, fake_services):
    client.post("/chat/new_session", json={"session_num": 1})
    response = client.post(
        "/chat/stream_chat",
        json={"prompt": "hello", "mode": "2", "task_mode": "work"},
    )

    assert response.status_code == 200
    assert fake_services.chat.last_task_mode == "work"


@pytest.mark.parametrize("path", ["/chat/single_chat", "/chat/stream_chat"])
def test_chat_rejects_unknown_task_mode(client, path):
    """前端传了非法任务轴要 422，不能静默退回 chat 把脏数据写进库"""
    client.post("/chat/new_session", json={"session_num": 1})
    response = client.post(path, json={"prompt": "hello", "task_mode": "sleep"})

    assert response.status_code == 422
    assert response.json()["detail"] == "未知对话模式"


def test_chat_mode_and_session_routes(client):
    new_response = client.post("/chat/new_session", json={"session_num": 1})
    mode_response = client.post("/chat/mode_change", json={"mode": "0"})
    chose_response = client.post("/chat/chose_session", json={"session_num": 1})
    delete_response = client.post("/chat/delete_session", json={"session_num": 1})

    assert mode_response.status_code == 200
    assert mode_response.json() is True
    assert new_response.status_code == 200
    assert new_response.json()["session_id"] == "session-1"
    assert new_response.json()["session_num"] == 1
    assert chose_response.status_code == 200
    assert chose_response.json()["ok"] is True
    assert delete_response.status_code == 200
    assert delete_response.json() is True


def test_chat_history_and_status_routes(client):
    client.post("/chat/new_session", json={"session_num": 1})
    sessions_response = client.get("/chat/sessions")
    history_response = client.get("/chat/history")
    status_response = client.post("/chat/get_status")

    assert sessions_response.status_code == 200
    assert "sessions" in sessions_response.json()
    assert history_response.status_code == 200
    assert history_response.json() == []
    assert status_response.status_code == 200
    assert status_response.json()["affection"] == 50


def test_chat_route_requires_selected_save(client):
    response = client.get("/chat/history")

    assert response.status_code == 409
    assert response.json()["detail"] == "请先选择存档"


def test_history_can_be_read_per_live_mode(client, fake_services):
    """直播窗口进入主题时要能单独拉取该主题(mode)的上下文"""
    client.post("/chat/new_session", json={"session_num": 1})
    fake_services.chat.history_by_mode["2"] = [
        {"created_at": "2026-09-05 12:00:00", "user": "你好", "assistant": "jine_pien_ok"},
    ]

    mode_response = client.get("/chat/history?mode=2")
    other_response = client.get("/chat/history?mode=3")
    default_response = client.get("/chat/history")

    assert mode_response.status_code == 200
    assert len(mode_response.json()) == 1
    assert mode_response.json()[0]["user"] == "你好"
    assert other_response.json() == []
    assert default_response.json() == []


def test_history_with_mode_still_requires_selected_save(client):
    response = client.get("/chat/history?mode=1")

    assert response.status_code == 409
    assert response.json()["detail"] == "请先选择存档"
