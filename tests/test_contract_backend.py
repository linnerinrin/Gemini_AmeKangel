import importlib
import json

import pytest

from conftest import CONTRACT_PATH, assert_shape, contract_entries

# 契约文件在收集阶段就要读：用例参数直接由已登记的 (path, method) 条目生成，
# 补登记后自然进入参数化，不会出现"登记了但没被测到"的情况。
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

SETTINGS_PREFIX = "/chat/settings"
# 设置类接口要打桩（读写设置文件 / 连通性测试），与普通路由分开跑
ROUTE_CASES = [
    (path, entry["method"])
    for path, entry in contract_entries(CONTRACT)
    if not path.startswith(SETTINGS_PREFIX)
]
SETTINGS_CASES = [
    (path, entry["method"])
    for path, entry in contract_entries(CONTRACT)
    if path.startswith(SETTINGS_PREFIX)
]


def _entry(contracts, path, method):
    """按 (path, method) 取契约条目，取不到直接失败而不是静默跳过。"""
    for entry in contracts[path]["methods"]:
        if entry["method"] == method:
            return entry
    raise AssertionError(f"契约未登记 {method} {path}")


@pytest.mark.parametrize("path,method", ROUTE_CASES)
def test_backend_routes_follow_contract(client, contracts, path, method):
    contract = _entry(contracts, path, method)

    if path in {"/chat/single_chat", "/chat/mode_change", "/chat/get_status", "/chat/history"}:
        client.post("/chat/new_session", json={"session_num": 1})
    elif path in {"/chat/chose_session", "/chat/delete_session"}:
        client.post(
            "/chat/new_session",
            json={"session_num": contract["request"]["session_num"]},
        )

    response = client.request(
        method,
        path,
        json=contract["request"],
    )

    assert response.status_code == 200
    assert_shape(response.json(), contract["response"])


def test_history_response_item_shape_follows_contract(client, fake_services, contracts):
    """历史每轮的字段受契约保护（R12 起每轮还带 llm_model / llm_provider）"""
    contract = _entry(contracts, "/chat/history", "GET")
    assert "mode" in contract.get("query", {}), "契约要登记 /chat/history 的 mode 查询参数"

    item = {
        "created_at": "2026-01-01 00:00:00",
        "user": "你好",
        "assistant": "回复",
        "thinking": [{"kind": "text", "text": "想了"}],
        "task_mode": "chat",
        "llm_model": "deepseek-v4-flash",
        "llm_provider": "deepseek",
    }
    client.post("/chat/new_session", json={"session_num": 1})
    fake_services.chat.history_by_mode["2"] = [item]

    payload = client.get("/chat/history?mode=2").json()

    assert len(payload) == 1
    assert set(payload[0]) == set(contract["response_item"])
    assert_shape(payload[0], contract["response_item"])
    # 老数据（新增列之前的历史行）这两列是 NULL，必须仍然满足契约
    assert_shape(dict(item, task_mode=None, llm_model=None, llm_provider=None),
                 contract["response_item"])


@pytest.mark.parametrize("path,method", SETTINGS_CASES)
def test_backend_settings_routes_follow_contract(client, contracts, path, method, monkeypatch):
    """设置接口也要按契约校验形状；连通性测试与落盘都打桩，不碰真实网络/设置文件。"""
    contract = _entry(contracts, path, method)
    settings_routes = importlib.import_module("app.api.chat.settings_routes")

    payload = contract["request"]
    if path == "/chat/settings/llm/test":
        async def fake_candidate(provider, profile):
            return {"ok": True, "message": "连接成功，模型可用"}

        monkeypatch.setattr(settings_routes, "test_candidate", fake_candidate)
        # 契约请求只登记前端必定发送的 provider；base_url/model/api_key 是可选覆盖项，
        # 这里补上候选值，免得连通性校验依赖本机 .env 里有没有 key
        payload = {**payload, **contract.get("optional_request", {})}
    elif method == "POST":
        monkeypatch.setattr(settings_routes.llm_settings, "apply_update", lambda payload: None)

    response = client.request(method, path, json=payload)

    assert response.status_code == 200
    assert_shape(response.json(), contract["response"])


def test_auth_routes_are_removed(client):
    assert client.post("/auth/login", json={}).status_code == 404


def test_stream_chat_uses_sse_contract(client):
    client.post("/chat/new_session", json={"session_num": 1})
    response = client.post("/chat/stream_chat", json={"prompt": "hello", "mode": "2"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "thinking_delta" in response.text
    assert "answer_delta" in response.text
    assert "done" in response.text
