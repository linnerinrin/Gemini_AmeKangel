"""P0-1 回归：保存设置端点的「凭据—地址绑定」与出站校验。

覆盖审计报告 P0-1：只改 base_url 不重输 key 时，旧 key 不得被绑定到新地址落盘。
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config.llm_import import load_llm_settings

PLAINTEXT_KEY = "sk-p0-1234567890abcdef-SECRET"
NEW_KEY = "sk-p0-9876543210fedcba-NEWKEY"
EVIL_URL = "https://gateway.attacker-example.net/v1"


@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    """与生产同一个加载器：settings_routes 与测试拿到同一份 llm_settings 模块对象。"""
    llm_settings = load_llm_settings()
    monkeypatch.setattr(llm_settings, "settings_path", lambda: tmp_path / "data" / "llm_settings.json")
    monkeypatch.delenv("LLM_ALLOW_PRIVATE_BASE_URL", raising=False)
    return llm_settings


@pytest.fixture
def client(fake_services, settings_env):
    import importlib
    module = importlib.import_module("app.main")
    return TestClient(module.app)


def _post(client, profiles, active="deepseek"):
    return client.post("/chat/settings/llm", json={"active": active, "profiles": profiles})


def _seed_deepseek(client, base_url=None, api_key=PLAINTEXT_KEY):
    profile = {"model": "deepseek-v4-flash"}
    if base_url is not None:
        profile["base_url"] = base_url
    if api_key is not None:
        profile["api_key"] = api_key
    response = _post(client, {"deepseek": profile})
    assert response.status_code == 200, response.text
    return response


# ═══════════════════════════════════════
# 1. 改地址但不提交 key → 422，且不落盘
# ═══════════════════════════════════════

def test_changing_base_url_without_key_is_rejected(client, settings_env, tmp_path):
    _seed_deepseek(client)

    response = _post(client, {"deepseek": {"base_url": EVIL_URL, "model": "deepseek-v4-flash"}})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "base_url" in detail and "API Key" in detail
    #旧 key 仍只属于旧地址：生效配置没有被改写，磁盘上也没有新地址
    effective = settings_env.resolve_profile("deepseek")
    assert effective["base_url"] == "https://api.deepseek.com"
    assert effective["api_key"] == PLAINTEXT_KEY
    saved = json.loads((tmp_path / "data" / "llm_settings.json").read_text(encoding="utf-8"))
    assert EVIL_URL not in json.dumps(saved)


def test_empty_key_string_on_new_base_url_is_rejected(client):
    """前端改地址时把 key 输入框清空会提交空串——同样属于“未重输 key”。"""
    _seed_deepseek(client)

    response = _post(client, {"deepseek": {"base_url": EVIL_URL, "api_key": ""}})

    assert response.status_code == 422
    assert "API Key" in response.json()["detail"]


def test_mask_replay_on_new_base_url_is_rejected(client, settings_env):
    """把掩码原样回传也是“沿用旧 key”，不允许跟随地址迁移。"""
    _seed_deepseek(client)
    masked = settings_env.settings_view()["profiles"]["deepseek"]["api_key"]

    response = _post(client, {"deepseek": {"base_url": EVIL_URL, "api_key": masked}})

    assert response.status_code == 422
    assert "API Key" in response.json()["detail"]


def test_first_time_base_url_without_key_reports_missing_key(client):
    """首次配置某厂商（没有可搬运的旧 key）时，提示应是“未配置 API Key”，
    而不是把地址变更误报成“已保存 Key 被转发”。"""
    response = _post(
        client,
        {"openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"}},
        active="openai",
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "未配置 API Key" in detail
    assert "转发" not in detail


# ═══════════════════════════════════════
# 2. 显式重输 key 后可改地址；仅归一化差异不算改地址
# ═══════════════════════════════════════

def test_changing_base_url_with_explicit_key_is_allowed(client, settings_env):
    _seed_deepseek(client)

    response = _post(client, {"deepseek": {"base_url": EVIL_URL, "api_key": NEW_KEY}})

    assert response.status_code == 200
    effective = settings_env.resolve_profile("deepseek")
    assert effective["base_url"] == EVIL_URL
    assert effective["api_key"] == NEW_KEY
    assert NEW_KEY not in json.dumps(response.json(), ensure_ascii=False)


def test_trailing_slash_and_case_do_not_require_new_key(client, settings_env):
    """大小写/末尾斜杠归一化后地址相同，沿用旧 key 属合法操作。"""
    _seed_deepseek(client)

    response = _post(client, {"deepseek": {"base_url": "HTTPS://API.DEEPSEEK.COM/"}})

    assert response.status_code == 200
    effective = settings_env.resolve_profile("deepseek")
    assert effective["api_key"] == PLAINTEXT_KEY


# ═══════════════════════════════════════
# 3. 保存路径的出站校验
# ═══════════════════════════════════════

def test_saving_private_base_url_is_blocked(client, settings_env):
    response = _post(
        client,
        {"custom": {"base_url": "http://127.0.0.1:11434/v1", "model": "local", "api_key": NEW_KEY}},
    )

    assert response.status_code == 400
    assert "不允许外发" in response.json()["detail"]
    assert settings_env.load_settings().profiles == {}


def test_private_base_url_allowed_by_explicit_env_switch(client, settings_env, monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_PRIVATE_BASE_URL", "1")

    response = _post(
        client,
        {"custom": {"base_url": "http://127.0.0.1:11434/v1", "model": "local", "api_key": NEW_KEY}},
        active="custom",
    )

    assert response.status_code == 200
    assert settings_env.resolve_profile("custom")["base_url"] == "http://127.0.0.1:11434/v1"
