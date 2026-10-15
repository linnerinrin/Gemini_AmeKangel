"""P5a：模型设置（读写/掩码/校验/.env 回退）+ ModelManager 实时替换

关键断言：任何接口响应都不得出现明文 key。
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import load_source_module
from app.config.llm_import import load_llm_settings

PLAINTEXT_KEY = "sk-p5a-1234567890abcdef-SECRET"


# ═══════════════════════════════════════
# 通用夹具：把设置文件指到临时目录
# ═══════════════════════════════════════

@pytest.fixture
def settings_env(tmp_path, monkeypatch):
    """llm_settings.settings_path() 指向 tmp_path/data/llm_settings.json。

    用与生产代码同一个加载器，保证与 settings_routes 用的是同一份模块对象。
    """
    llm_settings = load_llm_settings()
    monkeypatch.setattr(llm_settings, "settings_path", lambda: tmp_path / "data" / "llm_settings.json")
    return llm_settings


@pytest.fixture
def client(fake_services, settings_env):
    """复用 conftest 的假服务夹具（避免 import 真实模型），并带上设置路由。"""
    import importlib
    module = importlib.import_module("app.main")
    return TestClient(module.app)


def assert_no_plaintext(payload) -> None:
    """任何响应体都不能含明文 key。"""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    assert PLAINTEXT_KEY not in text


# ═══════════════════════════════════════
# 1. 设置读写与掩码
# ═══════════════════════════════════════

def test_default_settings_fall_back_to_provider_presets(settings_env):
    view = settings_env.settings_view()

    assert view["active"] == "deepseek"
    assert set(view["profiles"]) == {"deepseek", "openai", "custom"}
    assert view["profiles"]["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert view["profiles"]["openai"]["base_url"] == "https://api.openai.com/v1"
    # .env 与设置文件都没有 key：必须给出明确状态与可读文案，而不是抛 401
    assert view["configured"] is False
    assert "API Key" in view["message"]
    assert view["profiles"]["deepseek"]["api_key"] == ""


def test_save_writes_file_atomically_and_masks_read(settings_env):
    saved = settings_env.apply_update({
        "active": "openai",
        "profiles": {
            "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key": PLAINTEXT_KEY},
        },
    })

    path = settings_env.settings_path()
    assert path.exists()
    # 落盘的是明文（服务端自己要能用），但读接口只给掩码
    assert PLAINTEXT_KEY in path.read_text(encoding="utf-8")
    assert saved.active == "openai"
    # 目录里不残留半截临时文件
    assert [p.name for p in path.parent.iterdir() if p.suffix == ".tmp"] == []

    view = settings_env.settings_view()
    masked = view["profiles"]["openai"]["api_key"]
    assert masked != PLAINTEXT_KEY
    assert masked.startswith("sk-p5a-")
    assert set(masked) & {"*"}
    assert_no_plaintext(view)


def test_mask_keeps_head_and_tail_only(settings_env):
    key = "sk-" + "5e1b0e68b7e145eebe56f6eb92c4c57e"
    masked = settings_env.mask_api_key(key)

    assert masked == "sk-5e1b****c57e"
    assert len(masked) < len(key)
    assert settings_env.mask_api_key("") == ""


def test_masked_key_from_frontend_is_not_stored_as_plaintext(settings_env):
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": PLAINTEXT_KEY}},
    })
    masked = settings_env.settings_view()["profiles"]["deepseek"]["api_key"]

    # 前端把掩码原样回传：应当沿用旧 key，而不是把掩码当明文存下去
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": masked}},
    })
    assert settings_env.active_config()["api_key"] == PLAINTEXT_KEY

    # 掩码和已存 key 对不上时必须报错，不能悄悄存下去
    with pytest.raises(settings_env.LLMSettingsError):
        settings_env.apply_update({
            "active": "deepseek",
            "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "m", "api_key": "sk-aaaa****zzzz"}},
        })


# ═══════════════════════════════════════
# 2. 非法输入校验
# ═══════════════════════════════════════

@pytest.mark.parametrize("profile,expected", [
    ({"base_url": "", "model": "m", "api_key": "sk-x"}, "base_url"),
    ({"base_url": "ftp://api.deepseek.com", "model": "m", "api_key": "sk-x"}, "http(s)"),
    ({"base_url": "api.deepseek.com", "model": "m", "api_key": "sk-x"}, "http(s)"),
    ({"base_url": "https://api.deepseek.com", "model": "", "api_key": "sk-x"}, "模型名称"),
    ({"base_url": "https://api.deepseek.com", "model": "m", "api_key": ""}, "API Key"),
])
def test_validate_profile_rejects_bad_input(settings_env, profile, expected):
    with pytest.raises(settings_env.LLMSettingsError) as excinfo:
        settings_env.validate_profile("deepseek", profile)

    assert expected in str(excinfo.value)


def test_validate_profile_rejects_unknown_provider(settings_env):
    with pytest.raises(settings_env.LLMSettingsError) as excinfo:
        settings_env.validate_profile("gemini", {"base_url": "https://x", "model": "m", "api_key": "k"})

    assert "未知模型厂商" in str(excinfo.value)


def test_apply_update_rejects_unknown_active(settings_env):
    with pytest.raises(settings_env.LLMSettingsError):
        settings_env.apply_update({"active": "gemini", "profiles": {}})


def test_switching_to_unconfigured_provider_is_rejected(settings_env):
    """切到没有 key 的厂商只会换来 401，必须在保存时就拦住。"""
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": PLAINTEXT_KEY}},
    })
    with pytest.raises(settings_env.LLMSettingsError) as excinfo:
        settings_env.apply_update({"active": "openai", "profiles": {}})

    assert "API Key" in str(excinfo.value)


# ═══════════════════════════════════════
# 3. .env 回退链
# ═══════════════════════════════════════

def test_env_fallback_when_settings_file_missing(settings_env, monkeypatch):
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_API_KEY", "sk-env-fallback-key")
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_MODEL", "deepseek-env-model")
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_BASE_URL", "https://env.example.com/v1")

    assert not settings_env.settings_path().exists()
    resolved = settings_env.active_config()

    assert resolved["api_key"] == "sk-env-fallback-key"
    assert resolved["model"] == "deepseek-env-model"
    assert resolved["base_url"] == "https://env.example.com/v1"
    assert resolved["key_source"] == "env"
    assert resolved["configured"] is True
    # .env 提供的 key 也只以掩码形式出现在读接口里
    assert_no_plaintext(settings_env.settings_view())


def test_settings_file_wins_over_env(settings_env, monkeypatch):
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_API_KEY", "sk-env-fallback-key")
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://file.example.com/v1", "model": "file-model", "api_key": PLAINTEXT_KEY}},
    })

    resolved = settings_env.active_config()
    assert resolved["api_key"] == PLAINTEXT_KEY
    assert resolved["model"] == "file-model"
    assert resolved["base_url"] == "https://file.example.com/v1"
    assert resolved["key_source"] == "settings"


def test_no_key_anywhere_raises_readable_error(settings_env, monkeypatch):
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_API_KEY", "")

    with pytest.raises(settings_env.NotConfiguredError) as excinfo:
        settings_env.require_api_key()

    message = str(excinfo.value)
    assert "API Key" in message and "控制面板" in message
    assert "401" not in message


def test_broken_settings_file_falls_back_to_defaults(settings_env):
    path = settings_env.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")

    view = settings_env.settings_view()
    assert view["active"] == "deepseek"
    assert view["profiles"]["deepseek"]["base_url"] == "https://api.deepseek.com"


# ═══════════════════════════════════════
# 4. HTTP 接口（GET/POST/test）——响应用不含明文 key
# ═══════════════════════════════════════

def test_get_settings_never_returns_plaintext_key(client, settings_env):
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": PLAINTEXT_KEY}},
    })

    response = client.get("/chat/settings/llm")

    assert response.status_code == 200
    assert_no_plaintext(response.text)
    body = response.json()
    assert body["active"] == "deepseek"
    assert body["configured"] is True
    assert body["profiles"]["deepseek"]["api_key"] != PLAINTEXT_KEY
    assert body["providers"][0]["name"] == "deepseek"


def test_post_settings_saves_and_switches_active(client, settings_env):
    response = client.post("/chat/settings/llm", json={
        "active": "custom",
        "profiles": {
            "custom": {"base_url": "https://my-gateway.example.com/v1", "model": "my-model", "api_key": PLAINTEXT_KEY},
        },
    })

    assert response.status_code == 200
    assert_no_plaintext(response.text)
    body = response.json()
    assert body["active"] == "custom"
    assert body["model"] == "my-model"
    assert body["profiles"]["custom"]["base_url"] == "https://my-gateway.example.com/v1"
    # 落盘后 ModelManager 下次请求读到的就是这一份
    assert settings_env.active_config()["base_url"] == "https://my-gateway.example.com/v1"


def test_post_settings_rejects_bad_input_with_detail(client):
    response = client.post("/chat/settings/llm", json={
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "not-a-url", "model": "m", "api_key": "sk-x"}},
    })

    assert response.status_code == 422
    assert "http(s)" in response.json()["detail"]


def test_post_settings_is_rejected_while_generating(client, settings_env):
    from app.api.chat import generation_lock

    generation_lock.set_generation_busy(True)
    try:
        response = client.post("/chat/settings/llm", json={
            "active": "deepseek",
            "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "m", "api_key": "sk-x"}},
        })
    finally:
        generation_lock.set_generation_busy(False)

    assert response.status_code == 409
    assert "生成" in response.json()["detail"]
    # 生成中拒绝时不能落盘
    assert not settings_env.settings_path().exists()


def test_test_endpoint_uses_candidate_config_and_hides_key(client, settings_env, monkeypatch):
    from app.api.chat import settings_routes

    captured = {}

    async def fake_test(provider, profile):
        captured.update({"provider": provider, "profile": profile})
        return {"ok": True, "message": "连接成功，模型可用"}

    monkeypatch.setattr(settings_routes, "test_candidate", fake_test)

    response = client.post("/chat/settings/llm/test", json={
        "provider": "custom",
        "base_url": "https://candidate.example.com/v1",
        "model": "candidate-model",
        "api_key": PLAINTEXT_KEY,
    })

    assert response.status_code == 200
    assert_no_plaintext(response.text)
    assert response.json()["ok"] is True
    assert captured["profile"]["base_url"] == "https://candidate.example.com/v1"
    assert captured["profile"]["model"] == "candidate-model"
    assert captured["profile"]["api_key"] == PLAINTEXT_KEY


def test_test_endpoint_reports_failure_without_key(client, settings_env, monkeypatch):
    from app.api.chat import settings_routes

    async def fake_test(provider, profile):
        return {"ok": False, "message": "API Key 无效或已过期（401），请确认后重新填写"}

    monkeypatch.setattr(settings_routes, "test_candidate", fake_test)

    response = client.post("/chat/settings/llm/test", json={"provider": "deepseek", "api_key": PLAINTEXT_KEY})

    assert response.status_code == 200
    assert_no_plaintext(response.text)
    assert response.json()["ok"] is False
    assert "401" in response.json()["message"]


def test_test_endpoint_without_any_key_gives_readable_422(client, settings_env, monkeypatch):
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_API_KEY", "")

    response = client.post("/chat/settings/llm/test", json={"provider": "deepseek"})

    assert response.status_code == 422
    assert "API Key" in response.json()["detail"]


# ═══════════════════════════════════════
# 5. ModelManager 实时替换
# ═══════════════════════════════════════

class FakeAsyncOpenAI:
    """记录构造参数，并让自己看起来像一个能用的 OpenAI 客户端。"""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.created = []
        self.chat = types.SimpleNamespace(completions=FakeCompletions(self))
        FakeAsyncOpenAI.instances.append(self)


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.created.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=[]), finish_reason="stop")]
        )


@pytest.fixture
def manager_module(monkeypatch):
    """按既有测试的方式注入假 openai 并加载 ModelManager 源码。"""
    openai_module = types.ModuleType("openai")
    openai_module.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    FakeAsyncOpenAI.instances = []
    return load_source_module("model_manager_settings_under_test", "app/managers/ModelManager.py")


def _fake_client(manager):
    """给 manager 挂一个假 client（走 _owned_client 所有权判定）。"""
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions(types.SimpleNamespace(created=[]))),
        created=[],
    )
    client.chat.completions = FakeCompletions(client)
    manager.client = client
    manager._owned_client = False
    return client


def test_init_does_not_build_client_at_import_time(manager_module):
    """import 期（构造时）不能构造唯一客户端：设置改完必须能换。"""
    manager = manager_module.ModelManager()

    assert manager.client is None
    assert FakeAsyncOpenAI.instances == []


def test_generate_reads_active_profile_every_request(manager_module, settings_env):
    manager = manager_module.ModelManager()
    manager.tools = []

    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": "sk-first-key"}},
    })
    asyncio.run(manager.generate([], {}, use_tools=False))
    first = FakeAsyncOpenAI.instances[-1]
    assert first.kwargs["api_key"] == "sk-first-key"
    assert first.kwargs["base_url"] == "https://api.deepseek.com"
    assert first.created[0]["model"] == "deepseek-v4-flash"

    # 改设置：下一个请求必须走新 base_url/key/model，无需重启
    settings_env.apply_update({
        "active": "custom",
        "profiles": {"custom": {"base_url": "https://switched.example.com/v1", "model": "switched-model", "api_key": "sk-second-key"}},
    })
    asyncio.run(manager.generate([], {}, use_tools=False))
    second = FakeAsyncOpenAI.instances[-1]

    assert len(FakeAsyncOpenAI.instances) == 2
    assert second is not first
    assert second.kwargs["api_key"] == "sk-second-key"
    assert second.kwargs["base_url"] == "https://switched.example.com/v1"
    assert second.created[0]["model"] == "switched-model"
    assert manager.client is second


def test_client_is_cached_while_config_unchanged(manager_module, settings_env):
    manager = manager_module.ModelManager()
    settings_env.apply_update({
        "active": "deepseek",
        "profiles": {"deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash", "api_key": "sk-same-key"}},
    })

    asyncio.run(manager.generate([], {}, use_tools=False))
    asyncio.run(manager.generate([], {}, use_tools=False))

    # (base_url, api_key) 没变 => 复用同一个客户端
    assert len(FakeAsyncOpenAI.instances) == 1


def test_stream_generate_also_uses_fresh_config(manager_module, settings_env):
    manager = manager_module.ModelManager()
    manager.tools = []
    settings_env.apply_update({
        "active": "custom",
        "profiles": {"custom": {"base_url": "https://stream.example.com/v1", "model": "stream-model", "api_key": "sk-stream-key"}},
    })

    class FakeStream:
        def __aiter__(self):
            async def iterate():
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason="stop", delta=types.SimpleNamespace(content="hi", reasoning_content=None, tool_calls=[]))])
            return iterate()

        async def close(self):
            return None

    created = {}

    class StreamCompletions:
        async def create(self, **kwargs):
            created.update(kwargs)
            return FakeStream()

    manager.client = None
    manager._owned_client = True
    manager._client(settings_env.active_config())
    manager.client.chat = types.SimpleNamespace(completions=StreamCompletions())

    result = asyncio.run(manager.stream_generate([], {}, use_tools=False))

    assert result.content == "hi"
    assert created["stream"] is True
    assert created["model"] == "stream-model"


def test_generate_raises_readable_error_when_no_key_anywhere(manager_module, settings_env, monkeypatch):
    monkeypatch.setattr(settings_env.config, "DEEPSEEK_API_KEY", "")
    manager = manager_module.ModelManager()

    with pytest.raises(settings_env.NotConfiguredError) as excinfo:
        asyncio.run(manager.generate([], {}, use_tools=False))

    assert "API Key" in str(excinfo.value)
    assert FakeAsyncOpenAI.instances == []


def test_injected_client_is_never_replaced(manager_module, settings_env):
    """既有测试通过替换 mgr.client 注入假客户端：那条路径不能被实时替换逻辑覆盖。"""
    manager = manager_module.ModelManager()
    injected = _fake_client(manager)

    result = asyncio.run(manager.generate([], {}, use_tools=False))

    assert str(result) == "ok"
    assert manager.client is injected
    assert FakeAsyncOpenAI.instances == []
