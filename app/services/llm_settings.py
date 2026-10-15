"""
services/llm_settings.py  模型设置服务（P5a：只做 OpenAI 兼容厂商）

职责：
1. 把「厂商 / base_url / model / api_key」持久化到 data/llm_settings.json
   （写临时文件 + os.replace 原子替换，避免断电/异常留下半截 JSON）；
2. 读取时把 api_key 掩码，任何接口与日志都不得回传明文；
3. 设置文件不存在（或缺少 key）时回退到 .env 里的 config.DEEPSEEK_*；
4. 提供校验，让调用方拿到「请在控制面板配置 API Key」这类可读提示，
   而不是直接抛一个看不懂的 401。
"""

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.config.config import config

# 掩码里固定保留的头尾长度：sk-5e1****c57e 这种形状
MASK_HEAD = 7
MASK_TAIL = 4
MASK_CHAR = "*"

# OpenAI 兼容厂商预设：custom 的 base_url 由用户自己填
PROVIDER_CATALOG = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "custom": {
        "label": "自定义（OpenAI 兼容）",
        "base_url": "",
        "model": "",
    },
}

DEFAULT_ACTIVE = "deepseek"
# 回退链只认 DeepSeek：.env 里目前只有 DEEPSEEK_* 三项
ENV_FALLBACK_PROVIDER = "deepseek"


def settings_path() -> Path:
    """设置文件路径。每次读都重新算，方便测试 monkeypatch 到临时目录。"""
    return Path(config.BASE_PATH) / "data" / "llm_settings.json"


class ProviderProfile(BaseModel):
    base_url: str = ""
    model: str = ""
    # 明文 key 只在内存/磁盘里流转，绝不出现在任何响应中
    api_key: str = ""


class LLMSettings(BaseModel):
    active: str = DEFAULT_ACTIVE
    profiles: dict[str, ProviderProfile] = Field(default_factory=dict)


class LLMSettingsError(ValueError):
    """设置校验失败；message 直接给前端看，必须是人话。"""


def mask_api_key(api_key: str | None) -> str:
    """把 key 掩码成 sk-5e1****c57e；空值回空串，长度不足时也至少遮住中间。"""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= MASK_HEAD + MASK_TAIL:
        if len(key) <= 2:
            return MASK_CHAR * len(key)
        return key[0] + MASK_CHAR * (len(key) - 2) + key[-1]
    return key[:MASK_HEAD] + MASK_CHAR * 4 + key[-MASK_TAIL:]


def is_masked_api_key(api_key: str | None) -> bool:
    """判断前端回传的 key 是不是我们自己发出去的掩码串。

    面板打开时输入框里填的就是掩码值，用户没改就直接保存必须当成"沿用旧 key"，
    否则会把 sk-5e1****c57e 当明文 key 存下去。
    """
    key = (api_key or "").strip()
    if not key or MASK_CHAR not in key:
        return False
    if key.startswith(MASK_CHAR):
        return True
    prefix, sep, suffix = key.partition(MASK_CHAR)
    if not sep:
        return False
    return len(prefix) <= MASK_HEAD and len(suffix.lstrip(MASK_CHAR)) <= MASK_TAIL


def default_profiles() -> dict[str, ProviderProfile]:
    return {
        name: ProviderProfile(
            base_url=preset["base_url"],
            model=preset["model"],
            api_key="",
        )
        for name, preset in PROVIDER_CATALOG.items()
    }


def _normalize(settings: LLMSettings) -> LLMSettings:
    """补齐缺失厂商、修正非法 active，保证读出来的形状总是完整的。"""
    profiles = default_profiles()
    for name, profile in (settings.profiles or {}).items():
        if name not in profiles:
            # 未知厂商不静默保留，避免前端 select 出现空选项
            continue
        merged = profiles[name]
        profiles[name] = ProviderProfile(
            base_url=(profile.base_url or merged.base_url).strip(),
            model=(profile.model or merged.model).strip(),
            api_key=(profile.api_key or "").strip(),
        )
    active = settings.active if settings.active in profiles else DEFAULT_ACTIVE
    return LLMSettings(active=active, profiles=profiles)


def load_settings() -> LLMSettings:
    """读取设置文件里**实际存下来的内容**（不掺预设默认值）。

    文件不存在/坏掉/字段缺失时返回空 profiles 或空字段：这样上层才分得清
    "用户在面板里填过" 与 "没填过、该回退 .env"。预设默认值由 resolve_profile
    与 settings_view 补齐，不要在这里提前写死。
    """
    path = settings_path()
    if not path.exists():
        return LLMSettings(active=DEFAULT_ACTIVE, profiles={})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return LLMSettings(active=DEFAULT_ACTIVE, profiles={})
    try:
        stored = LLMSettings.model_validate(raw)
    except Exception:
        return LLMSettings(active=DEFAULT_ACTIVE, profiles={})

    profiles = {}
    for name, profile in (stored.profiles or {}).items():
        if name not in PROVIDER_CATALOG:
            continue
        profiles[name] = ProviderProfile(
            base_url=(profile.base_url or "").strip(),
            model=(profile.model or "").strip(),
            api_key=(profile.api_key or "").strip(),
        )
    active = stored.active if stored.active in PROVIDER_CATALOG else DEFAULT_ACTIVE
    return LLMSettings(active=active, profiles=profiles)


def save_settings(settings: LLMSettings) -> LLMSettings:
    """原子写入设置文件；父目录不存在就创建。"""
    normalized = _normalize(settings)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalized.model_dump(), ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="llm_settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return normalized


def env_api_key(provider: str | None = None) -> str:
    """`.env` 回退：目前只有 DeepSeek 三项，其它厂商没有 env 兜底。"""
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_API_KEY or "").strip()


def env_base_url(provider: str | None = None) -> str:
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_BASE_URL or "").strip()


def env_model(provider: str | None = None) -> str:
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_MODEL or "").strip()


def resolve_profile(provider: str | None = None, settings: LLMSettings | None = None) -> dict:
    """解析某个厂商最终生效的配置：设置文件优先，缺哪个字段才回退 .env。

    返回 {"active", "provider", "base_url", "model", "api_key", "key_source", "configured"}；
    这里的 api_key 是明文，只允许喂给 SDK，禁止进响应体或日志。
    """
    current = settings or load_settings()
    name = provider or current.active
    if name not in PROVIDER_CATALOG:
        name = current.active
    # 注意：这里不能拿 default_profiles() 当兜底，否则预设值会盖掉 .env 回退
    profile = current.profiles.get(name) or ProviderProfile()

    api_key = (profile.api_key or "").strip()
    key_source = "settings"
    if not api_key:
        api_key = env_api_key(name)
        key_source = "env" if api_key else "none"
    base_url = (profile.base_url or "").strip() or env_base_url(name) or PROVIDER_CATALOG[name]["base_url"]
    model = (profile.model or "").strip() or env_model(name) or PROVIDER_CATALOG[name]["model"]
    return {
        "active": current.active,
        "provider": name,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "key_source": key_source,
        "configured": bool(api_key),
    }


def active_config() -> dict:
    """ModelManager 每次请求都调这个，拿到的就是当前即时生效的配置。"""
    return resolve_profile()


def missing_key_message(provider: str | None = None) -> str:
    label = PROVIDER_CATALOG.get(provider or "", {}).get("label", "当前厂商")
    return (
        f"尚未配置 API Key（{label}）：请在控制面板「模型厂商 → API Key」里填写并保存，"
        "然后点「测试」确认可用"
    )


class NotConfiguredError(LLMSettingsError):
    """当前厂商没有可用 key；调用方应把它转成可读提示，而不是抛 401。"""


def require_api_key() -> dict:
    """生成前调用：没 key 就抛 NotConfiguredError，让上层给出明确提示。"""
    resolved = active_config()
    if not resolved["configured"]:
        raise NotConfiguredError(missing_key_message(resolved["provider"]))
    return resolved


def _stored_api_key(provider: str) -> str:
    stored = load_settings().profiles.get(provider)
    return (stored.api_key if stored else "") or env_api_key(provider)


def resolve_api_key(provider: str, raw_key: str) -> str:
    """把草稿里的 key 归一化成明文。

    - 空值：沿用已保存的 key（没有就回退 .env）
    - 掩码串：沿用已保存的 key；只有与已保存的掩码完全一致才认，否则报错
      （防止把 sk-**** 这种掩码当真 key 存下去）
    - 其它：就是用户刚粘贴的明文
    """
    key = (raw_key or "").strip()
    stored = _stored_api_key(provider)
    if not key:
        return stored
    if not is_masked_api_key(key):
        return key
    if stored and mask_api_key(stored) == key:
        return stored
    raise LLMSettingsError("API Key 不能填掩码值，请重新粘贴完整 Key（或留空沿用已保存的 Key）")


def validate_profile(provider: str, profile: dict, *, require_key: bool = True) -> ProviderProfile:
    """校验单个厂商配置，失败抛 LLMSettingsError（detail 直接给前端）。

    require_key=False 用于连通性测试：候选 key 可以暂时为空（由调用方补齐）。
    """
    if provider not in PROVIDER_CATALOG:
        raise LLMSettingsError(f"未知模型厂商: {provider}")
    if not isinstance(profile, dict):
        raise LLMSettingsError(f"厂商配置格式错误: {provider}")

    base_url = str(profile.get("base_url") or "").strip()
    model = str(profile.get("model") or "").strip()

    if not base_url:
        raise LLMSettingsError("base_url 不能为空，请填写 OpenAI 兼容接口地址")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LLMSettingsError("base_url 必须是 http(s):// 开头的合法地址")
    if not model:
        raise LLMSettingsError("模型名称不能为空，请填写要调用的模型版本")

    api_key = resolve_api_key(provider, str(profile.get("api_key") or ""))
    if require_key and not api_key:
        label = PROVIDER_CATALOG[provider]["label"]
        raise LLMSettingsError(f"未配置 API Key：请为「{label}」填写 API Key 后保存")

    return ProviderProfile(base_url=base_url.rstrip("/") or base_url, model=model, api_key=api_key)


def apply_update(payload: dict) -> LLMSettings:
    """把前端提交的 {active, profiles} 合并进当前设置并原子落盘。

    只接受 OpenAI 兼容厂商的字段；profiles 里没提到的厂商保持原样。
    """
    if not isinstance(payload, dict):
        raise LLMSettingsError("请求体必须是 JSON 对象")

    current = load_settings()
    incoming = payload.get("profiles")
    if incoming is None:
        incoming = {}
    if not isinstance(incoming, dict):
        raise LLMSettingsError("profiles 必须是对象")

    merged = {name: profile.model_copy() for name, profile in current.profiles.items()}
    for name, profile in incoming.items():
        merged[name] = validate_profile(name, profile)

    active = str(payload.get("active") or current.active).strip()
    if active not in PROVIDER_CATALOG:
        raise LLMSettingsError(f"未知模型厂商: {active}")

    # 切过去的厂商必须真的能用，否则用户切完只会拿到 401
    target = merged.get(active) or ProviderProfile()
    if not (target.api_key or env_api_key(active)):
        raise LLMSettingsError(
            f"未配置 API Key：请为「{PROVIDER_CATALOG[active]['label']}」填写 API Key 后保存"
        )

    return save_settings(LLMSettings(active=active, profiles=merged))


def settings_view() -> dict:
    """给前端看的设置视图：key 一律掩码，附带"当前是否已配置可用 key"。"""
    settings = load_settings()
    resolved = resolve_profile(settings=settings)
    profiles = {}
    for name in PROVIDER_CATALOG:
        stored = settings.profiles.get(name) or default_profiles()[name]
        effective = resolve_profile(name, settings=settings)
        profiles[name] = {
            "label": PROVIDER_CATALOG[name]["label"],
            "base_url": stored.base_url or PROVIDER_CATALOG[name]["base_url"],
            "model": stored.model or PROVIDER_CATALOG[name]["model"],
            "model_placeholder": PROVIDER_CATALOG[name]["model"],
            # 掩码只由明文算出；文件里没有 key 就掩码 .env 的 key，同样不回明文
            "api_key": mask_api_key(stored.api_key or env_api_key(name)),
            "configured": bool(effective["api_key"]),
            "key_source": effective["key_source"],
        }
    return {
        "active": settings.active,
        "profiles": profiles,
        "providers": [
            {"name": name, "label": PROVIDER_CATALOG[name]["label"]}
            for name in PROVIDER_CATALOG
        ],
        "configured": bool(resolved["api_key"]),
        "key_source": resolved["key_source"],
        "base_url": resolved["base_url"],
        "model": resolved["model"],
        # 未配置时的提示文案，前端直接显示，别再让用户对着 401 猜
        "message": "" if resolved["api_key"] else missing_key_message(settings.active),
    }
