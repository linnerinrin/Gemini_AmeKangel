#"""
#架构作用：设置路由与模型适配器共用的配置服务，负责磁盘存储、回退、掩码和校验
#数据流：JSON/环境/预设→生效配置→ SDK；写入先校验再临时文件原子替换
#明文凭据只应供服务端调用；settings_view生成面板视图，不能用掩码作为SDK密钥
#"""
"""
services/llm_settings.py模型设置服务（只做OpenAI兼容厂商）

职责：
1.把「厂商/ base_url / model / api_key」持久化到data/llm_settings.json
   （写临时文件+ os.replace原子替换，避免断电/异常留下半截JSON）；
2.读取时把api_key掩码，任何接口与日志都不得回传明文；
3.设置文件不存在（或缺少key）时回退到.env里的config.DEEPSEEK_*；
4.提供校验，让调用方拿到「请在控制面板配置API Key」这类可读提示，
而不是直接抛一个看不懂的401
"""

import ipaddress
import json
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.agent_framework.security.credential_policy import (
    CredentialPolicy,
    normalize_base_url,
)
from app.config.config import config

#凭据—地址绑定策略：保存路径与探测端点共用同一套判定，避免两处规则漂移
_credential_policy = CredentialPolicy()

#掩码里固定保留的头尾长度：sk-5e1****c57e这种形状
MASK_HEAD = 7
MASK_TAIL = 4
MASK_CHAR = "*"

#OpenAI兼容厂商预设：custom的base_url由用户自己填
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
#回退链只认DeepSeek：.env里目前只有DEEPSEEK_*三项
ENV_FALLBACK_PROVIDER = "deepseek"


#"""
#将磁盘位置绑定配置根目录；每次计算允许测试替换BASE_PATH，不在import时读密钥
#"""
def settings_path() -> Path:
    """设置文件路径。每次读都重新算，方便测试monkeypatch到临时目录。"""
    return Path(config.BASE_PATH) / "data" / "llm_settings.json"


#"""
#Pydantic数据边界：每厂商一份URL/模型/密钥，总设置保存当前厂商与档案映射
#Field(default_factory=dict)避免跨实例共享可变字典；模型本身不执行网络校验
#"""
class ProviderProfile(BaseModel):
    base_url: str = ""
    model: str = ""
    #明文key只在内存/磁盘里流转，绝不出现在任何响应中
    api_key: str = ""


class LLMSettings(BaseModel):
    active: str = DEFAULT_ACTIVE
    profiles: dict[str, ProviderProfile] = Field(default_factory=dict)


class LLMSettingsError(ValueError):
    """设置校验失败；message直接给前端看，必须是人话。"""


#"""
#仅生成展示掩码，按长度保护中间字符；掩码不构成加密存储或访问控制
#"""
def mask_api_key(api_key: str | None) -> str:
    """把key掩码成sk-5e1****c57e；空值回空串，长度不足时也至少遮住中间。"""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= MASK_HEAD + MASK_TAIL:
        if len(key) <= 2:
            return MASK_CHAR * len(key)
        return key[0] + MASK_CHAR * (len(key) - 2) + key[-1]
    return key[:MASK_HEAD] + MASK_CHAR * 4 + key[-MASK_TAIL:]


#"""
#识别面板掩码形状，用于保存时沿用旧值；后续仍须与已有密钥掩码精确比对
#"""
def is_masked_api_key(api_key: str | None) -> bool:
    """判断前端回传的key是不是我们自己发出去的掩码串。

    面板打开时输入框里填的就是掩码值，用户没改就直接保存必须当成"沿用旧key"，
    否则会把sk-5e1****c57e当明文key存下去。
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


#"""
#根据厂商目录创建独立档案对象，供归一化补齐形状，api_key默认空值
#"""
def default_profiles() -> dict[str, ProviderProfile]:
    return {
        name: ProviderProfile(
            base_url=preset["base_url"],
            model=preset["model"],
            api_key="",
        )
        for name, preset in PROVIDER_CATALOG.items()
    }


#"""
#写入前统一字段、补预设并过滤未知厂商；不读取环境密钥，避免把回退值意外写盘
#"""
def _normalize(settings: LLMSettings) -> LLMSettings:
    """补齐缺失厂商、修正非法active，保证读出来的形状总是完整的。"""
    profiles = default_profiles()
    for name, profile in (settings.profiles or {}).items():
        if name not in profiles:
            #未知厂商不静默保留，避免前端select出现空选项
            continue
        merged = profiles[name]
        profiles[name] = ProviderProfile(
            base_url=(profile.base_url or merged.base_url).strip(),
            model=(profile.model or merged.model).strip(),
            api_key=(profile.api_key or "").strip(),
        )
    active = settings.active if settings.active in profiles else DEFAULT_ACTIVE
    return LLMSettings(active=active, profiles=profiles)


#"""
#读取实际保存字段，不提前混入预设，以保留“已保存/需回退”的来源判断
#"""
def load_settings() -> LLMSettings:
    """读取设置文件里**实际存下来的内容**（不掺预设默认值）。

    文件不存在/坏掉/字段缺失时返回空profiles或空字段：这样上层才分得清
    "用户在面板里填过"与"没填过、该回退.env"。预设默认值由resolve_profile
    与settings_view补齐，不要在这里提前写死。
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


#"""
#归一化后在同目录写临时文件，flush/fsync后原子替换；异常尽力删除临时文件
#原子替换避免半截文件，但读-改-写没有并发版本检测；凭据在磁盘仍为明文
#"""
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


#"""
#环境回退仅对DeepSeek开放；返回值仅供服务端解析，其他厂商空串继续走自身预设
#"""
def env_api_key(provider: str | None = None) -> str:
    """`.env`回退：目前只有DeepSeek三项，其它厂商没有env兜底。"""
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_API_KEY or "").strip()


#"""
#提取DeepSeek的环境地址，剔除首尾空白；不做URL可信性检查
#"""
def env_base_url(provider: str | None = None) -> str:
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_BASE_URL or "").strip()


#"""
#提取DeepSeek环境模型名；空字段会由resolve_profile继续回退到厂商预设
#"""
def env_model(provider: str | None = None) -> str:
    if (provider or DEFAULT_ACTIVE) != ENV_FALLBACK_PROVIDER:
        return ""
    return (config.DEEPSEEK_MODEL or "").strip()


#"""
#合并生效配置：保存字段优先，缺项走环境再走预设；结果包含明文key，只供内部
#provider省略使用active，注入settings可避免重复读取并方便离线测试
#"""
def resolve_profile(provider: str | None = None, settings: LLMSettings | None = None) -> dict:
    """解析某个厂商最终生效的配置：设置文件优先，缺哪个字段才回退.env。

    返回{"active", "provider", "base_url", "model", "api_key", "key_source", "configured"}；
    这里的api_key是明文，只允许喂给SDK，禁止进响应体或日志。
    """
    current = settings or load_settings()
    name = provider or current.active
    if name not in PROVIDER_CATALOG:
        name = current.active
    #这里不能拿default_profiles()当兜底，否则预设值会盖掉.env回退
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


#"""
#当前档案解析的便捷入口，调用时读取最新文件，不缓存旧的模型选择
#"""
def active_config() -> dict:
    """ModelManager每次请求都调这个，拿到的就是当前即时生效的配置。"""
    return resolve_profile()


#"""
#只生成可读的缺密钥提示，不插入凭据值；厂商标签来自受控目录
#"""
def missing_key_message(provider: str | None = None) -> str:
    label = PROVIDER_CATALOG.get(provider or "", {}).get("label", "当前厂商")
    return (
        f"尚未配置 API Key（{label}）：请在控制面板「模型厂商 → API Key」里填写并保存，"
        "然后点「测试」确认可用"
    )


class NotConfiguredError(LLMSettingsError):
    """当前厂商没有可用key；调用方应把它转成可读提示，而不是抛401。"""


#"""
#模型调用前最后一道非空检查，失败抛业务错误，成功返回内部生效配置
#"""
def require_api_key() -> dict:
    """生成前调用：没key就抛NotConfiguredError，让上层给出明确提示。"""
    resolved = active_config()
    if not resolved["configured"]:
        raise NotConfiguredError(missing_key_message(resolved["provider"]))
    return resolved


#"""
#为候选配置取已存或环境回退密钥；该函数本身不将key与端点绑定
#"""
def _stored_api_key(provider: str) -> str:
    stored = load_settings().profiles.get(provider)
    return (stored.api_key if stored else "") or env_api_key(provider)


#"""
#空输入沿用已存key；掩码必须匹配当前真实值才沿用，否则要求重新输入
#"""
def resolve_api_key(provider: str, raw_key: str) -> str:
    """把草稿里的key归一化成明文。

    - 空值：沿用已保存的key（没有就回退.env）
    - 掩码串：沿用已保存的key；只有与已保存的掩码完全一致才认，否则报错
      （防止把sk-****这种掩码当真key存下去）
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


#"""
#本地模型（如Ollama）确需私网/环回地址时，设环境变量LLM_ALLOW_PRIVATE_BASE_URL=1跳过拦截；
#判定集中在allow_private_base_url()，保存端点也复用同一开关，避免两处口径不一致
#"""
def allow_private_base_url() -> bool:
    """是否允许base_url指向环回/私网（本地模型场景的显式逃生门）。"""
    return os.environ.get("LLM_ALLOW_PRIVATE_BASE_URL") == "1"


#"""
#拒绝环回/私网/链路本地/保留/组播/未指定地址，避免凭据随请求被转发到内网或本机端口
#域名先经getaddrinfo解析再逐个检查；解析失败/无结果时打印提示并放行（请求发不出去，且测试/离线环境常见）
#"""
def _assert_public_host(parsed) -> None:
    """校验base_url的host不指向环回/私网/链路本地/保留地址。"""
    host = parsed.hostname
    if not host:
        raise LLMSettingsError("base_url 缺少主机名")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            print(f"[llm_settings] base_url主机名无法解析，跳过内网检查: {host}")
            return
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
        if not addresses:
            print(f"[llm_settings] base_url主机名没有可用IP地址，跳过内网检查: {host}")
            return
    if allow_private_base_url():
        return
    for addr in addresses:
        if (addr.is_loopback or addr.is_private or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            raise LLMSettingsError("base_url 不允许指向环回/私网/链路本地/保留地址，请使用公网接口地址（本地模型可设LLM_ALLOW_PRIVATE_BASE_URL=1）")


#"""
#校验厂商、URL语法、模型非空并解析key；require_key=False允许保存未激活空档案
#host经_assert_public_host拒绝环回/私网/链路本地/保留地址，防止凭据被转发到内网
#"""
def validate_profile(provider: str, profile: dict, *, require_key: bool = True) -> ProviderProfile:
    """校验单个厂商配置，失败抛LLMSettingsError（detail直接给前端）。

    require_key=False用于连通性测试：候选key可以暂时为空（由调用方补齐）。
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
    _assert_public_host(parsed)
    if not model:
        raise LLMSettingsError("模型名称不能为空，请填写要调用的模型版本")

    api_key = resolve_api_key(provider, str(profile.get("api_key") or ""))
    if require_key and not api_key:
        label = PROVIDER_CATALOG[provider]["label"]
        raise LLMSettingsError(f"未配置 API Key：请为「{label}」填写 API Key 后保存")

    return ProviderProfile(base_url=base_url.rstrip("/") or base_url, model=model, api_key=api_key)


#"""
#保存路径上的凭据—地址绑定：只改base_url不重输key时，resolve_api_key会静默沿用旧key，
#落盘后就变成“新地址+旧密钥”。这里在合并落盘前拦截：地址（归一化后）发生变化，
#而本次没有提交新的明文key（留空或回传掩码），一律拒绝保存。
#归一化（大小写/末尾斜杠）差异不算地址变更，复用CredentialPolicy的同一规则。
#"""
def _assert_key_binding_unchanged(provider: str, profile: dict) -> None:
    """地址变更必须显式重输API Key，防止已存密钥被转发到新端点。"""
    if not isinstance(profile, dict):
        return  #格式错误交给validate_profile统一报
    candidate_url = str(profile.get("base_url") or "").strip()
    if not candidate_url:
        return  #缺地址同样交给validate_profile
    saved = resolve_profile(provider)
    saved_url = str(saved["base_url"] or "")
    if normalize_base_url(candidate_url) == normalize_base_url(saved_url):
        return
    #当前厂商根本没有已保存的key（首次配置）：没有可搬运的凭据，
    #交给validate_profile报“未配置API Key”，不要误报成“已保存Key被转发”。
    if not saved.get("api_key"):
        return
    raw_key = str(profile.get("api_key") or "").strip()
    #空key与掩码都等于“沿用已保存的key”，两者都不能跨地址搬运
    carried = (not raw_key) or is_masked_api_key(raw_key)
    if carried:
        label = PROVIDER_CATALOG[provider]["label"]
        raise LLMSettingsError(
            f"修改「{label}」的 base_url 时必须重新填写并提交 API Key："
            "不允许把已保存的 Key 转发到新地址"
        )


#"""
#读取旧档案→合并请求中的厂商→验证激活档案有key→一次原子保存；中途异常不写盘
#"""
def apply_update(payload: dict) -> LLMSettings:
    """把前端提交的{active, profiles}合并进当前设置并原子落盘。

    只接受OpenAI兼容厂商的字段；profiles里没提到的厂商保持原样。
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
        if name not in PROVIDER_CATALOG:
            raise LLMSettingsError(f"未知模型厂商: {name}")
        #先查“地址变了却没重输key”，再走常规字段校验
        _assert_key_binding_unchanged(name, profile)
        merged[name] = validate_profile(name, profile)

    active = str(payload.get("active") or current.active).strip()
    if active not in PROVIDER_CATALOG:
        raise LLMSettingsError(f"未知模型厂商: {active}")

    #切过去的厂商必须真的能用，否则用户切完只会拿到401
    target = merged.get(active) or ProviderProfile()
    if not (target.api_key or env_api_key(active)):
        raise LLMSettingsError(
            f"未配置 API Key：请为「{PROVIDER_CATALOG[active]['label']}」填写 API Key 后保存"
        )

    return save_settings(LLMSettings(active=active, profiles=merged))


#"""
#构建面板DTO：用掩码替代所有key，附生效模型/来源，避免内部解析对象直接序列化
#"""
def settings_view() -> dict:
    """给前端看的设置视图：key一律掩码，附带"当前是否已配置可用key"。"""
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
            #掩码只由明文算出；文件里没有key就掩码.env的key，同样不回明文
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
        #未配置时的提示文案，前端直接显示，别再让用户对着401猜
        "message": "" if resolved["api_key"] else missing_key_message(settings.active),
    }

#_assert_key_binding_unchanged在落盘前拦截“地址变更但未提交新key”的保存；
#_assert_public_host在DNS解析失败时打印提示并放行，开关口径统一在allow_private_base_url。
