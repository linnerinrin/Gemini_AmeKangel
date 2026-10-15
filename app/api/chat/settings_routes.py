"""
api/chat/settings_routes.py  模型设置接口（P5a：只做 OpenAI 兼容厂商）

- GET  /chat/settings/llm        读取设置（key 一律掩码）
- POST /chat/settings/llm        保存并切换 active；生成中拒绝
- POST /chat/settings/llm/test   用候选配置发一次最小请求做连通性测试
"""

import asyncio

import httpx
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.api.chat.chat_routes import require_idle
from app.api.chat.settings_schemas import LLMSettingsUpdate, LLMTestRequest
from app.config.llm_import import load_llm_settings

# 按文件路径加载（app.services 在测试里会被换成替身包，包级导入会失败）
llm_settings = load_llm_settings()

router = APIRouter()

# 连通性测试只验证"能连上、key 有效、模型名认得"，所以请求体压到最小
TEST_PROMPT = "ping"
TEST_MAX_TOKENS = 4
TEST_TIMEOUT_SECONDS = 20.0


def _draft(provider: str, submitted: dict) -> dict:
    """把"只提交改动字段"的草稿补成完整候选配置，再交给统一校验。"""
    effective = llm_settings.resolve_profile(provider)
    draft = {
        "base_url": submitted.get("base_url") or effective["base_url"],
        "model": submitted.get("model") or effective["model"],
    }
    # 没提交 key 字段 => 沿用已保存/.env；提交了（哪怕是空串）就按提交值处理
    if submitted.get("api_key") is not None:
        draft["api_key"] = submitted["api_key"]
    return draft


@router.get("/settings/llm")
async def read_llm_settings():
    """返回 active、各厂商 base_url/model、掩码后的 key 与"是否已配置可用 key"。"""
    return llm_settings.settings_view()


@router.post("/settings/llm")
async def update_llm_settings(request: LLMSettingsUpdate):
    """保存设置并切换 active；生成中禁止修改（否则改的是正在进行的那一轮的配置）。"""
    require_idle()
    payload = {
        "active": request.active,
        "profiles": {
            name: _draft(name, profile.model_dump(exclude_unset=True))
            for name, profile in request.profiles.items()
        },
    }
    try:
        llm_settings.apply_update(payload)
    except llm_settings.LLMSettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # 保存后立刻回读一次：前端拿到的就是落盘后的真实状态（key 仍是掩码）
    return llm_settings.settings_view()


def _describe_error(exc: BaseException) -> str:
    """把 SDK 异常翻成人话，同时保证不把 key 带出来。"""
    text = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return f"连接失败：{text}"
    if isinstance(exc, httpx.TimeoutException):
        return f"请求超时（{TEST_TIMEOUT_SECONDS:.0f}s）：{text}"
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "API Key 无效或已过期（401），请确认后重新填写"
    if status == 403:
        return "API Key 无权访问该模型（403）"
    if status == 404:
        return "接口或模型不存在（404），请检查 base_url 与模型版本"
    if status == 429:
        return "请求过于频繁或额度不足（429）"
    if status:
        return f"接口返回 {status}：{text}"
    return text


async def test_candidate(provider: str, profile: dict) -> dict:
    """用候选配置发一次最小请求；返回 {"ok": bool, "message": str}，永不回显 key。"""
    client = None
    try:
        client = AsyncOpenAI(
            api_key=profile["api_key"],
            base_url=profile["base_url"],
            timeout=TEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=profile["model"],
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=TEST_MAX_TOKENS,
        )
        if not getattr(response, "choices", None):
            return {"ok": False, "message": "接口有响应但没有返回任何选择项，请检查模型版本"}
        return {"ok": True, "message": "连接成功，模型可用"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # 网络/SDK/鉴权 全部收敛成可读字符串
        return {"ok": False, "message": _describe_error(exc)}
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass


@router.post("/settings/llm/test")
async def test_llm_settings(request: LLMTestRequest):
    """连通性校验：key / base_url / 模型都可以覆盖，留空则沿用已保存值。"""
    provider = request.provider
    if provider not in llm_settings.PROVIDER_CATALOG:
        raise HTTPException(status_code=422, detail=f"未知模型厂商: {provider}")
    effective = llm_settings.resolve_profile(provider)
    draft = {
        "base_url": request.base_url or effective["base_url"],
        "model": request.model or effective["model"],
        "api_key": request.api_key or effective["api_key"],
    }
    try:
        # 这里允许 key 为空，由下面统一给出"未配置 API Key"的提示
        profile = llm_settings.validate_profile(provider, draft, require_key=False)
    except llm_settings.LLMSettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not profile.api_key:
        raise HTTPException(status_code=422, detail=llm_settings.missing_key_message(provider))
    result = await test_candidate(
        provider,
        {"base_url": profile.base_url, "model": profile.model, "api_key": profile.api_key},
    )
    return {"provider": provider, "model": profile.model, "base_url": profile.base_url, **result}