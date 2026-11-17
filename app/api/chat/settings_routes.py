#"""
#架构作用：模型配置的HTTP读写及连通性探测边界，由设置页调用
#数据流：请求草稿+生效配置→校验/保存，或携带密钥请求候选模型地址→返回状态
#"""
"""
api/chat/settings_routes.py模型设置接口（只做OpenAI兼容厂商）

- GET  /chat/settings/llm读取设置（key一律掩码）
- POST /chat/settings/llm保存并切换active；生成中拒绝
- POST /chat/settings/llm/test用候选配置发一次最小请求做连通性测试
"""

import asyncio

import httpx
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.api.chat.chat_routes import require_idle
from app.api.chat.settings_schemas import LLMSettingsUpdate, LLMTestRequest
from app.config.llm_import import load_llm_settings

#按文件路径加载（app.services在测试里会被换成替身包，包级导入会失败）
llm_settings = load_llm_settings()

router = APIRouter()

#连通性测试只验证"能连上、key有效、模型名认得"，所以请求体压到最小
TEST_PROMPT = "ping"
TEST_MAX_TOKENS = 4
TEST_TIMEOUT_SECONDS = 20.0


#"""
#部分更新合并：地址/模型回退到生效值，密钥只有显式提交才进入保存草稿
#"""
def _draft(provider: str, submitted: dict) -> dict:
    """把"只提交改动字段"的草稿补成完整候选配置，再交给统一校验。"""
    effective = llm_settings.resolve_profile(provider)
    draft = {
        "base_url": submitted.get("base_url") or effective["base_url"],
        "model": submitted.get("model") or effective["model"],
    }
    #没提交key字段=>沿用已保存/.env；提交了（哪怕是空串）就按提交值处理
    if submitted.get("api_key") is not None:
        draft["api_key"] = submitted["api_key"]
    return draft


#"""
#读取入口：设置服务输出掩码视图；掩码不等同于接口鉴权
#"""
@router.get("/settings/llm")
async def read_llm_settings():
    """返回active、各厂商base_url/model、掩码后的key与"是否已配置可用key"。"""
    return llm_settings.settings_view()


#"""
#保存入口：检查生成空闲后合并补丁，交给设置服务落盘并回读掩码视图
#"""
@router.post("/settings/llm")
async def update_llm_settings(request: LLMSettingsUpdate):
    """保存设置并切换active；生成中禁止修改（否则改的是正在进行的那一轮的配置）。"""
    require_idle()
    payload = {
        "active": request.active,
        "profiles": {
            name: _draft(name, profile.model_dump(exclude_unset=True))
            for name, profile in request.profiles.items()
        },
    }
    #保存路径上的凭据-地址绑定由llm_settings.apply_update内的
    #_assert_key_binding_unchanged统一校验（地址变更必须重输key），这里不重复拦
    try:
        llm_settings.apply_update(payload)
    except llm_settings.LLMSettingsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    #保存后立刻回读一次：前端拿到的就是落盘后的真实状态（key仍是掩码）
    return llm_settings.settings_view()


#"""
#错误翻译：常见状态码输出固定提示，其余分支仍包含底层异常文本
#"""
def _describe_error(exc: BaseException) -> str:
    """把SDK异常翻成人话，同时保证不把key带出来。"""
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
    #通用分支不外发异常原文，只进服务端日志
    print(f"[settings/llm/test] 连通性测试异常: {exc!r}")
    return "测试请求失败，请检查配置后重试"


#"""
#创建短生命周期客户端执行最小生成；超时20秒、禁重试，finally尽力关闭客户端
#取消继续向外传播，普通异常转成前端可展示结果
#"""
async def test_candidate(provider: str, profile: dict) -> dict:
    """用候选配置发一次最小请求；返回{"ok": bool, "message": str}，永不回显key。"""
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
    except Exception as exc:  #网络/SDK/鉴权全部收敛成可读字符串
        return {"ok": False, "message": _describe_error(exc)}
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass


#"""
#探测入口：按厂商补全候选配置，再发起外部请求；没有保存动作
#URL校验仅约束HTTP(S)形状，未将密钥与可信目标绑定。不可信调用者可达时可诱使携密钥外发或访问内网
#"""
@router.post("/settings/llm/test")
async def test_llm_settings(request: LLMTestRequest):
    """连通性校验：key / base_url /模型都可以覆盖，留空则沿用已保存值。"""
    require_idle()
    provider = request.provider
    if provider not in llm_settings.PROVIDER_CATALOG:
        raise HTTPException(status_code=422, detail=f"未知模型厂商: {provider}")
    effective = llm_settings.resolve_profile(provider)
    candidate_base_url = request.base_url or effective["base_url"]
    key_overridden = bool(request.api_key)
    #凭据-地址绑定：候选地址与生效地址不一致且未显式携带key时，
    #不得把已保存key发往候选地址
    if not key_overridden and llm_settings.normalize_base_url(candidate_base_url) != llm_settings.normalize_base_url(effective["base_url"]):
        raise HTTPException(
            status_code=422,
            detail="覆盖 base_url 时必须显式提供 api_key，不允许复用已保存密钥",
        )
    draft = {
        "base_url": candidate_base_url,
        "model": request.model or effective["model"],
        "api_key": request.api_key or effective["api_key"],
    }
    try:
        #这里允许key为空，由下面统一给出"未配置API Key"的提示
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

#保存与探测路径的地址校验统一在llm_settings：validate_profile内的_assert_public_host
#按LLM_ALLOW_PRIVATE_BASE_URL决定是否放行回环/私网，_assert_key_binding_unchanged
#在落盘前拒绝“地址变更但未提交新key”。
