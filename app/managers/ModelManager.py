# """
#架构作用：图节点与长记忆使用的OpenAI兼容适配器，统一普通/流式生成结果
#数据流：当前设置+消息+参数→ SDK/ChatClientPort → GenerationResult；工具先聚合校验后交给图
#配置惰性解析，客户端按地址/密钥缓存；注入替身不归本对象管理
# """
"""
manager/ModelManager推理模块管理器
Abstract on IModel
OpenAI兼容客户端按当前生效设置惰性构造并缓存；改设置后下一个请求即生效，无需重启
非流式/聚合路径（use_tools=False 且为真实配置）收敛到 ChatClientPort
（agent_framework OpenAICompatibleChatClient，错误归一为 ModelClientError）；
注入替身与流式路径保持 SDK 形状不变
"""
import asyncio
import json
import re
from app.interfaces.IModel import IModel
from app.config.config import config
from app.config.llm_import import load_llm_settings
from app.agent_framework.contracts import ModelProfile
from app.agent_framework.model_adapter.client import OpenAICompatibleChatClient
from openai import AsyncOpenAI


class InvalidToolCallError(ValueError):
    """工具调用不完整或无效；禁止执行，也不自动重试。"""


class EmptyModelResponseError(RuntimeError):
    """模型响应没有任何候选，属于可读业务错误而不是底层IndexError。"""


# """
#先拒绝截断，再检查名称字符/注册表及对象参数；此处不是完整工具参数schema校验
#全部调用校验完成之前不得执行，避免不完整JSON或半个函数名触发副作用
# """
def validate_tool_calls(tool_calls, *, truncated=False, allowed_names=None):
    if tool_calls and truncated:
        raise InvalidToolCallError("模型输出已截断，禁止执行工具调用")
    for call in tool_calls:
        name = call.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise InvalidToolCallError("工具名称缺失或格式无效")
        if allowed_names is not None and name not in allowed_names:
            raise InvalidToolCallError("工具名称未注册或不完整")
        if not isinstance(call.get("args"), dict):
            raise InvalidToolCallError("工具参数必须为 JSON 对象")


def _parse_tool_call(name, arguments, call_id):
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidToolCallError("工具参数不是完整有效的 JSON") from exc
    call = {"name": name, "args": args, "id": call_id}
    validate_tool_calls([call])
    return call


class GenerationResult:
    def __init__(self, content="", tool_calls=None, thinking="", truncated=False, finish_reason=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.thinking = thinking
        self.finish_reason = finish_reason
        self.truncated = bool(truncated or finish_reason == "length")

    def __str__(self):
        return self.content


class ModelManager(IModel):
    """OpenAI 兼容模型管理器。

    关键点：客户端不在 __init__ 里构造，而是每个请求读一次当前设置再惰性构造，
    并以 (base_url, api_key) 为缓存键复用。所以控制面板改完设置后，下一个请求就走新客户端。
    测试里可以直接给 mgr.client 赋一个假客户端，此时这里不会覆盖它。
    """

    def __init__(self):
        super().__init__()
        self.tools = config.TOOLS_SCHEMAS
        self.client = None
        #外部注入的client（测试/替身）不归本管理器所有，绝不覆盖
        self._owned_client = False
        #最近一次请求实际解析出的厂商/模型，供历史落库排查"这轮为什么风格不同"
        self.last_provider = None
        self.last_model = None

    def _remember_resolved(self, resolved):
        """记下本次实际解析出的 provider/model（来自 resolve_profile，不是 config.DEEPSEEK_MODEL）。

        resolved 为空表示当前 client 是外部注入的替身，没有真实厂商可记，如实写 None。
        """
        self.last_provider = resolved.get("provider") if resolved else None
        self.last_model = resolved.get("model") if resolved else None
        return resolved

    def _load_config(self):
        """每次请求现读设置：data/llm_settings.json 优先，缺字段回退 .env 的 DEEPSEEK_*。

        返回 {} 表示当前 client 是外部注入的替身，不需要（也不该）解析真实设置。
        """
        if getattr(self, "client", None) is not None and not getattr(self, "_owned_client", False):
            return {}
        return load_llm_settings().require_api_key()

    # """
    #外部client原样复用；自持client的配置键变化时新建，下一轮立即生效
    # """
    def _client(self, resolved):
        """按 (base_url, api_key) 惰性构造并缓存客户端；外部注入的 client 直接沿用。"""
        client = getattr(self, "client", None)
        if client is not None and not getattr(self, "_owned_client", False):
            return client
        key = (resolved.get("base_url"), resolved.get("api_key"))
        if client is None or getattr(self, "_client_key", None) != key:
            #替换前先关掉旧的自持client，避免HTTP连接池泄漏
            self._close_owned_client()
            client = AsyncOpenAI(api_key=resolved.get("api_key"), base_url=resolved.get("base_url"))
            self.client = client
            self._client_key = key
            self._owned_client = True
        return client

    def _close_owned_client(self):
        """尽力关闭自持client；外部注入的替身不归本管理器关闭。"""
        client = getattr(self, "client", None)
        if client is None or not getattr(self, "_owned_client", False):
            return
        close = getattr(client, "close", None)
        if close is None:
            #测试替身等无close方法的client直接跳过
            return
        try:
            asyncio.get_running_loop().create_task(close())
        except RuntimeError:
            #没有运行中的事件循环时同步关一次，忽略关闭失败
            try:
                asyncio.run(client.close())
            except Exception:
                pass

    async def aclose(self):
        """关闭自持HTTP客户端并复位缓存；进程退出或会话销毁时由上层调用。"""
        client = getattr(self, "client", None)
        if client is not None and getattr(self, "_owned_client", False):
            await client.close()
        self.client = None
        self._client_key = None
        self._owned_client = False

    # """
    #普通请求读取当前配置并记录模型元信息；拿到第一候选后转统一结果，再校验工具
    #阶段4接线：use_tools=False 且走真实配置（resolved非空）时收敛到 ChatClientPort；
    #注入替身（resolved={}）保持SDK形状不变，测试契约不动；流式路径保持现状
    #注意：判据是「自持客户端」（外部注入替身即使monkeypatch了_load_config也走SDK形状，
    #测试契约不动）；端口payload只含model/messages/temperature，paras里的top_p/max_tokens
    #不再下发，且端口无SDK默认重试——两处行为变化点已在合并手册留档
    # """
    async def generate(self, messages, paras, use_tools=True):
        resolved = self._remember_resolved(self._load_config())
        injected = getattr(self, "client", None) is not None and not getattr(self, "_owned_client", False)
        if not use_tools and resolved and not injected:
            return await self._generate_via_port(messages, paras, resolved)
        response = await self._client(resolved).chat.completions.create(
            model=resolved.get("model") or config.DEEPSEEK_MODEL,
            messages=messages,
            **paras,
            tools=self.tools if use_tools else None,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise EmptyModelResponseError("模型响应为空：没有返回任何候选回复")
        choice = choices[0]
        result = self._result_from_message(choice.message, getattr(choice, "finish_reason", None))
        self._validate_tools(result.tool_calls, result.truncated, use_tools)
        return result

    # """
    #ChatClientPort非流式出口：同步complete经to_thread调用，错误已被端口归一为
    # ModelClientError（且_sanitize保证不泄api_key），直接向上抛交上层原有错误处理
    # """
    async def _generate_via_port(self, messages, paras, resolved):
        client = getattr(self, "_port_client", None)
        if client is None:
            client = OpenAICompatibleChatClient()
            self._port_client = client
        profile = ModelProfile(
            base_url=resolved.get("base_url") or "",
            model=resolved.get("model") or config.DEEPSEEK_MODEL,
            api_key=resolved.get("api_key") or "",
            temperature=float(paras.get("temperature", 1.0)),
            #SDK默认超时600秒；端口无重试
            timeout=600.0,
            #P2修复：采样上限随端口透传，避免端口路径悄悄丢掉max_tokens/top_p
            max_tokens=int(paras.get("max_tokens") or 0),
            top_p=float(paras.get("top_p") or 0.0),
        )
        reply = await asyncio.to_thread(client.complete, messages, profile=profile)
        return GenerationResult(content=reply.content or "", finish_reason=reply.finish_reason or None)

    def _validate_tools(self, tool_calls, truncated, use_tools):
        names = {tool["function"]["name"] for tool in (self.tools or [])} if use_tools else set()
        validate_tool_calls(tool_calls, truncated=truncated, allowed_names=names)

    @staticmethod
    def _result_from_message(msg, finish_reason=None):
        raw_calls = getattr(msg, "tool_calls", None) or []
        if raw_calls and finish_reason == "length":
            raise InvalidToolCallError("模型输出已截断，禁止执行工具调用")
        tool_calls = [
            _parse_tool_call(tc.function.name, tc.function.arguments, tc.id)
            for tc in raw_calls
        ]
        return GenerationResult(
            content=msg.content or "",
            tool_calls=tool_calls,
            thinking=getattr(msg, "reasoning_content", "") or "",
            finish_reason=finish_reason,
        )

    # """
    #按index聚合碎片化工具名/参数；正文与推理增量逐事件await，向上游传递背压
    # finally关闭单次响应流，包括异常/取消；聚合完成后才解析JSON、验证并发工具名
    # """
    async def stream_generate(self, messages, paras, use_tools=True, on_event=None):
        """流式输出统一适配器：thinking/content 按增量推送，tool 名称一次推送。"""
        resolved = self._remember_resolved(self._load_config())
        stream = await self._client(resolved).chat.completions.create(
            model=resolved.get("model") or config.DEEPSEEK_MODEL,
            messages=messages,
            stream=True,
            **paras,
            tools=self.tools if use_tools else None,
        )
        content_parts = []
        thinking_parts = []
        tool_state = {}
        finish_reason = None
        try:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = getattr(delta, "content", None) or ""
                if reasoning:
                    thinking_parts.append(reasoning)
                    if on_event:
                        await on_event({"type": "thinking_delta", "text": reasoning})
                if content:
                    content_parts.append(content)
                    if on_event:
                        await on_event({"type": "answer_delta", "text": content})
                for tc in (getattr(delta, "tool_calls", None) or []):
                    index = tc.index
                    current = tool_state.setdefault(index, {"id": tc.id or "", "name": "", "args": ""})
                    if tc.id:
                        current["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            current["name"] += fn.name
                        if fn.arguments:
                            current["args"] += fn.arguments
        finally:
            if hasattr(stream, "close"):
                await stream.close()
        truncated = finish_reason == "length"
        if tool_state and truncated:
            raise InvalidToolCallError("模型输出已截断，禁止执行工具调用")
        tool_calls = [
            _parse_tool_call(item["name"], item["args"], item["id"])
            for item in tool_state.values()
        ]
        self._validate_tools(tool_calls, truncated, use_tools)
        if on_event:
            for call in tool_calls:
                await on_event({"type": "tool_call", "name": call["name"]})
        return GenerationResult(
            "".join(content_parts), tool_calls, "".join(thinking_parts),
            finish_reason=finish_reason,
        )

#进程级单例统一在app/managers/__init__.py创建，本模块不再保留第二实例

# [总审计 2026-11-13] P1：llm_settings.NotConfiguredError 在本文件中未捕获（generate/stream_generate），
# 未配置 key 时单次生成返回 500、流式只回“生成过程中出现错误”，与 README 承诺的“界面明确提示尚未配置
# API Key”不符；建议在 chat_routes 或本层转成可读提示。
# P2：_generate_via_port 的 payload 只带 model/messages/temperature，top_p/max_tokens 被丢弃。
