"""
manager/ModelManager 推理模块管理器
Abstract on IModel
OpenAI 兼容客户端按当前生效设置惰性构造并缓存；改设置后下一个请求即生效，无需重启。
"""
import json
import re
from app.interfaces.IModel import IModel
from app.config.config import config
from app.config.llm_import import load_llm_settings
from openai import AsyncOpenAI


class InvalidToolCallError(ValueError):
    """工具调用不完整或无效；禁止执行，也不自动重试。"""


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
        # 外部注入的 client（测试/替身）不归本管理器所有，绝不覆盖
        self._owned_client = False
        # 最近一次请求实际解析出的厂商/模型，供历史落库排查"这轮为什么风格不同"
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

    def _client(self, resolved):
        """按 (base_url, api_key) 惰性构造并缓存客户端；外部注入的 client 直接沿用。"""
        client = getattr(self, "client", None)
        if client is not None and not getattr(self, "_owned_client", False):
            return client
        key = (resolved.get("base_url"), resolved.get("api_key"))
        if client is None or getattr(self, "_client_key", None) != key:
            client = AsyncOpenAI(api_key=resolved.get("api_key"), base_url=resolved.get("base_url"))
            self.client = client
            self._client_key = key
            self._owned_client = True
        return client

    async def generate(self, messages, paras, use_tools=True):
        resolved = self._remember_resolved(self._load_config())
        response = await self._client(resolved).chat.completions.create(
            model=resolved.get("model") or config.DEEPSEEK_MODEL,
            messages=messages,
            **paras,
            tools=self.tools if use_tools else None,
        )
        choice = response.choices[0]
        result = self._result_from_message(choice.message, getattr(choice, "finish_reason", None))
        self._validate_tools(result.tool_calls, result.truncated, use_tools)
        return result

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

model_manager=ModelManager()
