# """
#架构作用：阶段5模型调用适配模块，ChatClientPort的默认实现（OpenAI兼容端点）
#调用方：未来适配器注入运行路径；当前为纯新增模块，未被app/任何模块import
#数据流：messages + ModelProfile → POST {base_url}/chat/completions → ChatReply
# """
"""
agent_framework/model_adapter/client.py  OpenAI兼容聊天客户端（非流式）

实现contracts.ChatClientPort：
-只支持非流式调用（stream=False），聚合choices[0].message.content进ChatReply；
-超时取自ModelProfile.timeout；base_url去尾斜杠后拼/chat/completions；
- api_key为空时直接抛ModelClientError，不发出任何请求；
-错误归一：HTTP 4xx/5xx、网络异常、JSON解析失败、响应缺choices全部包装为
  ModelClientError（可读原因），原因中绝不包含api_key原文；其余异常不外抛；
-调用日志用print输出（目标base_url、model、耗时、usage），日志严禁出现api_key

传输选型：默认传输为httpx（由requirements.txt第1行openai~=1.68.0传递引入，
已确认安装于venv）；httpx缺失时回退requests，再缺失时回退标准库urllib
（urllib兜底分支标注「传输选型待确认」）。不新增任何第三方依赖
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import ChatClientPort, ChatReply, ModelClientError, ModelProfile

# """
#传输层探测：httpx优先（openai SDK传递依赖，requirements.txt第1行），
#其次requests，最后标准库urllib兜底；导入失败不报错，到调用时才定生死
# """
try:  # httpx：默认传输
    import httpx  # type: ignore
except ImportError:  # pragma: no cover -环境相关
    httpx = None  # type: ignore[assignment]

try:  # requests：次选传输
    import requests  # type: ignore
except ImportError:  # pragma: no cover -环境相关
    requests = None  # type: ignore[assignment]

import urllib.error
import urllib.request

# """
# HTTP响应统一形状：状态码+解码后的文本；与具体传输库解耦，便于测试替身
#类型签名：HttpTransport = Callable[[url, headers, body, timeout], (status, text)]
# """
HttpTransport = Callable[[str, Dict[str, str], bytes, float], Tuple[int, str]]


def _httpx_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, str]:
    """用 httpx 发送 POST 并返回 (状态码, 响应文本)。"""
    response = httpx.post(url, headers=headers, content=body, timeout=timeout)  # type: ignore[union-attr]
    return response.status_code, response.text


def _requests_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, str]:
    """用 requests 发送 POST 并返回 (状态码, 响应文本)。"""
    response = requests.post(url, headers=headers, data=body, timeout=timeout)  # type: ignore[union-attr]
    return response.status_code, response.text


def _urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[int, str]:
    """用标准库 urllib 发送 POST 并返回 (状态码, 响应文本)。

    [传输选型待确认] urllib 为兜底传输：无连接池、错误语义粗糙，
    仅在 httpx 与 requests 均不可用时启用。
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # HTTPError也是响应：读出正文与状态码，交给上层统一归一
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _default_transport() -> HttpTransport:
    """按 httpx → requests → urllib 顺序挑选默认传输。"""
    if httpx is not None:
        return _httpx_transport
    if requests is not None:
        return _requests_transport
    return _urllib_transport


# """
#客户端主体：同步阻塞实现，匹配contracts.ChatClientPort的同步签名；
#构造时可注入transport（http客户端工厂），测试用替身替换，无需monkeypatch
# """


class OpenAICompatibleChatClient:
    """ChatClientPort 默认实现：调用 OpenAI 兼容端点的 /chat/completions。

    非流式、无重试、无会话状态；所有失败归一为 ModelClientError。
    """

    def __init__(self, transport: Optional[HttpTransport] = None) -> None:
        """注入 http 客户端工厂（可选）；缺省按已装库自动选型。"""
        self._transport: HttpTransport = transport or _default_transport()

    # """
    #主流程：空key拒绝→组包→发请求→状态码归一→解析归一→打日志
    #除ModelClientError外不向调用方抛任何异常
    # """
    def complete(self, messages: Sequence[Dict[str, str]], *,
                 profile: ModelProfile) -> ChatReply:
        """发送 messages（OpenAI chat 形状）并返回聚合回复；失败抛 ModelClientError。"""
        if not profile.api_key:
            #凭据缺失：不发请求直接拒绝，原因不含任何凭据内容
            raise ModelClientError("api_key 为空，拒绝发起模型调用，请先配置凭据")

        base_url = profile.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": profile.model,
            "messages": list(messages),
            "temperature": profile.temperature,
            "stream": False,
        }
        #0表示“使用厂商默认”，显式传0会被部分端点当成“最多0个token”
        if profile.max_tokens > 0:
            payload["max_tokens"] = profile.max_tokens
        if profile.top_p > 0:
            payload["top_p"] = profile.top_p
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {profile.api_key}",
        }
        body = json.dumps(payload).encode("utf-8")

        started = time.monotonic()
        status_code, text = self._send(url, headers, body, profile.timeout,
                                       secret=profile.api_key)
        reply = self._parse_reply(status_code, text, profile.model,
                                  secret=profile.api_key)
        elapsed_ms = (time.monotonic() - started) * 1000

        #调用日志：地址/模型/耗时/用量，绝不含api_key
        print(
            f"[model_adapter] POST {base_url} model={reply.model or profile.model} "
            f"耗时={elapsed_ms:.0f}ms usage_tokens={reply.usage_tokens}"
        )
        return reply

    # """
    #传输发送：网络层一切异常在此归一为ModelClientError；不区分具体传输库
    # """
    def _send(self, url: str, headers: Dict[str, str], body: bytes,
              timeout: float, *, secret: str = "") -> Tuple[int, str]:
        """经注入的 transport 发请求；网络异常包装为 ModelClientError。"""
        try:
            return self._transport(url, headers, body, timeout)
        except ModelClientError:
            raise
        except Exception as exc:  #网络/超时/连接异常统一归一
            reason = f"网络请求失败：{type(exc).__name__} {exc}"
            raise ModelClientError(self._sanitize(reason, secret)) from exc

    # """
    #响应归一：状态码→错误；JSON解析失败→错误；缺choices →错误；
    # usage.total_tokens缺失写0；content允许为空字符串（模型确实没说话）
    # """
    def _parse_reply(self, status_code: int, text: str, fallback_model: str,
                     *, secret: str = "") -> ChatReply:
        """把 (状态码, 响应文本) 解析为 ChatReply；一切异常形状归一为 ModelClientError。"""
        if status_code >= 400:
            reason = self._extract_error_message(text, secret)
            raise ModelClientError(f"模型服务返回 HTTP {status_code}：{reason}")

        try:
            data: Dict[str, Any] = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ModelClientError(f"响应 JSON 解析失败：{exc}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelClientError("响应缺少 choices 字段或为空，无法聚合回复")

        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

        total_tokens = usage.get("total_tokens")
        return ChatReply(
            content=str(message.get("content") or ""),
            model=str(data.get("model") or fallback_model),
            finish_reason=str(choice0.get("finish_reason") or ""),
            usage_tokens=total_tokens if isinstance(total_tokens, int) else 0,
        )

    # """
    #错误正文摘要：尽量取厂商error.message，失败则截断原文；
    #任何输出先过_sanitize，保证api_key原文不外泄
    # """
    def _extract_error_message(self, text: str, secret: str = "") -> str:
        """从错误响应正文提取可读原因；解析失败时截断原文前 200 字。"""
        try:
            data = json.loads(text)
            error = data.get("error") if isinstance(data, dict) else None
            if isinstance(error, dict) and error.get("message"):
                return self._sanitize(str(error["message"]), secret)
        except (ValueError, TypeError):
            pass
        return self._sanitize(text[:200], secret)

    @staticmethod
    def _sanitize(reason: str, secret: str = "") -> str:
        """兜底脱敏：原因文本中出现凭据原文或 Bearer 串，一律掩码，不留原文。"""
        #审计修复：厂商可能在error.message回显api_key原文，先按原文精确掩码
        if secret:
            reason = reason.replace(secret, "***")
        #防御性兜底：正常路径不会携带key；一旦发现Bearer形态即掩码
        if "Bearer " in reason:
            reason = reason.replace(reason.split("Bearer ", 1)[1].split()[0], "***") \
                if reason.split("Bearer ", 1)[1].split() else reason
        return reason
