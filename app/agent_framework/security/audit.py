#"""
#架构作用：审计落点模块，AuditSinkPort的两个默认实现
#调用方：适配器在敏感动作（切模型/切会话/鉴权失败等）处调用record(event)
#数据流：AuditEvent →补时间戳→内存deque留存/ print脱敏摘要
#"""
"""
agent_framework/security/audit.py审计落点（骨架）

实现contracts.AuditSinkPort：
- InMemoryAuditSink：deque(maxlen=1000)环形留存AuditEvent，提供tail(n)读取；
- PrintAuditSink：print一行脱敏摘要，对齐原项目print日志风格

两个实现的record内部都try/except吞掉异常：审计失败不得影响主流程
timestamp为0（构造方未填）时由本层用time.time()补齐（dataclasses.replace）
"""
from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import Deque, List

from ..contracts import AuditEvent, AuditSinkPort

#敏感字段黑名单：detail中出现即整体打码，避免key/token落日志
_SENSITIVE_KEYS = frozenset({"api_key", "apikey", "token", "secret", "password", "key"})


def _sanitize_detail(event: AuditEvent) -> str:
    """detail脱敏摘要：只保留键名与安全值，敏感键值替换为***。"""
    if not event.detail:
        return "-"
    parts = []
    for k, v in event.detail.items():
        if str(k).lower() in _SENSITIVE_KEYS:
            parts.append(f"{k}=***")
        else:
            text = str(v)
            if len(text) > 64:
                text = text[:61] + "..."
            parts.append(f"{k}={text}")
    return ", ".join(parts)


def _with_timestamp(event: AuditEvent) -> AuditEvent:
    """timestamp为0时用time.time()补齐；frozen数据类走replace。"""
    if event.timestamp and event.timestamp > 0:
        return event
    return dataclasses.replace(event, timestamp=time.time())


class InMemoryAuditSink(AuditSinkPort):
    """内存环形审计落点：最多留存1000条，超出自动丢弃最旧。"""

    def __init__(self, maxlen: int = 1000) -> None:
        self._events: Deque[AuditEvent] = deque(maxlen=maxlen)

    def record(self, event: AuditEvent) -> None:
        """记录一条审计事件；存储失败静默吞掉，绝不影响主流程。"""
        try:
            self._events.append(_with_timestamp(event))
        except Exception:
            #审计失败不得影响主流程：静默丢弃
            pass

    def tail(self, n: int = 50) -> List[AuditEvent]:
        """读取最近n条事件（按时间升序返回）。"""
        if n <= 0:
            return []
        items = list(self._events)
        return items[-n:]


class PrintAuditSink(AuditSinkPort):
    """print审计落点：一行脱敏摘要，对齐原项目print日志风格。"""

    def record(self, event: AuditEvent) -> None:
        """打印一行审计摘要；打印失败静默吞掉，绝不影响主流程。"""
        try:
            event = _with_timestamp(event)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
            print(
                f"[AUDIT] {ts} action={event.action} identity={event.identity} "
                f"trace={event.trace_id or '-'} detail={_sanitize_detail(event)}"
            )
        except Exception:
            #审计失败不得影响主流程：静默丢弃
            pass
