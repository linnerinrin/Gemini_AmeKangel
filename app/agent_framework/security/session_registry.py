# """
#架构作用：阶段0.2会话映射注册表模块，SessionRegistryPort的实现
#调用方：未来适配器按session_id取会话实例与每会话生成锁
#数据流：session_id →全局锁保护下的映射查询/惰性创建→会话实例/ asyncio.Lock
# """
"""
agent_framework/security/session_registry.py会话映射注册表

实现contracts.SessionRegistryPort：
-内部dict[str, Any]维护session_id →会话实例映射（不import ChatSession，
用Any持有，避免反向依赖services包）；
-内部dict[str, asyncio.Lock]维护每会话一把生成锁（按会话隔离互斥）；
-映射读写为事件循环内同步代码，无await交错，暂不引入全局互斥锁；
-会话实例由构造函数注入的session_factory惰性创建；
  factory为None时get_or_create抛ValueError（唯一允许的异常路径）
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from ..contracts import SessionRegistryPort

#工厂类型别名：接收session_id，返回会话实例
SessionFactory = Callable[[str], Any]
LockFactory = Callable[[], asyncio.Lock]


class SessionRegistry(SessionRegistryPort):
    """SessionRegistryPort 的实现：进程内映射，协程安全。"""

    def __init__(
        self,
        session_factory: Optional[SessionFactory] = None,
        lock_factory: LockFactory = asyncio.Lock,
    ) -> None:
        """注入会话工厂与锁工厂；lock_factory 默认 asyncio.Lock。"""
        self._session_factory: Optional[SessionFactory] = session_factory
        self._lock_factory: LockFactory = lock_factory
        self._sessions: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        # _mu当前未启用：映射操作均为单线程事件循环内的同步代码，无await交错，暂不需要互斥
        #若未来跨线程使用本注册表，应恢复此锁并包住映射读写


    # """
    #查询与创建：同步读写映射，协程间无await交错；工厂调用同步执行
    # """

    def get(self, session_id: str) -> Optional[Any]:
        """按 session_id 取已绑定的会话实例；不存在返回 None。"""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Any:
        """取会话，不存在则经注入工厂惰性创建；工厂缺失时抛 ValueError。"""
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        if self._session_factory is None:
            raise ValueError(
                f"SessionRegistry 未注入 session_factory，无法创建会话: {session_id!r}"
            )
        session = self._session_factory(session_id)
        #并发场景下可能已被其他协程创建，以先写入者为准
        return self._sessions.setdefault(session_id, session)

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """返回该会话专属的 asyncio 锁；不存在则惰性创建。"""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._lock_factory()
            lock = self._locks.setdefault(session_id, lock)
        return lock

    # """
    #实现类新增能力（不改冻结端口）：删档后回收会话实例与锁，避免注册表无限增长
    # """
    def evict(self, session_id: str) -> None:
        """移除指定会话的实例与锁；不存在时静默返回，不做任何事。"""
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
