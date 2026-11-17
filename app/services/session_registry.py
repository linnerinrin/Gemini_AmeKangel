"""
services/session_registry.py会话映射注册表
"""

"""
架构作用：会话映射注册表模块，按session_id持有会话实例与每会话生成锁
调用方：app/services/__init__.py的ChatFacade取会话、取锁、删档回收
数据流：session_id →映射查询/惰性创建→会话实例/ asyncio.Lock
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from app.agent_framework.contracts import SessionRegistryPort

#工厂类型别名：接收session_id，返回会话实例
SessionFactory = Callable[[str], Any]
LockFactory = Callable[[], asyncio.Lock]


class SessionRegistry(SessionRegistryPort):
    """SessionRegistryPort的实现：进程内映射，协程安全。"""

    def __init__(
        self,
        session_factory: Optional[SessionFactory] = None,
        lock_factory: LockFactory = asyncio.Lock,
    ) -> None:
        """注入会话工厂与锁工厂；lock_factory默认asyncio.Lock。"""
        self._session_factory: Optional[SessionFactory] = session_factory
        self._lock_factory: LockFactory = lock_factory
        self._sessions: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        #_mu当前未启用：映射操作均为单线程事件循环内的同步代码，无await交错，暂不需要互斥
        #若未来跨线程使用本注册表，应恢复此锁并包住映射读写

    """
    查询与创建：同步读写映射，协程间无await交错；工厂调用同步执行
    """

    def get(self, session_id: str) -> Optional[Any]:
        """按session_id取已绑定的会话实例；不存在返回None。"""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Any:
        """取会话，不存在则经注入工厂惰性创建；工厂缺失时抛ValueError。"""
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        if self._session_factory is None:
            raise ValueError(
                f"SessionRegistry 未注入 session_factory，无法创建会话: {session_id!r}"
            )
        session = self._session_factory(session_id)
        #并发场景下该会话可能被其他协程先行创建，setdefault以先写入者为准
        return self._sessions.setdefault(session_id, session)

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """返回该会话专属的asyncio锁；不存在则惰性创建。"""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._lock_factory()
            lock = self._locks.setdefault(session_id, lock)
        return lock

    """
    实现类新增能力（不改冻结端口）：删档后回收会话实例与锁，避免注册表无限增长
    """
    def evict(self, session_id: str) -> None:
        """移除指定会话的实例与锁；不存在时静默返回，不做任何事。"""
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
