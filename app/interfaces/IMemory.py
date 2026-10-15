"""
interface/IMemory
用于管理全量/短期上下文、用户与智能体信息
Abstract on IChat
    refresh 用于记忆的启动与重启
    update/load short_memory 对短期记忆(最近n条上下文)的读写
    update/load long_memory 对长期记忆(用户状态)的读写
    update/load status 对智能体状态的读写
    load history 对全量上下文的读取
"""
from abc import ABC, abstractmethod

from app.interfaces.IChat import IChat


class IMemory(IChat):
    @abstractmethod
    def refresh(self, session_id: str) -> None:
        ...

    @abstractmethod
    def update_short_memory(self, query: str, reply: str) -> None:
        ...

    @abstractmethod
    def load_short_memory(self) -> None:
        ...

    @abstractmethod
    async def update_long_memory(self, memory: str) -> None:
        ...

    @abstractmethod
    def load_long_memory(self):
        ...

    @abstractmethod
    def update_status(self, followers=None, affection=None, stress=None, mental=None) -> None:
        ...

    @abstractmethod
    def load_status(self):
        ...

    @abstractmethod
    def load_history(self) -> None:
        ...
