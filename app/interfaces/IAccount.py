"""
interface/IAccount
用于管理存档会话
    ***_session 对***会话进行管理
"""
from abc import ABC,abstractmethod


class IAccount(ABC):

    def __init__(self):
        ...

    @abstractmethod
    def list_sessions(self) -> list[dict]:
        ...

    @abstractmethod
    def create_new_session(self, session_num: int) -> str | None:
        ...

    @abstractmethod
    def chose_session(self, session_num: int):
        ...

    @abstractmethod
    def delete_session(self, session_num: int) -> bool:
        ...
