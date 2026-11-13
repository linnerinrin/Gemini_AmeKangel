# """
#架构作用：存档会话管理的抽象接口，供账户管理器实现列表、创建、选择和删除
#数据流：路由请求槽位操作→ IAccount实现→存档元数据/活动会话
# """
"""
interface/IAccount
用于管理存档会话
    ***_session对***会话进行管理
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
