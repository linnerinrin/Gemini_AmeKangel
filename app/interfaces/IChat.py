"""
interfaces/IChat
用于chat session的抽象
    current_mode 会话模式

    load_mode 加载模式

"""

from abc import ABC,abstractmethod


class IChat(ABC):
    def __init__(self):
        self.current_mode: str = "0"

    @abstractmethod
    def load_mode(self,*args):
        ...