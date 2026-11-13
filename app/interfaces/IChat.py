# """
#架构作用：为会话相关管理器提供模式切换的最小抽象；调用方是具体管理器
#数据流：模式标识由会话传入load_mode，具体实现决定如何加载配置与状态
# """
"""
interfaces/IChat
用于chat session的抽象
    current_mode会话模式

    load_mode加载模式

"""

from abc import ABC,abstractmethod


class IChat(ABC):
    def __init__(self):
        self.current_mode: str = "0"

    # """
    #子类必须实现模式加载；抽象层不保证参数类型、状态隔离或返回语义
    # """
    @abstractmethod
    def load_mode(self,*args):
        ...
