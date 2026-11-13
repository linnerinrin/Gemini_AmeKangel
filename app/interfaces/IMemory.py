#"""
#架构作用：会话记忆与角色状态的抽象边界，供服务和工具管理器使用
#数据流：存档标识绑定→短期/长期/状态读写→历史读取；持久化由具体类承担
#"""
"""
interface/IMemory
用于管理全量/短期上下文、用户与智能体信息
Abstract on IChat
    refresh用于记忆的启动与重启
    update/load short_memory对短期记忆(最近n条上下文)的读写
    update/load long_memory对长期记忆(用户状态)的读写
    update/load status对智能体状态的读写
    load history对全量上下文的读取
"""
from abc import ABC, abstractmethod

from app.interfaces.IChat import IChat


class IMemory(IChat):
    #"""
    #存档切换后刷新记忆视图；调用方必须先确认存档归属
    #"""
    @abstractmethod
    def refresh(self, session_id: str) -> None:
        ...

    #"""
    #将一组问答加入短期记忆；容量与持久化策略由具体实现决定
    #"""
    @abstractmethod
    def update_short_memory(self, query: str, reply: str) -> None:
        ...

    #"""
    #加载短期上下文到管理器状态，不在接口层暴露返回对象
    #"""
    @abstractmethod
    def load_short_memory(self) -> None:
        ...

    #"""
    #长期记忆写入口，异步调用允许具体实现执行外部或持久化工作
    #"""
    @abstractmethod
    async def update_long_memory(self, memory: str) -> None:
        ...

    #"""
    #长期记忆读取；实际返回形状未在抽象层声明
    #"""
    @abstractmethod
    def load_long_memory(self):
        ...

    #"""
    #角色状态写入口；None与显式数值的语义由实现解释
    #"""
    @abstractmethod
    def update_status(self, followers=None, affection=None, stress=None, mental=None) -> None:
        ...

    #"""
    #读取当前角色状态，供路由和图构造提示词使用
    #"""
    @abstractmethod
    def load_status(self):
        ...

    #"""
    #全量历史加载入口；抽象层未表达场景/任务轴及分页策略
    #"""
    @abstractmethod
    def load_history(self) -> None:
        ...
