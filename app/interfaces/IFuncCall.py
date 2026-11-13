#"""
#架构作用：工具管理器抽象，供具体FuncCallManager构建可执行工具集合
#数据流：会话模式/管理器依赖→子类_create_tools →图的工具节点
#"""
"""
interfaces/IFuncCall
用于为agent配置tools
abstract on IChat
    _create_tools tools集合返回工具列表


"""
from abc import ABC, abstractmethod
from app.interfaces.IChat import IChat


class IFuncCall(IChat):
    #"""
    #工具集合的抽象工厂；子类自行决定依赖绑定及返回形状
    #"""
    @abstractmethod
    def _create_tools(self):
        ...
