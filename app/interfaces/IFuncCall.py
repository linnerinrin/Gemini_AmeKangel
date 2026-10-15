"""
interfaces/IFuncCall
用于为agent配置tools
abstract on IChat
    _create_tools tools集合 返回工具列表


"""
from abc import ABC, abstractmethod
from app.interfaces.IChat import IChat


class IFuncCall(IChat):
    @abstractmethod
    def _create_tools(self):
        ...


