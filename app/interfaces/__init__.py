#"""
#架构作用：集中重导出常用接口类型，供服务层统一导入
#"""
from app.interfaces.IAccount import IAccount
from app.interfaces.IChat import IChat
from app.interfaces.IGraph import IGraph
from app.interfaces.IMemory import IMemory
from app.interfaces.IModel import IModel
from app.interfaces.IRAG import IRAG
from app.interfaces.IFuncCall import IFuncCall

__all__ = [
    "IAccount",
    "IChat",
    "IGraph",
    "IMemory",
    "IModel",
    "IRAG",
    "IFuncCall",
]
