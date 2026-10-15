"""
interfaces/IGraph
用于绘制agent数据流
abstract on IChat
    build 总构建函数 返回可用于invoke的stategraph
    create_node 建立用于图绘制的节点 返回节点集合的dict
    create_router 建立图所需要的路由 返回路由集合的dict
    create_graph 由node与router建立langgraph的模拟工作流
"""
from abc import abstractmethod
from app.interfaces.IChat import IChat

class IGraph(IChat):


    @abstractmethod
    def build(self,*args):
        ...

    @abstractmethod
    def create_nodes(self,*args):
        ...

    @abstractmethod
    def create_routers(self,*args):
        ...

    @abstractmethod
    def create_graph(self,nodes,routers,tools):
        ...