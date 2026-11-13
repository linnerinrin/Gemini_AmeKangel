# """
#架构作用：图编排管理器抽象，定义节点、路由、图以及总装配入口
#数据流：会话/工具依赖→节点与分支映射→可运行图；不在接口层执行推理
# """
"""
interfaces/IGraph
用于绘制agent数据流
abstract on IChat
    build总构建函数返回可用于invoke的stategraph
    create_nodes建立用于图绘制的节点返回节点集合的dict
    create_routers建立图所需要的路由返回路由集合的dict
    create_graph由node与router建立langgraph的模拟工作流
"""
from abc import abstractmethod
from app.interfaces.IChat import IChat

class IGraph(IChat):


    # """
    #总装配入口：调用子类节点、路由和图构建逻辑，返回运行体
    # """
    @abstractmethod
    def build(self,*args):
        ...

    # """
    #节点工厂：将模型、工具或记忆操作转换为图节点映射
    # """
    @abstractmethod
    def create_nodes(self,*args):
        ...

    # """
    #分支工厂：给出状态到后继节点的路由函数映射
    # """
    @abstractmethod
    def create_routers(self,*args):
        ...

    # """
    #接线入口：合并节点、分支和工具，具体实现负责状态结构与图编译
    # """
    @abstractmethod
    def create_graph(self,nodes,routers,tools):
        ...
