#"""
#架构作用：检索管理器抽象，向工具层暴露上下文更新、召回、重排和综合检索
#数据流：文本集合/查询→具体向量召回→重排→拼接结果；caller区分检索来源
#"""
"""
interface/IRAG
用于从长文本中检索目标文本
Abstract on IChat
    retrieve用于检索与目标查询匹配的文档片段

"""

from abc import ABC, abstractmethod

from app.interfaces.IChat import IChat


class IRAG(IChat):

    #"""
    #更新上下文索引的入口；隔离范围与存储寿命由具体实现决定
    #"""
    @abstractmethod
    def update_context(self, context: list) -> None:
        ...

    #"""
    #候选召回：caller选择知识或上下文来源，输出候选文档列表
    #"""
    @abstractmethod
    def recall(self, query: str, caller: str) -> list:
        ...

    #"""
    #重排：按查询为候选文档重新排序，供综合检索入口调用
    #"""
    @abstractmethod
    def rerank(self, query: str, documents: list) -> list:
        ...

    #"""
    #综合检索：异步入口返回可供模型消费的文本，不在接口层承诺授权隔离
    #"""
    @abstractmethod
    async def retrieve(self, query: str, caller: str = "knowledge") -> str:
        ...
