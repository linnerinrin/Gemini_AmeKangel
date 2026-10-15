"""
interface/IRAG
用于从长文本中检索目标文本
Abstract on IChat
    retrieve 用于检索与目标查询匹配的文档片段

"""

from abc import ABC, abstractmethod

from app.interfaces.IChat import IChat


class IRAG(IChat):

    @abstractmethod
    def update_context(self, context: list) -> None:
        ...

    @abstractmethod
    def recall(self, query: str, caller: str) -> list:
        ...

    @abstractmethod
    def rerank(self, query: str, documents: list) -> list:
        ...

    @abstractmethod
    async def retrieve(self, query: str, caller: str = "knowledge") -> str:
        ...
