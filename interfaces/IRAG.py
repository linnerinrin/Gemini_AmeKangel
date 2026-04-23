from abc import ABC, abstractmethod

class IRAG(ABC):
    @abstractmethod
    async def read(self,**kwargs):
        ...

    @abstractmethod
    async def recall(self, **kwargs):
        ...

    @abstractmethod
    async def rerank(self, **kwargs):
        ...

    @abstractmethod
    async def retrieve(self, **kwargs):
        ...