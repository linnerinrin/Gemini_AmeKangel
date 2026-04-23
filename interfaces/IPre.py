from abc import abstractmethod
from interfaces.IModel import IModel

class IPre(IModel):
    @abstractmethod
    async def generate(self, **kwargs):
        ...

    @abstractmethod
    async def rewrite(self, **kwargs):
        ...

    @abstractmethod
    async def compress(self, **kwargs):
        ...

