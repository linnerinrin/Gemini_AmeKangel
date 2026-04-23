from abc import ABC, abstractmethod

class IModel(ABC):
    def __init__(self):
        self.base_model = None

    @abstractmethod
    async def generate(self,**kwargs):
        ...