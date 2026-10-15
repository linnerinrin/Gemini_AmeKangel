"""
interface/IModel
模型根据提示词推理结果
    generate 根据提示词推理一串文本
"""
from abc import ABC, abstractmethod
from typing import Any


class IModel(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], paras: dict[str, Any], use_tools: bool = True) -> Any:
        ...
