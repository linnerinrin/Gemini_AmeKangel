# """
#架构作用：模型管理器的推理抽象，由调用链传入消息、生成参数与工具开关
#数据流：messages/paras →具体厂商实现→调用方消费返回结果；此处不发起网络请求
# """
"""
interface/IModel
模型根据提示词推理结果
    generate根据提示词推理一串文本
    stream_generate按增量推送推理结果（阶段4补声明，实现早已存在于ModelManager）
"""
from abc import ABC, abstractmethod
from typing import Any


class IModel(ABC):
    # """
    #声明异步单次生成能力；具体实现负责解释参数和返回值
    # """
    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], paras: dict[str, Any], use_tools: bool = True) -> Any:
        ...

    # """
    #声明异步流式生成能力；实现方按增量回调on_event，返回聚合结果
    # """
    @abstractmethod
    async def stream_generate(self, messages: list[dict[str, Any]], paras: dict[str, Any], use_tools: bool = True, on_event=None) -> Any:
        ...
