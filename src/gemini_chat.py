"""
核心类         --综合所有manager
ChatSession         --完成所有逻辑结合并适配流式输出
        init        --加载其他manager并维护一个上下文队列
        _load_mode      --加载指定模式 加载此模式的所有配置参数 切换lora为当前模式 并将上一个模式的结束会话与这个模式的开始会话贴进history
        async_chat      --一次会话 异步流水线先后实现 查询改写/RAG检索/上下文压缩/模型推理 四个步骤 并将模型推理格式转化为SSE（“data: {text}\n\n”）
        closing/opening         --输出结束语/开场白
        continue_chat       --循环调用async_chat 仅测试用
"""

from collections import deque
import yaml
import asyncio
import logging
from interfaces.IRAG import IRAG
from interfaces.IModel import IModel
from interfaces.IPre import IPre

from pathlib import Path
from typing import AsyncGenerator


BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(self, config: dict,ModelManager:IModel,RAGManager:IRAG,PreManager:IPre):
        #加载全局配置
        self.config=config
        self.mem_len = config['mem_len']
        self.current_mode = config['initial_mode']

        #加载manager
        self.model_mgr = ModelManager
        self.rag_mgr = RAGManager
        self.pre_mgr=PreManager

        #维护队列
        self.history = deque(maxlen=self.mem_len)
        self._load_mode(self.current_mode,init=True)

    def _load_mode(self, mode: str,init=False):
        #第一次不贴上一次模式（当前模式）的结束语 其他时候贴
        if not init:self.history.append({"role": "assistant", "content": self.fixed_chat["closing"]})

        #加载当前模式配置
        self.current_mode=mode
        self.sys_prompt=[{"role":"system","content":self.config['sys_prompt'][mode]}]
        self.fixed_chat=self.config['fixed_chat'][mode]
        self.knowledge=self.config['knowledge_prompt'][mode]
        self.inference_para=self.config['inference_para'][mode]

        #加载当前模式适配器
        self.model_mgr.base_model.set_adapter(mode)

        #贴当前模式的开场白
        self.history.append({"role":"assistant","content":self.fixed_chat["opening"]})


    async def async_chat(self, user_input: str) -> AsyncGenerator[str, None]:

        try:
            yield f"data: {'超天酱：' if self.current_mode=='kangel' else '糖糖：'}\n\n"
            yield f"data: \n\n"

            #查询改写
            rewritten = await self.pre_mgr.rewrite(user_input, self.history)

            # RAG检索
            docs = await self.rag_mgr.retrieve(rewritten,self.knowledge)

            #上下文压缩
            context = await self.pre_mgr.compress(self.history, docs)
            context=[{"role":"system","content":context}]

            #模型推理
            response=""
                #调用异步生成器输出
            async for chunk in self.model_mgr.generate(self.sys_prompt+context, rewritten,self.inference_para):
                response+=chunk
                yield f"data: {chunk}\n\n"

            #记录上次对话到历史队列
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})
        except Exception as e:
            logger.error(f"Chat failed: {e}", exc_info=True)
            yield "系统错误，稍后再试吧。"

    async def closing(self):
        yield f"data: 【{self.current_mode.upper()} 模式结束】\n\n"
        yield f"data: \n\n"
        for chunk in self.fixed_chat["closing"]:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.1)

    async def opening(self):
        yield f"data: 【{self.current_mode.upper()} 模式启动】\n\n"
        yield f"data: \n\n"
        for chunk in self.fixed_chat["opening"]:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.1)

    async def continue_chat(self):
        while True:
            user_input = input("你：").strip()
            if user_input.lower() == "exit":
                self.closing()
                break
            if user_input.lower() == "change":
                self.closing()
                self._load_mode("ame" if self.current_mode=="kangel" else "kangel")
                self.opening()
                continue
            if user_input.lower() == "history":
                print(self.history)
                continue
            if not user_input:
                continue
            async for chunk in self.async_chat(user_input):
                print(chunk,flush=True,end='')
            print()