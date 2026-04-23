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
        self.config=config
        self.model_mgr = ModelManager
        self.rag_mgr = RAGManager
        self.pre_mgr=PreManager
        self.mem_len = config['mem_len']
        self.history = deque(maxlen=self.mem_len)
        self.current_mode = config['initial_mode']
        self._load_mode(self.current_mode,init=True)

    def _load_mode(self, mode: str,init=False):
        if not init:self.history.append({"role": "assistant", "content": self.fixed_chat["closing"]})
        self.current_mode=mode
        self.sys_prompt=[{"role":"system","content":self.config['sys_prompt'][mode]}]
        self.fixed_chat=self.config['fixed_chat'][mode]
        self.knowledge=self.config['knowledge_prompt'][mode]
        self.inference_para=self.config['inference_para'][mode]
        self.model_mgr.base_model.set_adapter(mode)
        self.history.append({"role":"assistant","content":self.fixed_chat["opening"]})

    async def async_chat(self, user_input: str) -> AsyncGenerator[str, None]:

        try:
            yield f"data: {'超天酱：' if self.current_mode=='kangel' else '糖糖：'}\n\n"
            yield f"data: \n\n"
            # Query Rewrite
            rewritten = await self.pre_mgr.rewrite(user_input, self.history)
            # RAG Retrieve
            docs = await self.rag_mgr.retrieve(rewritten,self.knowledge)
            # Context Compress
            context = await self.pre_mgr.compress(self.history, docs)
            context=[{"role":"system","content":context}]
            # Inference
            response=""
            async for chunk in self.model_mgr.generate(self.sys_prompt+context, rewritten,self.inference_para):
                response+=chunk
                yield f"data: {chunk}\n\n"
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