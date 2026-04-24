"""
普通模型推理类        --实现模型推理 模块核心
ModelManager        --实现IModel接口 包含generate方法
    init        --初始化模型/分词器/适配器/参数/模板
    generate        --返回一个async generator 用于流式输出
"""


from pathlib import Path
import asyncio
from typing import AsyncGenerator
from interfaces.IModel import IModel
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel, get_chat_template

BASE_DIR = Path(__file__).resolve().parent.parent

class ModelManager(IModel):
    def __init__(self,config:dict):
        # 在配置文件中加载 模型/分词器/参数
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(BASE_DIR / config['base_model_paths']),
            max_seq_length=config['max_seq_length'],
            load_in_4bit=True,
        )

        #加载lora适配器
        self.base_model.load_adapter(BASE_DIR / config['mode_paths']['ame'], adapter_name="ame")
        self.base_model.load_adapter(BASE_DIR / config['mode_paths']['kangel'], adapter_name="kangel")

        # 加载对话模板 统一格式化确保模型理解对话结构
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=config["chat_template"])

        #padtoken 用于将不同长度的序列补齐到相同长度，便于批处理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    #生成函数
    async def generate(self,context,query,paras) -> AsyncGenerator[str, None]:
        #初始化一个流式生成器 可以让模型推理一个字就返回一个字
        streamer=TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        #将经过改写的上下文与经过改写的查询组合
        messages=context+[{"role":"user","content":query}]

        #给当前提示词套模板 方便模型推理
        inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

        #打包参数
        generation_kwargs=dict(
                input_ids=inputs,
                streamer=streamer,
                **paras,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        #创建异步任务
        #to_thread 将同步函数放到线程池中执行 返回一个可在事件循环中 await 的协程 这里的协程本质上是await 当前线程是否完毕
        #create_task 将协程包装成Task对象 并立即调度到事件循环中执行
        #组合起来就是 to_thread返回一个可await的协程进create_task 然后create_task将这个协程放进循环 循环每次到这个协程时看一眼线程完毕没有
        await asyncio.create_task(
                asyncio.to_thread(self.base_model.generate, **generation_kwargs)
            )

        #返回一个异步迭代器 返回一个字符后await一下 继续返回
        #与普通迭代器的区别为有没有await 调用需要用async for
        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0.1)