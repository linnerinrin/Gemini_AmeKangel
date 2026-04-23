import asyncio
from pathlib import Path
from interfaces.IPre import IPre
from unsloth import FastLanguageModel, get_chat_template

BASE_DIR = Path(__file__).resolve().parent.parent

class PreManager(IPre):
    def __init__(self, config):
        self.config=config
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(BASE_DIR / config['bart_paths']),
            max_seq_length=config['max_seq_length'],
            load_in_4bit=True,
        )
        self.paras=config["inference_para"]
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=config["chat_template"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def rewrite(self,user_input,history):
        if not history:
            return user_input

        history_text = ""
        for msg in history:
            role = msg["role"]
            history_text += f"{role}：{msg['content']}\n"

        prompt = f"""
        请结合以下对话历史，若你认为用户的输入存在指代不清等问题，如出现“他”、“这个”等代词，将用户当前问题根据上下文重写为一个独立、完整的问题，保留核心语义并解决指代不清问题，
        仅输出问题，若你认为不需要改写，则保留用户输入完全不变，严格禁止给出其他解释或前后缀：
        对话历史：
        {history_text}
        当前问题：{user_input}
        重写后的问题：""".strip()
        result=await self.generate(prompt,paras=self.paras["rewrite"])
        return result if result else user_input


    async def compress(self,history,docs):
        if not history:
            return ""

        history_text = ""
        for msg in history:
            role =msg["role"]
            history_text += f"{role}：{msg['content']}\n"

        prompt1 = f"""
        请压缩以下对话内容，保留关键信息，语言简洁，要求每一段对话都进行处理，不削减或增加对话
        仅输出压缩后的内容，无需任何前后缀，保留输入格式不变：
        {history_text}
        压缩后：
        """.strip()
        prompt2 = f"""
        请压缩以下文档，保留关键信息，语言简洁：
        {docs}
        压缩后：
        """.strip()

        result_hst=await self.generate(prompt1,self.paras["compress"])
        result_rag=await self.generate(prompt2,self.paras["compress"])
        result=result_hst+"\n"+result_rag
        return result if result else ""


    async def generate(self,prompt,paras) -> str:
        messages=[{"role":"user","content":prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        input_length = inputs.shape[1]
        generation_kwargs = dict(
            input_ids=inputs,
            **paras,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        result=await asyncio.create_task(
            asyncio.to_thread(self.base_model.generate, **generation_kwargs)
        )
        response = self.tokenizer.decode(result[0][input_length:], skip_special_tokens=True)
        print(f"我是response:{response}")
        return response

