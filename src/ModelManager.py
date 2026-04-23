from pathlib import Path
import asyncio
from typing import AsyncGenerator
from interfaces.IModel import IModel
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel, get_chat_template

BASE_DIR = Path(__file__).resolve().parent.parent

class ModelManager(IModel):
    def __init__(self,config:dict):
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(BASE_DIR / config['base_model_paths']),
            max_seq_length=config['max_seq_length'],
            load_in_4bit=True,
        )

        self.base_model.load_adapter(BASE_DIR / config['mode_paths']['ame'], adapter_name="ame")
        self.base_model.load_adapter(BASE_DIR / config['mode_paths']['kangel'], adapter_name="kangel")
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=config["chat_template"])


        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate(self,context,query,paras) -> AsyncGenerator[str, None]:
        streamer=TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        messages=context+[{"role":"user","content":query}]
        inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

        generation_kwargs=dict(
                input_ids=inputs,
                streamer=streamer,
                **paras,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        await asyncio.create_task(
                asyncio.to_thread(self.base_model.generate, **generation_kwargs)
            )

        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0.1)