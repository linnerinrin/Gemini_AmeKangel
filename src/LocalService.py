import asyncio
from pathlib import Path
import yaml

from src.ModelManager import ModelManager
from src.RAGManager import RAGManager
from src.PreManager import PreManager
from src.gemini_chat import ChatSession

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "config" / "chat_config.yaml","r",encoding="utf-8") as f:
    chat_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "pre_config.yaml","r",encoding="utf-8") as f:
    pre_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "model_config.yaml","r",encoding="utf-8") as f:
    model_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "rag_config.yaml","r",encoding="utf-8") as f:
    rag_cfg=yaml.safe_load(f)

if __name__ == '__main__':
    gemini=ChatSession(
        ModelManager=ModelManager(model_cfg),
        RAGManager=RAGManager(rag_cfg),
        PreManager=PreManager(pre_cfg),
        config=chat_cfg
    )
    asyncio.run(gemini.continue_chat())