import asyncio
import unsloth
import yaml
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "config" / "chat_config.yaml","r",encoding="utf-8") as f:
    chat_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "pre_config.yaml","r",encoding="utf-8") as f:
    pre_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "model_config.yaml","r",encoding="utf-8") as f:
    model_cfg=yaml.safe_load(f)

with open(BASE_DIR / "config" / "rag_config.yaml","r",encoding="utf-8") as f:
    rag_cfg=yaml.safe_load(f)

# 全局变量，初始为 None
_gemini = None
_pre_manager = None
_model_manager = None
_rag_manager = None


def get_gemini():
    global _gemini, _pre_manager, _model_manager, _rag_manager

    if _gemini is None:
        from src.PreManager import PreManager
        _pre_manager = PreManager(pre_cfg)

        from src.ModelManager import ModelManager
        _model_manager = ModelManager(model_cfg)

        from src.RAGManager import RAGManager
        _rag_manager = RAGManager(rag_cfg)

        from src.gemini_chat import ChatSession
        _gemini = ChatSession(
            config=chat_cfg,
            PreManager=_pre_manager,
            ModelManager=_model_manager,
            RAGManager=_rag_manager
        )

    return _gemini


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(
        asyncio.to_thread(get_gemini)
    )
    yield

app = FastAPI(title="我的AI工具服务", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
if not static_dir.exists():
    static_dir.mkdir()

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def _():
    with open("../static/x.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class PostData(BaseModel):
    content: str

@app.post("/api/post-demo")
async def post_demo(data: PostData):
    gemini_instance = get_gemini()

    return StreamingResponse(
        gemini_instance.async_chat(data.content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/history")
async def get_history():
    """获取聊天历史记录"""
    gemini_instance = get_gemini()
    history = list(gemini_instance.history)  # 转换为列表以便序列化
    return {"history": history}


@app.post("/api/change-with-animation")
async def change_mode_with_animation(data: PostData):
    """带流式动画的模式切换"""
    gemini_instance = get_gemini()

    async def generate():
        old_mode = gemini_instance.current_mode
        # 先输出旧模式的 closing 动画（带主题标识）
        async for chunk in gemini_instance.closing():
            yield chunk
        # 执行模式切换
        gemini_instance._load_mode(data.content)
        # 输出新模式的 opening 动画（带主题标识）
        async for chunk in gemini_instance.opening():
            yield chunk
        # 最后输出切换完成标识
        yield f"data: \n\n"
        yield f"data: ✅ 已切换到 {data.content.upper()} 模式\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == '__main__':
    uvicorn.run(
        "APIService:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        workers=1
    )