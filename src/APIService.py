"""
普通APIService        --创建fastapi用于前后端交互与逻辑实现
open        --加载yaml文件

get_gemini      --设置单例函数 保证模型全局只加载一次 防止oom

lifespan        --定义生命周期函数

app.        --api初始化
    add_middleware      --跨域
    mount       --挂载

app.post/get        --定义api接口
    get /       --获取html
    PostData        --定义请求体
    post /api/post      --返回流式响应
    post /api/history       --获取消息历史
    post /api/change        --带sse的opening/closing/load_mode实现
"""

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


#======
#加载yaml
#======

BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / "config" / "chat_config.yaml","r",encoding="utf-8") as f:
    chat_cfg=yaml.safe_load(f)
with open(BASE_DIR / "config" / "pre_config.yaml","r",encoding="utf-8") as f:
    pre_cfg=yaml.safe_load(f)
with open(BASE_DIR / "config" / "model_config.yaml","r",encoding="utf-8") as f:
    model_cfg=yaml.safe_load(f)
with open(BASE_DIR / "config" / "rag_config.yaml","r",encoding="utf-8") as f:
    rag_cfg=yaml.safe_load(f)


#======
#设置单例函数 保证模型全局只加载一次 防止oom
#======

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


#======
#异步上下文装饰器 管理异步资源的获取和释放 读完自动释放
#======

@asynccontextmanager
async def lifespan(app: FastAPI):
    #建立连接时自动加载模型 防止堵塞进程
    task = asyncio.create_task(
        asyncio.to_thread(get_gemini)
    )
    yield


#======
#api初始化
#======

#定义标题/生命周期
app = FastAPI(title="Gemini Ame/Kangel", lifespan=lifespan)

#跨域 为了让前端能访问api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#挂载 为了让api能访问文件 api由前端http请求访问服务器 因此要访问本地文件一定是此文件的相对路径 比如x.html要src同目录的js和css也要经过这个文件夹
static_dir = Path(__file__).parent.parent / "static"
if not static_dir.exists():
    static_dir.mkdir()
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


#======
#制定api接口
#======

#获取html
@app.get("/")
def _():
    with open(static_dir/"x.html", "r", encoding="utf-8") as f:
        #用htmlresponse读取x.html字符串 收到后渲染成网页
        return HTMLResponse(content=f.read())

#定义请求体
class PostData(BaseModel):
    content: str

#返回流式响应 StreamingResponse 为流式响应 第一个参数为带yield的生成器或者异步生成器 第二个参数为sse信息
@app.post("/api/post")
async def _(data: PostData):
    gemini_instance = get_gemini()

    return StreamingResponse(
        #async_generator
        gemini_instance.async_chat(data.content),
        #sse
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

#获取gemini.history
@app.post("/api/history")
async def get_history():
    """获取聊天历史记录"""
    gemini_instance = get_gemini()
    history = list(gemini_instance.history)
    return {"history": history}

#带sse的load_mode实现 依旧返回streamingresponse 先调用closing生成器输出结语 再用load_mode方法 再调用opening生成器输出欢迎语
@app.post("/api/change")
async def change_mode_with_animation(data: PostData):
    gemini_instance = get_gemini()

    async def generate():
        old_mode = gemini_instance.current_mode
        async for chunk in gemini_instance.closing():
            yield chunk

        gemini_instance._load_mode(data.content)

        async for chunk in gemini_instance.opening():
            yield chunk

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