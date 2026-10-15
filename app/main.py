"""
core/main.py 测试入口
"""
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from app.config.config import config
from app.mcp_server.server import mcp_lifespan, mount_mcp
from fastapi import FastAPI
from app.api import chat_router, settings_router


@asynccontextmanager
async def lifespan(_app):
    # Run schema creation/migration before the application accepts requests.
    from repository.init_db import init as init_database
    init_database()
    # MCP 的 HTTP 子应用有自己的会话管理器，必须在其父 app 的 lifespan 里进出，
    # 否则 /mcp 能匹配到路由但工具调用会失败。
    async with mcp_lifespan(_app):
        yield


def app_create():
    app = FastAPI(
        title=config.APP_NAME,
        version=config.APP_VERSION,
        description="",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        html = (static_dir / "boot.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    app.include_router(router=chat_router, prefix='/chat')
    # 模型设置（P5a）：与 chat 同一 prefix，保持 /chat/settings/llm 的路径形状
    app.include_router(router=settings_router, prefix='/chat')
    # 把工具以 MCP 协议对外暴露（P6b）：只给 5 个无状态工具，见 app/mcp_server/server.py。
    # 必须在 lifespan 进入前挂好，子应用的会话管理器由上面的 mcp_lifespan 启动。
    mount_mcp(app)
    return app


app = app_create()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )
