# """
#架构作用：FastAPI组合根，装配网页、聊天/设置API与MCP工具出口
#调用方：ASGI服务或直接启动；请求由路由进入服务层，启动阶段先准备数据库再启动MCP
#数据流：配置→应用装配→生命周期→请求处理；模块导入时即创建app
# """
"""
core/main.py测试入口
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from app.config.config import config
from app.mcp_server.server import mcp_lifespan, mount_mcp
from fastapi import FastAPI
from app.api import chat_router, settings_router


# """
#生命周期入口：服务接收请求前同步创建/迁移表，再进入MCP会话管理器
# yield把控制权交还ASGI；退出上下文时MCP资源得到回收
# """
@asynccontextmanager
async def lifespan(_app):
    # Run schema creation/migration before the application accepts requests.
    from repository.init_db import init as init_database
    init_database()
    # MCP的HTTP子应用有自己的会话管理器，必须在其父app的lifespan里进出，
    #否则/mcp能匹配到路由但工具调用会失败
    async with mcp_lifespan(_app):
        yield


# """
#应用工厂：配置中间件、静态资源和两类协议入口，返回完整ASGI应用
# """
def app_create():
    app = FastAPI(
        title=config.APP_NAME,
        version=config.APP_VERSION,
        description="",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    #跨域来源从环境变量ALLOWED_ORIGINS读取，逗号分隔；默认仅放行本机常见端口
    origins_env = os.environ.get("ALLOWED_ORIGINS", "")
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()] or [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # """
    #首页读取固定静态文件并返回HTML；路径由项目位置构造，不取请求输入
    # """
    @app.get("/")
    async def root():
        html = (static_dir / "boot.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    app.include_router(router=chat_router, prefix='/chat')
    #模型设置（P5a）：与chat同一prefix，保持/chat/settings/llm的路径形状
    app.include_router(router=settings_router, prefix='/chat')
    #把工具以MCP协议对外暴露（P6b）：只给5个无状态工具，见app/mcp_server/server.py
    #必须在lifespan进入前挂好，子应用的会话管理器由上面的mcp_lifespan启动
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
