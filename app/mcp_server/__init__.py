# """
#架构作用：MCP出口的包边界，用PEP 562的模块级__getattr__惰性转发公开入口
#数据流：调用方按名字取属性→首次导入app.mcp_server.server →复用进程级管理器
#设计约束：不在导入父包时提前拉入server子模块，避免`python -m`触发runpy警告
# """
"""
app/mcp_server把app/tools/registry.py里的工具暴露为MCP Server（P6b）

公开入口集中在app/mcp_server/server.py；本包用PEP 562的模块级__getattr__做惰性转发，
这样`from app.mcp_server import build_server`可用，而`python -m app.mcp_server.server`又不会在
import父包时就提前把app.mcp_server.server拉进sys.modules（runpy会为此发RuntimeWarning）
"""
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  #仅供IDE/类型检查，运行时不导入
    from app.mcp_server.server import (  # noqa: F401
        INSTRUCTIONS,
        MCP_HTTP_PATH,
        MCP_SERVER_NAME,
        MCP_SERVER_VERSION,
        build_server,
        get_server,
        main,
        mcp_lifespan,
        mount_mcp,
    )

__all__ = [
    "INSTRUCTIONS",
    "MCP_HTTP_PATH",
    "MCP_SERVER_NAME",
    "MCP_SERVER_VERSION",
    "build_server",
    "get_server",
    "main",
    "mcp_lifespan",
    "mount_mcp",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from app.mcp_server import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(__all__)
