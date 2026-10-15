"""
app/mcp_server 把 app/tools/registry.py 里的工具暴露为 MCP Server（P6b）

公开入口集中在 app/mcp_server/server.py；本包用 PEP 562 的模块级 __getattr__ 做惰性转发，
这样 `from app.mcp_server import build_server` 可用，而 `python -m app.mcp_server.server` 又不会在
import 父包时就提前把 app.mcp_server.server 拉进 sys.modules（runpy 会为此发 RuntimeWarning）。
"""
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # 仅供 IDE/类型检查，运行时不导入
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
