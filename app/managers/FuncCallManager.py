#"""
#架构作用：工具绑定层，ChatSession注入RAG/记忆/模型实例后交给GraphManager使用
#数据流：统一注册表→绑定本管理器的工具函数→ ToolNode；具体工具的副作用在注册表中
#当前对象为进程级共享实例，轮次标志不是按session/request隔离的存储
#"""
"""
managers/FuncCallManager.py
工具仓库管理器，为主agent提供tools
abstract on IFuncCall
base on RAGManager/MemoryManager/ModelManager

工具的名称/描述/参数schema/实现统一存放在app/tools/registry.py（唯一事实来源）；
本文件只负责把注册表里的工具绑定到当前管理器实例上，不自维护一份工具定义
"""
from __future__ import annotations

import os
from pathlib import Path

from app.agent_framework.readonly.fs_read_tools import ReadOnlyFileTools
from app.agent_framework.readonly.path_guard import PathGuard
from app.interfaces.IFuncCall import IFuncCall
from app.tools.adapters.langgraph import build_tools

#只读工具允许读取的根目录：默认放宽到用户目录（桌面/文档等可直接读），
#需要更多根目录时用READONLY_ROOTS（os.pathsep分隔）显式追加；
#凭据类文件（.env/.ssh/*.pem/Cookies等）仍由PathGuard黑名单拒绝
_DEFAULT_READONLY_ROOT = Path.home()


def _readonly_roots() -> list[str]:
    """读取只读工具的允许根目录；环境变量非法时回退到用户目录。"""
    raw = os.environ.get("READONLY_ROOTS", "").strip()
    if not raw:
        return [str(_DEFAULT_READONLY_ROOT)]
    parts = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    return parts or [str(_DEFAULT_READONLY_ROOT)]


class FunctionCallManager(IFuncCall):
    #"""
    #构造时尚未绑定依赖；空工具表避免在服务组装完成前创建带错误引用的闭包
    #readonly_tools：挂到本管理器属性上，注册表末尾四个只读工具的impl闭包经
    #getattr(mgr, "readonly_tools")取用；PathGuard无会话态，每实例一份即可
    #mcp_client：注册表末尾MCP桥接工具的impl闭包经getattr(mgr, "mcp_client")取用；
    #StdioMcpClient每次操作独立起子进程、无会话态，全实例共享注册表的进程级单例
    #（未配置时为显式失败的NotConfigured占位实现）
    #"""
    def __init__(self):
        super().__init__()
        self.tools = []
        self._status_changed_this_turn = False
        self.readonly_tools = ReadOnlyFileTools(PathGuard(_readonly_roots()))
        from app.tools.registry import MCP_CLIENT  #函数内导入：与模块顶部导入链对齐
        self.mcp_client = MCP_CLIENT

    #"""
    #会话每轮开始时重置状态变更标志，供状态工具限制同轮重复变更使用
    #"""
    def begin_turn(self):
        self._status_changed_this_turn = False


    #"""
    #组装期保存具体管理器引用，再创建工具；工具执行时通过self读取当前依赖
    #这些依赖包含同步数据库操作；此处的绑定本身既不提交事务，也不执行工具
    #"""
    def init(self,rag_mgr,memory_mgr,model_mgr):
        self.rag_mgr=rag_mgr
        self.memory_mgr=memory_mgr
        self.model_mgr=model_mgr
        self.tools=self._create_tools()

    #"""
    #仅更新模式字段；不重建工具，也不重置记忆、RAG或轮次标志
    #"""
    def load_mode(self,mode:str):
        self.current_mode=mode

    #"""
    #适配器将注册表实现绑定到当前owner；参数schema的协议一致性应在适配器边界验证
    #"""
    def _create_tools(self):
        """按注册表顺序构建工具；顺序是对外契约（test_func_call_manager按位置解包）。"""
        return build_tools(self)
