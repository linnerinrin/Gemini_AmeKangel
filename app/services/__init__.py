"""

"""

"""
架构作用：会话运行时接线——SessionRegistry按session_id持有ChatSession实例，
路由仍通过chat门面访问，对外调用形状与全局单例时期完全一致
调用方：聊天API导入chat；数据流为请求→门面→按session_id分派的ChatSession→图/模型/记忆
原有模块说明提到auth，但此处没有实现认证或客户端归属校验
"""

"""
会话运行时为「注册表+门面」结构：
-每个session_id拥有独立的ChatSession（含独立Memory/RAG/FuncCall/Graph管理器，
 仅ModelManager进程级共享——它按请求现读设置、无会话态）
-默认会话保留全局管理器组合，供未绑定请求与MCP延迟访问器路径使用
"""


from app.services.session_registry import SessionRegistry
from app.managers import (
    rag_manager, model_manager, account_manager, memory_manager,
    functioncall_manager, graph_manager,
)
from app.managers.RAGManager import RAGManager
from app.managers.MemoryManager import MemoryManager
from app.managers.FuncCallManager import FunctionCallManager
from app.managers.GraphManager import GraphManager
from app.services.chat_session import ChatSession


"""
每会话实例工厂：记忆/工具/图/RAG全部新建（RAG编码器权重进程内共享，代价可承受），
ModelManager共享——它是无会话态的设置执行器
"""
def _build_session() -> ChatSession:
    return ChatSession(
        model_manager,
        FunctionCallManager(),
        RAGManager(),
        MemoryManager(),
        GraphManager(),
    )


"""
注册表工厂：惰性创建后立即按session_id绑定；存档不存在时bind抛ValueError，
实例不会进入注册表，由路由转成409业务响应
"""
def _session_factory(session_id: str) -> ChatSession:
    session = _build_session()
    session.bind_session_id(session_id)
    return session


"""
门面：对路由暴露与旧全局chat完全相同的接口（bind_session_id/history/
single_chat/stream_chat/memory_mgr/...），内部按当前指针分派到注册表会话；
未绑定时回落到默认会话（全局管理器组合，供MCP路径与未绑定轮询使用）
"""
class ChatFacade:
    def __init__(self, default_session: ChatSession, registry: SessionRegistry) -> None:
        self._default = default_session
        self._registry = registry
        self._current_id = None

    """
    当前会话解析：未绑定或注册表缺失（删档后）时回落默认会话
    """
    def _current(self) -> ChatSession:
        if self._current_id is None:
            return self._default
        session = self._registry.get(self._current_id)
        return session if session is not None else self._default

    """
    给路由一个“此刻已确定的会话实例”。路由在发起生成/创建任务之前
    先取快照，后续调用一律落在该实例上，避免执行期间指针被换绑导致串档。
    """
    def resolve_current(self) -> ChatSession:
        """返回当前指针指向的具体会话实例；未绑定或删档后回落默认会话。"""
        return self._current()

    """
    换绑=移动指针，不重建或重载会话；不存在时经工厂创建，失败抛ValueError
    """
    def bind_session_id(self, session_id):
        if session_id is None:
            self._current_id = None
            return
        session_id = str(session_id)
        self._registry.get_or_create(session_id)
        self._current_id = session_id

    """
    每会话生成锁：同会话并发409、跨会话并行的互斥原语由注册表提供
    """
    def lock_for(self, session_id):
        return self._registry.lock_for(str(session_id))

    """
    清空当前会话缓存（沿用ChatSession原语义），用于删档等场景
    """
    def clear_session(self):
        self._current().clear_session()

    """
    删档回收：无论是否为当前会话都清缓存并逐出注册表，释放每会话RAG索引
    """
    def discard_session(self, session_id):
        session_id = str(session_id)
        session = self._registry.get(session_id)
        if session is not None:
            session.clear_session()
            self._registry.evict(session_id)
        if self._current_id == session_id:
            self._current_id = None

    """
    其余属性（history/memory_mgr/single_chat/stream_chat/load_mode/...）
    一律委托当前会话；需要跨await的调用方必须先调resolve_current()取快照，
    不能依赖__getattr__在调用时才解析（那会读到换绑后的新指针）
    """
    def __getattr__(self, name):
        return getattr(self._current(), name)


"""
实例构造会建立管理器间引用并编译图；默认会话沿用全局管理器组合（MCP依赖），
注册表会话按需惰性创建
"""
_default_session = ChatSession(model_manager, functioncall_manager, rag_manager, memory_manager, graph_manager)
session_registry = SessionRegistry(session_factory=_session_factory)
chat = ChatFacade(_default_session, session_registry)

