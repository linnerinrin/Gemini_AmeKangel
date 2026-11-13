# """
#架构作用：管理器组合入口，供services、聊天路由和MCP延迟访问器导入
#数据流：导入具体类型→立即构造进程级实例→ services将实例绑定为ChatSession
# Python模块缓存使常规导入共享这些实例；这不是按请求或存档创建的容器
# """
from app.managers.GraphManager import GraphManager
from app.managers.RAGManager import RAGManager
from app.managers.ModelManager import ModelManager
from app.managers.AccountManager import AccountManager
from app.managers.MemoryManager import MemoryManager
from app.managers.FuncCallManager import FunctionCallManager

# """
#组装阶段：RAG最先创建并加载资源；后续会话构造才绑定工具依赖与图
# ModelManager进程级实例只在本模块创建，ModelManager.py内不再有第二实例
# """
rag_manager=RAGManager()
model_manager=ModelManager()
account_manager=AccountManager()
memory_manager=MemoryManager()
functioncall_manager=FunctionCallManager()
graph_manager=GraphManager()
