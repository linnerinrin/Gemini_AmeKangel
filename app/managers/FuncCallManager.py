"""
managers/FuncCallManager.py
工具仓库管理器，为主agent提供tools
abstract on IFuncCall
base on RAGManager/MemoryManager/ModelManager

工具的名称/描述/参数 schema/实现统一存放在 app/tools/registry.py（唯一事实来源）；
本文件只负责把注册表里的工具绑定到当前管理器实例上，不再自己维护一份工具定义。
"""
from app.interfaces.IFuncCall import IFuncCall
from app.tools.adapters.langgraph import build_tools


class FunctionCallManager(IFuncCall):
    def __init__(self):
        super().__init__()
        self.tools = []
        self._status_changed_this_turn = False

    def begin_turn(self):
        self._status_changed_this_turn = False


    def init(self,rag_mgr,memory_mgr,model_mgr):
        self.rag_mgr=rag_mgr
        self.memory_mgr=memory_mgr
        self.model_mgr=model_mgr
        self.tools=self._create_tools()

    def load_mode(self,mode:str):
        self.current_mode=mode

    def _create_tools(self):
        """按注册表顺序构建工具；顺序是对外契约（test_func_call_manager 按位置解包）。"""
        return build_tools(self)
