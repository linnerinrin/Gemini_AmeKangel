#"""
#架构作用：将注册表绑定到FuncCallManager，生成图中ToolNode可调用的工具对象
#数据流：ToolSpec +管理器→带闭包的实现函数→ LangChain工具；本层不重复定义描述
#"""
"""
tools/adapters/langgraph.py注册表→ LangChain工具（ToolNode执行体）

唯一事实来源是app/tools/registry.py；本文件只负责把ToolSpec包装成LangChain工具：
-名称与描述取自注册表（注入__name__ / __doc__，所以实现函数不需要写docstring）；
-参数schema由实现的真实签名推导，签名必须与注册表parameters保持一致

这里不使用bind_tools：模型可见的工具载荷始终来自adapters/openai.py
（tests/unit/test_model_manager.py明确禁止ModelManager出现bind_tools）
"""
from typing import Any, List

from app.tools.registry import TOOL_REGISTRY, ToolSpec


#"""
#单个工具绑定：延迟导入框架，给新建闭包补名称和描述，再由真实签名推导参数
#注册表schema与函数签名必须同步；该适配器本身不执行副作用
#"""
def build_tool(spec: ToolSpec, mgr: Any):
    """把单个ToolSpec绑定到管理器实例上，返回LangChain工具对象。"""
    #函数内导入：测试会替换sys.modules["langchain_core.tools"]，模块级导入会把真实实现固化下来
    from langchain_core.tools import tool

    fn = spec.impl(mgr)
    fn.__name__ = spec.name
    fn.__doc__ = spec.description
    return tool(fn)


#"""
#批量入口供工具管理器使用；保留注册表顺序，将同一个管理器绑定到全部闭包
#"""
def build_tools(mgr: Any) -> List[Any]:
    """按注册表顺序构建全部工具（顺序是对外契约，测试按位置解包）。"""
    return [build_tool(spec, mgr) for spec in TOOL_REGISTRY]
