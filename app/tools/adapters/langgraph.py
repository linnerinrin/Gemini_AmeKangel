"""
tools/adapters/langgraph.py 注册表 → LangChain 工具（ToolNode 执行体）

唯一事实来源是 app/tools/registry.py；本文件只负责把 ToolSpec 包装成 LangChain 工具：
- 名称与描述取自注册表（注入 __name__ / __doc__，所以实现函数不需要写 docstring）；
- 参数 schema 由实现的真实签名推导，签名必须与注册表 parameters 保持一致。

这里不使用 bind_tools：模型可见的工具载荷始终来自 adapters/openai.py
（tests/unit/test_model_manager.py 明确禁止 ModelManager 出现 bind_tools）。
"""
from typing import Any, List

from app.tools.registry import TOOL_REGISTRY, ToolSpec


def build_tool(spec: ToolSpec, mgr: Any):
    """把单个 ToolSpec 绑定到管理器实例上，返回 LangChain 工具对象。"""
    # 函数内导入：测试会替换 sys.modules["langchain_core.tools"]，模块级导入会把真实实现固化下来
    from langchain_core.tools import tool

    fn = spec.impl(mgr)
    fn.__name__ = spec.name
    fn.__doc__ = spec.description
    return tool(fn)


def build_tools(mgr: Any) -> List[Any]:
    """按注册表顺序构建全部工具（顺序是对外契约，测试按位置解包）。"""
    return [build_tool(spec, mgr) for spec in TOOL_REGISTRY]
