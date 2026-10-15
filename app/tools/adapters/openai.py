"""
tools/adapters/openai.py 注册表 → 模型可见的 OpenAI function 载荷

唯一事实来源是 app/tools/registry.py；本文件只做形状转换，不新增任何工具描述。
config.TOOLS_SCHEMAS 由 to_openai_schemas() 生成（取代原来的 app/config/tools_schemas.yaml）：
ModelManager 把它原样放进 OpenAI 请求的 tools=，同时它也是工具调用白名单来源。

需要人类可读快照（原来那份 yaml 的等价物）时随时可以重新生成：

    python -m app.tools.adapters.openai
"""
import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from app.tools.registry import TOOL_REGISTRY, ToolSpec


def to_openai_schemas(specs: Optional[Sequence[ToolSpec]] = None) -> List[Dict[str, Any]]:
    """把注册表转换成 OpenAI function dict 列表，顺序与注册表一致。"""
    specs = TOOL_REGISTRY if specs is None else specs
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                # 深拷贝：调用方拿到的是副本，改它不会污染注册表里的唯一来源
                "parameters": copy.deepcopy(spec.parameters),
            },
        }
        for spec in specs
    ]


if __name__ == "__main__":
    print(json.dumps(to_openai_schemas(), ensure_ascii=False, indent=2))
