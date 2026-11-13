# """
#架构作用：将唯一工具注册表转换为模型API所需的function描述，供配置与模型管理器使用
#数据流：ToolSpec序列→名称/描述/参数副本→ OpenAI tools请求字段；本层不执行工具
# """
"""
tools/adapters/openai.py注册表→模型可见的OpenAI function载荷

唯一事实来源是app/tools/registry.py；本文件只做形状转换，不新增任何工具描述
config.TOOLS_SCHEMAS由to_openai_schemas()生成（取代原来的app/config/tools_schemas.yaml）：
ModelManager把它原样放进OpenAI请求的tools=，同时它也是工具调用白名单来源

需要人类可读快照（原来那份yaml的等价物）时随时可以重新生成：

    python -m app.tools.adapters.openai
"""
import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from app.tools.registry import TOOL_REGISTRY, ToolSpec


# """
#转换入口：None表示使用全表；显式空序列保持为空，避免误打开工具权限
#深拷贝隔离可变参数字典，名称与描述仍以注册表为准
# """
def to_openai_schemas(specs: Optional[Sequence[ToolSpec]] = None) -> List[Dict[str, Any]]:
    """把注册表转换成 OpenAI function dict 列表，顺序与注册表一致。"""
    specs = TOOL_REGISTRY if specs is None else specs
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                #深拷贝：调用方拿到的是副本，改它不会污染注册表里的唯一来源
                "parameters": copy.deepcopy(spec.parameters),
            },
        }
        for spec in specs
    ]


if __name__ == "__main__":
    print(json.dumps(to_openai_schemas(), ensure_ascii=False, indent=2))
