# """
#架构作用：把只读工具包装为原项目ToolSpec形状的规格列表，是readonly包接入注册表的桥
#调用方：接入适配层（负责把ReadOnlyFileTools实例挂到mgr.readonly_tools并合并规格）
#数据流：build_readonly_tool_specs(workspace_root) → list[ToolSpec] →追加进TOOL_REGISTRY末尾
# """
"""
agent_framework/readonly/tool_specs.py只读工具规格（与app/tools/registry.py完全兼容）

约定（对齐registry.py的实现约定）：
- ToolSpec / SCOPE_INTERNAL函数内延迟导入：本文件在接入前不得触发对原项目的import，
保持readonly包「除本文件外不依赖原项目」的边界；
- impl为_bind_*(mgr)闭包工厂：工厂只捕获mgr，真正读取在被调用时；
- mgr语义（待确认）：通过getattr(mgr, "readonly_tools")取得ReadOnlyFileTools实例——
这是接入点约定，接入时由适配层把实例挂到FuncCallManager或包装owner上；
-实现函数返回str（ReadResult序列化为可读文本，含文件:行号与截断提示）；
-实现函数不写docstring：描述只放ToolSpec.description，由适配器写入__doc__

重要：本文件不注册进TOOL_REGISTRY，接入前需人工评估风险
"""


# """
#模块一：结果序列化
#把ReadResult拍平为模型可读文本；失败给原因，截断给显式提示
# """
def _format_result(result) -> str:
    """ReadResult → 可读文本：失败带原因，截断带提示（内部函数，不进规格）。"""
    if not result.ok:
        return f"操作失败：{result.reason}"
    lines = []
    if result.content:
        lines.append(result.content)
    else:
        lines.append("（无内容）")
    if result.truncated:
        lines.append(f"[已截断] 结果过长被截断；文件/结果总行数：{result.total_lines}")
    return "\n".join(lines)


# """
#模块二：工具实现闭包（impl工厂）
#每个_bind_*返回带真实签名的工具函数；mgr上挂readonly_tools是接入约定（待确认）
# """
def _bind_read_file(mgr):
    def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
        tools = getattr(mgr, "readonly_tools")
        result = tools.read_file(path, offset=offset, limit=limit or None)
        return f"文件 {path} 内容（从第 {offset} 行起）：\n" + _format_result(result)

    return read_file


def _bind_list_dir(mgr):
    def list_dir(path: str) -> str:
        tools = getattr(mgr, "readonly_tools")
        result = tools.list_dir(path)
        return f"目录 {path} 清单：\n" + _format_result(result)

    return list_dir


def _bind_glob(mgr):
    def glob(pattern: str) -> str:
        tools = getattr(mgr, "readonly_tools")
        result = tools.glob_files(pattern)
        return f"匹配 {pattern} 的文件：\n" + _format_result(result)

    return glob


def _bind_grep(mgr):
    def grep(pattern: str, path: str = ".") -> str:
        tools = getattr(mgr, "readonly_tools")
        result = tools.grep(pattern, path=path)
        return f"在 {path} 中搜索 {pattern} 的结果（文件:行号:内容）：\n" + _format_result(result)

    return grep


# """
#模块三：规格构建入口
#延迟导入ToolSpec / SCOPE_INTERNAL；返回的list顺序即追加顺序（read/list/glob/grep）
# """
def build_readonly_tool_specs(workspace_root: str) -> list:
    """构建四个只读工具的 ToolSpec 列表；workspace_root 仅作文档语义保留，实例由 mgr 注入。"""
    from app.tools.registry import SCOPE_INTERNAL, ToolSpec  #延迟导入：见模块docstring

    return [
        ToolSpec(
            name="read_file",
            description="读取工作区内文件内容（只读）。当需要查看代码、配置或文档内容时使用；offset 为起始行号（从 1 开始），limit 为读取行数（不填则读到大小上限）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "工作区内文件路径"},
                    "offset": {"type": "integer", "description": "起始行号（1-based），默认 1"},
                    "limit": {"type": "integer", "description": "读取行数，可选"},
                },
                "required": ["path"],
            },
            impl=_bind_read_file,
            scope=SCOPE_INTERNAL,
        ),
        ToolSpec(
            name="list_dir",
            description="列出工作区内目录的一层清单（只读），目录在前文件在后。当需要了解项目结构时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "工作区内目录路径"},
                },
                "required": ["path"],
            },
            impl=_bind_list_dir,
            scope=SCOPE_INTERNAL,
        ),
        ToolSpec(
            name="glob",
            description="按通配符在工作区内查找文件（只读），如 **/*.py。当需要按文件名模式定位文件时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "通配符模式，如 **/*.py"},
                },
                "required": ["pattern"],
            },
            impl=_bind_glob,
            scope=SCOPE_INTERNAL,
        ),
        ToolSpec(
            name="grep",
            description="在工作区内按正则搜索文本内容（只读），返回 文件:行号:内容。当需要定位符号、字符串出现位置时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式"},
                    "path": {"type": "string", "description": "搜索范围（文件或目录），默认整个工作区"},
                },
                "required": ["pattern"],
            },
            impl=_bind_grep,
            scope=SCOPE_INTERNAL,
        ),
    ]
