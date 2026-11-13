#"""
#架构作用：agent_framework只读工具子包的包边界；当前不接入运行路径，调用方按需导入子模块
#本文件仅说明职责与依赖方向，无独立执行数据流
#"""
"""
agent_framework/readonly子包：只读文件工具（read_file / list_dir / glob / grep）

- path_guard.py路径安全判定：PathGuardPort的默认实现，前缀校验+黑名单+逃逸防护
- fs_read_tools.py只读文件操作：ReadOnlyFileTools，四个同步只读方法，统一返回ReadResult
- tool_specs.py工具规格适配：把四个只读工具包装为app/tools/registry.py的ToolSpec形状

依赖方向：本包只依赖contracts.py的端口与数据类，不反向依赖原项目任何模块
（tool_specs.py对app.tools.registry的导入在函数内延迟进行，接入时才生效）
当前状态：纯新增文件，未被app/现有任何模块import，不改变现有行为
本文件不导出任何类：调用方显式import子模块，保持边界清晰
"""
