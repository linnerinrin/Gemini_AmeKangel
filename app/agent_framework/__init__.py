#"""
#架构作用：agent框架的包边界；调用方按需导入子模块
#本文件仅说明职责与依赖方向，无独立执行数据流
#"""
"""
agent_framework包：框架层，调用方按需导入子模块

- contracts.py统一契约层（端口/数据类，冻结版，全包唯一共享依赖）
- security/出站校验/凭据策略/会话映射/审计骨架
- readonly/只读文件工具（read_file / list_dir / glob / grep）
- context_budget/token计数端口与轮/消息窗口策略
- model_adapter/模型调用适配（OpenAI兼容客户端，端口先行）
- mcp_client/MCP客户端骨架（只读工具桥接，端口先行）

回复简短化由既有reflection审查节点承担，框架不提供独立的确定性检查层。

依赖方向：各模块只依赖contracts.py，模块之间不得互相import；
对接原项目时由适配器依赖本包抽象，本包不反向依赖运行路径
接入状态（2026-11）：security / readonly / context_budget / mcp_client四个子包均已接入运行路径，
分别由app/api/chat/{chat_routes,settings_routes}.py、app/managers/{FuncCallManager,MemoryManager}.py、
app/tools/registry.py消费；仅model_adapter为半接入（见该子包__init__.py）
"""
