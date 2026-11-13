#"""
#架构作用：mcp_client子包的包边界；MCP客户端骨架模块的集合入口说明
#调用方：调用方按需导入具体模块（from .client import NotConfiguredMcpClient等）
#本文件仅说明职责与依赖方向，无独立执行数据流
#"""
"""
agent_framework/mcp_client子包：MCP客户端骨架（纯新增，未接入运行路径）

- client.py        MCP客户端实现：NotConfiguredMcpClient（占位默认实现，端口先行）
与FakeMcpClient（测试用内存实现）
- tool_bridge.py   MCP工具→原项目ToolSpec的桥接：build_mcp_tool_specs
                   （默认只桥接read_only=True的工具，对齐只读改造方向）

依赖方向：各模块只依赖contracts（McpClientPort / McpToolDescriptor / McpClientError）
以及标准库；不导出具体类到包级命名空间，调用方按需导入，避免包级import带来重依赖；
本包不反向依赖任何实现，除tool_bridge.py函数内延迟导入registry外不import原项目
"""
