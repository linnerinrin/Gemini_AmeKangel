#"""
#架构作用：工具注册与协议适配的包边界；调用方按需导入具体模块
#本文件仅说明职责，工具定义经registry派生到各出口，无独立执行数据流
#"""
"""
tools包：工具的唯一注册表与各出口适配器

- registry.py工具唯一事实来源（name / description / parameters / impl / scope）
- adapters/openai.py    →模型可见的OpenAI function载荷（config.TOOLS_SCHEMAS）
- adapters/langgraph.py → LangChain工具对象（ToolNode执行体）
- app/mcp_server/server.py → MCP Server出口（只暴露SCOPE_EXTERNAL工具）
"""
