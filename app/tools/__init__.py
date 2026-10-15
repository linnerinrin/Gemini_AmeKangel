"""
tools 包：工具的唯一注册表与各出口适配器

- registry.py          工具唯一事实来源（name / description / parameters / impl / scope）
- adapters/openai.py    → 模型可见的 OpenAI function 载荷（config.TOOLS_SCHEMAS）
- adapters/langgraph.py → LangChain 工具对象（ToolNode 执行体）
- adapters/mcp.py       → （P6b 后续）MCP Server 出口
"""
