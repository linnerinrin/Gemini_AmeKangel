# MCP Server（P6b）

把本项目**变成 MCP Server**，让其他模型/客户端（Claude Desktop、Cursor、其他 agent）接入本项目的工具。
反向（把外部 MCP 工具接进本图）不在范围内。

- 实现：`app/mcp_server/server.py`（对外入口）、`app/mcp_server/__init__.py`（惰性转发）
- 测试：`tests/unit/test_mcp_server.py`
- 依赖：`mcp>=2.3.0`（**注意是 2.x**：1.x 的 `FastMCP` / `mcp.server.fastmcp` 在 2.x 已改名，
  `mcp.server.fastmcp` 现在是「导入即报错」的迁移提示模块）

---

## 1. 暴露了哪 5 个工具

唯一事实来源是 `app/tools/registry.py` 的 `TOOL_REGISTRY`（7 个 `ToolSpec`）。
MCP 侧只取 `scope == SCOPE_EXTERNAL` 的 5 个（方案 R5）：

| 工具 | 参数 | 说明 |
|---|---|---|
| `output_time` | 无 | 返回当前日期时间（不依赖任何管理器） |
| `retrieve_knowledge` | `query` | 角色知识库检索 |
| `retrieve_context` | `query` | 近期对话上下文检索 |
| `rewrite` | `query` | 结合对话历史改写问题 |
| `search_web` | `query` | 互联网实时搜索 |

### 为什么 `save_memory` / `change_status` 不外放

这两个工具依赖「**当前存档**」这类**进程内状态**，而外部 MCP 调用**没有会话上下文**：

- `save_memory` 写的是 `mgr.memory_mgr`（当前存档的长期记忆）；
- `change_status` 改的是当前存档的好感/压力/阴暗，且用 `_status_changed_this_turn`
  做「本轮只结算一次」的回合闸门。

外部客户端没有 `session_id`，放出去只会写进某个不确定的存档，或者让回合闸门语义错乱。
方案里给过三种取舍（显式传 `session_id` / 只暴露无状态工具 / 独立外部会话），
本轮按 **R5 取「只暴露无状态工具」**，风险最低；将来若要外放状态类工具，走 (a) 显式传参升级。

名字、描述、参数 schema **一律取自注册表**，MCP 侧不新增第二份定义（见 §4）。

---

## 2. 用法

### 2.1 stdio（本地客户端直连）

```bash
python -m app.mcp_server.server
# 显式指定：python -m app.mcp_server.server --transport stdio
```

客户端配置示例（Claude Desktop / Cursor 的 `mcpServers`）：

```json
{
  "mcpServers": {
    "ame-kangel-tools": {
      "command": "C:\\...\\Gemini_AmeKangel\\venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server.server"],
      "cwd": "C:\\...\\Gemini_AmeKangel"
    }
  }
}
```

> `cwd` 建议填项目根目录：`python -m app.mcp_server.server` 依赖项目根在 `sys.path` 上。
> stdio 模式下**不要**往 stdout 打印任何东西（stdout 是协议通道），日志都走 stderr。

### 2.2 HTTP（供局域网 / 其他模型接入）

两种方式，二选一：

**(a) 挂到现有 FastAPI app 上（推荐，供 main.py 接线）**

```python
from app.mcp_server import mount_mcp, mcp_lifespan

app = FastAPI()

mount_mcp(app)                              # 在 /mcp 挂上 StreamableHTTP 入口
app.router.lifespan_context = mcp_lifespan  # 必须：子应用不会自己跑 lifespan
```

已有自己的 lifespan 时嵌套使用：

```python
@asynccontextmanager
async def lifespan(app):
    async with mcp_lifespan(app):
        yield
```

**为什么必须接 `mcp_lifespan`**：`streamable_http_app()` 返回的是一个 **Starlette 子应用**，
Starlette 的挂载**不会**执行子应用的 lifespan，`session_manager` 不启动请求就会失败。
`mcp_lifespan()` 就是把它挂到**父 app** 的 lifespan 上（内部 `async with server.session_manager.run()`）。

对外地址即 `http://<host>:<port>/mcp`，例如 `http://127.0.0.1:8000/mcp`。

> 挂载用的是 `Route(path, endpoint=子应用)` 而不是 `app.mount(path, 子应用)`：
> Starlette 的 `Mount` 正则要求前缀后有斜杠，`app.mount("/mcp", 子应用)` 会让 `POST /mcp`
> 先吃一个 **307 跳到 `/mcp/`**，而 MCP 客户端默认不跟随重定向。`Route` 把子应用当 ASGI app
> 直接调用，路径原样传入，因此对外就是干净的 `/mcp`。

**(b) 独立起一个 HTTP 服务**

```bash
python -m app.mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8931
```

### 2.3 Host 头校验（默认关闭）

SDK 在 host 为 localhost 时会默认只放行 `127.0.0.1` / `localhost` / `[::1]` 的 `Host` 头；
本服务是给本机或局域网里其他模型接入的（`Host` 是 `192.168.x.x:port` 或反代域名），
那份默认白名单会直接回 **421 Invalid Host header**，所以 `mount_mcp` / `main` 默认关闭该校验。
需要收紧时：

```python
mount_mcp(app, allowed_hosts=["192.168.1.5:*", "my-proxy.example.com:*"])
```

---

## 3. 管理器从哪来

5 个工具里 `retrieve_knowledge` / `retrieve_context` / `rewrite` 需要进程内管理器（RAG、模型），
`output_time` / `search_web` 不需要。`build_server()` 默认传入一个 `LazyManager` 代理：

- 只有工具**真正访问 `mgr.xxx`** 时才 `from app.managers import ...`；
- HTTP 挂载场景复用的是 Web 进程里已被 `ChatSession.init()` 装配过的 `functioncall_manager`；
- 独立 stdio 进程里没有 `ChatSession`，就地把 `functioncall_manager.init(rag_manager, memory_manager, model_manager)`
  补一次。

因此「只连 `output_time`」的客户端不会顺手把 torch / 向量模型 / 数据库拉起来。
测试通过 `build_server(manager=替身)` 注入假管理器，**完全不联网、不需要 API key**。

---

## 4. schema 保真：为什么不是「让 SDK 从函数签名推导」

`MCPServer.add_tool()` 会从函数签名推导 input schema。若放任推导，注册表里的显式 JSON Schema
与推导结果就成了**两份定义**，正是 P6a 刚消灭的漂移陷阱（改了注册表、对外形状却不变，且无人提示）。

**探针结论（mcp 2.3.0 实测）：SDK 允许把注册表 schema 直接喂进去**，所以不需要退回签名推导。
落地为 `app/mcp_server/server.py::_build_tool()` 三步：

1. `Tool.from_function(invoke, name=..., description=..., structured_output=False)`
   造出 MCP 内部的注册模型（`mcp.server.mcpserver.tools.Tool`）；
2. **`tool.parameters = copy.deepcopy(spec.parameters)`** 覆盖掉签名推导结果 ——
   `MCPServer.list_tools()` 会原样把它当作 `inputSchema` 发出去
   （SDK 源码：`MCPTool(name=..., input_schema=info.parameters, ...)`）；
   用副本是为了不让调用方改到注册表里的唯一来源；
3. `invoke` 故意写成 `**kwargs`（SDK 看不到真实形参），改用
   `signature_from_schema(spec.parameters)` 按注册表的 `properties` / `required` 生成 `__signature__`。
   于是「对客户端公布的 `inputSchema`」与「mcp 侧参数校验用的 `arg_model`」也来自同一份注册表，
   不会出现「公布了 A、实际按 B 校验」的分裂。

`structured_output=False`：注册表没有输出 schema，就不允许 SDK 为 `str` 返回值自动造一个 `outputSchema`。

**等价性由测试钉死**（不靠人工同步）：

- `test_mcp_schemas_are_the_registry_schemas`：`tool.input_schema == spec.parameters`（deep-equal）；
- `test_schema_survives_the_wire_over_an_official_client`：走官方客户端 SDK 的真实协议后再比一次；
- `test_http_initialize_and_list_tools`：HTTP 上的 `inputSchema` 再比一次。

---

## 5. 自检

```bash
python -m pytest tests/unit/test_mcp_server.py -q
```

覆盖：对外集合恰好 5 个（且 `save_memory` / `change_status` 不可调用）、schema deep-equal、
五个工具真能执行（检索/改写/联网均为替身，不联网）、`mount_mcp(app)` 后 app 上有 `/mcp` 路由、
`mcp_lifespan()` 能正常进出、HTTP 上 initialize + tools/list + tools/call 全通。
