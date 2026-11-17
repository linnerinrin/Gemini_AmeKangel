# Gemini_AmeKangel · 双子角色扮演聊天系统

基于大语言模型的角色扮演聊天系统。糖糖（ame）与超天酱 / 堕天酱（kangel / dangel）可在**私聊（JINE）**与**直播**两种场景下切换，内置 LangGraph 生成管线、RAG 检索、思考链、状态栏（好感 / 压力 / 心理 / 粉丝）、表情协议、MCP 工具出口，以及一套 Windows 98 风格的桌面式纯前端。

---

## 1. 如何开始

### 环境要求

- Windows + Python 3.10+
- MySQL（启动时会自动建表与迁移）
- GPU 可选：向量检索用 faiss-cpu（CPU 版），但嵌入/重排模型由 `RAG_DEVICE` 决定落在哪个设备上。
  **默认值是 `cuda`，没有 N 卡时必须在 `.env` 里设 `RAG_DEVICE=cpu`，否则加载模型会直接报错。**

### 首次安装

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 启动

```bat
start.bat            :: 默认端口 5577
start.bat 5599       :: 端口被占用时指定其他端口
```

启动后浏览器打开 <http://127.0.0.1:5577/>，按 `Ctrl+C` 停止服务。

常见问题：端口被占用就换个端口启动；MySQL 没启动会导致建表失败；模型 API Key 没配请在页面里的**控制面板**填好并点「测试」。

---

## 2. 项目结构、核心文件与技术栈

### 目录总览

```
Gemini_AmeKangel/
├─ start.bat            # Windows 一键启动脚本（uvicorn app.main:app）
├─ requirements.txt     # Python 依赖
├─ contracts/           # 前后端契约（api_contract.json、role2idx.json 角色映射）
├─ prompts/             # 提示词：base / role(ame,kangel,dangel,ame_live) / task(chat) / 知识库
├─ docs/                # 设计文档：mcp-server.md、plan-chat-work-mode-mcp-llm.md（含架构审计）
├─ repository/          # 数据层：conn.py（线程本地连接，非连接池）、init_db.py（建表）
├─ static/              # 前端：boot.html / desktop.html / css / js / assets（角色逐帧素材）
├─ tests/               # pytest 单测与集成测试 + Node .cjs 前端契约测试
└─ app/                 # 后端主体（FastAPI）
   ├─ main.py           # 装配：静态目录、/chat 路由、/mcp 挂载、lifespan
   ├─ config/           # config.py（全部配置项）、status_rules.py（状态随机规则）、
   │                    #   llm_import.py（按文件加载避免测试遮蔽）、schema.py（幂等迁移）
   ├─ api/chat/         # chat_routes.py（聊天路由）、settings_routes.py（模型设置路由）、
   │                    #   *_schemas.py（请求边界）、generation_lock.py（生成互斥）
   ├─ services/         # chat_session.py（单会话运行时）、llm_settings.py（设置读写与校验）、
   │                    #   __init__.py（会话注册表+门面 ChatFacade）
   ├─ managers/         # GraphManager（LangGraph 管线）、MemoryManager（记忆与状态落库）、
   │                    #   ModelManager（模型调用）、RAGManager（向量检索+重排）、
   │                    #   FuncCallManager（工具执行）、AccountManager（3 个存档位）
   ├─ tools/            # registry.py（工具唯一来源：7 核心 + 4 只读 + MCP 外部工具）、adapters/
   ├─ mcp_server/       # server.py：把 registry 的 external 工具暴露为 MCP Server
   └─ agent_framework/  # 能力端口层：contracts.py（端口契约）+ 5 个子包
                        #   security 审计/出站校验/凭据策略/会话注册表（已接入 api/chat）
                        #   readonly 只读文件工具+PathGuard（已接入 tools/registry、FuncCallManager）
                        #   context_budget token 计数与会话窗口（已接入 MemoryManager）
                        #   mcp_client stdio MCP 客户端+工具桥（已接入 tools/registry）
                        #   model_adapter OpenAI 兼容客户端（半接入：仅非流式路径）
```

### 核心后端文件作用

| 文件 | 作用 |
|---|---|
| `app/main.py` | FastAPI 装配：挂 `/static`、`/chat` 两组路由、`/mcp` 的 StreamableHTTP 入口，驱动 MCP lifespan |
| `app/services/chat_session.py` | 单个存档的运行时：装配系统提示词、跑生成管线、写记忆、结算状态与涨粉 |
| `app/services/__init__.py` | 会话注册表 + `ChatFacade` 门面；`resolve_current()` 在 await 前锁定会话快照，防止换绑串档 |
| `app/services/llm_settings.py` | 模型设置的读写、掩码、校验；凭据-地址绑定（改 base_url 必须重填 key），出站地址校验 |
| `app/managers/GraphManager.py` | LangGraph 生成管线：四节点 agent → generate_reply → reflection，tools 节点按需介入；全程推送思考链事件 |
| `app/managers/MemoryManager.py` | 短期/长期记忆、状态栏（好感/压力/心理/粉丝）落库；直播模式每轮按概率涨粉 |
| `app/managers/ModelManager.py` | 模型统一入口：OpenAI 兼容协议（`AsyncOpenAI` SDK 为主，流式与非流式都走它；`agent_framework.model_adapter` 只兜底非流式路径）、超时/重试、参数透传（top_p/max_tokens 按 >0 条件携带） |
| `app/managers/RAGManager.py` | RAG：bi-encoder 向量召回（faiss）+ cross-encoder 重排，本地模型目录离线加载 |
| `app/managers/FuncCallManager.py` | 工具执行调度；只读文件工具的读取边界由 `READONLY_ROOTS` 决定 |
| `app/tools/registry.py` | 工具唯一事实来源：schema、执行闭包、scope；同一份定义喂给模型、LangGraph 与 MCP |
| `app/mcp_server/server.py` | 把 5 个无状态工具（output_time / retrieve_knowledge / retrieve_context / rewrite / search_web）暴露为 MCP Server，schema 直接取自注册表 |
| `app/agent_framework/model_adapter/` | OpenAI 兼容客户端（httpx，`ChatClientPort` 默认实现）。**半接入**：仅当「非流式 + 未注入替身」时由 ModelManager 走这里，主链路与全部流式调用走 openai SDK |
| `app/agent_framework/security/` | 审计事件、出站地址校验（SSRF 防护）、凭据策略、每会话锁注册表 |
| `repository/conn.py` | MySQL 访问门面：`threading.local` 线程本地单连接（首次执行 SQL 才建连），非连接池；无事务上下文，调用方显式 commit |
| `repository/init_db.py` | 启动建表；老库由 `app/config/schema.py` 幂等迁移 |

### 技术栈

- **后端**：FastAPI + Uvicorn、Pydantic v2 / pydantic-settings、PyMySQL（MySQL）
- **生成管线**：LangGraph + langchain-core
- **RAG**：sentence-transformers（bi-encoder + cross-encoder）、faiss-cpu、HuggingFace 本地模型
- **模型接入**：OpenAI 兼容协议，主链路用 `openai` SDK 的 `AsyncOpenAI`（支持流式），`httpx` 用于设置页探测端点与 `model_adapter` 的非流式兜底；支持 deepseek / openai / custom 等厂商
- **工具协议**：MCP（Server 出口 + stdio 客户端入口）
- **其他**：tiktoken（token 计数）、ddgs（联网搜索）、sentence-transformers 及其后端 torch / transformers（本地向量与重排推理，无需 GPU）
- **前端**：纯原生 HTML/CSS/JS，无构建步骤；SSE 流式渲染、逐帧角色动画
- **测试**：pytest（单元+集成）、Node .cjs（前端契约测试）

---

## 3. 可用 API 一览

HTTP 接口统一挂在 `/chat` 前缀下（FastAPI，端口见启动参数）。

### 聊天与存档

| 方法与路径 | 说明 | 请求体 |
|---|---|---|
| `POST /chat/single_chat` | 单轮生成，等待完整结果 | `{prompt, session_id?, mode?}` |
| `POST /chat/stream_chat` | 流式生成（SSE），实时推送思考链与分段回复 | `{prompt, session_id?, mode?}` |
| `GET /chat/history` | 读取历史；带 `mode` 参数按场景读取 | `?mode=&session_id=` |
| `POST /chat/mode_change` | 切换场景（JINE / 各直播档） | `{mode, session_id?}` |
| `POST /chat/get_status` | 读取状态栏（好感/压力/心理/粉丝） | `{session_id?}` |
| `GET /chat/sessions` | 存档列表（共 3 个槽位） | — |
| `POST /chat/new_session` | 新建存档并绑定 | `{session_num}` |
| `POST /chat/chose_session` | 选择已有存档 | `{session_num}` |
| `POST /chat/delete_session` | 删除存档 | `{session_num}` |
| `POST /chat/start_session` | 绑定指定存档为活动会话 | `{session_id}` |

### 模型设置（控制面板）

| 方法与路径 | 说明 |
|---|---|
| `GET /chat/settings/llm` | 读取设置视图（active 厂商、各厂商 base_url/model、掩码 key） |
| `POST /chat/settings/llm` | 保存设置并切换 active；改 base_url 必须重填 key；生成中禁止修改 |
| `POST /chat/settings/llm/test` | 连通性探测：key/base_url/model 可覆盖，留空沿用已保存值 |

### MCP 工具出口

- 挂载点：`/mcp`（StreamableHTTP）；stdio 直连：`python -m app.mcp_server.server`
- 暴露的 5 个无状态工具：`output_time`、`retrieve_knowledge`、`retrieve_context`、`rewrite`、`search_web`
- 其余：`GET /` 返回 `static/boot.html`（开机动画页，随后进入 `static/desktop.html` 主界面）；`/static/*` 为静态资源

---

## 4. 工作流：配置 API Key → JINE → 直播

### ① 配置 API Key

1. 启动服务后打开页面，进入**控制面板**（设置页）。
2. 选择厂商（deepseek / openai / custom 等 OpenAI 兼容端点），填入 `base_url`、`model`、`API Key`，点「测试」确认连通，再保存。
3. 设置持久化在 `data/llm_settings.json`；响应里只回传掩码 key，永不回传明文。
4. 安全规则：改 `base_url` 必须重新填写 key（防止已保存凭据被转发到新地址）；默认拒绝回环/内网地址，自建本地推理服务时设 `LLM_ALLOW_PRIVATE_BASE_URL=1`。

### ② JINE 私聊（mode 0）

- 角色为**糖糖 / 雨（ame）**，桌面上的 JINE 窗口即私聊界面。
- 回复按换行分段发送；支持 `jine_pien_*` 表情协议（单独成条发送，前端渲染为表情图）。
- 状态栏由模型通过 `change_status` 工具表达方向，系统按概率与步长随机决定幅度；好感度决定立绘差分（`_dark/_low/_high/_peak`），压力决定待机动画。
- 长期记忆、短期记忆与状态栏都按存档独立持久化在 MySQL。

### ③ 直播（mode 1/2/3）

- 三个直播档：**超天酱（kangel）、堕天酱（dangel）、糖糖直播（ame_live）**，各有独立角色提示词与知识库。
- 直播窗口为弹幕/观众语境；ame_live 复用状态动画素材，同样随好感/压力换帧。
- **涨粉**：直播档每轮生成成功后按概率涨粉（默认 50% 概率涨 1~30），私聊不涨；粉丝数无上限，越过 1000 / 1万 / 10万 / 100万 时主播自觉提示词自动升级。
  - 调参（`.env`）：`FOLLOWERS_GROW_PROBABILITY`、`FOLLOWERS_GROW_MIN_STEP`、`FOLLOWERS_GROW_MAX_STEP`

### 生成管线（每轮消息的内部旅程）

```
用户消息 → 路由绑定会话快照 → GraphManager(LangGraph)
  → agent 节点：模型决定直接回答，还是调用工具
      ├─ 需检索/查时间/搜索/改写 → tools 节点执行 → 回到 agent（可多轮）
      └─ 否则 → generate_reply 节点
  → reflection 审查节点（不通过则带问题重写，上限 MAX_REFLECTION_TURNS）
  → SSE 分段推送 / 整段返回
  → MemoryManager 落库（历史+记忆+状态）→ 直播档结算涨粉
```

> RAG（`retrieve_knowledge` / `retrieve_context`）是 **agent 可选的工具**，不是管线的固定一环：
> 模型认为需要外部资料时才会经由 tools 节点检索，闲聊轮次不触发。

---

## 测试

```bat
venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/test_contract_backend.py tests/test_stream_behavior.py -q
for %f in (tests\*.cjs) do node %f
```
