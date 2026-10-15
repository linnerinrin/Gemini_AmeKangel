# Gemini_AmeKangel · 双子角色扮演聊天系统

基于大语言模型的双角色扮演聊天系统。糖糖（阴暗少女）与超天酱 / 堕天酱（元气主播）可在**私聊**与**直播**两种场景下切换，内置 RAG 知识库检索、查询改写、状态栏（好感 / 压力 / 心理）、思考链、表情协议，以及一套桌面式（Windows 98 风格）浏览器前端。

---

## 目录

- [功能](#功能)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型配置](#模型配置)
- [聊天模式与工作模式](#聊天模式与工作模式)
- [MCP 工具出口](#mcp-工具出口)
- [项目结构](#项目结构)
- [接口一览](#接口一览)
- [测试](#测试)
- [仓库里没有的东西](#仓库里没有的东西)
- [文档](#文档)

---

## 功能

| 模块 | 说明 |
|---|---|
| **双角色扮演** | 私聊（JINE，糖糖 / 雨）与直播（超天酱、堕天酱、糖糖）四个场景，各自独立系统提示词与知识库 |
| **聊天 / 工作双模式** | 同一角色语境下切换任务轴：聊天模式专注角色扮演、按换行分段发送；工作模式面向技术任务、允许代码与 Markdown、整条不分段 |
| **RAG 检索** | 向量召回（bi-encoder）+ 重排（cross-encoder），`retrieve_knowledge` / `retrieve_context` 两个工具；支持查询改写 `rewrite` |
| **思考链** | 每个节点的思考、工具调用与审查结论都会实时推送到前端，并在该条回复内折叠展示；落库后历史回放可还原 |
| **审查节点** | 回复生成后由独立节点审查，不通过则带着具体问题重写（次数上限见 `MAX_REFLECTION_TURNS`） |
| **状态栏** | 好感 / 压力 / 心理 / 粉丝数，由模型通过 `change_status` 表达方向、系统按配置随机决定幅度，变化时有浮字动画 |
| **表情协议** | `jine_pien_*` 系列表情代码，仅允许单独发送；前端按文本自动渲染为表情图片，历史旧数据的中文标签也能还原 |
| **存档系统** | 3 个存档位，各自独立会话、短期记忆、长期记忆与状态栏 |
| **桌面式前端** | 窗口拖拽、开始菜单、BGM / 音效、角色逐帧动画，纯原生 JS，无构建步骤 |

---

## 运行环境

- **Python 3.11**（开发环境 3.11.4）
- **MySQL**（默认库名 `gemini`，启动时自动建表与迁移）
- Windows（`start.bat` 与前端动画按 Windows 设计；其他平台可用 uvicorn 直接启动）
- 一个 **OpenAI 兼容**的大模型接口（DeepSeek / OpenAI / 任意兼容网关）

依赖见 [requirements.txt](requirements.txt)。其中 `unsloth` / `trl` / `datasets` 仅供 `utils/` 下的训练脚本使用，只跑服务可以不安。

---

## 快速开始

```bat
:: 1. 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

:: 2. 准备配置
copy .env.example .env      :: 然后按需修改（MySQL 账号、路径等）

:: 3. 准备本地模型（见"仓库里没有的东西"）

:: 4. 启动
start.bat
```

启动后浏览器访问 <http://127.0.0.1:5577/>。

也可以用参数换端口（原端口被占用时）：

```bat
start.bat 6000
```

或直接：

```bat
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5577
```

> 接口文档：<http://127.0.0.1:5577/docs>

---

## 模型配置

**源码里没有任何 API Key 或模型名的硬编码**，来源有两条，优先级从高到低：

1. **控制面板（推荐）** —— 打开桌面上的「控制面板」，在「模型厂商 / 模型版本 / API Key」处填写并保存，数据写入 `data/llm_settings.json`。
   - 内置厂商预设：`DeepSeek`、`OpenAI`、`custom`（自己填 `base_url`）
   - **保存后下一个请求即生效，无需重启**
   - 「测试」按钮会发一次最小请求验证连通性
   - 界面上 API Key 只显示掩码（如 `sk-5e1b****c57e`），明文永不回传
2. **`.env`** —— 未使用控制面板时的回退来源：
   ```
   DEEPSEEK_MODEL=deepseek-v4-flash
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   DEEPSEEK_API_KEY=sk-...
   ```

若两者都没有配置，界面会明确提示「尚未配置 API Key」，而不是抛出难以理解的 401。

**每轮对话都会记录实际使用的厂商与模型**（`chat.llm_provider` / `chat.llm_model`），方便排查同一存档里风格变化的原因。

---

## 聊天模式与工作模式

两条**正交**的轴：

| 轴 | 取值 | 决定 |
|---|---|---|
| 场景轴 `mode` | `0` 糖糖私聊 / `1` 超天酱直播 / `2` 堕天酱直播 / `3` 糖糖直播 | 角色人设、知识库、直播主题 |
| 任务轴 `task_mode` | `chat` / `work` | 输出纪律、是否分段、审查规则 |

最终提示词按 `base + role/{mode} + task/{task_mode}` 拼接：

- `prompts/base.txt` —— 跨角色跨任务共用的正文边界
- `prompts/role/*.txt` —— 四个角色的人设与示例
- `prompts/task/chat.txt` / `task/work.txt` —— 两种任务的纪律

**聊天模式**（默认）：

- 严格禁止输出工作类内容（代码、脚本、技术方案、长篇结构化文档）
- 用户问报错或技术问题时，可以简短解释原因与思路，但**不给完整代码块**，并提示切换到工作模式
- 保留 `jine_pien_*` 表情协议
- **按换行分段发送**：模型输出的每个换行都会被当作"另发一条消息"
  - 私聊：每段是独立的「头像 + 气泡」
  - 直播：弹幕区逐条追加，左侧字幕逐段切换
  - **段与段之间至少间隔 1 秒**才开始显示下一段；输入框会一直锁到最后一段出现
  - 历史回放同样按换行分段，但**一次性铺开、不加延时**
  - 代码围栏（三个反引号包起来的块）内的换行**不会**触发分段，防止代码被拆碎

**工作模式**：

- 面向技术问题、代码、文档、资料整理
- 允许并鼓励 Markdown、代码块、列表、表格、英文标识符与命令
- **不按换行分段**，回复始终作为一条消息，换行与缩进原样保留
- 使用独立的工作版审查规则（放行代码块，仍拦截泄露私密推理）

每轮的 `task_mode` 会写入 `chat` 表并随 `/chat/history` 返回，因此前端能对历史正确渲染（聊天模式的历史拆成多条，工作模式的保持整条）。前端开关在控制面板的「对话模式」，选择结果保存在浏览器 `localStorage`。

---

## MCP 工具出口

本项目的工具以 [Model Context Protocol](https://modelcontextprotocol.io) 对外暴露，其他模型 / 客户端可以直接接入。工具的**名称、描述、参数 Schema 只有一份定义**（`app/tools/registry.py`），同时派生给模型载荷、LangGraph 执行体与 MCP 三个出口。

对外只暴露 **5 个无状态工具**：

`output_time`、`retrieve_knowledge`、`retrieve_context`、`rewrite`、`search_web`

> `save_memory` 与 `change_status` 依赖「当前存档」这一进程内状态，**不对外暴露**，避免外部调用误改你正在玩的存档。

**stdio**（本地客户端直连）：

```bat
venv\Scripts\python.exe -m app.mcp_server.server
```

**HTTP**（其他机器 / 模型经网络接入）：应用启动后即挂载在 `/mcp`。

```bat
venv\Scripts\python.exe -m app.mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8931
```

> ⚠️ `/mcp` **目前没有认证**，任何能访问该端口的人都能调用这 5 个工具。若要暴露到公网，请自行加反向代理与鉴权。

详细用法见 [docs/mcp-server.md](docs/mcp-server.md)。

---

## 项目结构

```
app/
├── main.py                 # FastAPI 入口（装配路由、挂载 /mcp、lifespan 内建表迁移）
├── api/chat/               # 聊天与会话路由 + 设置路由 + 生成互斥锁
├── config/                 # 配置、提示词装配、表结构迁移
├── managers/               # 模型 / RAG / 记忆 / 工具 / 图 五个管理器
├── services/               # 会话编排（ChatSession）、模型设置服务
├── tools/                  # 工具唯一注册表 + OpenAI / LangGraph 适配器
└── mcp_server/             # MCP Server（stdio 与 HTTP 两种传输）
repository/                 # MySQL 连接与建表
prompts/                    # 基础 / 角色 / 任务提示词
static/                     # 前端（原生 JS，无构建）
tests/                      # pytest + 前端行为测试（*_behavior.cjs）
contracts/                  # 前后端接口契约（契约测试消费）
docs/                       # 设计文档
utils/                      # 训练与数据处理脚本
start.bat                   # Windows 启动脚本
```

### 对话编排（LangGraph）

```
START → agent（决策 / 工具调用）
          ├─ 需要工具 → tools → agent   （轮次上限 MAX_TOOL_TURNS）
          └─ 不需工具 → generate_reply（生成正文）
                          → reflection（审查）
                              ├─ 通过 → END
                              └─ 退回 → generate_reply（带具体问题重写）
```

三个节点各自读取会话上按 `(mode, task_mode)` 重算好的提示词，因此新增任务模式无需改动图结构。

---

## 接口一览

前缀 `/chat`，另有 `/mcp` 与 `/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat/start_session` | 绑定存档 |
| POST | `/chat/new_session` / `chose_session` / `delete_session` | 存档增删选 |
| GET | `/chat/sessions` | 存档列表 |
| GET | `/chat/history?mode=` | 历史（每行含 `task_mode` / `llm_provider` / `llm_model`） |
| POST | `/chat/single_chat` | 非流式对话 |
| POST | `/chat/stream_chat` | **流式对话（SSE）** |
| POST | `/chat/mode_change` | 切换场景轴 |
| POST | `/chat/get_status` | 读取状态栏 |
| GET / POST | `/chat/settings/llm` | 读取 / 保存模型设置（Key 只回掩码） |
| POST | `/chat/settings/llm/test` | 模型连通性测试 |

生成期间会拒绝切换存档 / 模式 / 模型（返回 409），避免中途串档。

---

## 测试

```bat
venv\Scripts\python.exe -m pytest -q
```

当前：**259 passed**。

- `tests/unit/` —— 各管理器、配置、迁移、工具注册表、MCP Server 的单元测试
- `tests/integration/` —— 路由与对话流程（含契约驱动）
- `tests/*_behavior.cjs` —— 前端行为测试，由 `pytest` 自动通过 node 子进程执行，覆盖分段渲染、字幕、思考链、控制面板、状态浮字、动画帧等
- `tests/manual_chat_flow.py` —— 手动跑通整条图链路（含真实 langchain 工具执行）

---

## 仓库里没有的东西

以下内容因体积或机密原因**不在仓库中**，clone 后需要自行准备，否则服务无法启动：

| 缺失内容 | 说明 | 如何准备 |
|---|---|---|
| `models/all-MiniLM-L6-v2/` | RAG 向量召回模型 | 从 HuggingFace 下载后放到该目录 |
| `models/ms-marco-MiniLM-L-6-v2/` | RAG 重排模型 | 同上 |
| `static/assets/audio/` | BGM 与音效（部分 wav 超过 GitHub 单文件 100MB 上限） | 自行放入 |
| `.env` | 内含 MySQL 密码、`JWT_SECRET`、SMTP 密码 | 由 `.env.example` 复制修改 |
| `data/llm_settings.json` | 控制面板保存的模型设置（含 API Key） | 启动后在控制面板填写 |
| `models/qwen2.5/` | 仅训练脚本需要，运行服务不需要 | 可选 |

路径由 `.env` 的 `BI_ENCODER_PATH` / `CROSS_ENCODER_PATH` 指定。

---

## 文档

- [改造方案与决策记录](docs/plan-chat-work-mode-mcp-llm.md) —— 双模式、MCP、模型设置的设计与取舍，含已知问题与后续项
- [MCP Server 使用说明](docs/mcp-server.md) —— 两种传输的接法、工具清单、Schema 保真做法

---

## 已知限制

- `/mcp` 无认证，暴露到公网前必须自行加鉴权
- 会话状态为**进程内全局单例**，多客户端同时使用会互相串档；生成期间的切换虽有互斥保护，但跨客户端隔离尚未实现
- 状态类工具（`change_status`）在审查通过前即写入数据库，审查失败时状态变更不回滚
- 长期记忆与 RAG 上下文**不按任务轴隔离**（仅短期记忆按 `mode + task_mode` 隔离）
- 聊天模式的换行分段依赖提示词约束模型「一条消息内部不换行」；模型若为排版换行，仍可能出现一条话被拆成两条气泡
