# 改造方案：聊天/工作双模式 · tools 转 MCP · 模型设置面板

> 本文档为**方案与审计**，本轮不修改代码。
> 所有结论均基于本轮只读勘察，附 `文件:行号` 证据；不确定项集中在 §6。

---

## 0. 现状事实（方案的前提）

| 事实 | 证据 |
|---|---|
| 提示词按 4 个场景 mode 硬分文件，每个文件都把「输出规范 + 表情协议」整段复制一遍（**已改造，见 §5**） | [config.py:116-121](../app/config/config.py#L116-L121) |
| 有一条测试专门守护这 4 份复制的同步 | [test_config.py:54-67](../tests/unit/test_config.py#L54-L67) |
| `mode` 列**已存在**，但语义是场景/角色（0-3），不是任务类型 | [init_db.py:41](../repository/init_db.py#L41)、[chat_routes.py:22-24](../app/api/chat/chat_routes.py#L22-L24) |
| 短期记忆**按 mode 过滤** | [MemoryManager.py:57](../app/managers/MemoryManager.py#L57) |
| `/history` 返回的每行**不含 mode** | [MemoryManager.py:118-125](../app/managers/MemoryManager.py#L118-L125) |
| 工具**已收敛为一份定义**（**已改造，见 §5 P6a/P6b**）：`registry.py` 是唯一来源，模型可见载荷、ToolNode 执行体、MCP 出口全部由它派生；旧 `app/config/tools_schemas.yaml` 已删除 | [registry.py:144-228](../app/tools/registry.py#L144-L228)、[adapters/openai.py:19-33](../app/tools/adapters/openai.py#L19-L33)、[adapters/langgraph.py:16-29](../app/tools/adapters/langgraph.py#L16-L29)、[mcp/server.py](../app/mcp_server/server.py) |
| 一致性改由测试按 `registry.py` 钉死（不再是「两份定义强制同名」） | [test_tools_registry.py:54-84](../tests/unit/test_tools_registry.py#L54-L84)、[test_mcp_server.py](../tests/unit/test_mcp_server.py) |
| **API key 明文硬编码在源码里**，且 `.env` 中**没有**这三项 → 代码默认值就是线上生效值 | [config.py:31-33](../app/config/config.py#L31-L33)、[.env](../.env) 无 `DEEPSEEK_*` |
| 全仓**没有任何热更新机制**（`importlib.reload` / `setattr(config` 零命中） | 勘察确认 |
| `MEMORY_FILE_PATH` 是死字段；`data/` 目录不存在 | [config.py:66](../app/config/config.py#L66) |
| `app/api/logger`、`app/api/vision` 是**空文件**；`app/api/auth` 源码已删只剩 `.pyc` | 勘察确认 |
| `ame_knowledge.txt` 与 `kangel_knowledge.txt` **内容完全相同**（同 hash）；`.env` 里 `KNOWLEDGE_AME/KANGEL` 是失效残留 | 勘察确认 |
| `GraphManager.load_mode` / `FuncCallManager.load_mode` 存的 `current_mode` **从未被读取**（死代码） | [GraphManager.py:240-241](../app/managers/GraphManager.py#L240-L241)、[FuncCallManager.py:32-33](../app/managers/FuncCallManager.py#L32-L33) |
| `mcp` 2.3.0 与 `mcp-types` 2.3.0 **已安装**（P6b 落地；注意 2.x 里 `FastMCP` 已改名 `MCPServer`，`mcp.server.fastmcp` 是「导入即报错」的迁移提示模块） | `pip list`、[mcp/server.py](../app/mcp_server/server.py) |
| `.gitignore` 是 **UTF-16LE** 编码，已忽略 `prompts/` → **提示词文件不受版本控制** | [.gitignore](../.gitignore) |
| `.gitignore` **未包含 `.env`**，也未包含 `data/` → 含 MySQL/JWT/SMTP 明文密钥的 `.env` 有被 `git add -A` 提交的风险 | [.gitignore](../.gitignore) |
| 模型客户端在 import 时构造，base_url/key 冻结 | [ModelManager.py:56-59](../app/managers/ModelManager.py#L56-L59) |
| 控制面板只有 3 个音量相关控件，无任何下拉/文本输入 | [desktop.html:580-614](../static/desktop.html#L580-L614) |
| 控制面板窗口高度受限，加控件会溢出 | [desktop.css:1902](../static/css/desktop.css#L1902) `height: clamp(200px,35vh,350px)` |
| 前端**完全没有** localStorage/sessionStorage/IndexedDB | 全 static 检索 0 命中 |
| `/chat/mode_change` 接口存在但前端从未调用（死代码） | [mode.js:9-17](../static/js/data/api/mode.js#L9-L17) |
| `TWEET_PROMPT` 只被赋值，`single_tweet` 是空实现 | [chat_session.py:164-165](../app/services/chat_session.py#L164-L165) |
| `logger` / `vision` 路由存在但未挂载（main.py 只挂 chat_router） | [main.py:50](../app/main.py#L50)、[api/__init__.py](../app/api/__init__.py) |
| MCP 相关依赖**已安装**：`mcp` 2.3.0 / `mcp-types` 2.3.0（已写进 `requirements.txt`） | `pip list` |

---

## 1. 聊天模式 / 工作模式

### 1.1 核心设计：把「场景」与「任务」拆成两条正交轴

现在只有一条轴，且已被占满：

| 轴 | 取值 | 决定什么 | 状态 |
|---|---|---|---|
| **场景轴** `mode` | `0/1/2/3` | 角色人设、知识库、直播主题 | 已存在，**不动** |
| **任务轴** `task_mode` | `chat` / `work` | 输出纪律、是否按 `\n` 分段、工具与审查规则 | **新增** |

不建议把任务类型塞进现有 `mode`（例如编码成 `0|work`）：会同时打破 `validate_mode` 白名单、`load_short_memory` 的等值过滤、以及前端 topic→mode 的映射，改动面和回归风险都远大于新增一列。

### 1.2 提示词拼接

```
generate_reply 节点的 system 提示词
  = base.txt                      ← 跨角色共享（从 4 份复制中抽出）
  + role/{mode}.txt               ← 场景人设（现有 4 份文件去掉共享段）
  + task/{task_mode}.txt          ← 任务纪律（新）

agent 节点  = base.txt + task/{task_mode}.txt + call_{chat|work}.txt
reflection  = base.txt + task/{task_mode}.txt + reflection_{chat|work}.txt
```

**文件职责**

| 文件 | 内容 | 来源 |
|---|---|---|
| `prompts/base.txt`（新） | 输出边界、括号/旁白规则、`__ACCEPT__` 协议例外（**不含**表情协议） | 从 4 份公共段抽出（16 行，逐字节相同） |
| `prompts/role/{ame,kangel,dangel,ame_live}.txt`（新） | 人设、说话特点、人物动机、示例引言与全部示例（**不含**表情映射表） | 现有 `*_sys.txt` 删除公共段，角色文本逐字保留 |
| `prompts/task/chat.txt`（新） | 保持角色扮演，**禁止输出工作/编程内容**（可解释报错但不给完整代码块）；`jine_pien_*` 表情映射表与表情纪律；约定「**一个换行** = 另发一条」（一条消息内部不要换行） | 新增 |
| `prompts/task/work.txt`（新） | 面向工作任务，允许 Markdown/代码块/多行/英文标识符；**不按换行分段**，换行原样保留在同一条消息里 | 新增 |

> 已落地（P2a 拆分 + P2b 装配，均经独立复核）：`base.txt` 5 行 / `task/chat.txt` 22 行 / `task/work.txt` 7 行 / `role/*.txt` 69-72 行 / `call_chat.txt` 83 行 / `call_work.txt` 42 行 / `reflection_chat.txt` 10 行 / `reflection_work.txt` 11 行；`base` 新增行 **0**（纯抽取）；6 个旧文件（4 份 `*_sys.txt` + `reflection_prompt.txt` + `call_prompt.txt`）**已删除**，全仓无残留引用。
>
> 装配公式：`SYS_PROMPT[mode]` = `base + role`（保持字符串，兼容旧契约）；`config.sys_prompt_for(mode, task_mode)` = `base + role + task`（顺序 0→role→task，用 `\n\n` 连接）。`GraphManager` 三个节点读 `session.sys_prompt/call_prompt/reflection_prompt`，由 `chat_session._assemble_prompts()` 在两轴任一变化时重算，因此 **`GraphManager.py` 零改动**。

**为什么任务提示词必须同时管到 reflection**：改造前的审查规则要求「退回动作描写」「对白以中文为主」。工作模式下模型必然输出代码块与英文标识符，**若不新增工作版审查规则，每一个代码回复都会被判不通过**，最终耗尽重试抛 `ReflectionError`。已落地：`prompts/reflection_work.txt` 允许代码/Markdown/英文，同时保留 `__ACCEPT__` / `需要修改：` 协议（这是 [GraphManager.py:165-166](../app/managers/GraphManager.py#L165-L166) 的代码硬依赖）。

### 1.3 每轮对话记录 flag

- **新增列** `chat.task_mode varchar(8) not null default 'chat'`。
- 迁移复用既有幂等写法：[schema.py:16-37](../app/config/schema.py#L16-L37) 的「重复列错误码即成功」模式，扩展为可注册多条 DDL 的列表；启动时由 [init_db.py:50](../repository/init_db.py#L50) 调用。
- `update_short_memory` 写入该列；`read_history` 的 SELECT 增加该列并放进返回字典，`/history` 据此把 `task_mode` 交给前端。
- **旧数据默认 `chat`**：历史里的旧代码类回复会被按聊天模式拆分 → 见 §6-R6。

### 1.4 渲染规则

| 场景 | 聊天模式 | 工作模式 |
|---|---|---|
| JINE 实时 | 按换行分隔的每段 = 一条独立「头像 + 气泡」（代码围栏块内的换行不切段） | 单条气泡，`\n` 原样保留（现有 `white-space: pre-wrap`） |
| 直播实时 | 右侧弹幕区逐条追加；左侧字幕**逐段切换** | 单条弹幕 + 整段字幕 |
| 历史回放 | 按 `task_mode` 分段铺开，**无延时** | 单条 |
| 输入锁 | 保持到**最后一段**出现 | 流式结束即释放 |

**数据库始终只存一条完整 reply**，分段纯粹是表现层；这是保证历史与实时一致的唯一稳妥做法。

### 1.5 上一轮分段实现的教训（必须避免重蹈）

上一轮同类改动被回退，审计确认了两个必然发生的卡死（见上一轮审计结论）：

1. **闸门死锁**：首段已由 16ms 合并重绘显示后，`\n` 才到达；此时闸门关闭且**没有定时器**，后续段永不出现。
2. **取消不释放**：`done` 后等待后续段时若清空历史/切主题，`finished` 与视图版本变化让等待者永不 resolve，`chatBusy` 永久占用。

**新实现必须满足的不变式**（建议直接写进测试）：
- 任何时刻「有未显示段」⇔「存在已排定的定时器」；两者不得同时为假。
- `await` 的等待对象必须能在 `cancel()`、`clearMessages()`、`chatSessionGeneration` 变化、以及定时器回调发现 `!visible()` 时**全部**被释放。
- 释放输入锁走 `finally`，不得依赖某个可能被放弃的 Promise。

---

## 2. tools 转 MCP

### 2.1 目标解读（存在歧义，见 §6-R4）

按「**以便其他模型接入**」的字面含义，应让本项目**成为 MCP Server**，把工具暴露给外部客户端
（Claude Desktop / Cursor / 其他 agent）；数量最终由 R5 锁定为 **5 个无状态工具**（见 §2.3）。
反向（把外部 MCP 工具接进本图）不在本方案范围内。

### 2.2 消除现存的双份定义

改造前是「yaml 给模型看 + `@tool` 给 ToolNode 执行」，靠测试维持同名。**现已收敛为唯一注册表**
（P6a 删掉 `tools_schemas.yaml`，P6b 补上 MCP 出口；见 §5）：

```
app/tools/registry.py        ← 唯一事实来源：name / description / JSON Schema / 实现 / 作用域
   ├─ adapters/openai.py     → 生成模型可见 tools 载荷（取代 tools_schemas.yaml）
   ├─ adapters/langgraph.py  → 包装成 LangChain tool 供 ToolNode 执行
   └─ mcp/server.py          → 暴露为 MCP tools（只取 SCOPE_EXTERNAL 的 5 个）
```

收益：新增/改名工具只需改一处；一致性测试改为断言「注册表 = 模型载荷 = MCP 列表」三者一致
（[test_tools_registry.py](../tests/unit/test_tools_registry.py) +
[test_mcp_server.py](../tests/unit/test_mcp_server.py)：MCP 的 `inputSchema` 与注册表 `parameters`
**deep-equal**，不靠人工同步）。

### 2.3 MCP Server 形态

| 项 | 方案 |
|---|---|
| 依赖 | `mcp>=2.3.0`（官方 Python SDK）**已安装**，并写进 [requirements.txt](../requirements.txt)。注意 2.x 的改名：`FastMCP` → `MCPServer`（`mcp.server.mcpserver`） |
| 传输 | ① stdio：`python -m app.mcp_server.server`，供本地客户端直连；② HTTP：`mount_mcp(app)` 挂 `/mcp`，供局域网/其他模型接入（父 app lifespan 里必须跑 `mcp_lifespan()`） |
| 挂载点 | 由 [mcp/server.py](../app/mcp_server/server.py) 提供 `mount_mcp(app)` + `mcp_lifespan()` 两个口子；[main.py:50](../app/main.py#L50) 的接线尚未接（P6b 明确不改 main.py） |
| 工具清单 | **R5 锁定 5 个无状态工具**：`output_time`、`retrieve_knowledge`、`retrieve_context`、`rewrite`、`search_web`（`save_memory` / `change_status` 不外放，原因见 [docs/mcp-server.md](mcp-server.md)） |

**最大设计障碍（§6-R5 必须决策）**：`save_memory` 与 `change_status` 依赖「当前存档」这一进程内状态（[MemoryManager.py:47](../app/managers/MemoryManager.py#L47)、[FuncCallManager.py:93-97](../app/managers/FuncCallManager.py#L93-L97)），而外部 MCP 调用**没有会话上下文**。三种取舍：

- **(a) 显式传参**：给这两个工具加 `session_id` 参数，仅 MCP 版本暴露；内部版本继续用隐式上下文。
- **(b) 只暴露无状态工具**：MCP 侧只给 `output_time` / `retrieve_*` / `rewrite` / `search_web`，状态类工具不外放。
- **(c) 独立「外部会话」**：为 MCP 调用方单独建 session，与游戏存档隔离。

推荐 **(b) 起步**（风险最低，且回答「其他模型接入」的主要诉求就是检索与时间），再按需升级到 (a)。
**已按 (b) 落地**：MCP 侧注册的是 `registry.external_specs()` 的 5 个工具（见 §5 P6b）。

---

## 3. 控制面板：厂商 / 模型 / Key + 实时替换

### 3.1 去硬编码

- 删除 [config.py:31-33](../app/config/config.py#L31-L33) 的模型名、base_url、**明文 API key** 默认值，改由 `.env` 提供（与既有的 `MYSQL_PASSWORD`、`JWT_SECRET` 同规格）。
- **安全前置项**：该 key 已随源码提交进 git 历史，**必须在厂商后台轮换作废**，否则只是把泄露点从代码搬到别处。→ §6-R2

### 3.2 运行时设置与实时替换

```
app/services/llm_settings.py      ← 配置读写 + 校验 + 掩码
   ├─ profiles: {provider: {base_url, model, api_key, params}}   ← 持久化
   └─ active: provider                                            ← 当前生效
持久化：data/llm_settings.json（加入 .gitignore），不写 .env、不改 config
```

> 注意：`data/` 目录**当前不存在**，且 [.gitignore](../.gitignore) 里既没有 `data/` 也没有 `.env` —— 落地时需一并补上，否则新写入的明文 key 会被下一次 `git add -A` 带进版本库。
> 另：`prompts/` 已被 `.gitignore` 忽略，**提示词拆分（§1.2）不会有 git 历史可回溯**，建议同时解除忽略或另做备份。

**实时替换的关键**：现在客户端在 import 期构造且 base_url/key 冻结（[ModelManager.py:56-59](../app/managers/ModelManager.py#L56-L59)）。改为**按 `(base_url, api_key)` 惰性构造并缓存**，每次 `generate` / `stream_generate` 读取当前 active profile → 改设置后**下一个请求即生效，无需重启**。模型名本来就按调用读取（[ModelManager.py:64](../app/managers/ModelManager.py#L64)、[:97](../app/managers/ModelManager.py#L97)），这部分无需改动。

### 3.3 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat/settings/llm` | 返回厂商/模型/base_url + **掩码后的 key**（永不回传明文） |
| POST | `/chat/settings/llm` | 保存并切换 active profile；**生成中拒绝**（复用 [require_idle()](../app/api/chat/chat_routes.py#L17-L19)） |
| POST | `/chat/settings/llm/test` | 用候选配置发一次最小请求做连通性校验 |

前端新增 `static/js/data/api/settings.js`（沿用 [core.js 的 `NGO.apiJson`](../static/js/data/api/core.js#L15-L30)）。

### 3.4 控制面板 UI

新增分区（沿用既有 `.control-section` / `.control-label`）：

1. **对话模式**：聊天 / 工作 二选一（建议复用 `.control-toggle` 或做成两段式按钮）
2. **模型厂商**：`<select>`（desktop.html 目前 0 个 select，需新增样式）
3. **模型版本**：文本输入或下拉
4. **API Key**：`type=password` 输入 + 「保存」「测试」按钮

**布局约束**：窗口固定高度 `clamp(200px,35vh,350px)`（[desktop.css:1902](../static/css/desktop.css#L1902)），塞入 4 个分区必然溢出。方案：`.control-body` 加 `overflow-y:auto`，或改成标签页（音量 / 对话 / 模型）。

**持久化**：对话模式需在刷新后保留 → 新建前端持久化（localStorage 或随存档走后端）。注意现状是**完全没有任何持久化**，控制面板音量滑块自身刷新后也会回到默认值。

---

## 4. 各节点的模型提示词构成（改造前 → 改造后）

### 4.1 改造前（现状实测）

| 节点 | messages 构成 | 参数 | 证据 |
|---|---|---|---|
| **agent**（决策/工具） | `system` = `CALL_PROMPT`（含工具清单与 5 组示例）＋ 若有工具结果则**再追加** `system`「上一轮的思考结果…工具返回结果…」；否则 `user` = 本次 query | `think_para`（temp 0.3 / top_p 0.85） | [GraphManager.py:61-85](../app/managers/GraphManager.py#L61-L85) |
| **generate_reply**（回复） | `system` = `SYS_PROMPT[mode]`；＋ `system` = 状态提示词；＋ `system` = 长期记忆；＋ 若被退回 `system` = 上轮反思意见；＋ `short_memory`（user/assistant 交替）；＋ 若有工具结果 `system`；＋ `user` = query | `inference_para`（temp 0.8 / top_p 0.9） | [GraphManager.py:109-141](../app/managers/GraphManager.py#L109-L141) |
| **reflection**（审查） | `system` = `REFLECTION_PROMPT`；`user` = 「用户查询：…／角色回复：…」；格式异常时追加一条 `system` 重试指令 | `reflect_para`（temp 0.3 / top_p 0.9） | [GraphManager.py:149-175](../app/managers/GraphManager.py#L149-L175) |
| **tools** | 无提示词，`ToolNode` 执行 | — | [GraphManager.py:224-229](../app/managers/GraphManager.py#L224-L229) |

`generate_reply` 的 system 直接拼**原始 prompt 文本**，状态/记忆是**追加的独立 system 消息**而非字符串拼接，这一点改造时应保留。

### 4.2 改造后

| 节点 | system 构成 |
|---|---|
| agent | `base + task/{task_mode} + call_{chat\|work}` |
| generate_reply | `[base + role/{mode} + task/{task_mode}]` ＋（状态 / 长期记忆 / 反思意见 / 工具结果，保持独立 system 追加） |
| reflection | `base + task/{task_mode} + reflection_{chat\|work}` |
| tools | 不变 |

> 说明：把 `base` 与 `task` 放进 `generate_reply` 的同一条 system，是为了避免 system 消息条数继续膨胀；`agent`/`reflection` 同理。若你希望它们在日志里可分离，也可以拆成多条 system，二者行为等价。

---

## 5. 实施进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 | `task_mode` 列 + 幂等迁移 + 读写链路 + `/history` 返回 flag + 前端管道 | ✅ 完成（191 passed，已独立复核） |
| P2a | 提示词拆分：`base.txt` + `role/*` + `task/*` | ✅ 完成（已独立复核：原文每行均被覆盖，`base` 零新增行） |
| P2b+P3 | `config` 装配 + `chat_session` 三属性随轴重算 + 工作版 `reflection_work`/`call_work` | ✅ 完成（8 组装配顺序 base→role→task 已复核；`GraphManager` 零改动） |
| P4 | 前端按**换行**分段渲染（含代码围栏保护）+ §1.5 全部不变式 + 回归测试 | ✅ 完成（15 例；原 `@@@` 方案已被 R10-撤销 推翻，见 §6） |
| P5a | 控制面板：对话模式开关 + 厂商/模型/Key + 实时替换（**只做 OpenAI 兼容**） | ✅ 完成（`data/llm_settings.json` 原子写；掩码 `sk-5e1b****c57e`；改配置下一请求即生效） |
| P5b | Gemini / Anthropic 原生适配层 | ❌ **已取消**（用户决定只用 OpenAI 兼容，改动面回到最小） |
| P6a | 工具注册表收敛为唯一来源 | ✅ 完成（`config.TOOLS_SCHEMAS` 与旧 yaml deep equal = True，已独立复核） |
| P6b | MCP Server（对外 5 个无状态工具，R5） | ✅ 完成（`app/mcp_server/server.py`：stdio + `mount_mcp(app)`/`mcp_lifespan()`；schema 由注册表直接喂给 MCP 并 deep-equal 锁死；真实 stdio 握手与 HTTP initialize 均实测通过，见 [docs/mcp-server.md](mcp-server.md)） |
| R12 | 每轮落库「实际使用的 provider/model」 | ✅ 完成（`chat.llm_model` / `chat.llm_provider`，幂等迁移，取 `resolve_profile` 的结果而非 `config.DEEPSEEK_MODEL`） |
| R14 | 契约登记 `/chat/history` + 3 个 settings 路径 | ✅ 完成（7 → 10 路径 / 11 method 条目；契约驱动参数化用例 7 → 11，并做了突变验证） |
| R13 | 零风险冗余清理 | ✅ 完成（删 `app/api/{logger,vision}` 6 个 1 字节文件、`TWEET_PROMPT` 死配置链与 `prompts/tweet_prompt.txt`；`mode.js` 经实测保留——删它会打破契约测试；`manual_chat_flow.py` 修好并实跑 exit 0） |
| — | 既有 P0（存档隔离、状态落库时机） | ⏳ 按 R11 只记录，不在本轮修 |

> **最终基线：`pytest -q` → 258 passed / 0 failed**（开工时 165）。
>
> 三处"防假绿/防挂死"的加固，都是被真实事故逼出来的：
> 1. [test_static_assets.py](../tests/test_static_assets.py) 两处 `subprocess.run(["node", ...])` 加 60s 超时并转成 `pytest.fail` —— 此前两个任务先后被半成品 JS **静默挂死 >10 分钟且无输出**。
> 2. `tests/segment_behavior.cjs` 加真实保活定时器 —— 加之前存在"断言没跑完就 exit 0"的假绿；加之后**立刻暴露 2 个实现真死锁**（`segDoneNow()` 不释放等待者；`cancel()` 因 `finished` 提前 return 导致 `done` 后清历史跳过 `cancelSeg()`），正是上一版被回退的病根。（该文件原名 `segment_marker_behavior.cjs`，`@@@` 方案推翻后按换行规则重写并改名。）
> 3. `test_tools_registry.py` 的 MCP 边界断言从「全局 `sys.modules` 无 mcp」改为**子进程验证** —— 原写法在 `app/main.py` 挂载 MCP 后变成随执行顺序时红时绿的假失败（已实测证明 `import app.main` 会让 `mcp` 进 `sys.modules`）。


---

## 6. 已锁定的决策（本轮确认）

| 编号 | 决策 | 对实现的影响 |
|---|---|---|
| R1 | 新增**工作版审查**：允许代码/Markdown/英文，仍拦截泄露私密推理 | 需 `reflection_work.txt`；`reflection_node` 按 `task_mode` 选规则 |
| R3 | 工作模式**仍绑定角色**，两条轴正交 | `SYS = base + role/{mode} + task/{task_mode}`，4 份角色卡全部保留 |
| R4 | 本项目作为 **MCP Server** 对外暴露工具 | 新增 MCP 服务出口（§2） |
| R5 | MCP 侧**只暴露无状态工具** | 对外 5 个：`output_time`/`retrieve_knowledge`/`retrieve_context`/`rewrite`/`search_web`；`save_memory`/`change_status` 仅内部 |
| R6 | **还要支持 Gemini / Anthropic 原生 SDK** | ⚠️ 见下方代价说明 |
| R7 | 短期记忆**按任务轴隔离** | `load_short_memory` 的过滤条件从 `mode=%s` 变为 `mode=%s and task_mode=%s` |
| R9 | 聊天模式**允许解释报错，但不给完整代码块** | `task/chat.txt` 需写明「可解释原因与思路，不得输出多行代码块」 |
| R2 | 旧 key **已在厂商后台轮换** | 只需完成代码去硬编码（§3.1） |
| R8 | 旧行（默认 `chat`）**一律按 chat 规则分段** | 不需要内容启发式识别 |
| R10 | ~~改用显式分隔标记 `@@@`（单独一行）~~ **已推翻**：改回**按 `\n` 分段**（`@@@` 旧协议取消） | ⚠️ 用户明确要求，原因见下方「R10 的推翻与连带影响」 |
| R11 | 既有 P0（全局单例串存档、状态在审查前落库）**只记录，本轮不修** | 另开一轮处理 |
| R12 | 生成中**拒绝**切换模型；每轮**落库记录**所用厂商/模型 | 需再加一列存模型标识 |
| R13 | 只清理**零风险**冗余（死代码/空文件/失效配置） | 不动重复知识库与提示词去重 |
| R14 | 补登记 `/chat/history` 到 [api_contract.json](../contracts/api_contract.json) | 让 `task_mode` 字段受契约保护 |
| P5 | **拆两步**：先 OpenAI 兼容 + 打通面板，再补 Gemini/Anthropic 适配 | P5a 不装新依赖；P5b 才需要 `google-genai`/`anthropic` |

### R10 的推翻与连带影响（重要）

R10 曾把原需求「按 `\n` 分段」改成**显式分隔标记 `@@@`**（单独一行、trim 后完全相等），理由是 `\n` 在 Markdown、代码、列表、引用里都有排版含义。
**该决策现已推翻**：实测 `@@@` 这类自定义标记对模型不自然、**触发率低**（模型经常忘记或写错），用户明确要求改回**按 `\n` 分段**。当前落地的是：

1. **提示词**（`prompts/task/chat.txt`）：写明「你输出的每一个换行都会被当成另发一条消息」——一条消息内部不要换行（不要为了排版而换行）；需要连发多条时每条占一行；并明确「不要输出 `@@@` 之类的分隔符（旧协议已取消）」。工作类内容禁令（不写代码/脚本/技术方案/长文档）因此更重要：代码里必然有换行，会被拆成多条消息。
2. **前端分段器**（[chat.js](../static/js/service/chat.js)）：聊天模式按 `\n`（兼容 `\r\n`）切段；**段内容不 trim**（行首空格、缩进原样保留）；**只含空白的段丢弃**（连续空行不产生空气泡）。
3. **代码围栏保护**（`@@@` 方案曾靠"标记完全在表情词法之外"回避的风险，现在必须正面处理）：某行 trim 后以三个反引号开头的那一行算围栏分隔行；**从开围栏到配对闭围栏之间的所有换行都不触发切段**，整块（含围栏行本身）作为同一个段；流式期间围栏还没等到闭合时按"仍在围栏内"处理，先不切。这样模型万一输出代码，也不会被拆成多条气泡。
4. **工作模式**：完全不按换行分段，`\n` 原样保留在同一条消息里（现有 `white-space: pre-wrap` 已支持）。
5. **单行回复（整条没有 `\n`）行为零变化**：不进入分段模式、无闸门定时器、不延长输入锁——这是最重要的回归安全网。
6. **闸门与锁的既有语义全部保留**：段间 ≥1 秒（`SEG_DELAY = 1000`）、输入锁保持到最后一段出现、历史回放同步铺开无延时、两条死锁修复（`segDoneNow()` 必须 settle 等待者；`cancel()` 必须在 `finished` 判断之前释放等待者）一条都不动——**本轮只换"怎么切段"**。
7. **兼容旧数据**：旧行默认 `chat` → 含换行的旧回复会被按换行拆成多条气泡（含代码/Markdown 的旧回复也在内）；这是 `\n` 分段的固有代价，见 §6b-R8。


### R6 的代价（选择原生 SDK 后新增的工作与风险）

| 项 | 事实 |
|---|---|
| 依赖 | `google-genai`、`anthropic` **均未安装**（venv 实测 MISSING），需要联网安装 |
| 需新增 | provider 适配层：把「messages + tools + 流式 delta」抽象成统一接口，再分别落到 OpenAI / Gemini / Anthropic 三套 SDK |
| 会打破的测试 | [test_model_manager.py:9-19](../tests/unit/test_model_manager.py#L9-L19)（以 `FakeAsyncOpenAI` 替换 `openai.AsyncOpenAI` 的方式失效）、[test_model_stream.py:32-114](../tests/unit/test_model_stream.py#L32-L114)（依赖 OpenAI `delta.reasoning_content` / `tool_calls[].index` 结构）→ 两个文件需重写为按 provider 参数化 |
| 仍受约束 | [test_model_manager.py:62-79](../tests/unit/test_model_manager.py#L62-L79) 断言 `mgr.tools` 是 OpenAI function dict 形状 → 适配层必须做「内部统一 schema → 各厂商格式」的转换，而不是替换内部 schema |
| 工期影响 | P5 从「改 base_url + model」升级为「新建适配层 + 重写 2 个测试文件」，约为原工作量的 3-4 倍 |

> 若想先拿到可用结果，建议 P5 分两步：**先做 OpenAI 兼容（含 DeepSeek 与任何兼容端点）并打通控制面板**，再补 Gemini/Anthropic 适配。其余阶段（P1-P4、P6）不受 R6 影响。

---

## 6b. 其余待决项（不阻塞开工）

### 仍需在实现中决定（不阻塞开工）

> 已决项见上方 §6 表格；以下为剩余细节。
> 附注（R7 的连带事实）：长期记忆（`session.memory`）与 RAG 上下文库**本来就没有模式维度**，跨 mode 共享，仅以 `{mode}:` 前缀标记（[RAGManager.py:49-54](../app/managers/RAGManager.py#L49-L54)）。本次只隔离短期记忆（`chat` 表），长期记忆是否也要拆分留待后续。

**R8｜旧数据默认值**
已决：旧行（默认 `chat`）一律按聊天规则分段。分段依据改回换行后，旧回复**只要含换行就会被拆成多条气泡**（例如旧的工作类回复里的代码/Markdown 会被拆碎）；这是 `\n` 分段的固有代价：代价只落在表现层，数据库里始终是一条完整 reply。

**R11｜会话隔离尚未修复（既有 P0）**
上一轮审计已确认：全局单例 `ChatSession`（[services/__init__.py:12](../app/services/__init__.py#L12)）+ 不校验请求身份的 `/history`、`/get_status`（[chat_routes.py:43-49](../app/api/chat/chat_routes.py#L43-L49)）→ 多客户端切存档会串号。**同时**，状态类工具在审查通过前就落库（[MemoryManager.py:79-99](../app/managers/MemoryManager.py#L79-L99)）。这两项与本次改造正交，但设置面板会引入新的「全局可变状态」（active profile），建议一并规划。

**R12｜模型切换的并发语义**
生成中是否允许切换？（方案默认拒绝，复用 `require_idle`）。另外切换是否应写入历史/日志以便复现某轮回复用的是哪个模型？

**R13｜已确认的冗余（可顺手清理，非阻塞）**
- `/chat/mode_change` 前端从未调用（[mode.js](../static/js/data/api/mode.js)）
- `#bgm-autoplay` 的 change 监听是空函数（[control.js:23-26](../static/js/ui/control.js#L23-L26)）
- `TWEET_PROMPT` 已加载但 `single_tweet` 为空实现（[chat_session.py:164](../app/services/chat_session.py#L164)）
- `app/api/logger`、`app/api/vision` 是空文件；`app/api/auth` 只剩 `.pyc`（[api/__init__.py](../app/api/__init__.py) 只导出 chat_router）
- `MemoryManager.load_status` 与 `refresh` 重复计算状态提示词
- `ame_knowledge.txt` 与 `kangel_knowledge.txt` 内容完全相同（同 hash）→ 可合并为一个文件，但会削弱「每个 mode 独立知识库」的扩展位，建议保留但注明
- `MemoryManager.MEMORY_FILE_PATH`（[config.py:66](../app/config/config.py#L66)）与 `.env` 的 `KNOWLEDGE_AME/KNOWLEDGE_KANGEL` 均为死配置
- `GraphManager.load_mode` / `FuncCallManager.load_mode` 存的 `current_mode` 从未被读取

**R14｜契约测试未覆盖 `/chat/history`**
[api_contract.json](../contracts/api_contract.json) 只登记 7 个路径，不含 `/chat/history`。给历史响应加 `task_mode` 字段不会被契约测试发现回归 → 建议先登记再改。

**R15｜MCP 出口必须与既有 OpenAI 形状并存**
[test_model_manager.py:62-79](../tests/unit/test_model_manager.py#L62-L79) 断言 `mgr.tools` 就是 OpenAI function dict 列表。因此 §2.2 的注册表必须是「**一个来源、多个出口**」，MCP 的 `inputSchema` 不能取代内部使用的 `parameters`，否则会连模型调用一起改坏。

### 测试影响（必须同步改）

**硬约束**——以下断言会直接锁死某些实现方式，改造前必须先决定怎么改测试：

| 测试 | 断言内容 | 对方案的约束 |
|---|---|---|
| [test_tools_schema_sync.py:18-41](../tests/unit/test_tools_schema_sync.py#L18-L41) | `entry["function"]["name"]` 集合 == `@tool` 名字集合；`change_status` 参数集精确匹配 | 收敛注册表后改为「注册表 = MCP 列表 = 模型载荷」三方一致；**不能只留 MCP 的 `inputSchema` 命名** |
| [test_model_manager.py:29-32](../tests/unit/test_model_manager.py#L29-L32) | 显式断言 `not hasattr(mgr,'bind_tools')` | 禁止引入 LangChain `bind_tools` 绑定工具 |
| [test_model_manager.py:62-79](../tests/unit/test_model_manager.py#L62-L79) | `mgr.tools` 必须是 OpenAI dict 列表，且原样进 `calls[0]["tools"]` | 注册表**必须仍能产出 OpenAI function 形状**（MCP 是并列出口，不是替代出口） |
| [test_model_manager.py:9-19](../tests/unit/test_model_manager.py#L9-L19) | 用 `FakeAsyncOpenAI` 替换 `openai.AsyncOpenAI` 后加载模块 | 惰性建 client 时**不能改变"从 `openai` 模块取 `AsyncOpenAI`"这一可替换点** |
| [test_model_stream.py:32-114](../tests/unit/test_model_stream.py#L32-L114) | 依赖 OpenAI 流式 delta 结构（`reasoning_content`、`tool_calls[].index`） | 厂商范围（§6-R6）若扩到非 OpenAI SDK，这两个文件需大改 |
| [test_config.py:41-43](../tests/unit/test_config.py#L41-L43) | 三个 `MAX_TOKENS` 字段**类默认值必须为 0** | 模型参数持久化应放独立 store，**不要写进 config 字段默认值** |
| [test_func_call_manager.py:41](../tests/unit/test_func_call_manager.py#L41)、[:83-96](../tests/unit/test_func_call_manager.py#L83-L96) | 硬编码 7 个工具 + 按**位置**解包 | 增删工具或改顺序都会破，需同步 |
| [test_graph_manager.py:180](../tests/unit/test_graph_manager.py#L180) | 断言 `messages[0]["content"] == "system prompt"`（桩值） | 提示词改为「base+role+task」拼接后，此桩值语义变化，需同步 |
| [test_memory_manager.py:51-54](../tests/unit/test_memory_manager.py#L51-L54) | `read_history` 返回 4 字段字典 | 加 `task_mode` 后需同步 |
| [test_config.py:54-67](../tests/unit/test_config.py#L54-L67) | 4 份 prompt 都含同一输出规范段 | 拆分为 base 后**必然失败**，改为断言组合结果 |

**覆盖缺口**——新增测试：

| 新增测试 | 目标 |
|---|---|
| 分段不变式 | §1.5 三条（无定时器⇔无待显示段；取消必释放；`finally` 释放锁） |
| 工作模式审查放行 | 含代码块/英文标识符的回复必须通过 reflection |
| 设置接口掩码 | GET 不得回传明文 key |
| 契约补齐 | `/chat/history` 未登记在 [api_contract.json](../contracts/api_contract.json)，新增 `task_mode` 字段**不受契约测试保护**，建议一并登记 |
| [test_status_rules.py](../tests/unit/test_status_rules.py) | 零幅度、负界、独立概率（上一轮审计已提出） |

---

## 7. 验收标准（建议）

1. 聊天模式：一句用换行分成 3 段的回复 → 3 条独立气泡（JINE）/ 3 条弹幕 + 3 次字幕切换（直播），段间 ≥1 秒；工作模式同一文本 → 1 条（换行原样保留、不分段）；代码围栏块内的换行不拆段；单行回复与改造前完全一致。
2. 刷新页面重新加载历史：分段结果与实时一致，且**无延时**。
3. 生成中切换主题 / 清空历史 / 切换存档 / 关闭窗口 → 输入锁**必定释放**（不得出现永久 `chatBusy`）。
4. 控制面板改厂商/模型/Key → **不重启**，下一个请求生效；GET 接口返回的 key 为掩码。
5. `pytest` 基线不低于 165 passed，且新增测试覆盖 §6 的每个决策点。
