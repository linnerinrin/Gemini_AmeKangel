#"""
#架构作用：工具名称、描述、schema、实现工厂及出口范围的唯一来源
#调用方：配置的OpenAI适配器、FuncCallManager的图适配器及MCP Server
#数据流：ToolSpec →适配器→带管理器闭包的函数→检索/模型/持久化服务
#"""
"""
tools/registry.py工具唯一注册表（唯一事实来源）

工具的名称、模型可见描述、参数JSON Schema、实现函数、作用域只在本文件定义一次；
所有对外形状都由TOOL_REGISTRY派生，不要再新增第二份工具定义：

    TOOL_REGISTRY
      ├─ app/tools/adapters/openai.py    →模型可见的OpenAI function载荷（config.TOOLS_SCHEMAS）
      ├─ app/tools/adapters/langgraph.py → LangChain工具对象（FuncCallManager._create_tools → ToolNode）
      └─ app/mcp_server/server.py               → MCP Server的tools列表（只暴露无状态工具）

原app/config/tools_schemas.yaml的内容已原样迁移到本文件并删除该文件：留着那份yaml只会
成为漂移陷阱（改了它不会被加载，也不会有任何提示）。需要人类可读快照时随时重新生成：

    python -m app.tools.adapters.openai

实现约定：
- ToolSpec.impl(mgr)返回带真实签名的工具函数；签名是LangChain侧的参数校验依据，必须与
  parameters保持一致（reason必填、三个amount有默认值、output_time保留空参兜底）
-实现函数不写docstring：描述只保留在ToolSpec.description，由适配器写入__doc__ /
  description，避免同一句描述维护两处
-依赖config / status_rules / ddgs的实现使用函数内延迟导入，原因有二：
  ① app.config.config会反向导入本注册表来生成TOOLS_SCHEMAS，而status_rules又导入config，
模块级导入会形成循环导入；
  ② ddgs延迟到调用时导入，测试注入的sys.modules["ddgs"]替身才会生效
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

#项目根（本文件位于app/tools/下，向上三级即项目根）；不取config.PROJECT_ROOT
#是为了避免config→注册表→config的循环导入，且路径语义与本文件位置天然一致
#只读工具的读取边界不在这里：实际根目录由FuncCallManager按READONLY_ROOTS（默认用户目录）注入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#注册表导入期的警告走标准logging（本文件不导入config，避免循环导入）
logger = logging.getLogger(__name__)

#作用域：internal只给内部模型与ToolNode；external额外通过MCP对外暴露
#save_memory / change_status依赖「当前存档」这类进程内状态，不对外暴露
SCOPE_INTERNAL = "internal"
SCOPE_EXTERNAL = "external"


#"""
#模块一：ToolSpec与参数schema工厂
#一个工具=名字+模型可见的description + JSON Schema parameters +实现工厂+作用域；
#这里是全项目工具定义的唯一事实来源，config.TOOLS_SCHEMAS与MCP出口都由它派生
#"""
@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整定义；description / parameters就是模型可见的那一份。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    impl: Callable[[Any], Callable[..., Any]]
    scope: str = SCOPE_INTERNAL


def _query_parameters(description: str) -> Dict[str, Any]:
    """只有一个必填query参数的JSON Schema（与迁移前的yaml完全一致）。"""
    return {
        "type": "object",
        "properties": {"query": {"type": "string", "description": description}},
        "required": ["query"],
    }


#"""
#模块二：工具实现闭包（impl工厂）
#每个_bind_*返回一个可被ToolNode调用的协程；工厂只捕获管理器，不做事，真正副作用在被调用时
#共性是延迟导入config / ddgs / status_rules以避免循环依赖与替身注入需求；
#代价是副作用时机与执行线程都藏在闭包里，审查前写库的问题就在这里（见各函数行内标记）
#"""
def _bind_output_time(mgr):
    def output_time(str=""):
        return "现在时间是" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    return output_time


def _bind_retrieve_knowledge(mgr):
    async def retrieve_knowledge(query):
        result = await mgr.rag_mgr.retrieve(query, caller="knowledge")
        return "得到文档内容：" + result

    return retrieve_knowledge


def _bind_retrieve_context(mgr):
    async def retrieve_context(query):
        result = await mgr.rag_mgr.retrieve(query, caller="context")
        return "得到与查询存在关联的上下文内容：" + result

    return retrieve_context


def _bind_rewrite(mgr):
    async def rewrite(query):
        from app.config.config import config  #延迟导入：见模块docstring（避免循环导入）
        prompt = f"请结合对话历史，将以下问题改写为独立完整的查询，仅输出改写结果：\n{query}"
        result = await mgr.model_mgr.generate(
            [{"role": "user", "content": prompt}],
            {"max_tokens": config.REWRITE_MAX_TOKENS, "temperature": 0.3},
            use_tools=False
        )
        return f"改写后的查询：{result}"

    return rewrite


def _bind_save_memory(mgr):
    async def save_memory(memory: str) -> str:
        await mgr.memory_mgr.update_long_memory(memory)
        return ""

    return save_memory


def _bind_search_web(mgr):
    async def search_web(query: str) -> str:
        from app.config.config import config  #延迟导入：见模块docstring（避免循环导入）
        from ddgs import DDGS  #延迟导入：测试会用替身替换ddgs

        #DDGS.text是同步调用，放进工作线程执行，避免在async工具内阻塞事件循环
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=config.SEARCH_MAX_RESULTS))

        try:
            results = await asyncio.to_thread(_search)
            if not results:
                return "未搜索到相关结果"
            formatted = "\n".join(
                f"{i+1}. {r['title']}\n   {r['body']}\n   链接：{r['href']}"
                for i, r in enumerate(results)
            )
            return f"互联网搜索结果：\n{formatted}"
        except Exception as e:
            return f"搜索失败：{str(e)}"

    return search_web


def _bind_change_status(mgr):
    async def change_status(reason: str, amount_affection: int = 0,
                            amount_stress: int = 0, amount_mental: int = 0) -> str:
        from app.config.status_rules import StatusArgumentError  #延迟导入：见模块docstring
        if mgr._status_changed_this_turn:
            return "本轮状态已经结算，后续状态变化未生效。"
        try:
            deltas = mgr.memory_mgr.apply_status_change({
                "affection": amount_affection,
                "stress": amount_stress,
                "mental": amount_mental,
            })
        except StatusArgumentError as exc:
            return f"状态变化参数无效，本次未生效：{exc}"
        mgr._status_changed_this_turn = True
        return reason

    return change_status


#"""
#模块三：注册表与访问器
#列表顺序就是对外契约（测试按位置解包、ToolNode按名派发）；scope决定是否经MCP暴露
#增删或换序必须同步适配器、MCP出口与相关测试
#"""
#顺序即对外契约：tests/unit/test_func_call_manager.py按位置解包前7个工具，
#ToolNode也按顺序注册；增删/换序必须同步改测试（只读工具追加在末尾）
TOOL_REGISTRY: List[ToolSpec] = [
    ToolSpec(
        name="output_time",
        description="获取当前日期和时间。当用户询问现在几点、今天几号时使用。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        impl=_bind_output_time,
        scope=SCOPE_EXTERNAL,
    ),
    ToolSpec(
        name="retrieve_knowledge",
        description="在角色知识库中查找背景设定信息，当用户询问角色设定、背景故事等相关信息时使用，若已经尝试查找背景设定但无果，则无需再继续调用。",
        parameters=_query_parameters("搜索查询"),
        impl=_bind_retrieve_knowledge,
        scope=SCOPE_EXTERNAL,
    ),
    ToolSpec(
        name="retrieve_context",
        description="在近期对话上下文中搜索相关信息，当需要根据对话历史中提到的内容回答时使用，若已经尝试查找上下文但无果，则无需再继续调用。",
        parameters=_query_parameters("搜索查询"),
        impl=_bind_retrieve_context,
        scope=SCOPE_EXTERNAL,
    ),
    ToolSpec(
        name="rewrite",
        description="对用户的输入进行改写，在用户的输入存在指代不清和需要分治处理时使用。",
        parameters=_query_parameters("用户的原始问题"),
        impl=_bind_rewrite,
        scope=SCOPE_EXTERNAL,
    ),
    ToolSpec(
        name="save_memory",
        description="保存用户的关键信息，如姓名、爱好、年龄等。请严格按照'键: 值'格式填写（例如'年龄: 20'），同一键名自动覆盖。当用户提供了个人信息时调用。",
        parameters={
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "提取到的用户信息",
                },
            },
            "required": ["memory"],
        },
        impl=_bind_save_memory,
        scope=SCOPE_INTERNAL,
    ),
    ToolSpec(
        name="search_web",
        description="在互联网上搜索实时信息，当用户询问最新新闻、实时资讯或角色不知道的信息时使用",
        parameters=_query_parameters("搜索关键词"),
        impl=_bind_search_web,
        scope=SCOPE_EXTERNAL,
    ),
    ToolSpec(
        name="change_status",
        description="当用户发言影响角色状态时调用。参数只表达好感、压力、阴暗三个状态的变化方向，实际变化幅度由系统随机决定。",
        parameters={
            "type": "object",
            "properties": {
                "amount_affection": {
                    "type": "integer",
                    "description": "好感变化方向：1=上升，-1=下降，0=不变。 正数表示用户夸奖、关心、支持角色；负数表示用户冷落、批评、伤害角色。 实际变化幅度由系统在配置范围内随机决定。\n",
                },
                "amount_stress": {
                    "type": "integer",
                    "description": "压力变化方向：1=增加，-1=减少，0=不变。 正数表示用户提到敏感话题、施加压力、催促；负数表示用户安抚或让角色放松。 实际变化幅度由系统在配置范围内随机决定。\n",
                },
                "amount_mental": {
                    "type": "integer",
                    "description": "阴暗变化方向：1=增加，-1=减少，0=不变。 正数表示用户提及负面/黑暗内容或角色陷入消极思考；负数表示用户带来积极情绪。 实际变化幅度由系统在配置范围内随机决定。\n",
                },
                "reason": {
                    "type": "string",
                    "description": "状态变化的原因",
                },
            },
            "required": ["reason"],
        },
        impl=_bind_change_status,
        scope=SCOPE_INTERNAL,
    ),
]


#"""
#模块三·附：agent_framework只读工具追加到注册表末尾
#build_readonly_tool_specs内部延迟导入本模块取ToolSpec/SCOPE_INTERNAL，此处调用时
#本模块已定义完这两个名字；工具实现经mgr.readonly_tools取实例，
#此处传入的workspace_root仅作文档语义，真正的读取边界由FuncCallManager的READONLY_ROOTS决定
#"""
from app.agent_framework.readonly.tool_specs import build_readonly_tool_specs

TOOL_REGISTRY.extend(
    build_readonly_tool_specs(str(PROJECT_ROOT))
)


def tool_names() -> List[str]:
    """注册表里的工具名，顺序即契约（测试按位置解包、ToolNode按名派发）。"""
    return [spec.name for spec in TOOL_REGISTRY]


def get_spec(name: str) -> ToolSpec:
    """按名字取定义；名字不存在直接抛KeyError，避免静默降级。"""
    for spec in TOOL_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"未注册的工具：{name}")


def external_specs() -> List[ToolSpec]:
    """对外（MCP）暴露的无状态工具。"""
    return [spec for spec in TOOL_REGISTRY if spec.scope == SCOPE_EXTERNAL]


#"""
#模块三·附二：MCP stdio客户端接线
#server清单读data/mcp_servers.json（json直读，不import config——config反向依赖本模块，
#import即循环）；文件缺失/无servers→MCP_CLIENT为NotConfigured占位实现，注册表不追加任何工具；
#配置形状：{"servers": [{"name","command","args":[...],"env":{...},"timeout":30}]}
#限制：tool_bridge的impl经mgr.mcp_client取单一客户端，多server只会用第一份配置
#与既有名字的冲突在extend前检查（_resolve_name只解决MCP内部冲突）：冲突项跳过并记录警告
#"""
import json as _json

from app.agent_framework.mcp_client.client import NotConfiguredMcpClient
from app.agent_framework.mcp_client.stdio_client import StdioMcpClient
from app.agent_framework.mcp_client.tool_bridge import build_mcp_tool_specs


def _build_mcp_client():
    """按data/mcp_servers.json构造MCP客户端；未配置时返回显式失败的占位实现。"""
    config_path = PROJECT_ROOT / "data" / "mcp_servers.json"
    try:
        servers = _json.loads(config_path.read_text(encoding="utf-8")).get("servers") or []
    except FileNotFoundError:
        servers = []
    except Exception as exc:  #配置文件损坏不应拖垮整个注册表导入
        logger.warning("[mcp_client] 读取 %s 失败，按未配置处理：%s", config_path, exc)
        servers = []
    if not servers:
        return NotConfiguredMcpClient()
    if len(servers) > 1:
        logger.warning("[mcp_client] 暂只支持单个 server，已忽略多余配置")
    first = servers[0]
    return StdioMcpClient(
        first["command"],
        first.get("args") or [],
        first.get("env"),
        timeout=float(first.get("timeout", 30.0)),
        server=str(first.get("name") or ""),
    )


MCP_CLIENT = _build_mcp_client()
"""进程级 MCP 客户端单例：注册表桥接与 FuncCallManager.mcp_client 共用同一份配置。"""

_mcp_specs = build_mcp_tool_specs(MCP_CLIENT)  #失败（含未配置）返回空列表
_known_names = set(tool_names())
for _spec in _mcp_specs:
    if _spec.name in _known_names:
        logger.warning("[mcp_client] MCP 工具 %s 与既有工具重名，已跳过", _spec.name)
        continue
    _known_names.add(_spec.name)
    TOOL_REGISTRY.append(_spec)

#导入期警告经logging.warning记录；注册表不导入config以保持可先于config导入。
#前7个核心工具的名称与顺序固定；只读工具（read_file/list_dir/glob/grep）追加在末尾；
#MCP外部工具去重后追加且scope=SCOPE_EXTERNAL。
