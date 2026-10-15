"""
tools/registry.py 工具唯一注册表（唯一事实来源）

工具的名称、模型可见描述、参数 JSON Schema、实现函数、作用域只在本文件定义一次；
所有对外形状都由 TOOL_REGISTRY 派生，不要再新增第二份工具定义：

    TOOL_REGISTRY
      ├─ app/tools/adapters/openai.py    → 模型可见的 OpenAI function 载荷（config.TOOLS_SCHEMAS）
      ├─ app/tools/adapters/langgraph.py → LangChain 工具对象（FuncCallManager._create_tools → ToolNode）
      └─ app/mcp_server/server.py               → MCP Server 的 tools 列表（R5：只暴露无状态工具）

原 app/config/tools_schemas.yaml 的内容已原样迁移到本文件并删除该文件：留着那份 yaml 只会
成为漂移陷阱（改了它不会被加载，也不会有任何提示）。需要人类可读快照时随时重新生成：

    python -m app.tools.adapters.openai

实现约定：
- ToolSpec.impl(mgr) 返回带真实签名的工具函数；签名是 LangChain 侧的参数校验依据，必须与
  parameters 保持一致（reason 必填、三个 amount 有默认值、output_time 保留空参兜底）。
- 实现函数不写 docstring：描述只保留在 ToolSpec.description，由适配器写入 __doc__ /
  description，避免同一句描述维护两处。
- 依赖 config / status_rules / ddgs 的实现使用函数内延迟导入，原因有二：
  ① app.config.config 会反向导入本注册表来生成 TOOLS_SCHEMAS，而 status_rules 又导入 config，
     模块级导入会形成循环导入；
  ② ddgs 延迟到调用时导入，测试注入的 sys.modules["ddgs"] 替身才会生效。
"""
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# 作用域：internal 只给内部模型与 ToolNode；external 额外通过 MCP 对外暴露。
# 方案 R5：save_memory / change_status 依赖「当前存档」这类进程内状态，不对外暴露。
SCOPE_INTERNAL = "internal"
SCOPE_EXTERNAL = "external"


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整定义；description / parameters 就是模型可见的那一份。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    impl: Callable[[Any], Callable[..., Any]]
    scope: str = SCOPE_INTERNAL


def _query_parameters(description: str) -> Dict[str, Any]:
    """只有一个必填 query 参数的 JSON Schema（与迁移前的 yaml 完全一致）。"""
    return {
        "type": "object",
        "properties": {"query": {"type": "string", "description": description}},
        "required": ["query"],
    }


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
        from app.config.config import config  # 延迟导入：见模块 docstring（避免循环导入）
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
        from app.config.config import config  # 延迟导入：见模块 docstring（避免循环导入）
        from ddgs import DDGS  # 延迟导入：测试会用替身替换 ddgs
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=config.SEARCH_MAX_RESULTS))
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
        from app.config.status_rules import StatusArgumentError  # 延迟导入：见模块 docstring
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


# 顺序即对外契约：tests/unit/test_func_call_manager.py 按位置解包这 7 个工具，
# ToolNode 也按顺序注册；增删/换序必须同步改测试。
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


def tool_names() -> List[str]:
    """注册表里的工具名，顺序即契约（测试按位置解包、ToolNode 按名派发）。"""
    return [spec.name for spec in TOOL_REGISTRY]


def get_spec(name: str) -> ToolSpec:
    """按名字取定义；名字不存在直接抛 KeyError，避免静默降级。"""
    for spec in TOOL_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"未注册的工具：{name}")


def external_specs() -> List[ToolSpec]:
    """对外（MCP）暴露的工具：方案 R5 只暴露无状态工具。"""
    return [spec for spec in TOOL_REGISTRY if spec.scope == SCOPE_EXTERNAL]
