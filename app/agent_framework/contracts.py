#"""
#架构作用：新增agent框架的统一契约层，定义只读/回复/上下文/模型/MCP各类端口
#调用方：app/agent_framework下各模块 + app/managers与app/api的接入点（见各默认实现处的引用标注）
#数据流：端口定义→各模块默认实现→接入方在调用点直接实例化使用
#"""
"""
agent_framework/contracts.py框架统一契约（冻结版）

本文件是全框架唯一允许被所有模块共同依赖的抽象层：
-端口（Protocol/ABC）只描述「需要什么能力」，不知道实现长什么样；
-默认数据类（不可变）描述模块间交换的形状；
-各模块（readonly / response / context_budget / model_adapter / mcp_client）只依赖本文件，
模块之间不得互相import，保证依赖方向单向、无环

依赖方向：具体模块→ contracts（抽象），contracts不依赖任何实现，符合依赖倒置
冻结说明：本文件在实现期冻结；任何修改必须由主Agent统一进行并通知所有分派方
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


#"""
#模块一：只读工具端口
#路径安全判定与文件读取分离；读取结果统一走ReadResult形状
#"""


@runtime_checkable
class PathGuardPort(Protocol):
    """路径安全判定端口：realpath前缀校验+黑名单，拒绝越界与逃逸。"""

    def resolve(self, path: str) -> "ResolvedPath":
        """把用户输入路径解析为安全结果。

        返回ResolvedPath；任何越界/黑名单/逃逸情况都不抛异常，
        通过ResolvedPath.allowed=False与reason表达。
        """
        ...


@dataclass(frozen=True)
class ResolvedPath:
    """路径判定结果：解析后的绝对路径+是否放行+拒绝原因。"""

    absolute: str = ""     #realpath后的绝对路径；拒绝时为空串
    allowed: bool = False
    reason: str = ""       #拒绝原因（越界/黑名单/符号链接等），放行时为空串


@dataclass(frozen=True)
class ReadResult:
    """只读工具统一返回形状：内容+元信息，截断显式可见。"""

    ok: bool               #是否成功读取
    content: str = ""      #文本内容（可能被截断）
    reason: str = ""       #失败/拒绝原因，成功时为空串
    total_lines: int = 0   #文件总行数；截断时调用方可见原始规模
    truncated: bool = False  #是否因超上限被截断


#"""
#模块三：回复检查端口
#确定性检查（句数/字数/格式），零成本不过即打回；口吻判断不在本层
#"""


@runtime_checkable
class BrevityCheckerPort(Protocol):
    """回复简短性检查端口：纯函数判定，无副作用。"""

    def check(self, text: str) -> "BrevityVerdict":
        """检查回复文本。"""
        ...


@dataclass(frozen=True)
class BrevityVerdict:
    """检查结果：是否通过+违规项列表（供打回重生成时说明原因）。"""

    passed: bool
    violations: List[str] = field(default_factory=list)  #如["句数超限", "含列表"]
    char_count: int = 0
    sentence_count: int = 0


#"""
#模块四：上下文预算端口
#计数与窗口策略分离；tiktoken为可选增强，默认实现不得强依赖它
#"""


@runtime_checkable
class TokenCounterPort(Protocol):
    """token计数端口：只负责数数，不负责裁剪。"""

    def count(self, text: str) -> int:
        """返回文本的token估算数；实现必须保证任何输入都有结果。"""
        ...


@runtime_checkable
class WindowPolicyPort(Protocol):
    """窗口策略端口：把「轮数上限」统一换算成「消息条数上限」。

    对接原项目MemoryManager.py:81的轮/消息语义不一致问题：
    SQL LIMIT数「轮」、deque maxlen数「消息」，本端口给出唯一换算口径。
    """

    def message_limit(self, turn_limit: int) -> int:
        """轮数上限 → 消息条数上限（默认一轮= user + assistant两条）。"""
        ...


@runtime_checkable
class AuthVerifierPort(Protocol):
    """鉴权校验端口：校验请求携带的凭据是否有效。

    默认实现为静态token比对（AUTH_TOKEN），不引入python-jose；
    未来如需JWT，新增实现替换即可，调用方不变。
    """

    def verify(self, credential: Optional[str]) -> bool:
        """凭据有效返回True；None或无效返回False，不抛异常。"""
        ...


#"""
#模块五：会话注册端口
#只声明「按身份取会话」的能力；ChatSession的具体类型由适配器决定，
#这里用Any持有，避免契约层反向依赖services包
#"""


class SessionRegistryPort(ABC):
    """会话映射端口：session_id → 会话实例+每会话一把生成锁。

    抽象类而非Protocol：需要实现方显式承诺线程/协程安全语义。
    """

    @abstractmethod
    def get(self, session_id: str) -> Optional[Any]:
        """按session_id取已绑定的会话实例；不存在返回None。"""
        ...

    @abstractmethod
    def get_or_create(self, session_id: str) -> Any:
        """按session_id取会话，不存在则经注入工厂惰性创建。"""
        ...

    @abstractmethod
    def lock_for(self, session_id: str) -> Any:
        """返回该会话专属的asyncio锁（按会话隔离生成互斥）。"""
        ...


#"""
#模块六：模型适配端口
#统一聊天调用抽象：调用方只认ChatClientPort，不绑死任何厂商SDK/HTTP细节
#"""


@dataclass(frozen=True)
class ModelProfile:
    """一次模型调用的完整配置：地址、模型、凭据、参数。

    对接原项目services/llm_settings.py的resolve_profile输出语义；
    api_key允许为空（由实现决定如何报错或拒绝，不得在日志中打印）。
    """

    base_url: str          #OpenAI兼容端点，如https://api.deepseek.com
    model: str             #模型名
    api_key: str = ""
    temperature: float = 0.8
    timeout: float = 60.0  #秒
    #采样上限：0表示不显式传入（与config._build_para的“0即省略”语义一致）
    max_tokens: int = 0
    top_p: float = 0.0


@dataclass(frozen=True)
class ChatReply:
    """一次聊天补全的统一返回形状，厂商差异在本层抹平。"""

    content: str           #文本内容（非流式聚合结果）
    model: str = ""        #实际使用的模型
    finish_reason: str = ""  #stop / length /厂商原值
    usage_tokens: int = 0  #总token数；厂商未返回时为0


@runtime_checkable
class ChatClientPort(Protocol):
    """聊天补全端口：只描述「发消息、拿回复」，不关心传输实现。"""

    def complete(self, messages: Sequence[Dict[str, str]], *,
                 profile: ModelProfile) -> ChatReply:
        """发送messages（OpenAI chat形状）并返回聚合回复。

        实现失败必须抛ModelClientError，不得返回半成品或静默吞错。
        """
        ...


class ModelClientError(Exception):
    """模型调用统一异常：携带可读原因，调用方按此做降级/重试判断。"""


#"""
#模块七：MCP客户端端口
#外部MCP server的只读工具接入：端口先行，真实传输实现后补
#"""


class McpClientError(Exception):
    """MCP调用统一异常：连接/协议/工具执行失败都归一到此处。"""


@dataclass(frozen=True)
class McpToolDescriptor:
    """一个MCP工具的描述快照：名称、说明、参数模式、只读标记。"""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)  #JSON Schema
    read_only: bool = False  #由server注解或名称推断；False时按写工具处理
    server: str = ""         #来源server标识


@runtime_checkable
class McpClientPort(Protocol):
    """MCP客户端端口：列工具与调工具两件事，不关心stdio/HTTP传输。"""

    def list_tools(self) -> List[McpToolDescriptor]:
        """枚举server暴露的工具；失败抛McpClientError。"""
        ...

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用指定工具并返回文本结果；失败抛McpClientError。"""
        ...

#BrevityCheckerPort(:130-136)与AuthVerifierPort(:177-184)当前无实现/调用方，属预留契约
