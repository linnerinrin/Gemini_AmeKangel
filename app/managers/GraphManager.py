"""
managers/GraphManager.py  LangGraph图管理器
Abstract on IGraph
"""
import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from app.config.config import config
from app.interfaces.IGraph import IGraph

MAX_TOOL_TURNS = config.MAX_TOOL_TURNS
MAX_REFLECTION_TURNS = config.MAX_REFLECTION_TURNS


class ReflectionError(RuntimeError):
    """审查未通过；本轮草稿不得提交为正式历史。"""


class AgentState(TypedDict):
    agent_message:Annotated[List[dict],operator.add]  #agentnode节点自迭代传递的信息 仅用于收集  tooltext

    think_text:Annotated[List[str],operator.add] #思考链 agentnode的思考文本集合
    tool_text:Annotated[List[str],operator.add] #工具链 agentnode的工具调用集合
    reflection_text:Annotated[List[str],operator.add] #反思链 reflection的反思文本集合

    query:str #查询
    reply:str #回复

    tool_finish: bool    #是否完成工具调用
    tool_turns: int     #tool次数
    reflection_finish: bool    #是否完成reflection
    reflection_turns: int   #reflection次数
    reflection_review_retries: int  #单轮请求共享，跨重写最多一次审查重试
    stream_event: object    #可选的异步流式事件回调


class GraphManager(IGraph):
    def __init__(self):
        super().__init__()
        self.app = None
        self.router=None

    def init(self,session):
        self.session=session

    def build(self):
        nodes = self.create_nodes()
        routers = self.create_routers()
        self.app = self.create_graph(nodes,routers, self.session.fcall_mgr.tools)
        return self.app

    #图节点
    async def agent_node(self, state: AgentState):
        ####    PART    决策阶段
        session = self.session
        msg= [
            {"role": "system", "content": session.call_prompt},
        ]


        ####    PART    工具调用 写思考给自己/回复节点
        tool_results = [m for m in state["agent_message"] if isinstance(m, ToolMessage)]
        if tool_results:
            tool_text = "\n".join(f"工具({m.name})返回：{m.content}" for m in tool_results)
            msg.append({"role": "system", "content": f"上一轮的思考结果：{state['think_text'][-1]}\n\n工具返回结果：\n{tool_text}\n请结合上述结果判断是否应该继续调用工具"})
        else:
            tool_text = ""
            msg.append({"role": "user", "content": state["query"]})

        emit = state.get("stream_event")
        if emit:
            await emit({"type": "node", "name": "agent"})
            async def decision_event(event):
                if event["type"] == "answer_delta":
                    event = {**event, "type": "thinking_delta"}
                await emit(event)
            output = await session.model_mgr.stream_generate(
                msg, session.config.think_para, use_tools=True, on_event=decision_event)
        else:
            output = await session.model_mgr.generate(msg, session.config.think_para, use_tools=True)
        if output.tool_calls and (getattr(output, "truncated", False) or
                                  getattr(output, "finish_reason", None) == "length"):
            raise ValueError("模型输出已截断，禁止执行工具调用")
        print(f"tool_calls:{output.tool_calls}")
        print(f"thinking_text:{output.content}")
        if output.tool_calls and state.get("stream_event"):
            # 工具名称已在模型流中发过一次，这里仅发送阶段事件，不重复流式参数。
            await state["stream_event"]({"type": "tool_calls_ready", "calls": [c["name"] for c in output.tool_calls]})
        return {
            "agent_message": [AIMessage("", tool_calls=output.tool_calls)],

            "think_text": [getattr(output, "thinking", "") or output.content],
            "tool_text": [tool_text],

            "tool_finish": not bool(output.tool_calls),
            "tool_turns": state.get("tool_turns", 0) + 1,

            "reflection_finish": False,
        }

    async def generate_reply_node(self, state: AgentState):

        ####  PART 写入状态/长期记忆 提示词
        session = self.session
        session.memory_mgr.load_status()
        status_prompt = session.memory_mgr.status_prompts
        memory_text = session.memory_mgr.load_long_memory()
        msg = [
            {"role": "system", "content": session.sys_prompt},
        ]
        if status_prompt:
            msg.append({"role": "system", "content": status_prompt})
        if memory_text:
            msg.append({"role": "system", "content": memory_text})


        ####  PART 接收reflection提示词
        if state.get("reflection_text"):
            msg.append({"role": "system", "content": f"上一轮回复需要改进：{state['reflection_text'][-1]}"})
        msg += list(session.short_memory)


        ####  PART    接收工具调用 查询
        if state.get("tool_text"):
            msg.append({"role": "system", "content": state["tool_text"][-1]})
        msg.append({"role": "user", "content": state["query"]})

        emit = state.get("stream_event")
        if emit:
            await emit({"type": "node", "name": "generate_reply"})
            # Each reflection retry replaces the previous provisional answer.
            await emit({"type": "answer_reset"})
            output = await session.model_mgr.stream_generate(
                msg, session.inference_para, use_tools=False, on_event=emit)
        else:
            output = await session.model_mgr.generate(msg, session.inference_para, use_tools=False)
        if not output.content or not output.content.strip():
            raise ValueError("模型未返回回答正文，请重试")
        if getattr(output, "truncated", False) and state.get("stream_event"):
            # 被长度上限截断时只提示，不中断，也不丢弃已生成的内容
            await state["stream_event"]({"type": "thinking_delta", "text": "\n（本轮回答达到长度上限，已被截断）\n"})
        return {"reply": output.content}

    async def reflection_node(self, state: AgentState):
        session = self.session
        last_reply = state.get("reply")
        reply_text = last_reply["content"] if isinstance(last_reply, dict) else str(last_reply)
        messages = [
            {"role": "system", "content": session.reflection_prompt},
            {"role": "user", "content": f"用户查询：{state['query']}\n\n角色回复：{reply_text}"},
        ]
        emit = state.get("stream_event")
        if emit:
            await emit({"type": "node", "name": "reflection"})
        review_retries = state.get("reflection_review_retries", 0)
        while True:
            output = await session.model_mgr.generate(messages, session.config.reflect_para, use_tools=False)
            verdict = (output.content or "").strip()
            complete = not getattr(output, "truncated", False) and getattr(output, "finish_reason", None) == "stop"
            accepted = complete and verdict == "__ACCEPT__"
            rejected = complete and verdict.startswith("需要修改：") and bool(verdict[len("需要修改："):].strip())
            if accepted or rejected:
                break
            if review_retries >= 1:
                raise ReflectionError("审查结果为空、截断或格式异常，单次审查重试额度已耗尽")
            review_retries += 1
            if emit:
                await emit({"type": "thinking_delta", "text": "\n审查结果无效，正在进行本轮唯一一次审查重试\n"})
            messages.append({"role": "system", "content":
                "上一份审查结果无效。请重新审查，仅输出完整的 __ACCEPT__，或 需要修改：具体问题。不得输出其他格式。"})
        if emit:
            await emit({"type": "thinking_delta", "text":
                "\n审查结果：通过\n" if accepted else f"\n审查结果：{verdict}\n"})
        if accepted:
            return {"reflection_finish": True, "reflection_review_retries": review_retries}
        turns = state.get("reflection_turns", 0) + 1
        if turns > MAX_REFLECTION_TURNS:
            raise ReflectionError("审查明确不通过，回复重写次数已耗尽")
        return {
            "reflection_finish": False,
            "reflection_text": [verdict],
            "reflection_turns": turns,
            "reflection_review_retries": review_retries,
        }

    def create_nodes(self):
        return {
            "agent": self.agent_node,
            "generate_reply": self.generate_reply_node,
            "reflection": self.reflection_node,
        }

    #路由
    def toolcall_router(self, state):
        if state.get("tool_finish"):
            return "generate_reply"
        if state.get("tool_turns", 0) >= MAX_TOOL_TURNS:
            return "generate_reply"
        return "tools"

    def reflection_router(self, state):
        if state.get("reflection_finish"):
            return END
        if state.get("reflection_turns", 0) > MAX_REFLECTION_TURNS:
            raise ReflectionError("审查明确不通过，回复重写次数已耗尽")
        return "generate_reply"

    def create_routers(self):
        return {
            "agent_generate":self.toolcall_router,
            "generate_reflection":self.reflection_router,
        }

    #图构建
    def create_graph(self, nodes,routers, tools):
        builder = StateGraph(AgentState)
        builder.add_node("agent", nodes["agent"])
        builder.add_node("generate_reply", nodes["generate_reply"])
        tool_node = ToolNode(tools, messages_key="agent_message")
        async def run_tools(state):
            if state.get("stream_event"):
                await state["stream_event"]({"type": "node", "name": "tools"})
            return await tool_node.ainvoke(state)
        builder.add_node("tools", run_tools)
        builder.add_node("reflection", nodes["reflection"])

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", routers["agent_generate"])
        builder.add_edge("tools", "agent")
        builder.add_edge("generate_reply", "reflection")
        builder.add_conditional_edges("reflection", routers["generate_reflection"])

        return builder.compile()

    def load_mode(self, mode):
        self.current_mode=mode
