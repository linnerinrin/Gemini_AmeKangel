"""
services/chat_session.py  会话服务
"""

from collections import deque
from app.config.config import config
from app.config.emoji_labels import normalize_emoji_text
from app.config.thinking_trace import append_event, from_texts
from app.managers import ModelManager, FuncCallManager, RAGManager, MemoryManager
from app.managers.GraphManager import GraphManager, ReflectionError


class ChatSession:
    def __init__(self, model_mgr:ModelManager, fcall_mgr:FuncCallManager, rag_mgr:RAGManager, memory_mgr:MemoryManager,graph_mgr:GraphManager):
        #配置文件与提示词
        self.config = config
        self.mem_len = config.SHORT_MEMORY_LEN
        self.current_mode = config.INITIAL_MODE
        # 任务轴（chat=闲聊 / work=工作）：与场景轴 current_mode 正交，按轮次透传
        self.current_task_mode = "chat"
        self.status_prompt=None

        #各个模块管理器
        self.model_mgr = model_mgr
        self.fcall_mgr = fcall_mgr
        self.rag_mgr = rag_mgr
        self.memory_mgr = memory_mgr
        self.graph_mgr=graph_mgr

        #提示词装配：三个节点各读一个属性，先按 memory 里已有的任务轴落定一次
        self._sync_task_mode_from_memory()
        self._assemble_prompts()

        #初始化
        #热加载懒加载的模块
        self.fcall_mgr.init(self.rag_mgr, self.memory_mgr, self.model_mgr)
        self.graph_mgr.init(self)
        self.memory_mgr.init(self.model_mgr)

        #graph初始化
        self.app = self.graph_mgr.build()

        #memory初始化
        self.short_memory = deque(maxlen=self.mem_len)
        self.long_memory = ""
        self.status = None
        self.history = []
        self.session_id = None

        #model初始化
        self.inference_para = self.config.inference_para

    def bind_session_id(self,session_id):
        self.memory_mgr.bind_session_id(session_id)
        self.current_mode = config.INITIAL_MODE
        self.load_mode(self.current_mode)
        self.refresh(session_id)


    def clear_session(self):
        self.session_id = None
        self.memory_mgr.bind_session_id(None)
        self.short_memory.clear()
        self.long_memory = ""
        self.status = None
        self.history = []


    def _sync_task_mode_from_memory(self):
        """任务轴以 MemoryManager 为准（P1 的唯一状态源），不要在这里另起一套。"""
        task_mode = getattr(self.memory_mgr, "task_mode", None) or self.current_task_mode
        self.current_task_mode = self.validate_task_mode(task_mode)

    def _assemble_prompts(self):
        """按 (场景轴 mode, 任务轴 task_mode) 重新装配三个节点各自读的属性。

        GraphManager 的三个节点分别读 session.call_prompt / session.sys_prompt /
        session.reflection_prompt，所以两轴任一变化都要在这里刷新一次。
        """
        task_mode = self.current_task_mode
        # 顺序固定为 base -> role -> task
        self.sys_prompt = self.config.sys_prompt_for(self.current_mode, task_mode)
        self.call_prompt = self.config.call_prompt_for(task_mode)
        self.reflection_prompt = self.config.reflection_prompt_for(task_mode)

    def refresh(self,session_id):
        self.session_id=session_id
        self.memory_mgr.refresh(session_id)
        self.short_memory = self.memory_mgr.short_memory
        self.long_memory = self.memory_mgr.long_memory
        self.status=self.memory_mgr.status
        self.history = self.memory_mgr.history
        self.status_prompts=self.memory_mgr.status_prompts
        # 任务轴以 MemoryManager 为准，refresh 后同步回来，避免两边不一致
        self.current_task_mode = self.memory_mgr.task_mode
        self._assemble_prompts()

    def read_history(self, mode=None):
        """读取指定 mode 的历史记录（只读，供前端按直播主题拉取各自的上下文）"""
        return self.memory_mgr.read_history(mode)

    def last_model_meta(self):
        """本轮实际使用的 (provider, model)；ModelManager 每次生成后刷新，取不到就是 None。"""
        return (getattr(self.model_mgr, "last_provider", None),
                getattr(self.model_mgr, "last_model", None))

    def update_memory(self,prompt,reply,thinking=None,model=None,provider=None):
        # model/provider 记这一轮实际用的厂商与版本，缺省写 NULL（不因为拿不到模型名报错）
        self.memory_mgr.update_short_memory(prompt, reply, thinking, model=model, provider=provider)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.rag_mgr.update_context([[{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]])
        self.refresh(self.session_id)

    def load_mode(self, mode: str):
        """
        模式加载函数 通过不同字符串传入改变提示词。
        """
        mode = str(mode)
        if mode not in self.config.SYS_PROMPT:
            raise ValueError(f"未知聊天模式: {mode}")
        self.current_mode = mode
        self.graph_mgr.load_mode(mode)
        self.memory_mgr.load_mode(mode)
        self.rag_mgr.load_mode(mode)
        self.fcall_mgr.load_mode(mode)
        # MemoryManager.refresh replaces these collections; drop stale mode references.
        self.short_memory = self.memory_mgr.short_memory
        self.long_memory = self.memory_mgr.long_memory
        self.history = self.memory_mgr.history
        self.status = self.memory_mgr.status
        # 场景轴变了：正文提示词要重装配（任务轴随之保持不变）
        self._assemble_prompts()


    def validate_task_mode(self, task_mode):
        """校验任务轴：None 表示沿用当前值；非法值直接报错，不要静默退回默认。

        注意任务轴与场景轴是两条正交的轴，这里不碰 current_mode 的切换逻辑。
        """
        if task_mode is None:
            return self.current_task_mode
        task_mode = str(task_mode)
        if task_mode not in ("chat", "work"):
            raise ValueError("未知对话模式")
        return task_mode

    async def single_chat(self, prompt, mode=None, task_mode=None):
        """按请求模式执行一次对话，避免三个直播依赖共享的全局 mode。"""
        # 任务轴要在 load_mode/begin_turn 之前落定：load_mode 会按轴重载短期记忆
        task_mode = self.validate_task_mode(task_mode)
        self.memory_mgr.set_task_mode(task_mode)
        self.current_task_mode = task_mode
        if mode is not None and str(mode) != self.current_mode:
            self.load_mode(str(mode))
        # 任务轴变了：三个节点的提示词都要换成对应版本
        self._assemble_prompts()
        self.fcall_mgr.begin_turn()
        initial_state = {
            "agent_message": [],
            "think_text": [],
            "tool_text": [],
            "reflection_text": [],
            "query": prompt,
            "reply": "",
            "tool_finish": False,
            "tool_turns": 0,
            "reflection_finish": False,
            "reflection_turns": 0,
        }
        state_result = await self.app.ainvoke(initial_state)
        if state_result.get("reflection_finish") is not True:
            raise ReflectionError("回复未通过审查，禁止保存正式历史")
        reply = normalize_emoji_text(state_result["reply"])
        think_text = state_result.get("think_text", [])
        think = "\n".join(think_text) if isinstance(think_text, list) else str(think_text)
        # 非流式没有事件可收集，把纯文本思考包成文本项一起入库
        provider, model = self.last_model_meta()
        self.update_memory(prompt, reply, from_texts(think_text if isinstance(think_text, list) else [think]),
                           model=model, provider=provider)
        return reply, think

    async def stream_chat(self, prompt, mode=None, on_event=None, task_mode=None):
        """通过同一张图执行流式对话，保证三个直播和 JINE 共用相同编排逻辑。"""
        # 与 single_chat 相同：任务轴先落定，再按场景轴加载，最后才开始这一轮
        task_mode = self.validate_task_mode(task_mode)
        self.memory_mgr.set_task_mode(task_mode)
        self.current_task_mode = task_mode
        if mode is not None and str(mode) != self.current_mode:
            self.load_mode(str(mode))
        # 任务轴变了：三个节点的提示词都要换成对应版本
        self._assemble_prompts()
        self.fcall_mgr.begin_turn()
        # 一边把事件转给前端，一边累积结构化思考链，落库后历史回放才能还原同样的折叠块
        trace_items = []

        async def capture(event):
            append_event(trace_items, event)
            if on_event:
                await on_event(event)

        initial_state = {
            "agent_message": [], "think_text": [], "tool_text": [], "reflection_text": [],
            "query": prompt, "reply": "", "tool_finish": False, "tool_turns": 0,
            "reflection_finish": False, "reflection_turns": 0, "stream_event": capture,
        }
        if on_event:
            await on_event({"type": "status", "stage": "thinking"})
        state_result = await self.app.ainvoke(initial_state)
        if state_result.get("reflection_finish") is not True:
            raise ReflectionError("回复未通过审查，禁止保存正式历史")
        reply = normalize_emoji_text(state_result["reply"])
        think_text = state_result.get("think_text", [])
        think = "\n".join(think_text) if isinstance(think_text, list) else str(think_text)
        provider, model = self.last_model_meta()
        self.update_memory(prompt, reply, trace_items, model=model, provider=provider)
        if on_event:
            await on_event({"type": "done", "answer": reply, "think": think})
        return reply, think

    async def single_tweet(self):
        ...


    async def test_chat(self):
        while True:
            prompt=input('you:')
            if prompt=='change':
                mode='1' if self.current_mode=='0' else '0'
                self.load_mode(mode)
                continue
            await self.single_chat(prompt)