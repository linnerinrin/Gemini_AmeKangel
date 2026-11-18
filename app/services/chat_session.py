"""
services/chat_session.py
"""

"""
架构作用：聊天API的会话门面，将管理器、提示词装配、图执行和历史提交串起来
数据流：绑定存档→按场景装配→图生成及审查→持久化→ RAG/内存刷新
当前按session_id一实例：SessionRegistry惰性创建并持有，各manager的
模式/记忆/RAG索引均为会话私有；仅ModelManager进程级共享（并发由每会话锁串行化）。
ChatFacade门面只移动当前指针，跨await的调用方须先取resolve_current()快照
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
        self.status_prompt=None

        #各个模块管理器
        self.model_mgr = model_mgr
        self.fcall_mgr = fcall_mgr
        self.rag_mgr = rag_mgr
        self.memory_mgr = memory_mgr
        self.graph_mgr=graph_mgr

        #提示词装配：三个节点各读一个属性
        self._assemble_prompts()

        #初始化
        #热加载懒加载的模块
        self.fcall_mgr.init(self.rag_mgr, self.memory_mgr, self.model_mgr)
        self.graph_mgr.init(self)
        self.memory_mgr.init(self.model_mgr)

        #graph初始化
        self.app = self.graph_mgr.build()

        #memory初始化
        #mem_len是「轮」上限，deque按消息条数开窗（一轮=两条消息），
        #与MemoryManager.short_memory同一口径；本队列随后会被memory_mgr的队列替换；
        #替身（FakeMemory）没有window_policy时回落旧口径，保证测试契约不变
        _policy = getattr(self.memory_mgr, "window_policy", None)
        self.short_memory = deque(maxlen=_policy.message_limit(self.mem_len) if _policy else self.mem_len)
        self.long_memory = ""
        self.status = None
        self.history = []
        self.session_id = None

        #model初始化
        self.inference_para = self.config.inference_para

    """
    先把ID写入记忆管理器，再加载默认场景及刷新，成功后会话才完全同步
    """
    def bind_session_id(self,session_id):
        self.memory_mgr.bind_session_id(session_id)
        self.rag_mgr.bind_session_id(session_id)
        self.current_mode = config.INITIAL_MODE
        self.load_mode(self.current_mode)
        self.refresh(session_id)


    """
    清空会话可见缓存但不重建管理器；RAG context未清，因此不能视作完整隔离重置
    """
    def clear_session(self):
        deleted_id=self.session_id
        self.session_id = None
        self.memory_mgr.bind_session_id(None)
        self.rag_mgr.bind_session_id(None)
        self.short_memory.clear()
        self.long_memory = ""
        self.status = None
        self.history = []
        #只清被删存档的context文档，其他存档的聊天索引保留
        self.rag_mgr.clear_context(deleted_id)


    def _assemble_prompts(self):
        """按场景轴mode重新装配三个节点各自读的属性。

        GraphManager的三个节点分别读session.call_prompt / session.sys_prompt /
        session.reflection_prompt，所以场景变化都要在这里刷新一次。
        """
        #顺序固定为base -> role
        self.sys_prompt = self.config.sys_prompt_for(self.current_mode)
        self.call_prompt = self.config.call_prompt_for()
        self.reflection_prompt = self.config.reflection_prompt_for()

    """
    刷新后集合直接引用MemoryManager的对象，不是副本；模式重载替换队列时必须同步引用
    先改self.session_id再查询也意味着异常不是原子失败，需上层统一绑定事务
    """
    def refresh(self,session_id):
        self.session_id=session_id
        self.memory_mgr.refresh(session_id)
        self.short_memory = self.memory_mgr.short_memory
        self.long_memory = self.memory_mgr.long_memory
        self.status=self.memory_mgr.status
        self.history = self.memory_mgr.history
        self.status_prompts=self.memory_mgr.status_prompts
        self._assemble_prompts()

    """
    只读转发历史查询，不切换当前场景；返回的记录用于前端回放
    """
    def read_history(self, mode=None):
        """读取指定mode的历史记录（只读，供前端按直播主题拉取各自的上下文）"""
        return self.memory_mgr.read_history(mode)

    """
    读取最近模型调用元信息供历史落库；多节点共用管理器，值反映最后一次生成调用
    """
    def last_model_meta(self):
        """最近一次实际使用的(provider, model)；ModelManager每次生成后刷新，取不到就是None。"""
        return (getattr(self.model_mgr, "last_provider", None),
                getattr(self.model_mgr, "last_model", None))

    """
    提交顺序：SQL历史commit →内存追加→ RAG embedding/更新→全量刷新
    """
    def update_memory(self,prompt,reply,thinking=None,model=None,provider=None):
        #model/provider记这一轮实际用的厂商与版本，缺省写NULL（不因为拿不到模型名报错）
        self.memory_mgr.update_short_memory(prompt, reply, thinking, model=model, provider=provider)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.rag_mgr.update_context([[{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]])
        self.refresh(self.session_id)

    """
    校验场景后依次通知图、记忆、RAG、工具，再同步集合引用和提示词
    多个对象按顺序修改，没有原子发布；中途IO/模型失败可能留下部分更新状态
    RAG重建知识库是同步重型操作，这个调用不会因外层async而自动移出事件循环
    """
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
        #MemoryManager.refresh replaces these collections; drop stale mode references.
        self.short_memory = self.memory_mgr.short_memory
        self.long_memory = self.memory_mgr.long_memory
        self.history = self.memory_mgr.history
        self.status = self.memory_mgr.status
        #场景轴变了：正文提示词要重装配
        self._assemble_prompts()


    """
    执行完整图，必须通过审查才写正式问答；工具的数据库副作用在审查前可能已提交
    """
    async def single_chat(self, prompt, mode=None):
        """按请求模式执行一次对话，避免三个直播依赖共享的全局mode。"""
        if mode is not None and str(mode) != self.current_mode:
            self.load_mode(str(mode))
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
        #非流式没有事件可收集，把纯文本思考包成文本项一起入库
        provider, model = self.last_model_meta()
        self.update_memory(prompt, reply, from_texts(think_text if isinstance(think_text, list) else [think]),
                           model=model, provider=provider)
        self._settle_followers()
        return reply, think

    """
    与普通路径共享图；capture先累积结构化轨迹，再await转发，慢消费者会形成背压
    图通过审查后同步落库/索引，成功才发done；失败前已发送的增量仍是过程信息
    """
    async def stream_chat(self, prompt, mode=None, on_event=None):
        """通过同一张图执行流式对话，保证三个直播和JINE共用相同编排逻辑。"""
        if mode is not None and str(mode) != self.current_mode:
            self.load_mode(str(mode))
        self.fcall_mgr.begin_turn()
        #一边把事件转给前端，一边累积结构化思考链，落库后历史回放才能还原同样的折叠块
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
        self._settle_followers()
        if on_event:
            await on_event({"type": "done", "answer": reply, "think": think})
        return reply, think

    """
    直播档（mode非0）每轮生成成功后按概率涨粉；私聊档不增长
    替身（FakeMemory）没有apply_followers_gain时跳过，保证测试契约不变
    """
    def _settle_followers(self):
        if self.current_mode == "0":
            return
        settle = getattr(self.memory_mgr, "apply_followers_gain", None)
        if callable(settle):
            settle()

    """
    预留推文接口，目前省略号不执行生成或持久化，await后隐式返回None
    """
    async def single_tweet(self):
        ...
