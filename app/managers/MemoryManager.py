"""
managers/MemoryManager.py  记忆模块管理器
Abstract on IMemory
"""
import json
from collections import deque
from pathlib import Path
from app.config.status_rules import plan_status_change
from app.config.config import config
from app.config.thinking_trace import decode as decode_thinking, encode as encode_thinking
from app.interfaces import IMemory
from repository.conn import db

# 状态栏默认值由 config 统一管理：followers / affection / stress / mental = 0 / 50 / 0 / 0
STATUS_DEFAULTS = {
    "followers": config.DEFAULT_FOLLOWERS,
    "affection": config.DEFAULT_AFFECTION,
    "stress": config.DEFAULT_STRESS,
    "mental": config.DEFAULT_MENTAL,
}

# Schema migration is performed during application startup, not during refresh.

# 记忆模块 mysql session 和 chat 两表
#创建会话->创建session实例-> 看time字段 大于3报错->随机一个id->memory 空
#聊天->创建一个chat实例->存query reply
#短期记忆 where sissionid="" order by created_at 取前len query reply
#存长期记忆session实例 update memory
#状态变化 update status
#加载聊天页面chat where sessionid=""  order by created_at

class MemoryManager(IMemory):
    def __init__(self):
        super().__init__()
        self.session_id=None
        self.history=[]
        self.model_mgr=None
        self.status=None
        self.status_prompts=""
        self.mode = "0"
        # 任务轴（chat=闲聊 / work=工作），与场景轴 mode 正交，决定短期记忆走哪条上下文
        self.task_mode = "chat"

    def init(self,model_mgr):
        self.model_mgr=model_mgr


    def bind_session_id(self,session_id):
        self.session_id = session_id


    def set_task_mode(self, task_mode):
        """设置任务轴；非法值直接报错，避免把脏值写进 chat.task_mode。"""
        task_mode = str(task_mode)
        if task_mode not in ("chat", "work"):
            raise ValueError(f"未知对话模式: {task_mode}")
        self.task_mode = task_mode


    def update_short_memory(self, query, reply, thinking=None, model=None, provider=None):
        # thinking 存结构化思考链 JSON；没有思考内容时写 NULL
        # model/provider 记本轮实际使用的模型；拿不到就是 NULL，不能因为缺模型名挡住落库
        db("insert into chat (session_id, query, reply, mode, thinking, task_mode, llm_model, llm_provider) values (%s, %s, %s, %s, %s, %s, %s, %s)",
           (self.session_id, query, reply, self.mode, encode_thinking(thinking), self.task_mode, model, provider))
        db.commit()
    def load_short_memory(self):
        self.short_memory.clear()
        # 决策 R7：聊天与工作各自独立上下文，短期记忆要同时按场景轴与任务轴过滤
        db('select query,reply from chat where session_id=%s and mode=%s and task_mode=%s order by created_at desc limit %s;',(self.session_id,self.mode,self.task_mode,config.SHORT_MEMORY_LEN))
        for res in reversed(db.fetchall()):
            self.short_memory.append({"role":"user","content":res[0]})
            self.short_memory.append({"role": "assistant", "content":res[1]})

    async def update_long_memory(self, memory: str):
        messages = [
            {"role": "system", "content": "总结以下对话记忆，保留重要信息："},
            {"role": "user", "content": f"原有记忆：{self.long_memory or '无'}\n新内容：{memory}\n简洁总结："},
        ]
        result = await self.model_mgr.generate(messages, config.think_para, use_tools=False)
        self.long_memory = str(result)
        db('update session set memory=%s where id=%s;', (self.long_memory, self.session_id))
        db.commit()


    def load_long_memory(self):
        db('select memory from session where id=%s;', (self.session_id,))
        self.row = db.fetchone()
        self.long_memory = self.row[0] if self.row else None
        return self.long_memory

    def update_status(self, followers=None, affection=None, stress=None, mental=None):
        """按增量更新状态栏；followers 当前只读，不接入随机状态变化。"""
        updates = (
            ("followers", followers, False),
            ("affection", affection, True),
            ("stress", stress, True),
            ("mental", mental, True),
        )
        for name, amount, bounded in updates:
            if amount is None or amount == 0:
                continue
            expression = f"greatest(0,least(100,{name}+%s))" if bounded else f"{name}+%s"
            db(f"update session set {name}={expression} where id=%s;", (amount, self.session_id))
        db.commit()
        self.load_status()

    def apply_status_change(self, amounts, rng=None):
        deltas = plan_status_change(amounts, rng=rng)
        if deltas:
            self.update_status(**deltas)
        return deltas

    def load_status(self):
        db("select followers,affection,stress,mental from session where id=%s;", (self.session_id,))
        row = db.fetchone()
        values = row or (None, None, None, None)
        self.fol = values[0] if values[0] is not None else STATUS_DEFAULTS["followers"]
        self.aff = values[1] if values[1] is not None else STATUS_DEFAULTS["affection"]
        self.ste = values[2] if values[2] is not None else STATUS_DEFAULTS["stress"]
        self.men = values[3] if values[3] is not None else STATUS_DEFAULTS["mental"]
        self.set_status_prompts()
        self.status = {
            "followers": self.fol,
            "affection": self.aff,
            "stress": self.ste,
            "mental": self.men,
        }
        return self.status

    def read_history(self, mode=None):
        """读取指定 mode 的历史记录（只读，不改变当前记忆状态）

        每轮自带 task_mode，前端据此区分聊天/工作；本阶段只透传，不做分段渲染。
        每轮还带回当时实际使用的 llm_model/llm_provider（老数据为 None）。
        """
        target_mode = self.mode if mode is None else str(mode)
        db('select query,reply,created_at,thinking,task_mode,llm_model,llm_provider from chat where session_id=%s and mode=%s order by created_at asc;',(self.session_id,target_mode))
        return [
            {"created_at": str(res[2]), "user": res[0], "assistant": res[1],
             "thinking": decode_thinking(res[3]), "task_mode": res[4],
             "llm_model": res[5], "llm_provider": res[6]}
            for res in db.fetchall()
        ]

    def load_history(self, mode=None):
        self.history = self.read_history(mode)

    def set_status_prompts(self):
        self.status_prompts = ""
        if self.fol<1000:self.status_prompts += "你是一个底边主播。"
        elif self.fol<10000:self.status_prompts += "你的直播事业有所起色。"
        elif self.fol<100000:self.status_prompts += "你是一个比较出名的主播。"
        elif self.fol<1000000:self.status_prompts += "你是当今人气最高的主播之一。"
        else:self.status_prompts += "你是全天下最顶级的主播。"

        if self.aff<20:self.status_prompts += "你现在对用户很厌恶。"
        elif self.aff<40:self.status_prompts += "你现在不太喜欢用户。"
        elif self.aff<60:self.status_prompts += "你对用户感觉一般。"
        elif self.aff<80:self.status_prompts += "你对用户有些好感。"
        else:self.status_prompts += "你非常喜欢用户。"

        if self.ste<20:self.status_prompts += "你现在的精神压力不大。"
        elif self.ste<40:self.status_prompts += "你感到有些压力。"
        elif self.ste<60:self.status_prompts += "你有比较大的压力。"
        elif self.ste<80:self.status_prompts += "你感到非常焦虑。"
        else:self.status_prompts += "你已经在崩溃的边缘了。"

        if self.men<20:self.status_prompts += "你的心情还不错。"
        elif self.men<40:self.status_prompts += "你有一点消沉。"
        elif self.men<60:self.status_prompts += "你的精神状态不太好。"
        elif self.men<80:self.status_prompts += "你的内心非常阴暗。"
        else:self.status_prompts += "黑暗的情绪正吞没你。"

    def load_mode(self,mode):
        self.mode=mode
        self.refresh(self.session_id)


    def refresh(self,session_id):
        db('use gemini;')
        db('select id from session where id=%s;', (session_id,))
        self.session_info = db.fetchone()
        if not self.session_info:
            raise ValueError()
        self.session_id=session_id
        self.short_memory=deque(maxlen=config.SHORT_MEMORY_LEN)
        self.load_long_memory()
        self.load_status()
        self.load_history()
        self.load_short_memory()
        self.set_status_prompts()