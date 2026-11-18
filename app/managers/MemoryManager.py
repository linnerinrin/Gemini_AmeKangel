"""
managers/MemoryManager.py记忆模块管理器
"""


"""
架构作用：ChatSession与工具使用的记忆/角色状态门面，直接访问session、chat表
数据流：存档/场景/任务轴→ SQL →短记忆队列、长记忆、历史、状态提示词；写入显式提交
本对象由包级单例持有，字段随绑定与刷新改变，不是不可变的请求快照
"""

import json
from collections import deque
from pathlib import Path
from app.agent_framework.context_budget.counter import TiktokenCounter
from app.agent_framework.context_budget.window import TurnWindowPolicy
from app.config.status_rules import plan_followers_gain, plan_status_change
from app.config.config import config
from app.config.thinking_trace import decode as decode_thinking, encode as encode_thinking
from repository.conn import db

#轮/消息口径统一策略（一轮=user+assistant两条）与token计数器
#tiktoken精确计数，未安装时TiktokenCounter内部静默回退字符估算
_WINDOW_POLICY = TurnWindowPolicy()
_TOKEN_COUNTER = TiktokenCounter()

#状态栏默认值由config统一管理：followers / affection / stress / mental = 0 / 50 / 0 / 0
STATUS_DEFAULTS = {
    "followers": config.DEFAULT_FOLLOWERS,
    "affection": config.DEFAULT_AFFECTION,
    "stress": config.DEFAULT_STRESS,
    "mental": config.DEFAULT_MENTAL,
}

#记忆模块mysql session和chat两表
#创建会话->创建session实例->看time字段大于3报错->随机一个id->memory空
#聊天->创建一个chat实例->存query reply
#短期记忆where sissionid="" order by created_at取前len query reply
#存长期记忆session实例update memory
#状态变化update status
#加载聊天页面chat where sessionid=""  order by created_at

class MemoryManager:
    def __init__(self):
        super().__init__()
        self.session_id=None
        #短记忆队列类型与refresh一致；refresh前调用load_short_memory不会再AttributeError
        #maxlen按窗口策略换算为消息条数（SHORT_MEMORY_LEN是「轮」上限，
        #SQL LIMIT数轮、deque数消息，原同一数值两种口径会让deque只留一半轮次）
        self.short_memory=deque(maxlen=_WINDOW_POLICY.message_limit(config.SHORT_MEMORY_LEN))
        #token计数端口（默认tiktoken精确计数，未装回退字符估算）；当前只供预算核算，
        #截断策略保持原行为（deque窗口），不做基于token的裁剪
        self.token_counter=_TOKEN_COUNTER
        self.window_policy=_WINDOW_POLICY
        self.history=[]
        self.model_mgr=None
        self.status=None
        self.status_prompts=""
        self.mode = "0"

    def init(self,model_mgr):
        self.model_mgr=model_mgr


    """
    这里只赋值，不查库；合法性留给refresh，因此失败之前已经改变当前绑定
    """
    def bind_session_id(self,session_id):
        self.session_id = session_id


    """
    将一轮完整问答与场景、思考链和模型元信息写入chat，立即commit
    """
    def update_short_memory(self, query, reply, thinking=None, model=None, provider=None):
        #thinking存结构化思考链JSON；没有思考内容时写NULL
        #model/provider记当次实际使用的模型；拿不到就是NULL，不能因为缺模型名挡住落库
        db("insert into chat (session_id, query, reply, mode, thinking, llm_model, llm_provider) values (%s, %s, %s, %s, %s, %s, %s)",
           (self.session_id, query, reply, self.mode, encode_thinking(thinking), model, provider))
        db.commit()
    """
    按存档/场景筛选最近轮次，再反转为时间正序并拆成user/assistant消息
    """
    def load_short_memory(self):
        self.short_memory.clear()
        db('select query,reply from chat where session_id=%s and mode=%s order by created_at desc limit %s;',(self.session_id,self.mode,config.SHORT_MEMORY_LEN))
        for res in reversed(db.fetchall()):
            self.short_memory.append({"role":"user","content":res[0]})
            self.short_memory.append({"role": "assistant", "content":res[1]})

    """
    await模型生成摘要，GenerationResult.__str__提取正文，然后更新并提交session
    """
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

    """
    使用固定列名集合构造更新表达式，数值仍参数化；bounded三项由SQL钳制到0..100
    多列依次写入后一次commit，再加载状态及提示词；不在本层提供失败回滚
    """
    def update_status(self, followers=None, affection=None, stress=None, mental=None):
        """按增量更新状态栏；followers只增不减、无上限，其余三项钳制到0..100。"""
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

    def apply_followers_gain(self, rng=None):
        """直播模式每轮结算一次粉丝增长；返回实际增量，未命中为0。"""
        gain = plan_followers_gain(rng=rng)
        if gain:
            self.update_status(followers=gain)
        return gain

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

    """
    历史读取不改变当前场景；解码思考链并保留每轮模型元信息用于回放
    """
    def read_history(self, mode=None):
        """读取指定mode的历史记录（只读，不改变当前记忆状态）

        每轮带回当时实际使用的llm_model/llm_provider（老数据为None）。
        """
        target_mode = self.mode if mode is None else str(mode)
        db('select query,reply,created_at,thinking,llm_model,llm_provider from chat where session_id=%s and mode=%s order by created_at asc;',(self.session_id,target_mode))
        return [
            {"created_at": str(res[2]), "user": res[0], "assistant": res[1],
             "thinking": decode_thinking(res[3]),
             "llm_model": res[4], "llm_provider": res[5]}
            for res in db.fetchall()
        ]

    """
    以数据库结果整体替换历史快照；引用旧列表的调用方需通过ChatSession.refresh同步
    """
    def load_history(self, mode=None):
        self.history = self.read_history(mode)

    """
    将粉丝/好感/压力/精神数值按阈值映射成角色状态片段，供提示词装配使用
    这里只重建字符串，不调用模型；各维度分别追加，不能用其替代数据库事实
    """
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

    """
    更新场景轴并重新读取当前存档；任务轴是另一独立维度，决定短记忆过滤条件
    """
    def load_mode(self,mode):
        self.mode=mode
        self.refresh(self.session_id)


    """
    验证存档后重建队列并依次加载长记忆、状态、历史和短记忆；同步SQL无await
    """
    def refresh(self,session_id):
        db('use gemini;')
        db('select id from session where id=%s;', (session_id,))
        self.session_info = db.fetchone()
        if not self.session_info:
            raise ValueError()
        self.session_id=session_id
        #与__init__同一口径：SHORT_MEMORY_LEN（轮）经窗口策略换算为消息条数上限
        self.short_memory=deque(maxlen=_WINDOW_POLICY.message_limit(config.SHORT_MEMORY_LEN))
        self.load_long_memory()
        self.load_status()
        self.load_history()
        self.load_short_memory()
        self.set_status_prompts()

#insert_short_memory/load_short_memory按session_id+mode两重隔离短期记忆；
#token_counter仅用于计数上报，不参与窗口裁剪（裁剪由轮窗口策略负责）
