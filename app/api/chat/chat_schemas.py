from pydantic import BaseModel
from typing import Optional


class StartSessionRequest(BaseModel):
    session_id: str


class SingleChatRequest(BaseModel):
    prompt: str
    # 前端在发送时携带当前 JINE/直播 mode，避免依赖全局 mode_change 的竞态
    mode: Optional[str] = None
    # 任务轴（chat=闲聊 / work=工作），与场景轴 mode 正交；None 表示沿用会话当前值
    task_mode: Optional[str] = None


class ModeChangeRequest(BaseModel):
    mode: str


class SessionOperationRequest(BaseModel):
    session_num: int
