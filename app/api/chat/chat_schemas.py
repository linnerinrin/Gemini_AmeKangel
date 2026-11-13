#"""
#架构作用：聊天请求的Pydantic边界，由FastAPI解析请求体
#数据流：JSON →会话标识/提示词/场景轴→路由；业务合法值由路由继续校验
#"""
from pydantic import BaseModel
from typing import Optional


#"""
#存档绑定请求：传递标识，存在性检查由服务与路由处理
#"""
class StartSessionRequest(BaseModel):
    session_id: str


#"""
#单轮请求：提示词与场景轴一起提交，减少依赖前序切换请求
#"""
class SingleChatRequest(BaseModel):
    prompt: str
    #每轮显式携带session_id，路由按请求绑定会话，不信任全局绑定
    session_id: Optional[str] = None
    #前端在发送时携带当前JINE/直播mode，避免依赖全局mode_change的竞态
    mode: Optional[str] = None


#"""
#独立模式切换载荷，合法值由路由统一检查
#"""
class ModeChangeRequest(BaseModel):
    mode: str
    #与会话绑定的操作都要显式携带session_id
    session_id: Optional[str] = None


#"""
#状态读取载荷：会话归属由请求方显式声明
#"""
class GetStatusRequest(BaseModel):
    session_id: Optional[str] = None


#"""
#存档槽位请求，新增、删除和选择共用；容量和存在性由账户管理器处理
#"""
class SessionOperationRequest(BaseModel):
    session_num: int
