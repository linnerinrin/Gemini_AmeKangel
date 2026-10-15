"""
api/chat/generation_lock.py  生成中互斥状态

原先这个标志放在 chat_routes 里，P5a 之后 settings 路由也要复用同一份状态
（生成中禁止改模型设置），继续放在路由模块会形成 routes 之间互相 import，
所以单独抽到这里；chat_routes 仍然 re-export require_idle，保持既有调用点不变。
"""

from fastapi import HTTPException

_generation_busy = False


def require_idle():
    if _generation_busy:
        raise HTTPException(status_code=409, detail="正在生成回答，请完成后再切换存档或模式")


def set_generation_busy(value: bool) -> bool:
    global _generation_busy
    _generation_busy = bool(value)
    return _generation_busy
