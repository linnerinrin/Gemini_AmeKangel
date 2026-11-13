#"""
#架构作用：聊天与设置路由共享的生成忙碌标记；只在当前Python进程内有效
#数据流：生成前计数+1，finally计数-1，切换类请求通过require_idle检查
#计数式互斥：跨会话并行生成时，任一在途生成都视为全局忙碌
#"""
"""
api/chat/generation_lock.py生成中互斥状态

该标志供chat_routes与settings路由复用同一份状态（生成中禁止改模型设置），
放在任一路由模块都会形成routes之间互相import，
所以单独抽到这里；chat_routes仍然re-export require_idle，保持既有调用点不变

同会话互斥由SessionRegistry的每会话锁承担（chat_routes）；
本模块只保留「是否有任何生成在进行」的全局视图，供切换存档/模式/设置等
跨会话操作使用；_generation_busy名称与set_generation_busy签名保持兼容
"""

from fastapi import HTTPException

_generation_busy = False
#在途生成计数：不同会话允许并行，全部结束后才算空闲
_generation_count = 0


#"""
#前置检查供切换存档、模式及配置调用；不会排队等待，忙碌时直接返回409
#"""
def require_idle():
    if _generation_count > 0:
        raise HTTPException(status_code=409, detail="正在生成回答，请完成后再切换存档或模式")


#"""
#状态写入口：True登记一个在途生成，False结算一个；返回当前是否仍有在途生成
#"""
def set_generation_busy(value: bool) -> bool:
    global _generation_busy, _generation_count
    if value:
        _generation_count += 1
    else:
        _generation_count = max(0, _generation_count - 1)
    _generation_busy = _generation_count > 0
    return _generation_busy
