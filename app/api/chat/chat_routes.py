# """
#架构作用：聊天HTTP边界，将请求模型转换为ChatSession/AccountManager调用
#调用方：浏览器通过/chat请求；数据流为校验→按请求session_id绑定会话→ JSON或SSE
#会话归属按请求显式声明（方案二）：路由入口先bind_request_session，不信任全局绑定
# """
import asyncio
import json
from contextlib import suppress
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services import chat
from app.managers import account_manager
#P1修复：生成前缺 Key 时抛 NotConfiguredError，路由层要能识别并给出可读提示
from app.config.llm_import import load_llm_settings
#阶段1安全接线：切换会话等敏感动作落审计（默认print落点）
from app.agent_framework.security.audit import PrintAuditSink
from app.agent_framework.contracts import AuditEvent
#生成中互斥状态已抽到generation_lock（settings路由也要用同一份），这里继续
# re-export：既有的tests/test_stream_behavior.py会读routes._generation_busy
from app.api.chat.generation_lock import _generation_busy, require_idle, set_generation_busy
from app.api.chat.chat_schemas import (
    StartSessionRequest, SingleChatRequest,
    ModeChangeRequest, SessionOperationRequest, GetStatusRequest,
)

router = APIRouter()

#审计落点：默认print；落库实现待F3拍板后替换
audit_sink = PrintAuditSink()

#P1修复：把“缺 Key”异常映射成可读 422/流式 error 事件，而不是 500
llm_settings = load_llm_settings()


# """
#输入边界：场景轴接受四种模式；None留给会话层解释为沿用当前值
# """
def validate_mode(mode):
    if mode is not None and mode not in {"0", "1", "2", "3"}:
        raise HTTPException(status_code=422, detail="未知聊天模式")


# """
#会话归属按请求显式校验：缺session_id即409；阶段0后换绑只是门面指针移动
#（每会话独立实例），不再要求生成空闲窗口；同会话轮询语义不变
# """
def bind_request_session(session_id):
    if not session_id:
        raise HTTPException(status_code=409, detail="请先选择存档")
    if getattr(chat, "session_id", None) != session_id:
        #多实例后换绑只是门面指针移动，无共享状态改写，不再需要生成空闲窗口
        try:
            chat.bind_session_id(session_id)
        except ValueError:
            raise HTTPException(status_code=409, detail="会话不存在或已删除")


# """
#P1修复：取“此刻绑定的具体会话实例”快照。
#门面 __getattr__ 在调用时才解析 _current_id，跨 await/延迟执行会读到换绑后的新指针；
#路由在发起生成前先取一次实例，后续只对该实例调用。测试替身没有 resolve_current 时回落门面本身。
# """
def _bound_session():
    resolver = getattr(chat, "resolve_current", None)
    return resolver() if callable(resolver) else chat


# """
#绑定指定存档：检查生成空闲后修改共享会话，绑定失败转成业务响应
# """
@router.post("/start_session")
async def start_session(request: StartSessionRequest):
    require_idle()
    try:
        chat.bind_session_id(request.session_id)
        return {"ok": True}
    except ValueError:
        return {"ok": False, "msg": "会话不存在"}


# """
#历史读取：无mode返回当前缓存，指定mode则交给服务按场景读取
# """
@router.get("/history")
async def history(mode: str | None = None, session_id: str | None = None):
    bind_request_session(session_id)
    validate_mode(mode)
    if mode is None:
        return chat.history
    return chat.read_history(mode)


# """
#返回账户管理器的存档列表；列表来自进程所连接的统一数据库
# """
@router.get("/sessions")
async def sessions():
    return {"sessions": account_manager.list_sessions()}


# """
#单次生成：设置忙碌标记，等待完整结果；finally确保异常时也恢复空闲
# """
@router.post("/single_chat")
async def single_chat(request: SingleChatRequest):
    bind_request_session(request.session_id)
    #同会话互斥用每会话锁；替身（测试fake）没有lock_for时回落旧全局布尔语义
    lock_provider = getattr(chat, "lock_for", None)
    lock = lock_provider(request.session_id) if lock_provider is not None else None
    if lock is not None:
        if lock.locked():
            raise HTTPException(status_code=409, detail="正在生成回答，请完成后再切换存档或模式")
    else:
        require_idle()
    validate_mode(request.mode)
    #在await之前锁定具体会话实例，防止生成期间换绑指针导致串档
    session = _bound_session()
    if lock is not None:
        await lock.acquire()
    set_generation_busy(True)
    try:
        try:
            reply_text, think_text = await session.single_chat(request.prompt, request.mode)
        except llm_settings.NotConfiguredError as exc:
            #缺 Key 是配置问题而非服务器故障：422 + 面板引导文案
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"answer": reply_text, "think": think_text}
    finally:
        set_generation_busy(False)
        if lock is not None:
            lock.release()


# """
#流式入口：校验当前会话与场景轴，后台生产事件，响应迭代器消费事件
#有界队列提供背压；终止哨兵与业务事件分离，断开连接时取消生产任务
# """
@router.post("/stream_chat")
async def stream_chat(request: SingleChatRequest):
    """SSE：思考内容按增量、tool 名称一次、最终回答按增量发送。"""
    bind_request_session(request.session_id)
    #与single_chat一致：每会话锁判定同会话并发，替身路径回落旧全局布尔
    lock_provider = getattr(chat, "lock_for", None)
    lock = lock_provider(request.session_id) if lock_provider is not None else None
    if lock is not None:
        if lock.locked():
            raise HTTPException(status_code=409, detail="正在生成回答，请完成后再切换存档或模式")
    else:
        require_idle()
    validate_mode(request.mode)
    #P1修复：create_task之前锁定实例；run()只调用该快照，指针此后换绑也不影响本轮
    session = _bound_session()
    if lock is not None:
        await lock.acquire()
    set_generation_busy(True)
    #限制待消费事件数量，慢客户端会使生产端在put处等待
    queue = asyncio.Queue(maxsize=64)
    done = object()

    # """
    #图执行回调：把事件交给队列，不直接操作HTTP响应
    # """
    async def emit(event):
        await queue.put(event)

    # """
    #生产协程：执行流式会话，统一发错误事件并释放忙碌标记
    # """
    async def run():
        try:
            await session.stream_chat(request.prompt, request.mode, emit)
        except llm_settings.NotConfiguredError as exc:
            #P1修复：缺 Key 直接给面板引导文案（含厂商名与去哪填），不再落到通用错误
            await queue.put({"type": "error", "message": str(exc)})
        except Exception as exc:
            #外发仅通用文案，异常原文只打印到服务端日志
            print(f"[stream_chat] 生成异常: {exc!r}")
            await queue.put({"type": "error", "message": "生成过程中出现错误，请稍后重试"})
        finally:
            set_generation_busy(False)
            if lock is not None and lock.locked():
                lock.release()
            # Cancellation must not block on a full queue after disconnect.
            if not asyncio.current_task().cancelling():
                await queue.put(done)

    task = asyncio.create_task(run())

    # """
    #消费协程：事件编码为SSE；15秒仅是保活间隔，不是生成总超时
    # finally回收生产任务，避免客户端离开后仍继续生成
    # """
    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is done:
                    break
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# """
#场景切换：要求已有会话且未生成，更新共享ChatSession的场景配置
# """
@router.post("/mode_change")
async def mode_change(request: ModeChangeRequest):
    bind_request_session(request.session_id)
    require_idle()
    validate_mode(request.mode)
    chat.load_mode(request.mode)
    return True


# """
#新建存档：由账户管理器分配槽位，成功后立即绑定为活动会话
# """
@router.post("/new_session")
async def new_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.create_new_session(request.session_num)
    if session_id:
        chat.bind_session_id(str(session_id))
        return {"ok": True, "session_id": str(session_id), "session_num": request.session_num}
    return {"ok": False, "msg": "已达最大会话数(3)"}


# """
#删除存档：先记录槽位身份，再删除数据；若删除活动会话则清除共享绑定
# """
@router.post("/delete_session")
async def delete_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.chose_session(request.session_num)
    account_manager.delete_session(request.session_num)
    if session_id:
        #无论被删档是否为当前会话，都清缓存并逐出注册表（替身无此方法时按旧语义清理）
        discard = getattr(chat, "discard_session", None)
        if discard is not None:
            discard(str(session_id))
        elif getattr(chat, "session_id", None) == session_id:
            chat.clear_session()
    return True


# """
#选择存档：槽位解析成功后切换活动会话；不存在时返回业务失败
# """
@router.post("/chose_session")
async def chose_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.chose_session(request.session_num)
    if not session_id:
        return {"ok": False, "msg": "会话不存在"}
    chat.bind_session_id(str(session_id))
    #切换会话属敏感动作：审计埋点
    audit_sink.record(AuditEvent(action="switch_session", identity=str(session_id)))
    return {"ok": True, "msg": "会话已切换", "session_id": str(session_id)}


# """
#状态读取：从当前会话的记忆管理器取得状态，不在路由层重新计算
# """
@router.post("/get_status")
async def get_status(request: GetStatusRequest):
    bind_request_session(request.session_id)
    return chat.memory_mgr.load_status()

# [总审计 2026-11-13] P1 已修复：single_chat/stream_chat 在 await / create_task 之前调用
# _bound_session() 取具体会话快照，run() 只调用该实例；期间任何 bind 换指针都不再影响在途生成。
# 缺 Key 的 NotConfiguredError 在 single_chat 转 422，在流式里转带引导文案的 error 事件。
# P2 保留：single_chat(:95-102) 的 lock.locked() 判定与 acquire 之间非原子（并发同会话第二个请求排队而非立即 409）；
# 生成期间 mode_change/new_session 等仍走全局 require_idle，跨会话并行时会被 409 挡住。
