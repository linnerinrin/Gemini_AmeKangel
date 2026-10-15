import asyncio
import json
from contextlib import suppress
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services import chat
from app.managers import account_manager
# 生成中互斥状态已抽到 generation_lock（settings 路由也要用同一份），这里继续
# re-export：既有的 tests/test_stream_behavior.py 会读 routes._generation_busy。
from app.api.chat.generation_lock import _generation_busy, require_idle, set_generation_busy
from app.api.chat.chat_schemas import (
    StartSessionRequest, SingleChatRequest,
    ModeChangeRequest, SessionOperationRequest,
)

router = APIRouter()


def validate_mode(mode):
    if mode is not None and mode not in {"0", "1", "2", "3"}:
        raise HTTPException(status_code=422, detail="未知聊天模式")


def validate_task_mode(task_mode):
    """任务轴只允许 chat / work；None 表示沿用会话当前任务轴。"""
    if task_mode is not None and task_mode not in {"chat", "work"}:
        raise HTTPException(status_code=422, detail="未知对话模式")


def require_session():
    if not getattr(chat, "session_id", None):
        raise HTTPException(status_code=409, detail="请先选择存档")


@router.post("/start_session")
async def start_session(request: StartSessionRequest):
    require_idle()
    try:
        chat.bind_session_id(request.session_id)
        return {"ok": True}
    except ValueError:
        return {"ok": False, "msg": "会话不存在"}


@router.get("/history")
async def history(mode: str | None = None):
    require_session()
    validate_mode(mode)
    if mode is None:
        return chat.history
    return chat.read_history(mode)


@router.get("/sessions")
async def sessions():
    return {"sessions": account_manager.list_sessions()}


@router.post("/single_chat")
async def single_chat(request: SingleChatRequest):
    require_session()
    require_idle()
    validate_mode(request.mode)
    validate_task_mode(request.task_mode)
    set_generation_busy(True)
    try:
        reply_text, think_text = await chat.single_chat(request.prompt, request.mode, request.task_mode)
        return {"answer": reply_text, "think": think_text}
    finally:
        set_generation_busy(False)


@router.post("/stream_chat")
async def stream_chat(request: SingleChatRequest):
    """SSE：思考内容按增量、tool 名称一次、最终回答按增量发送。"""
    require_session()
    require_idle()
    validate_mode(request.mode)
    validate_task_mode(request.task_mode)
    set_generation_busy(True)
    queue = asyncio.Queue(maxsize=64)
    done = object()

    async def emit(event):
        await queue.put(event)

    async def run():
        try:
            await chat.stream_chat(request.prompt, request.mode, emit, request.task_mode)
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            set_generation_busy(False)
            # Cancellation must not block on a full queue after disconnect.
            if not asyncio.current_task().cancelling():
                await queue.put(done)

    task = asyncio.create_task(run())

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


@router.post("/mode_change")
async def mode_change(request: ModeChangeRequest):
    require_session()
    require_idle()
    validate_mode(request.mode)
    chat.load_mode(request.mode)
    return True


@router.post("/new_session")
async def new_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.create_new_session(request.session_num)
    if session_id:
        chat.bind_session_id(str(session_id))
        return {"ok": True, "session_id": str(session_id), "session_num": request.session_num}
    return {"ok": False, "msg": "已达最大会话数(3)"}


@router.post("/delete_session")
async def delete_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.chose_session(request.session_num)
    account_manager.delete_session(request.session_num)
    if session_id and getattr(chat, "session_id", None) == session_id:
        chat.clear_session()
    return True


@router.post("/chose_session")
async def chose_session(request: SessionOperationRequest):
    require_idle()
    session_id = account_manager.chose_session(request.session_num)
    if not session_id:
        return {"ok": False, "msg": "会话不存在"}
    chat.bind_session_id(str(session_id))
    return {"ok": True, "msg": "会话已切换"}


@router.post("/get_status")
async def get_status():
    require_session()
    return chat.memory_mgr.load_status()
