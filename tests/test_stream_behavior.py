import asyncio
import importlib
import json


def parse_sse(text):
    frames = text.split('\n\n')
    assert frames.pop() == ''
    return [json.loads(frame[6:]) for frame in frames if frame.startswith('data: ')]


def test_sse_frames_are_independently_parseable(client):
    client.post('/chat/new_session', json={'session_num': 1})
    response = client.post('/chat/stream_chat', json={'prompt': 'hello', 'session_id': 'session-1', 'mode': '2'})
    events = parse_sse(response.text)
    assert [e['type'] for e in events] == ['status', 'thinking_delta', 'answer_delta', 'done']
    assert events[-1]['answer'] == '回复'


def test_sse_error_is_terminal_not_done(client, fake_services):
    async def fail(prompt, mode, emit):
        await emit({'type': 'thinking_delta', 'text': '开始'})
        raise ValueError('offline')
    fake_services.chat.stream_chat = fail
    client.post('/chat/new_session', json={'session_num': 1})
    events = parse_sse(client.post('/chat/stream_chat', json={'prompt': 'hi', 'session_id': 'session-1'}).text)
    #异常原文只进服务端日志，客户端收通用文案
    assert events == [{'type': 'thinking_delta', 'text': '开始'}, {'type': 'error', 'message': '生成过程中出现错误，请稍后重试'}]


def test_stream_generation_blocks_mode_and_save_changes(client, fake_services):
    routes = importlib.import_module('app.api.chat.chat_routes')
    fake_services.chat.session_id = 'test'
    async def scenario():
        ready, finish = asyncio.Event(), asyncio.Event()
        async def generate(prompt, mode, emit):
            ready.set()
            await finish.wait()
            await emit({'type': 'done', 'answer': 'ok'})
        fake_services.chat.stream_chat = generate
        response = await routes.stream_chat(routes.SingleChatRequest(prompt='x', session_id='session-1', mode='2'))
        await ready.wait()
        for operation in [routes.mode_change(routes.ModeChangeRequest(mode='1', session_id='session-1')), routes.chose_session(routes.SessionOperationRequest(session_num=2)), routes.single_chat(routes.SingleChatRequest(prompt='second', session_id='session-1'))]:
            try:
                await operation
                assert False, 'must reject conflicting mutation'
            except routes.HTTPException as error:
                assert error.status_code == 409
        finish.set()
        text = ''.join([chunk async for chunk in response.body_iterator])
        assert parse_sse(text)[-1]['answer'] == 'ok'
        assert not routes._generation_busy
    asyncio.run(scenario())
