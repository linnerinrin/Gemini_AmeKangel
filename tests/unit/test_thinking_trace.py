"""思考链结构化存取：事件累积、编解码与坏数据降级"""
import json

import pytest

from app.config.thinking_trace import append_event, decode, encode, from_texts


def test_append_event_keeps_nodes_and_tools_in_order():
    items = []
    assert append_event(items, {"type": "node", "name": "agent"}) is True
    assert append_event(items, {"type": "tool_call", "name": "search_web"}) is True
    assert append_event(items, {"type": "node", "name": "generate_reply"}) is True

    assert items == [
        {"kind": "node", "name": "agent"},
        {"kind": "tool_call", "name": "search_web"},
        {"kind": "node", "name": "generate_reply"},
    ]


def test_append_event_merges_adjacent_text_and_splits_on_nodes():
    items = []
    append_event(items, {"type": "node", "name": "agent"})
    append_event(items, {"type": "thinking_delta", "text": "先"})
    append_event(items, {"type": "thinking_delta", "text": "想一下"})
    append_event(items, {"type": "node", "name": "generate_reply"})
    append_event(items, {"type": "thinking_delta", "text": "再写"})

    assert items == [
        {"kind": "node", "name": "agent"},
        {"kind": "text", "text": "先想一下"},
        {"kind": "node", "name": "generate_reply"},
        {"kind": "text", "text": "再写"},
    ]


@pytest.mark.parametrize("event", [
    {"type": "answer_delta", "text": "正文"},
    {"type": "done", "answer": "正文"},
    {"type": "status", "stage": "thinking"},
    {"type": "tool_calls_ready", "calls": ["a"]},
    {"type": "node"},
    {"type": "thinking_delta", "text": ""},
    {},
])
def test_append_event_ignores_non_thinking_events(event):
    items = []

    assert append_event(items, event) is False
    assert items == []


def test_encode_decode_round_trip():
    items = [{"kind": "node", "name": "agent"}, {"kind": "text", "text": "中文思考"}]
    raw = encode(items)

    assert json.loads(raw) == {"v": 1, "items": items}
    assert decode(raw) == items
    # 没有思考内容时写 NULL，而不是空数组
    assert encode([]) is None
    assert encode(None) is None


@pytest.mark.parametrize("raw,expected", [
    (None, []),
    ("", []),
    ("   ", []),
    ("不是 JSON", [{"kind": "text", "text": "不是 JSON"}]),
    ('{"v": 1}', []),
    ('{"v": 1, "items": "坏数据"}', []),
    ('[{"kind": "text", "text": "数组也能用"}, "噪声"]', [{"kind": "text", "text": "数组也能用"}]),
    ("42", [{"kind": "text", "text": "42"}]),
])
def test_decode_degrades_bad_data(raw, expected):
    assert decode(raw) == expected


def test_decode_accepts_bytes_from_mysql():
    assert decode('{"v": 1, "items": [{"kind": "text", "text": "字节"}]}'.encode("utf-8")) == [
        {"kind": "text", "text": "字节"},
    ]


def test_from_texts_skips_empty_entries():
    assert from_texts(["想", "", None, "更多"]) == [
        {"kind": "text", "text": "想"},
        {"kind": "text", "text": "更多"},
    ]
    assert from_texts(None) == []
