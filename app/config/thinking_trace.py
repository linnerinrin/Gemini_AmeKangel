"""
config/thinking_trace.py  思考链的结构化存取

前端把思考链渲染成一串"带类型的项"（节点 / 工具 / 文本），历史回放要还原成同样的折叠块，
所以入库前统一编码成 JSON，读出来再解码成同一形状。

存储格式（chat.thinking，mediumtext）：
    {"v": 1, "items": [{"kind": "node", "name": "agent"},
                       {"kind": "text", "text": "……"}]}
"""
import json

TRACE_VERSION = 1

NODE = "node"
TOOL = "tool_call"
TEXT = "text"


def append_event(items, event):
    """把一条 SSE 事件累积进思考链，返回是否发生了变化。

    只认识思考类事件（node / tool_call / thinking_delta），
    回答正文、字幕、状态之类的通知一律忽略。
    """
    if not isinstance(items, list) or not isinstance(event, dict):
        return False
    etype = event.get("type")
    if etype in ("node", "tool_call"):
        name = event.get("name")
        if not name:
            return False
        items.append({"kind": NODE if etype == "node" else TOOL, "name": name})
        return True
    if etype == "thinking_delta":
        text = event.get("text") or ""
        if not text:
            return False
        # 相邻文本合并成一项：前端逐条渲染，合并后回放效果一样，但存储小得多
        if items and items[-1].get("kind") == TEXT:
            items[-1]["text"] = (items[-1].get("text") or "") + text
        else:
            items.append({"kind": TEXT, "text": text})
        return True
    return False


def from_texts(texts):
    """非流式路径只有纯文本思考，包成文本项。"""
    items = []
    for text in (texts or []):
        if text:
            items.append({"kind": TEXT, "text": str(text)})
    return items


def encode(items):
    """编码成入库字符串；没有思考内容时返回 None（列保持 NULL）。"""
    if not items:
        return None
    return json.dumps({"v": TRACE_VERSION, "items": items}, ensure_ascii=False)


def decode(raw):
    """解码成前端可直接渲染的项数组；坏数据一律降级处理，不抛异常。"""
    if not raw:
        return []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str) and not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        # 手工写入或早期纯文本：按一段文本处理，至少不丢内容
        return [{"kind": TEXT, "text": str(raw)}]
    if isinstance(data, dict):
        items = data.get("items")
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    # JSON 标量（例如误写了一个数字）：当作文本保留，不假装有思考链
    return [{"kind": TEXT, "text": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}]
