import json
import re
from pathlib import Path

from conftest import contract_entries


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
PROJECT_ROOT = STATIC_ROOT.parent


def _frontend_source():
    parts = []
    for path in sorted((STATIC_ROOT / "js").rglob("*.js")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _extract_request_fields(source, api_path):
    pattern = re.compile(
        r"apiJson\(\s*['\"]" + re.escape(api_path) + r"['\"][\s\S]{0,500}?"
        r"JSON\.stringify\(\s*\{(?P<body>[\s\S]*?)\}\s*\)",
        re.MULTILINE,
    )
    fields = []
    for match in pattern.finditer(source):
        body = re.sub(r"//.*", "", match.group("body"))
        fields.append(set(re.findall(r"([A-Za-z_$][\w$]*)\s*:", body)))
    return fields


def test_frontend_calls_all_contract_paths(contracts):
    source = _frontend_source()
    for api_path in contracts:
        assert f"apiJson('{api_path}'" in source or f'apiJson("{api_path}"' in source


def test_frontend_request_fields_follow_contract(contracts):
    source = _frontend_source()
    for api_path, contract in contract_entries(contracts):
        if contract["method"] == "GET":
            continue
        expected = set(contract["request"].keys())
        if not expected:
            continue
        call_fields = _extract_request_fields(source, api_path)
        assert call_fields, f"no JSON.stringify body found for {api_path}"
        assert any(expected <= fields for fields in call_fields), {
            "path": api_path,
            "expected": sorted(expected),
            "actual_calls": [sorted(fields) for fields in call_fields],
        }


def test_frontend_reads_declared_response_fields(contracts):
    source = _frontend_source()
    for _api_path, contract in contract_entries(contracts):
        for field in contract.get("frontend_reads", []):
            assert re.search(r'\b(?:res|event)\.' + re.escape(field) + r'\b', source)


def test_window_manager_honors_close_configuration():
    source = (STATIC_ROOT / "js" / "service" / "window.js").read_text(encoding="utf-8")

    assert "typeof cfg.closable === 'function'" in source
    assert "if (closable === false) return;" in source
    assert "win.classList.add('hidden');" in source
    assert "style.setProperty('display', 'none', 'important')" in source


def test_save_close_is_only_handled_by_window_manager():
    save_source = (STATIC_ROOT / "js" / "service" / "save.js").read_text(encoding="utf-8")
    save_ui_source = (STATIC_ROOT / "js" / "ui" / "save.js").read_text(encoding="utf-8")

    assert "saveWinClose.addEventListener" not in save_source
    assert "saveWindowFromDesktop = false;" in save_ui_source


def test_live_topics_use_distinct_backend_modes():
    """每个直播主题一个独立 mode，否则不同直播会共用上下文/记忆"""
    source = (STATIC_ROOT / "js" / "data" / "topics.js").read_text(encoding="utf-8")
    modes = dict(re.findall(r'charVariant: "([\w-]+)", mode: "(\d*)"', source))

    assert modes["kangel"] == "1"
    assert modes["dangel"] == "2"
    assert modes["ame"] == "3"
    assert len({"1", "2", "3"}) == 3


def test_role_mapping_matches_contract():
    contract = json.loads((PROJECT_ROOT / "contracts" / "role2idx.json").read_text(encoding="utf-8"))
    source = (STATIC_ROOT / "js" / "data" / "role-mapping.js").read_text(encoding="utf-8")

    for role, idx in contract.items():
        pattern = r'"?' + re.escape(role) + r'"?\s*:\s*"' + re.escape(idx) + r'"'
        assert re.search(pattern, source), role


def test_history_api_supports_mode_parameter():
    """直播窗口要按主题拉历史，api 层必须能带上 mode"""
    source = (STATIC_ROOT / "js" / "data" / "api" / "chat.js").read_text(encoding="utf-8")

    assert "getHistory: function(mode)" in source
    assert "streamChat: async function(prompt, mode, taskMode, onEvent, signal)" in source
    assert "'/chat/stream_chat'" in source
    assert "'/chat/history'" in source
    assert "'?mode='" in source


def test_chat_api_carries_task_mode():
    """任务轴（chat/work）要跟场景轴 mode 一起发到后端，否则落库与 /history 都拿不到"""
    api_source = (STATIC_ROOT / "js" / "data" / "api" / "chat.js").read_text(encoding="utf-8")
    core_source = (STATIC_ROOT / "js" / "service" / "chat.js").read_text(encoding="utf-8")
    registry = (STATIC_ROOT / "js" / "data" / "registry.js").read_text(encoding="utf-8")

    assert api_source.count("task_mode: taskMode == null ? null : String(taskMode)") == 2
    # 发送时必须按 streamChat(prompt, mode, taskMode, onEvent, signal) 的顺序传参
    assert "channel.getTaskMode ? channel.getTaskMode() : 'chat'" in core_source
    assert "NGO.chatAPI.streamChat(text, mode, taskMode, function(event) {" in core_source
    # 两个 channel 都要能给出任务轴；P5 才会提供 UI 去写 NGO.taskMode
    assert registry.count("getTaskMode: function()") == 2
    assert "NGO.taskMode" in registry
    # 历史回填要读出每轮的 task_mode（P4 才按它分段渲染）
    assert "item.task_mode" in core_source


def test_chat_core_sends_mode_and_consumes_sse_events():
    source = (STATIC_ROOT / "js" / "service" / "chat.js").read_text(encoding="utf-8")

    assert "channel.getMode" in source
    assert "NGO.chatAPI.streamChat" in source
    assert "thinking_delta" in source
    assert "tool_call" in source
    assert "answer_delta" in source
    assert "playThinkingAction" in source
    assert "stopThinkingAction" in source


def test_live_subtitle_streams_and_box_auto_sizes():
    """直播字幕要跟着流式增量走，字幕框要随文字长宽自适应并保持长方形"""
    chat_source = (STATIC_ROOT / "js" / "service" / "chat.js").read_text(encoding="utf-8")
    live_source = (STATIC_ROOT / "js" / "ui" / "live.js").read_text(encoding="utf-8")
    registry = (STATIC_ROOT / "js" / "data" / "registry.js").read_text(encoding="utf-8")
    css = (STATIC_ROOT / "css" / "desktop.css").read_text(encoding="utf-8")

    # 流式阶段必须通知 channel（否则字幕只在回复结束时一次性出现）
    assert "channel.onAnswerLifecycle" in chat_source
    assert "onAnswerLifecycle" in registry
    assert "function updateStreamReply" in live_source
    # 旧的逐字定时器会与流式打架，应已移除
    assert "streamReplyTimer" not in live_source

    overlay = css.split(".live-reply-overlay {")[1].split("}")[0]
    assert "width: max-content" in overlay   # 宽度跟着文字长
    assert "max-width: 80%" in overlay       # 到上限后换行
    assert "min-width" in overlay            # 短句也保持长方形
    assert "white-space: pre-wrap" in overlay  # 换行会改变高度
