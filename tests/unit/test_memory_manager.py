import asyncio
from collections import deque

import pytest

from conftest import load_source_module


memory_module = load_source_module("memory_manager_under_test", "app/managers/MemoryManager.py")


class FakeDB:
    def __init__(self):
        self.calls = []
        self.fetchone_queue = []
        self.fetchall_queue = []
        self.committed = False

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self.fetchone_queue.pop(0) if self.fetchone_queue else None

    def fetchall(self):
        return self.fetchall_queue.pop(0) if self.fetchall_queue else []

    def commit(self):
        self.committed = True


def _sql(fake_db, prefix):
    """取最近一条以 prefix 开头的 SQL，顺带保证这类查询确实发生过。"""
    matched = [sql for sql, _ in fake_db.calls if sql.startswith(prefix)]
    assert matched, f"没有执行 {prefix} 查询"
    return matched[-1]


def test_refresh_loads_session_state(monkeypatch):
    fake_db = FakeDB()
    fake_db.fetchone_queue = [
        ("session-1",),
        ("long memory",),
        (60, 50, 20, 10),
    ]
    fake_db.fetchall_queue = [
        [("q1", "a1", "2026-01-01", '{"v": 1, "items": [{"kind": "text", "text": "想了"}]}',
          "deepseek-v4-flash", "deepseek")],
        [("q2", "a2")],
    ]
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.refresh("session-1")

    assert mgr.session_id == "session-1"
    assert mgr.long_memory == "long memory"
    assert mgr.fol == 60
    assert mgr.aff == 50
    assert mgr.ste == 20
    assert mgr.men == 10
    assert mgr.history == [
        {"created_at": "2026-01-01", "user": "q1", "assistant": "a1",
         "thinking": [{"kind": "text", "text": "想了"}],
         "llm_model": "deepseek-v4-flash", "llm_provider": "deepseek"},
    ]
    assert list(mgr.short_memory) == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    # 迁移已移到应用启动阶段，refresh 只负责读取状态
    assert not any("alter table chat add column thinking" in sql for sql, _ in fake_db.calls)


def test_read_history_decodes_thinking_and_tolerates_bad_data(monkeypatch):
    fake_db = FakeDB()
    fake_db.fetchall_queue = [[
        ("q1", "a1", "t1", '{"v": 1, "items": [{"kind": "node", "name": "agent"}]}', None, None),
        ("q2", "a2", "t2", None, "gpt-4o-mini", "openai"),
        ("q3", "a3", "t3", "早期纯文本思考", None, None),
    ]]
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    history = mgr.read_history("0")

    assert history[0]["thinking"] == [{"kind": "node", "name": "agent"}]
    assert history[1]["thinking"] == []
    assert history[2]["thinking"] == [{"kind": "text", "text": "早期纯文本思考"}]
    assert "thinking" in fake_db.calls[0][0]
    # 老数据没有模型列时也要正常返回 None，不能因为多两列就报错
    assert history[0]["llm_model"] is None
    assert history[0]["llm_provider"] is None
    assert (history[1]["llm_model"], history[1]["llm_provider"]) == ("gpt-4o-mini", "openai")
    assert "llm_model" in fake_db.calls[0][0]
    assert "llm_provider" in fake_db.calls[0][0]


def test_read_history_returns_llm_model_and_provider_of_each_turn(monkeypatch):
    """每轮带回当时实际使用的厂商/模型，排查'这轮为什么风格不同'"""
    fake_db = FakeDB()
    fake_db.fetchall_queue = [[
        ("q1", "a1", "t1", None, "deepseek-v4-flash", "deepseek"),
        ("q2", "a2", "t2", None, None, None),
    ]]
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    history = mgr.read_history("0")

    assert [item["llm_model"] for item in history] == ["deepseek-v4-flash", None]
    assert [item["llm_provider"] for item in history] == ["deepseek", None]


def test_update_short_memory_commits(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    mgr.update_short_memory("q", "a")

    assert fake_db.committed
    sql, params = fake_db.calls[-1]
    assert "thinking" in sql
    assert "llm_model" in sql
    assert "llm_provider" in sql
    assert params[4] is None      # 没有思考内容时写 NULL
    assert params[5] is None      # 拿不到模型名时写 NULL，不能报错
    assert params[6] is None


def test_update_short_memory_stores_structured_trace(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    trace = [{"kind": "node", "name": "agent"}, {"kind": "text", "text": "想了很久"}]
    mgr.update_short_memory("q", "a", trace)

    import json
    sql, params = fake_db.calls[-1]
    assert "thinking" in sql
    assert json.loads(params[4]) == {"v": 1, "items": trace}


def test_update_short_memory_writes_llm_model_and_provider(monkeypatch):
    """R12：每轮把实际使用的厂商/模型写进 chat 表，历史回放才能解释风格差异"""
    fake_db = FakeDB()
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    mgr.update_short_memory("q", "a", None, model="gpt-4o-mini", provider="openai")

    sql, params = fake_db.calls[-1]
    assert "llm_model" in sql
    assert "llm_provider" in sql
    assert params[5] == "gpt-4o-mini"
    assert params[6] == "openai"
    assert fake_db.committed


def test_load_short_memory_filters_by_session_and_mode(monkeypatch):
    """短期记忆按会话+场景轴过滤。"""
    fake_db = FakeDB()
    fake_db.fetchall_queue = [[("q", "a")]]
    monkeypatch.setattr(memory_module, "db", fake_db)

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    mgr.mode = "0"
    mgr.short_memory = deque(maxlen=3)
    mgr.load_short_memory()

    sql, params = fake_db.calls[-1]
    assert "mode=%s" in sql
    assert params == ("session-1", "0", memory_module.config.SHORT_MEMORY_LEN)
    assert list(mgr.short_memory) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_update_long_memory_uses_model(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(memory_module, "db", fake_db)

    class FakeModel:
        async def generate(self, messages, params, use_tools=False):
            return "summary"

    mgr = memory_module.MemoryManager()
    mgr.session_id = "session-1"
    mgr.long_memory = ""
    mgr.model_mgr = FakeModel()

    asyncio.run(mgr.update_long_memory("new memory"))

    assert mgr.long_memory == "summary"
    assert fake_db.committed
