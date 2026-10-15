import pytest
import asyncio
import sys
import types

from conftest import load_source_module


def fake_tool(func):
    return func


class FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results):
        return []


langchain_core = sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
tools_module = types.ModuleType("langchain_core.tools")
tools_module.tool = fake_tool
ddgs_module = types.ModuleType("ddgs")
ddgs_module.DDGS = FakeDDGS
sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.tools"] = tools_module
sys.modules["ddgs"] = ddgs_module

func_module = load_source_module("func_call_manager_under_test", "app/managers/FuncCallManager.py")


def test_init_creates_tools():
    mgr = func_module.FunctionCallManager()

    mgr.init(rag_mgr=object(), memory_mgr=object(), model_mgr=object())

    assert len(mgr.tools) == 7


def test_load_mode_records_mode():
    mgr = func_module.FunctionCallManager()
    mgr.load_mode("0")

    assert mgr.current_mode == "0"


class FakeRAG:
    async def retrieve(self, query, caller="knowledge"):
        return f"{caller}:{query}"


class FakeMemory:
    def __init__(self):
        self.long_memory = None
        self.status_changes = []

    async def update_long_memory(self, memory):
        self.long_memory = memory

    def apply_status_change(self, amounts):
        self.status_changes.append(amounts)
        return amounts


class FakeModel:
    async def generate(self, messages, paras, use_tools=False):
        self.messages = messages
        self.paras = paras
        self.use_tools = use_tools
        return "rewritten"


def test_created_tools_execute_core_branches():
    memory = FakeMemory()
    model = FakeModel()
    mgr = func_module.FunctionCallManager()
    mgr.init(rag_mgr=FakeRAG(), memory_mgr=memory, model_mgr=model)

    output_time, retrieve_knowledge, retrieve_context, rewrite, save_memory, search_web, change_status = mgr.tools

    assert output_time()
    assert "knowledge:query" in asyncio.run(retrieve_knowledge("query"))
    assert "context:query" in asyncio.run(retrieve_context("query"))
    assert "rewritten" in asyncio.run(rewrite("query"))
    assert asyncio.run(save_memory("age: 20")) == ""
    assert memory.long_memory == "age: 20"
    assert "未" in asyncio.run(search_web("query")) or "鏈" in asyncio.run(search_web("query"))
    assert asyncio.run(change_status(reason="test", amount_affection=1)) == "test"
    assert memory.status_changes == [{"affection": 1, "stress": 0, "mental": 0}]
    assert asyncio.run(change_status(reason="second", amount_affection=1)) == "本轮状态已经结算，后续状态变化未生效。"
    mgr.begin_turn()
    assert asyncio.run(change_status(reason="next", amount_affection=-1)) == "next"
    assert len(memory.status_changes) == 2

