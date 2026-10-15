"""
Manual chat flow test: runs the full LangGraph pipeline with mocked LLM.
Verifies message propagation across agent -> tools -> agent -> generate_reply -> reflection.

Usage: python tests/manual_chat_flow.py
"""
import asyncio
import importlib
import importlib.util
import sys
import types
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- mock heavy dependencies before any app imports ----
class PlaceholderDB:
    def __call__(self, *args, **kwargs): return None
    def commit(self): return None
    def fetchall(self): return []
    def fetchone(self): return None

repo_mod = types.ModuleType("repository")
repo_conn = types.ModuleType("repository.conn")
repo_conn.db = PlaceholderDB()
sys.modules.setdefault("repository", repo_mod)
sys.modules.setdefault("repository.conn", repo_conn)

# ---- load source modules directly (skips app/managers/__init__.py) ----
def load_source(module_name, rel_path):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

GraphManager = load_source("_graph_mod", "app/managers/GraphManager.py").GraphManager
GenerationResult = load_source("_model_mod", "app/managers/ModelManager.py").GenerationResult
FunctionCallManager = load_source("_fcall_mod", "app/managers/FuncCallManager.py").FunctionCallManager
config = load_source("_config_mod", "app/config/config.py").config


class MockModel:
    """Returns pre-defined GenerationResult in sequence."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0
        self.calls = []

    async def generate(self, messages, paras, use_tools=True):
        result = self.outputs.pop(0)
        self.call_count += 1
        self.calls.append((messages, paras, use_tools))
        role_tag = "agent" if use_tools else "reply" if self.call_count < 4 else "refl"
        print(f"    [{role_tag}] content={result.content[:60]!r}, tool_calls={len(result.tool_calls)}")
        return result


class MockMemory:
    status_prompts = ""
    def load_status(self): pass
    def load_long_memory(self): return None


class MockRAG:
    async def retrieve(self, query, caller="knowledge"): return ""


async def main():
    # Model output sequence (4 calls for a full cycle):
    #  1. agent_node (use_tools=True)  -> thinking + calls output_time
    #  2. agent_node (use_tools=True)  -> thinking + no tool_calls
    #  3. generate_reply_node          -> final reply
    #  4. reflection_node              -> __ACCEPT__
    # reflection_node 只接受 finish_reason == "stop" 的完整审查结果，
    # 所以桩输出必须带 finish_reason（默认 None 会被当成"截断/格式异常"而重试）
    model = MockModel([
        GenerationResult(content="用户想知道时间，调用output_time。", finish_reason="stop", tool_calls=[
            {"name": "output_time", "args": {}, "id": "call_1", "type": "tool_call"},
        ]),
        GenerationResult(content="已获取时间，准备回复。", finish_reason="stop", tool_calls=[]),
        GenerationResult(content="现在是2026年6月9日。", finish_reason="stop", tool_calls=[]),
        GenerationResult(content="__ACCEPT__", finish_reason="stop", tool_calls=[]),
    ])

    # FunctionCallManager with real tools
    fcall = FunctionCallManager()
    fcall.init(rag_mgr=MockRAG(), memory_mgr=MockMemory(), model_mgr=MockModel([]))
    fcall.memory_mgr = MockMemory()  # ensure change_status has a live mock

    session = types.SimpleNamespace(
        current_mode="0",
        call_prompt=config.CALL_PROMPT,
        sys_prompt=config.SYS_PROMPT["0"],
        reflection_prompt=config.REFLECTION_PROMPT,
        short_memory=deque(maxlen=config.SHORT_MEMORY_LEN),
        inference_para=config.inference_para,
        config=config,
        model_mgr=model,
        memory_mgr=MockMemory(),
        fcall_mgr=fcall,
        rag_mgr=MockRAG(),
    )

    mgr = GraphManager()
    mgr.init(session)
    app = mgr.build()

    print("=== Chat Flow Trace ===")
    print(f"Tools: {[t.name for t in fcall.tools]}\n")

    state = {
        "agent_message": [],
        "think_text": [],
        "tool_text": [],
        "reflection_text": [],
        "query": "现在几点？",
        "reply": "",
        "tool_finish": False,
        "tool_turns": 0,
        "reflection_finish": False,
        "reflection_turns": 0,
    }

    result = await app.ainvoke(state)

    print(f"\n=== Result ===")
    print(f"  reply:              {result.get('reply')!r}")
    print(f"  tool_turns:         {result.get('tool_turns')}")
    print(f"  reflection_turns:   {result.get('reflection_turns')}")
    print(f"  model calls:        {model.call_count}")
    print(f"  think rounds:       {len(result.get('think_text', []))}")
    print(f"  reflection_finish:  {result.get('reflection_finish')}")
    print()

    # Verify
    assert result.get("reply"), "Reply should not be empty"
    assert result["reply"] == "现在是2026年6月9日。"
    assert result["tool_turns"] == 2
    assert result["reflection_finish"] is True
    assert len(result["think_text"]) == 2
    assert len(result["tool_text"]) == 2
    assert model.call_count == 4
    print("=== All OK: message propagation verified ===")


if __name__ == "__main__":
    asyncio.run(main())
