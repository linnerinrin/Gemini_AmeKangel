import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "api_contract.json"
# 契约里表示"这一项允许为 null"的前缀，见 assert_shape
NULLABLE_PREFIX = "nullable:"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PlaceholderDB:
    def __call__(self, *args, **kwargs):
        return None

    def commit(self):
        return None

    def fetchall(self):
        return []

    def fetchone(self):
        return None


repository_module = types.ModuleType("repository")
repository_conn_module = types.ModuleType("repository.conn")
repository_conn_module.db = PlaceholderDB()
sys.modules.setdefault("repository", repository_module)
sys.modules.setdefault("repository.conn", repository_conn_module)


@pytest.fixture(scope="session")
def contracts():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def contract_entries(contracts):
    """把契约文件展平成 [(path, entry)]，每个 entry 是一个具体 method 的登记。

    一个路径可能有多个 method（例如 /chat/settings/llm 的 GET/POST），所以统一用
    value["methods"] 列表登记；两个契约测试都按条目迭代，新登记会自然被覆盖。
    """
    return [
        (path, entry)
        for path, value in contracts.items()
        for entry in value["methods"]
    ]


class FakeAccountManager:
    def __init__(self):
        self.sessions = {}

    def list_sessions(self):
        return [
            {"session_id": session_id, "session_num": session_num}
            for session_num, session_id in sorted(self.sessions.items())
        ]

    def create_new_session(self, session_num):
        session_id = self.sessions.get(session_num) or f"session-{session_num}"
        self.sessions[session_num] = session_id
        return session_id

    def delete_session(self, session_num):
        self.sessions.pop(session_num, None)
        return True

    def chose_session(self, session_num):
        return self.sessions.get(session_num)


class FakeMemoryManager:
    def load_status(self):
        return {"followers": 0, "affection": 50, "stress": 0, "mental": 0}


class FakeChatService:
    def __init__(self):
        self.history = []
        self.history_by_mode = {}
        self.memory_mgr = FakeMemoryManager()
        self.mode = None
        self.session_id = None
        # 记录路由透传过来的任务轴，供 API 层测试断言
        self.last_task_mode = None

    def bind_session_id(self, session_id):
        self.session_id = session_id

    def clear_session(self):
        self.session_id = None
        self.history = []

    async def single_chat(self, prompt, mode=None, task_mode=None):
        self.mode = mode if mode is not None else self.mode
        self.last_task_mode = task_mode
        return "回复", "思考"

    async def stream_chat(self, prompt, mode=None, on_event=None, task_mode=None):
        self.mode = mode if mode is not None else self.mode
        self.last_task_mode = task_mode
        if on_event:
            await on_event({"type": "status", "stage": "thinking"})
            await on_event({"type": "thinking_delta", "text": "思考"})
            await on_event({"type": "answer_delta", "text": "回复"})
            await on_event({"type": "done", "answer": "回复", "think": "思考"})
        return "回复", "思考"

    def load_mode(self, mode):
        self.mode = mode

    def read_history(self, mode=None):
        """按 mode(直播主题) 读取历史，供前端回填各自的上下文"""
        if mode is None:
            return list(self.history)
        return list(self.history_by_mode.get(str(mode), []))


@pytest.fixture
def fake_services(monkeypatch):
    services = types.ModuleType("app.services")
    services.chat = FakeChatService()
    monkeypatch.setitem(sys.modules, "app.services", services)

    managers = types.ModuleType("app.managers")
    managers.account_manager = FakeAccountManager()
    monkeypatch.setitem(sys.modules, "app.managers", managers)

    for name in list(sys.modules):
        if name == "app.main" or name == "app.api" or name.startswith("app.api."):
            sys.modules.pop(name, None)
    return services


@pytest.fixture
def client(fake_services):
    module = importlib.import_module("app.main")
    return TestClient(module.app)


def assert_shape(actual, expected):
    if expected == "string":
        assert isinstance(actual, str)
        return
    if expected == "number":
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        return
    if expected == "boolean":
        assert isinstance(actual, bool)
        return
    if expected == "dict":
        assert isinstance(actual, dict)
        return
    if expected == "list":
        assert isinstance(actual, list)
        return
    if isinstance(expected, str) and expected.startswith(NULLABLE_PREFIX):
        # "nullable:string"：老数据（如新增列之前的历史行）允许是 None
        if actual is None:
            return
        assert_shape(actual, expected[len(NULLABLE_PREFIX):])
        return
    if isinstance(expected, bool):
        assert actual is expected
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            assert_shape(actual[key], value)
        return
    assert actual == expected


def load_source_module(module_name, relative_path):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
