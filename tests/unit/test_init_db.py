"""建表与迁移：thinking/task_mode 两列的幂等补充，失败不能挡住启动"""
import pymysql
import pytest

from conftest import load_source_module
from app.config import schema as schema_module


init_db_module = load_source_module("init_db_under_test", "repository/init_db.py")


class FakeDB:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.committed = False

    def __call__(self, sql, params=None):
        self.calls.append(sql)
        if self.error is not None and sql.lstrip().upper().startswith("ALTER"):
            raise self.error
        return 1

    def commit(self):
        self.committed = True

    def fetchone(self):
        return None

    def fetchall(self):
        return []


def test_migrate_adds_column_and_commits(monkeypatch):
    schema_module.reset_state_for_tests()
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    assert init_db_module.migrate() is True
    assert any("alter table chat add column thinking" in sql for sql in fake_db.calls)
    assert any("alter table chat add column task_mode" in sql for sql in fake_db.calls)
    assert fake_db.committed
    assert schema_module._ready is True
    assert schema_module._task_mode_ready is True


def test_migrate_treats_duplicate_column_as_ready(monkeypatch):
    """并发启动或已有列时，1060 等价于迁移成功。"""
    schema_module.reset_state_for_tests()
    fake_db = FakeDB(error=pymysql.err.OperationalError(1060, "Duplicate column name 'thinking'"))
    monkeypatch.setattr(init_db_module, "db", fake_db)

    assert init_db_module.migrate() is True
    assert schema_module._ready is True
    assert schema_module._task_mode_ready is True
    assert schema_module._llm_model_ready is True
    assert schema_module._llm_provider_ready is True


def test_migrate_adds_llm_model_and_provider_columns(monkeypatch):
    """R12：chat 表要补 llm_model / llm_provider 两列（老库靠幂等 ALTER）"""
    schema_module.reset_state_for_tests()
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    assert init_db_module.migrate() is True
    assert any("alter table chat add column llm_model varchar(64)" in sql for sql in fake_db.calls)
    assert any("alter table chat add column llm_provider varchar(32)" in sql for sql in fake_db.calls)
    assert schema_module._llm_model_ready is True
    assert schema_module._llm_provider_ready is True


def test_llm_column_migration_is_idempotent(monkeypatch):
    """重复列（1060）视为迁移成功；失败不能被缓存成 ready。"""
    schema_module.reset_state_for_tests()
    duplicate_db = FakeDB(error=pymysql.err.OperationalError(1060, "Duplicate column name 'llm_model'"))
    monkeypatch.setattr(init_db_module, "db", duplicate_db)
    assert init_db_module.migrate() is True
    assert schema_module._llm_model_ready is True
    assert schema_module._llm_provider_ready is True

    schema_module.reset_state_for_tests()

    class LlmFailDB(FakeDB):
        def __call__(self, sql, params=None):
            self.calls.append(sql)
            if "llm_model" in sql or "llm_provider" in sql:
                raise pymysql.err.OperationalError(1045, "Access denied")
            return 1

    monkeypatch.setattr(init_db_module, "db", LlmFailDB())
    assert init_db_module.migrate() is False
    assert schema_module._llm_model_ready is False
    assert schema_module._llm_provider_ready is False
    assert schema_module._ready is True       # 前两列已就绪，不受影响
    assert schema_module._task_mode_ready is True


def test_migration_failure_leaves_state_retryable(monkeypatch, capsys):
    """失败不能被缓存为 ready，后续调用必须重试。"""
    schema_module.reset_state_for_tests()
    failed_db = FakeDB(error=pymysql.err.OperationalError(1045, "Access denied"))
    good_db = FakeDB()

    assert init_db_module.migrate(failed_db) is False
    assert schema_module._ready is False
    assert schema_module._task_mode_ready is False
    assert init_db_module.migrate(good_db) is True
    assert schema_module._ready is True
    assert schema_module._task_mode_ready is True
    assert any("alter table chat add column thinking" in sql for sql in good_db.calls)
    assert any("alter table chat add column task_mode" in sql for sql in good_db.calls)
    out = capsys.readouterr().out
    assert "thinking 列迁移跳过" in out
    assert "task_mode 列迁移跳过" in out


def test_startup_migration_failure_fails_fast(monkeypatch):
    """应用启动不能继续到接收请求，却在 INSERT 时才因缺列失败。"""
    schema_module.reset_state_for_tests()
    fake_db = FakeDB(error=pymysql.err.OperationalError(1045, "Access denied"))
    monkeypatch.setattr(init_db_module, "db", fake_db)

    with pytest.raises(RuntimeError, match="chat.thinking"):
        init_db_module.migrate(raise_on_error=True)
    assert schema_module._ready is False


def test_startup_task_mode_migration_failure_fails_fast(monkeypatch):
    """thinking 列已就绪、task_mode 列失败时，启动同样要中止。"""

    class TaskModeFailDB(FakeDB):
        def __call__(self, sql, params=None):
            self.calls.append(sql)
            if "task_mode" in sql:
                raise pymysql.err.OperationalError(1045, "Access denied")
            return 1

    schema_module.reset_state_for_tests()
    monkeypatch.setattr(init_db_module, "db", TaskModeFailDB())

    with pytest.raises(RuntimeError, match="chat.task_mode"):
        init_db_module.migrate(raise_on_error=True)
    assert schema_module._ready is True
    assert schema_module._task_mode_ready is False


def test_init_creates_table_with_thinking_column(monkeypatch):
    schema_module.reset_state_for_tests()
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists chat" in sql]
    assert create and "thinking mediumtext" in create[0]
    assert schema_module._ready is True


def test_init_creates_table_with_task_mode_column(monkeypatch):
    """新库直接建全：chat 表要自带任务轴列，且建表后两列迁移都算就绪。"""
    schema_module.reset_state_for_tests()
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists chat" in sql]
    assert create and "task_mode varchar(8) not null default 'chat'" in create[0]
    # 任务轴列必须紧跟在场景轴 mode 后面
    body = create[0].split("mode varchar(20) default '0',")[1]
    assert body.lstrip().startswith("task_mode")
    assert any("alter table chat add column task_mode" in sql for sql in fake_db.calls)
    assert schema_module._task_mode_ready is True


def test_init_creates_table_with_llm_columns(monkeypatch):
    """R12：新库建表就带 llm_model / llm_provider，且紧跟在 task_mode 之后。"""
    schema_module.reset_state_for_tests()
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists chat" in sql]
    assert create and "llm_model varchar(64) null" in create[0]
    assert "llm_provider varchar(32) null" in create[0]
    body = create[0].split("task_mode varchar(8) not null default 'chat',")[1]
    assert body.lstrip().startswith("llm_model")
    assert schema_module._llm_model_ready is True
    assert schema_module._llm_provider_ready is True


def test_importing_init_db_has_no_side_effects(monkeypatch):
    """import 不能顺手连库：以前模块级直接 init()，测试一导入就会去连 MySQL"""
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)
    fake_db.calls.clear()

    load_source_module("init_db_again_under_test", "repository/init_db.py")

    assert fake_db.calls == []
