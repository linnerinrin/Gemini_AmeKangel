"""建库建表：chat表必须一次建全thinking/llm_model/llm_provider三列"""
import pymysql
import pytest

from conftest import load_source_module


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


def test_init_creates_table_with_thinking_column(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists chat" in sql]
    assert create and "thinking mediumtext" in create[0]


def test_init_creates_table_with_llm_columns(monkeypatch):
    """新库建表就带llm_model / llm_provider，且紧跟在场景轴mode之后。"""
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists chat" in sql]
    assert create and "llm_model varchar(64) null" in create[0]
    assert "llm_provider varchar(32) null" in create[0]
    body = create[0].split("mode varchar(20) default '0',")[1]
    assert body.lstrip().startswith("llm_model")


def test_init_creates_session_table(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    create = [sql for sql in fake_db.calls if "create table if not exists session" in sql]
    assert create and "unique key uq_session_num(session_num)" in create[0]


def test_init_does_not_run_migrations(monkeypatch):
    """启动路径只建库建表，不执行任何ALTER。"""
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)

    init_db_module.init()

    alters = [sql for sql in fake_db.calls if sql.lstrip().upper().startswith("ALTER")]
    assert alters == []


def test_importing_init_db_has_no_side_effects(monkeypatch):
    """import不能顺手连库"""
    fake_db = FakeDB()
    monkeypatch.setattr(init_db_module, "db", fake_db)
    fake_db.calls.clear()

    load_source_module("init_db_again_under_test", "repository/init_db.py")

    assert fake_db.calls == []
