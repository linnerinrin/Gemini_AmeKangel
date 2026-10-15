import importlib.util
import sys
import threading
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeCursor:
    def __init__(self, execute_error=None, result=None):
        self.execute_error = execute_error
        self.result = result
        self.calls = []
        self.closed = False

    def execute(self, query, params=None):
        self.calls.append((query, params))
        if self.execute_error:
            error = self.execute_error
            self.execute_error = None
            raise error
        return 1

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursors, ping_error=None):
        self.cursors = list(cursors)
        self.ping_error = ping_error
        self.closed = False
        self.committed = False

    def ping(self, reconnect=True):
        if self.ping_error:
            error = self.ping_error
            self.ping_error = None
            raise error

    def cursor(self):
        return self.cursors.pop(0)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def load_connection_module(monkeypatch, connections):
    fake_pymysql = types.ModuleType("pymysql")

    class DatabaseError(Exception):
        pass

    class InterfaceError(DatabaseError):
        pass

    class OperationalError(DatabaseError):
        pass

    fake_pymysql.Error = DatabaseError
    fake_pymysql.InterfaceError = InterfaceError
    fake_pymysql.OperationalError = OperationalError
    fake_pymysql.connect = lambda **kwargs: connections.pop(0)
    monkeypatch.setitem(sys.modules, "pymysql", fake_pymysql)

    path = PROJECT_ROOT / "repository" / "conn.py"
    spec = importlib.util.spec_from_file_location("connection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconnects_before_query_when_ping_fails(monkeypatch):
    stale = FakeConnection([FakeCursor()])
    healthy_cursor = FakeCursor(result=("session-1",))
    healthy = FakeConnection([healthy_cursor])
    module = load_connection_module(monkeypatch, [healthy])
    connection = module.dbConnection()
    connection._local.conn = stale
    connection._local.cursor = FakeCursor()
    stale.ping_error = module.sql.InterfaceError(0, "")

    assert connection("select id from session") == 1
    assert connection.fetchone() == ("session-1",)
    assert stale.closed
    assert healthy_cursor.calls == [("select id from session", None)]


def test_retries_read_once_after_execute_disconnect(monkeypatch):
    failed_cursor = FakeCursor()
    healthy_cursor = FakeCursor(result=[])
    first = FakeConnection([failed_cursor])
    second = FakeConnection([healthy_cursor])
    module = load_connection_module(monkeypatch, [first, second])
    failed_cursor.execute_error = module.sql.InterfaceError(0, "")
    connection = module.dbConnection()

    assert connection("SELECT id FROM session WHERE session_num=%s", (1,)) == 1
    assert first.closed
    assert healthy_cursor.calls == [("SELECT id FROM session WHERE session_num=%s", (1,))]


def test_does_not_retry_write_after_execute_disconnect(monkeypatch):
    failed_cursor = FakeCursor()
    first = FakeConnection([failed_cursor])
    second = FakeConnection([FakeCursor()])
    module = load_connection_module(monkeypatch, [first, second])
    failed_cursor.execute_error = module.sql.InterfaceError(0, "")
    connection = module.dbConnection()

    try:
        connection("insert into session (id) values (%s)", ("session-1",))
    except module.sql.InterfaceError:
        pass
    else:
        raise AssertionError("write disconnect must be propagated")

    assert len(second.cursors) == 1


def test_uses_separate_connections_per_thread(monkeypatch):
    first_cursor = FakeCursor()
    second_cursor = FakeCursor()
    module = load_connection_module(
        monkeypatch,
        [FakeConnection([first_cursor]), FakeConnection([second_cursor])],
    )
    connection = module.dbConnection()

    threads = [threading.Thread(target=connection, args=("select 1",)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert first_cursor.calls == [("select 1", None)]
    assert second_cursor.calls == [("select 1", None)]
