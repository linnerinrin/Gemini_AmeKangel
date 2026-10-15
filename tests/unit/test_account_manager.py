from conftest import load_source_module


account_module = load_source_module("account_manager_under_test", "app/managers/AccountManager.py")


class FakeDB:
    def __init__(self, fetchone_value=None, fetchall_value=None):
        self.fetchone_value = fetchone_value
        self.fetchall_value = fetchall_value or []
        self.calls = []
        self.committed = False

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self.fetchone_value

    def fetchall(self):
        return self.fetchall_value

    def commit(self):
        self.committed = True


def test_list_sessions_maps_db_rows(monkeypatch):
    fake_db = FakeDB(fetchall_value=[("session-1", 1), ("session-2", 2)])
    monkeypatch.setattr(account_module, "db", fake_db)

    assert account_module.AccountManager().list_sessions() == [
        {"session_id": "session-1", "session_num": 1},
        {"session_id": "session-2", "session_num": 2},
    ]


def test_create_new_session_returns_none_for_invalid_slot(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(account_module, "db", fake_db)

    assert account_module.AccountManager().create_new_session(4) is None


def test_create_new_session_reuses_existing_slot(monkeypatch):
    fake_db = FakeDB(fetchone_value=("session-1",))
    monkeypatch.setattr(account_module, "db", fake_db)

    assert account_module.AccountManager().create_new_session(1) == "session-1"
    assert not fake_db.committed


def test_delete_session_commits(monkeypatch):
    fake_db = FakeDB()
    monkeypatch.setattr(account_module, "db", fake_db)

    assert account_module.AccountManager().delete_session(1) is True
    assert fake_db.committed


def test_chose_session_returns_db_value(monkeypatch):
    fake_db = FakeDB(fetchone_value=("session-1",))
    monkeypatch.setattr(account_module, "db", fake_db)

    assert account_module.AccountManager().chose_session(1) == "session-1"
