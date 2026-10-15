from threading import local

import pymysql as sql

from app.config.config import config


class dbConnection:
    _READ_STATEMENTS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}

    def __init__(self):
        self._local = local()

    def _connect(self):
        conn = sql.connect(
            host="localhost",
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
        )
        self._local.conn = conn
        self._local.cursor = None
        return conn

    def _invalidate(self):
        cursor = getattr(self._local, "cursor", None)
        conn = getattr(self._local, "conn", None)
        if cursor:
            try:
                cursor.close()
            except sql.Error:
                pass
        if conn:
            try:
                conn.close()
            except sql.Error:
                pass
        self._local.cursor = None
        self._local.conn = None

    def _prepare_cursor(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
        else:
            try:
                conn.ping(reconnect=True)
            except (sql.InterfaceError, sql.OperationalError):
                self._invalidate()
                conn = self._connect()

        cursor = getattr(self._local, "cursor", None)
        if cursor:
            cursor.close()
        cursor = conn.cursor()
        self._local.cursor = cursor
        return cursor

    def __call__(self, query, params=None):
        statement = query.lstrip().split(None, 1)[0].upper()
        is_read = statement in self._READ_STATEMENTS
        cursor = self._prepare_cursor()
        try:
            return cursor.execute(query, params) if params is not None else cursor.execute(query)
        except (sql.InterfaceError, sql.OperationalError):
            self._invalidate()
            if not is_read:
                raise
            cursor = self._prepare_cursor()
            return cursor.execute(query, params) if params is not None else cursor.execute(query)

    def commit(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            raise RuntimeError("No database connection is active")
        conn.commit()

    def fetchall(self):
        return self._local.cursor.fetchall()

    def fetchone(self):
        return self._local.cursor.fetchone()

    def close(self):
        self._invalidate()


db = dbConnection()
