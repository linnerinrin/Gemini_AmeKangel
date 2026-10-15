"""
managers/AccountManager 单会话管理器 用于提供目标session_id
Abstract on IAccount

"""

import secrets

from app.config.config import config
from app.interfaces.IAccount import IAccount
from repository.conn import db


class AccountManager(IAccount):
    def generate_id(self):
        sid = secrets.randbelow(9 * 10 ** 17) + 10 ** 17
        while True:
            db("SELECT 1 FROM session WHERE id = %s", (str(sid),))
            if not db.fetchone():
                return sid
            sid += 1

    def list_sessions(self):
        db('select id, session_num from session order by session_num asc;')
        return [
            {"session_id": str(row[0]), "session_num": int(row[1])}
            for row in db.fetchall()
        ]

    def create_new_session(self, session_num):
        if session_num < 1 or session_num > 3:
            return None
        db('select id from session where session_num=%s;', (session_num,))
        existing = db.fetchone()
        if existing:
            return str(existing[0])
        db('select id from session')
        if len(db.fetchall()) >= 3:
            return None
        session_id = self.generate_id()
        db(
            'insert into session (id, session_num, followers, affection, stress, mental) '
            'values (%s, %s, %s, %s, %s, %s);',
            (str(session_id), session_num, config.DEFAULT_FOLLOWERS,
             config.DEFAULT_AFFECTION, config.DEFAULT_STRESS, config.DEFAULT_MENTAL),
        )
        db.commit()
        return str(session_id)

    def chose_session(self, session_num):
        db('select id from session where session_num=%s', (session_num,))
        row = db.fetchone()
        return str(row[0]) if row else None

    def delete_session(self, session_num):
        db('delete from chat where session_id in (select id from session where session_num=%s);', (session_num,))
        db('delete from session where session_num=%s;', (session_num,))
        db.commit()
        return True
