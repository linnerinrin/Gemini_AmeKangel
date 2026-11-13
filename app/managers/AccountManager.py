# """
#架构作用：存档仓储与业务规则入口，由聊天路由调用，借助全局db执行同步SQL
#数据流：前端存档位编号→ session表→字符串session_id；本类不绑定聊天上下文
#事务边界：创建和删除显式commit，异常回滚依赖调用体系，目前没有统一工作单元
# """
"""
managers/AccountManager单会话管理器用于提供目标session_id
Abstract on IAccount

"""

import secrets

from app.config.config import config
from app.interfaces.IAccount import IAccount
from repository.conn import db


class AccountManager(IAccount):
    # """
    #生成十八位随机候选ID，再查库避开已存在ID；碰撞时递增重试
    #返回整数，公开接口再转字符串，避免前端JavaScript大整数精度损失
    #查重与插入是分离操作；唯一约束才是数据库侧的最终一致性保障
    # """
    def generate_id(self):
        sid = secrets.randbelow(9 * 10 ** 17) + 10 ** 17
        while True:
            db("SELECT 1 FROM session WHERE id = %s", (str(sid),))
            if not db.fetchone():
                return sid
            sid += 1

    # """
    #只列出存档标识和槽位，按槽位排序；不加载历史、状态，也不切换当前会话
    # SQL返回元组后在边界处转换为JSON友好的字段类型
    # """
    def list_sessions(self):
        db('select id, session_num from session order by session_num asc;')
        return [
            {"session_id": str(row[0]), "session_num": int(row[1])}
            for row in db.fetchall()
        ]

    # """
    #输入槽位只允许1..3；已有存档直接返回，避免重复点击创建新的逻辑存档
    #新建时写入配置中的初始状态，commit成功后才把ID返回调用方
    #容量查询属于应用检查；跨请求并发仍需数据库约束，不能仅凭查询保证互斥
    # """
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

    # """
    #将槽位解析成ID；不存在返回None，绑定与合法性处理由上层负责
    # """
    def chose_session(self, session_num):
        db('select id from session where session_num=%s', (session_num,))
        row = db.fetchone()
        return str(row[0]) if row else None

    # """
    #先删子表历史，再删session，最后提交；避免正常路径违反外键关联
    # """
    def delete_session(self, session_num):
        db('delete from chat where session_id in (select id from session where session_num=%s);', (session_num,))
        db('delete from session where session_num=%s;', (session_num,))
        db.commit()
        return True
