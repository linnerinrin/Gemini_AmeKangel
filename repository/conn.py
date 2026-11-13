# """
#架构作用：同步PyMySQL访问门面，由账号、记忆、建库模块共享db对象
#数据流：参数化SQL →当前线程连接/游标→ fetch/commit；首次执行才实际连接
# local隔离线程而非asyncio任务；同线程协程共享连接，不能据此认定无await片段竞态
#本层没有事务上下文，调用者负责显式commit；写操作驱动异常时自动rollback
# """
from threading import local

import pymysql as sql

from app.config.config import config


class dbConnection:
    _READ_STATEMENTS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}

    # """
    #只创建线程本地容器，此时尚无网络连接；游标在首次SQL调用时创建
    # """
    def __init__(self):
        self._local = local()

    # """
    #连接成功后发布到当前线程；直接指定配置库名，不提供无库的引导连接
    #目标库尚不存在的冷启动由init_db._ensure_database()用无库引导连接先建库
    # """
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

    # """
    #尽力关闭游标和连接，再清引用；清理异常只吞驱动错误，便于重新建立连接
    #关闭连接不等于恢复调用方已经完成的提交，也不构成多SQL的事务协议
    # """
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

    # """
    #惰性连接并ping保活，每次执行前关闭旧游标；fetch必须对应最近一次执行
    #重连只恢复连接可用性，不重放此前事务，也不能恢复未提交业务上下文
    # """
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

    # """
    #异常路径上的尽力回滚：写操作失败后连接不能留在坏事务里；
    #连接可能已断开或已损坏，回滚本身的任何失败都吞掉，不影响原异常继续抛出
    # """
    def _rollback_quietly(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass

    # """
    #依据首个SQL关键词区分可重试读取；断连后清理，写入直接抛错以免盲目重复副作用
    #写操作任何驱动异常都先rollback再抛，读重试只恢复连接
    #参数交给驱动绑定；这里只重试一次读取，不把事务commit自动化
    # """
    def __call__(self, query, params=None):
        statement = query.lstrip().split(None, 1)[0].upper()
        is_read = statement in self._READ_STATEMENTS
        cursor = self._prepare_cursor()
        try:
            return cursor.execute(query, params) if params is not None else cursor.execute(query)
        except (sql.InterfaceError, sql.OperationalError):
            if not is_read:
                self._rollback_quietly()
                self._invalidate()
                raise
            self._invalidate()
            cursor = self._prepare_cursor()
            return cursor.execute(query, params) if params is not None else cursor.execute(query)
        except sql.Error:
            #写路径上的非断连错误（如约束冲突）同样要回滚，避免半截事务残留
            if not is_read:
                self._rollback_quietly()
            raise

    # """
    #提交当前线程连接的全部待提交变更，而非仅最近SQL；没有连接则明确报错
    # """
    def commit(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            raise RuntimeError("No database connection is active")
        conn.commit()

    # """
    #从最近一次执行的游标取全部结果；不创建连接，也不替调用者执行查询
    # """
    def fetchall(self):
        return self._local.cursor.fetchall()

    # """
    #取下一行（无记录为None），调用者须自行处理缺失；游标状态属于线程
    # """
    def fetchone(self):
        return self._local.cursor.fetchone()

    # """
    #显式释放本线程资源；不会遍历和关闭其他线程的连接
    # """
    def close(self):
        self._invalidate()


db = dbConnection()
