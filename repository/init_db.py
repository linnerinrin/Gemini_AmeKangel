#"""
#架构作用：启动生命周期调用init建表，migrate委托schema模块补历史列
#数据流：无库引导连接建库→共享db门面→ DDL →四项幂等迁移；不在import时执行DDL
#"""
"""
repository/init_db.py建库/建表/表结构迁移

直接运行本文件即可初始化数据库；import本模块不会有任何副作用
"""
import pymysql as sql

from app.config.config import config
from app.config.schema import (
    add_llm_model_column,
    add_llm_provider_column,
    add_thinking_column,
)
from repository.conn import db


#"""
#冷启动引导：db门面的连接直配目标库，库不存在时连不上就没法建库；
#先用不指定database的引导连接执行CREATE DATABASE IF NOT EXISTS，再走正常连接
#"""
def _ensure_database():
    conn = sql.connect(
        host="localhost",
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f'create database if not exists {config.MYSQL_DATABASE} character set utf8mb4;'
            )
    finally:
        conn.close()


#"""
#显式破坏性维护入口：先删依赖外键的chat，再删session；正常init不调用它
#DDL不能按普通业务事务假设可回滚，调用者应自行确认目标和备份
#"""
def drop():
    db('use gemini;')
    db('drop table chat')
    db('drop table session')


#"""
#可注入连接便于离线测试；四个迁移按顺序执行，汇总布尔结果
#raise_on_error在启动传True，使迁移异常阻止应用以不兼容结构继续运行
#"""
def migrate(connection=None, *, raise_on_error=False):
    """幂等补充chat的thinking / llm_model / llm_provider三列；
    可在启动时要求失败即中止启动。"""
    conn = connection or db
    thinking_ok = add_thinking_column(conn, raise_on_error=raise_on_error)
    llm_model_ok = add_llm_model_column(conn, raise_on_error=raise_on_error)
    llm_provider_ok = add_llm_provider_column(conn, raise_on_error=raise_on_error)
    return thinking_ok and llm_model_ok and llm_provider_ok


#"""
#建库、选择库、创建父表和子表后迁移；IF NOT EXISTS不升级已有表的约束
#"""
def init():
    #先确保目标库存在（无库引导连接），否则db首次连接会因库不存在直接失败
    _ensure_database()
    db('create database if not exists gemini character set utf8mb4;')
    db('use gemini;')

    #单个会话信息表
    db('''create table if not exists session(
        id varchar(18) primary key,
        session_num int not null,
        memory text,
        followers int default 0,
        affection int default 50,
        stress int default 0,
        mental int default 0,
        created_at datetime default current_timestamp,
        unique key uq_session_num(session_num));
        ''')

    #单次问答表（thinking存该轮的思考链JSON，历史回放用；
    #llm_model/llm_provider记该轮实际使用的模型版本与厂商，老库由migrate()补列）
    db('''create table if not exists chat(
        session_id varchar(18) not null,
        mode varchar(20) default '0',
        llm_model varchar(64) null,
        llm_provider varchar(32) null,
        query text,
        reply text,
        thinking mediumtext,
        created_at datetime default current_timestamp,
        foreign key (session_id) references session(id));
        ''')

    #老库不会因为create table if not exists自动新增列，这里做幂等迁移
    migrate(raise_on_error=True)


if __name__ == '__main__':
    init()
