"""
repository/init_db.py  建库 / 建表 / 表结构迁移

直接运行本文件即可初始化数据库；import 本模块不会有任何副作用。
"""
from app.config.schema import (
    add_llm_model_column,
    add_llm_provider_column,
    add_task_mode_column,
    add_thinking_column,
)
from repository.conn import db


def drop():
    db('use gemini;')
    db('drop table chat')
    db('drop table session')


def migrate(connection=None, *, raise_on_error=False):
    """幂等补充 chat 的 thinking / task_mode / llm_model / llm_provider 四列；
    可在启动阶段要求失败即中止启动。"""
    conn = connection or db
    thinking_ok = add_thinking_column(conn, raise_on_error=raise_on_error)
    task_mode_ok = add_task_mode_column(conn, raise_on_error=raise_on_error)
    llm_model_ok = add_llm_model_column(conn, raise_on_error=raise_on_error)
    llm_provider_ok = add_llm_provider_column(conn, raise_on_error=raise_on_error)
    return thinking_ok and task_mode_ok and llm_model_ok and llm_provider_ok


def init():
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

    #单次问答表（thinking 存该轮的思考链 JSON，历史回放用；
    #task_mode 是任务轴 chat/work，与场景轴 mode 正交；
    #llm_model/llm_provider 记该轮实际使用的模型版本与厂商，老库由 migrate() 补列）
    db('''create table if not exists chat(
        session_id varchar(18) not null,
        mode varchar(20) default '0',
        task_mode varchar(8) not null default 'chat',
        llm_model varchar(64) null,
        llm_provider varchar(32) null,
        query text,
        reply text,
        thinking mediumtext,
        created_at datetime default current_timestamp,
        foreign key (session_id) references session(id));
        ''')

    # 老库不会因为 create table if not exists 自动新增列，这里做幂等迁移
    migrate(raise_on_error=True)


if __name__ == '__main__':
    init()
