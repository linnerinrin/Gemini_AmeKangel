# """
#架构作用：为既有chat表补充新字段，兼容旧库并保持重复执行幂等
#调用方：数据库初始化路径；接收SQL执行callable，不拥有连接生命周期
#数据流：模块内DDL →连接执行/提交→按数据库身份区分的进程内就绪标记
#就绪标记绑定配置库名：热切换MYSQL_DATABASE后身份失配，迁移会重新执行
# """
"""
config/schema.py表结构迁移（幂等）

新库由repository/init_db.py的create table直接建全；
已经存在的老库不会因为create table if not exists而多出列，所以需要这里补
"""

MYSQL_DUPLICATE_COLUMN = 1060

THINKING_COLUMN_DDL = 'alter table chat add column thinking mediumtext null;'

#每轮实际使用的模型：llm_model是模型版本，llm_provider是厂商（同名模型靠它分辨）
#两列都允许NULL：老数据拿不到模型信息，也不能让落库失败
LLM_MODEL_COLUMN_DDL = "alter table chat add column llm_model varchar(64) null;"
LLM_PROVIDER_COLUMN_DDL = "alter table chat add column llm_provider varchar(32) null;"

# """
#读取当前配置的目标库名作为就绪标记的身份；惰性import避免模块级依赖config
#读取失败返回None，此时退化为旧行为（不区分身份）
# """
def _current_db_identity():
    try:
        from app.config.config import config
        return getattr(config, "MYSQL_DATABASE", None)
    except Exception:
        return None


# Per-process state: only a successful ALTER or a duplicate-column response is ready.
#布尔标记保留原语义；*_ready_db记录成功时的数据库身份，身份变化（配置热切换）视为未就绪
_ready = False
_llm_model_ready = False
_llm_provider_ready = False
_ready_db = None
_llm_model_ready_db = None
_llm_provider_ready_db = None


def add_thinking_column(connection, *, raise_on_error=False):
    """Ensure ``chat.thinking`` exists without destroying existing data.

    A duplicate-column response is success: another initializer already added it.
    Unexpected failures leave the state unready, so a later explicit retry is possible.
    """
    global _ready, _ready_db
    db_id = _current_db_identity()
    if _ready and _ready_db == db_id:
        return True
    try:
        connection(THINKING_COLUMN_DDL)
        connection.commit()
        _ready = True
        _ready_db = db_id
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _ready = True
            _ready_db = db_id
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.thinking 数据库迁移") from exc
        print(f"thinking 列迁移跳过：{exc}")
        return False


def add_llm_model_column(connection, *, raise_on_error=False):
    """Ensure ``chat.llm_model`` exists without destroying existing data.

    Same idempotency rules as thinking: a duplicate-column response is
    success and unexpected failures stay unready so a later retry is possible.
    """
    global _llm_model_ready, _llm_model_ready_db
    db_id = _current_db_identity()
    if _llm_model_ready and _llm_model_ready_db == db_id:
        return True
    try:
        connection(LLM_MODEL_COLUMN_DDL)
        connection.commit()
        _llm_model_ready = True
        _llm_model_ready_db = db_id
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _llm_model_ready = True
            _llm_model_ready_db = db_id
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.llm_model 数据库迁移") from exc
        print(f"llm_model 列迁移跳过：{exc}")
        return False


def add_llm_provider_column(connection, *, raise_on_error=False):
    """Ensure ``chat.llm_provider`` exists without destroying existing data.

    Duplicate column (1060) counts as success; unexpected failures stay unready.
    """
    global _llm_provider_ready, _llm_provider_ready_db
    db_id = _current_db_identity()
    if _llm_provider_ready and _llm_provider_ready_db == db_id:
        return True
    try:
        connection(LLM_PROVIDER_COLUMN_DDL)
        connection.commit()
        _llm_provider_ready = True
        _llm_provider_ready_db = db_id
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _llm_provider_ready = True
            _llm_provider_ready_db = db_id
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.llm_provider 数据库迁移") from exc
        print(f"llm_provider 列迁移跳过：{exc}")
        return False


def reset_state_for_tests():
    """Reset process-local state; production code should never call this."""
    global _ready, _llm_model_ready, _llm_provider_ready
    global _ready_db, _llm_model_ready_db, _llm_provider_ready_db
    _ready = False
    _llm_model_ready = False
    _llm_provider_ready = False
    _ready_db = None
    _llm_model_ready_db = None
    _llm_provider_ready_db = None
