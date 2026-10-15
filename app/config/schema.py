"""
config/schema.py  表结构迁移（幂等）

新库由 repository/init_db.py 的 create table 直接建全；
已经存在的老库不会因为 create table if not exists 而多出列，所以需要这里补。
"""

MYSQL_DUPLICATE_COLUMN = 1060

THINKING_COLUMN_DDL = 'alter table chat add column thinking mediumtext null;'

# 任务轴：mode 是场景轴（糖糖私聊/各直播间），task_mode 是任务轴（chat/work），两者正交
TASK_MODE_COLUMN_DDL = "alter table chat add column task_mode varchar(8) not null default 'chat';"

# 每轮实际使用的模型：llm_model 是模型版本，llm_provider 是厂商（同名模型靠它分辨）
# 两列都允许 NULL：老数据拿不到模型信息，也不能让落库失败
LLM_MODEL_COLUMN_DDL = "alter table chat add column llm_model varchar(64) null;"
LLM_PROVIDER_COLUMN_DDL = "alter table chat add column llm_provider varchar(32) null;"

# Per-process state: only a successful ALTER or a duplicate-column response is ready.
_ready = False
_task_mode_ready = False
_llm_model_ready = False
_llm_provider_ready = False


def add_thinking_column(connection, *, raise_on_error=False):
    """Ensure ``chat.thinking`` exists without destroying existing data.

    A duplicate-column response is success: another initializer already added it.
    Unexpected failures leave the state unready, so a later explicit retry is possible.
    """
    global _ready
    if _ready:
        return True
    try:
        connection(THINKING_COLUMN_DDL)
        connection.commit()
        _ready = True
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _ready = True
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.thinking 数据库迁移") from exc
        print(f"thinking 列迁移跳过：{exc}")
        return False


def add_task_mode_column(connection, *, raise_on_error=False):
    """Ensure ``chat.task_mode`` exists without destroying existing data.

    Same idempotency rules as thinking: a duplicate-column response is success and
    unexpected failures stay unready so a later explicit retry is possible.
    """
    global _task_mode_ready
    if _task_mode_ready:
        return True
    try:
        connection(TASK_MODE_COLUMN_DDL)
        connection.commit()
        _task_mode_ready = True
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _task_mode_ready = True
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.task_mode 数据库迁移") from exc
        print(f"task_mode 列迁移跳过：{exc}")
        return False


def add_llm_model_column(connection, *, raise_on_error=False):
    """Ensure ``chat.llm_model`` exists without destroying existing data.

    Same idempotency rules as thinking/task_mode: a duplicate-column response is
    success and unexpected failures stay unready so a later retry is possible.
    """
    global _llm_model_ready
    if _llm_model_ready:
        return True
    try:
        connection(LLM_MODEL_COLUMN_DDL)
        connection.commit()
        _llm_model_ready = True
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _llm_model_ready = True
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.llm_model 数据库迁移") from exc
        print(f"llm_model 列迁移跳过：{exc}")
        return False


def add_llm_provider_column(connection, *, raise_on_error=False):
    """Ensure ``chat.llm_provider`` exists without destroying existing data.

    Duplicate column (1060) counts as success; unexpected failures stay unready.
    """
    global _llm_provider_ready
    if _llm_provider_ready:
        return True
    try:
        connection(LLM_PROVIDER_COLUMN_DDL)
        connection.commit()
        _llm_provider_ready = True
        return True
    except Exception as exc:
        if getattr(exc, "args", (None,))[:1] == (MYSQL_DUPLICATE_COLUMN,):
            _llm_provider_ready = True
            return True
        if raise_on_error:
            raise RuntimeError("无法完成 chat.llm_provider 数据库迁移") from exc
        print(f"llm_provider 列迁移跳过：{exc}")
        return False


def reset_state_for_tests():
    """Reset process-local state; production code should never call this."""
    global _ready, _task_mode_ready, _llm_model_ready, _llm_provider_ready
    _ready = False
    _task_mode_ready = False
    _llm_model_ready = False
    _llm_provider_ready = False
