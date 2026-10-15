"""
config/status_rules.py  状态变化规则

模型负责判断本次用户发言是否影响状态，并为各状态提供方向；实际幅度由系统随机决定。
"""
import random

from app.config.config import config

MAX_MODEL_STEP = 5
AFFECTION = "affection"
STRESS = "stress"
MENTAL = "mental"
DIMENSIONS = (AFFECTION, STRESS, MENTAL)


class StatusArgumentError(ValueError):
    """模型给出的状态方向参数非法。"""


def _bounds(min_step, max_step):
    low = max(0, int(min_step))
    high = max(0, int(max_step))
    return (low, high) if low <= high else (high, low)


def plan_status_change(amounts, *, probability=None, min_step=None, max_step=None, rng=None):
    """校验模型方向并随机化实际状态增量。"""
    probability = config.STATUS_CHANGE_PROBABILITY if probability is None else probability
    low, high = _bounds(
        config.STATUS_CHANGE_MIN_STEP if min_step is None else min_step,
        config.STATUS_CHANGE_MAX_STEP if max_step is None else max_step,
    )
    source = rng if rng is not None else random

    if not 0 <= probability <= 1:
        raise StatusArgumentError("状态变化概率必须在 0 到 1 之间")
    for name in DIMENSIONS:
        value = amounts.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise StatusArgumentError(f"{name} 必须是整数")
        if abs(value) > MAX_MODEL_STEP:
            raise StatusArgumentError(f"{name} 超出 ±{MAX_MODEL_STEP} 范围")

    deltas = {}
    for name in DIMENSIONS:
        direction = amounts.get(name, 0)
        if direction == 0 or source.random() >= probability:
            continue
        magnitude = source.randint(low, high)
        if magnitude:
            deltas[name] = magnitude if direction > 0 else -magnitude
    return deltas
