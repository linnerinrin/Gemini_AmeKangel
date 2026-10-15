import asyncio
import random

from conftest import load_source_module


status_module = load_source_module("status_rules_under_test", "app/config/status_rules.py")


def test_status_change_uses_direction_and_configured_range():
    rng = random.Random(4)
    result = status_module.plan_status_change(
        {"affection": 1, "stress": -1, "mental": 0},
        probability=1.0,
        min_step=2,
        max_step=2,
        rng=rng,
    )
    assert set(result) == {"affection", "stress"}
    assert result["affection"] == 2
    assert result["stress"] == -2


def test_status_change_probability_zero_has_no_effect():
    assert status_module.plan_status_change(
        {"affection": 1, "stress": 1, "mental": 1},
        probability=0,
        min_step=1,
        max_step=5,
        rng=random.Random(1),
    ) == {}


def test_status_change_rejects_invalid_input_before_changes():
    import pytest

    with pytest.raises(status_module.StatusArgumentError):
        status_module.plan_status_change({"affection": 6})
    with pytest.raises(status_module.StatusArgumentError):
        status_module.plan_status_change({"affection": True})
    with pytest.raises(status_module.StatusArgumentError):
        status_module.plan_status_change({"affection": 1}, probability=2)
