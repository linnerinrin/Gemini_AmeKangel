"""PathGuard 多根目录与凭据黑名单的边界测试。

背景：只读工具从“仅项目根”放宽到 READONLY_ROOTS（默认用户目录）后，
越界判定改为“命中任一允许根目录”，同时新增凭据类文件黑名单；
这里锁住放松边界后仍不能读到 SSH 私钥、.env、Cookie 等敏感文件。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agent_framework.readonly.path_guard import PathGuard


@pytest.fixture()
def roots(tmp_path: Path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.txt").write_text("a", encoding="utf-8")
    (root_b / "b.txt").write_text("b", encoding="utf-8")
    return root_a, root_b


def test_relative_path_resolves_against_first_root(roots):
    root_a, _ = roots
    guard = PathGuard([str(root_a), str(roots[1])])
    resolved = guard.resolve("a.txt")
    assert resolved.allowed, resolved.reason
    assert Path(resolved.absolute) == (root_a / "a.txt").resolve()


def test_absolute_path_under_second_root_allowed(roots):
    _, root_b = roots
    guard = PathGuard([str(roots[0]), str(root_b)])
    resolved = guard.resolve(str(root_b / "b.txt"))
    assert resolved.allowed, resolved.reason


def test_path_outside_all_roots_denied(roots, tmp_path):
    root_a, root_b = roots
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    guard = PathGuard([str(root_a), str(root_b)])
    resolved = guard.resolve(str(outside))
    assert not resolved.allowed
    assert "越出" in resolved.reason


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        "config/.env",
        ".env.local",
        "llm_settings.json",
        ".ssh/id_rsa",
        "certs/server.pem",
        "deploy.key",
        "Cookies",
        "Login Data",
    ],
)
def test_sensitive_paths_stay_denied_after_widening(roots, relative):
    root_a, _ = roots
    target = root_a / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret", encoding="utf-8")
    guard = PathGuard([str(root_a)])
    resolved = guard.resolve(str(target))
    assert not resolved.allowed, f"{relative} 不应被放行"
    assert "黑名单" in resolved.reason


@pytest.mark.parametrize(
    "raw",
    [
        "../root_b/b.txt",
        "file:///C:/Windows/win.ini",
        "\\\\server\\share\\secret.txt",
    ],
)
def test_escape_attempts_denied(roots, raw):
    guard = PathGuard([str(roots[0]), str(roots[1])])
    assert not guard.resolve(raw).allowed


def test_empty_roots_rejected():
    with pytest.raises(ValueError):
        PathGuard([])
