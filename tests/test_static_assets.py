import re
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "static"

# 每个 node 子进程的上限。必须存在：node --check 与 *_behavior.cjs 在遇到半成品/
# 写坏的文件时可能既不退出也不报错，没有超时就是整个 pytest 进程静默挂死
# （本项目已发生过两次，各 >10 分钟且输出为空）。
NODE_TIMEOUT_SECONDS = 60


def test_no_legacy_content_references():
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in STATIC_ROOT.rglob("*")
        if path.suffix.lower() in {".html", ".css", ".js"}
    )
    assert "../content" not in text
    assert "../../content" not in text
    assert "/content/" not in text


def test_static_html_and_css_references_exist():
    missing = []
    for path in list(STATIC_ROOT.rglob("*.html")) + list((STATIC_ROOT / "css").rglob("*.css")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r'(?:src|href)="([^"]+)"', text):
            ref = match.group(1).split("?")[0]
            if ref.startswith(("http", "/", "#")):
                continue
            if not (path.parent / ref).resolve().exists():
                missing.append((str(path.relative_to(STATIC_ROOT)), ref))
        for match in re.finditer(r"url\([\"']?([^\)\"']+)", text):
            ref = match.group(1).split("?")[0]
            if ref.startswith(("http", "data:", "/", "#")):
                continue
            if not (path.parent / ref).resolve().exists():
                missing.append((str(path.relative_to(STATIC_ROOT)), ref))
    assert missing == []


def test_assets_are_categorized():
    expected_dirs = [
        "audio",
        "images",
        "stream/ame",
        "stream/kangel",
    ]
    for rel in expected_dirs:
        path = STATIC_ROOT / "assets" / rel
        assert path.is_dir()
        assert any(path.iterdir())


def test_javascript_syntax_check():
    for path in sorted((STATIC_ROOT / "js").rglob("*.js")):
        # 必须带 timeout：node --check 遇到被写坏/半成品的文件可能不退出也不报错，
        # 没有超时会静默挂住整个 pytest（重定向到文件时连一个字都看不到）。
        try:
            result = subprocess.run(
                ["node", "--check", str(path.relative_to(PROJECT_ROOT))],
                cwd=PROJECT_ROOT,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=NODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"node --check 超时 {NODE_TIMEOUT_SECONDS}s：{path.relative_to(PROJECT_ROOT)}（文件可能被写坏或正在被写入）")
        if result.returncode != 0 and "EPERM" in result.stderr and "lstat" in result.stderr:
            pytest.skip(f"node cannot access the project path in this environment: {result.stderr}")
        assert result.returncode == 0, result.stderr


def test_frontend_behavior_suites():
    """把 tests/*_behavior.cjs 纳入 pytest。

    这些行为测试此前只能手动运行，回归时会被漏掉（node --check 只查语法，
    查不出"帧表缺动作"这类静默退化）。
    """
    suites = sorted((PROJECT_ROOT / "tests").glob("*_behavior.cjs"))
    assert suites, "未找到任何前端行为测试"

    for suite in suites:
        # 同 test_javascript_syntax_check：行为脚本里含 fake 定时器与 await，
        # 一旦逻辑写错会永久等待，没有超时就是整个测试进程挂死。
        try:
            result = subprocess.run(
                ["node", str(suite.relative_to(PROJECT_ROOT))],
                cwd=PROJECT_ROOT,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=NODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"{suite.name} 超时 {NODE_TIMEOUT_SECONDS}s（脚本可能在等待永不 resolve 的 Promise）")
        if result.returncode != 0 and "EPERM" in result.stderr and "lstat" in result.stderr:
            pytest.skip(f"node cannot access the project path in this environment: {result.stderr}")
        assert result.returncode == 0, (
            f"{suite.name} 失败:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
