"""仓库内的包名不得遮蔽第三方顶层模块。

背景（真实事故）：MCP 服务最初放在 `app/mcp/`。PyCharm 之类运行方式会把**脚本所在目录**
（即 `app/`）连同仓库根一起放进 `sys.path`，于是 `import mcp` 解析到我们自己的 `app/mcp`
（是包），`app/mcp/server.py` 里的 `from mcp.server.mcpserver import MCPServer` 就变成
`'mcp.server' is not a package` —— 直接启动失败。

这个坑在 pytest 下永远复现不出来：pytest 从仓库根以 `app.mcp` 的形式导入，顶层 `mcp`
仍然是 SDK。因此本测试**必须用子进程**，在干净解释器里人为把 `app/` 放到 `sys.path` 前面，
再检查这些第三方模块是否解析到了仓库内部。
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 容易被 app/ 下的同名子包遮蔽、且本项目实际依赖的顶层模块
PROBE_MODULES = (
    "mcp",
    "openai",
    "fastapi",
    "starlette",
    "pydantic",
    "langchain_core",
    "uvicorn",
    "anthropic",
    "google.genai",
)


def test_repo_packages_do_not_shadow_third_party_modules():
    # 注意判据：不能只判断"路径在仓库内"——venv/ 就在仓库里，那样所有 site-packages
    # 模块都会被误判为被遮蔽。真正的遮蔽是"从被加进 sys.path 的 app/ 目录解析出来"。
    shadow_root = REPO_ROOT / "app"
    code = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"sys.path.insert(0, {str(shadow_root)!r})\n"
        f"shadow_root = {str(shadow_root)!r}\n"
        f"names = {list(PROBE_MODULES)!r}\n"
        "shadowed = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        mod = importlib.import_module(name)\n"
        "    except Exception:\n"
        "        continue\n"          # 没装的模块不参与判断
        "    path = getattr(mod, '__file__', '') or ''\n"
        "    if path.startswith(shadow_root):\n"
        "        shadowed[name] = path\n"
        "print(json.dumps(shadowed, ensure_ascii=False))\n"
        "raise SystemExit(1 if shadowed else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, (
        "以下第三方模块被仓库内的同名包遮蔽了。任何把 app/ 放进 sys.path 的运行方式"
        "（例如 PyCharm 直接运行 app/main.py）都会因此崩溃：\n"
        f"{result.stdout.strip()}\n--- stderr ---\n{result.stderr}"
    )
    assert json.loads(result.stdout.strip() or "{}") == {}
