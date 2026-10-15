"""
config/llm_import.py  安全加载 app/services/llm_settings.py

为什么不让调用方直接 `from app.services import llm_settings`：
- tests/conftest.py 的 fake_services 夹具会把 app.services 换成没有 __path__ 的
  替身模块（为了不初始化真实的重型服务），此时包级导入会直接抛
  ModuleNotFoundError: 'app.services' is not a package；
- ModelManager 本身又会被 app.services.__init__ 反向导入，包级导入还存在循环。

所以统一按**文件路径**加载，并固定注册成 app.services.llm_settings：
sys.modules 里只有这一份，已加载时直接复用（monkeypatch 也照常生效）。
"""

import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "app.services.llm_settings"


def _module_path() -> Path:
    # 本文件在 app/config/ 下，项目根往上一级，再进 app/services/
    return Path(__file__).resolve().parent.parent / "services" / "llm_settings.py"


def load_llm_settings():
    """返回 settings 服务模块（模块级单例）。"""
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(MODULE_NAME, _module_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
