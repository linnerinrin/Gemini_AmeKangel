#"""
#架构作用：兼容加载LLM设置服务模块，规避测试替身包结构和服务导入环
#调用方：API/schema/config在需要时调用load_llm_settings
#数据流：文件路径→ importlib spec → sys.modules单例；加载后由Python执行模块
#"""
"""
config/llm_import.py安全加载app/services/llm_settings.py

为什么不让调用方直接`from app.services import llm_settings`：
- tests/conftest.py的fake_services夹具会把app.services换成没有__path__的
替身模块（为了不初始化真实的重型服务），此时包级导入会直接抛
  ModuleNotFoundError: 'app.services' is not a package；
- ModelManager本身又会被app.services.__init__反向导入，包级导入还存在循环

所以统一按**文件路径**加载，并固定注册成app.services.llm_settings：
sys.modules里只有这一份，已加载时直接复用（monkeypatch也照常生效）
"""

import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "app.services.llm_settings"


def _module_path() -> Path:
    #本文件在app/config/下，项目根往上一级，再进app/services/
    return Path(__file__).resolve().parent.parent / "services" / "llm_settings.py"


def load_llm_settings():
    """返回settings服务模块（模块级单例）。"""
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(MODULE_NAME, _module_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
