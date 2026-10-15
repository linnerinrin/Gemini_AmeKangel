import importlib.util
import sys
from pathlib import Path


def _load_parent_conftest():
    existing = sys.modules.get("conftest")
    parent_path = Path(__file__).resolve().parents[1] / "conftest.py"
    if existing and Path(getattr(existing, "__file__", "")).resolve() == parent_path:
        return existing

    module_name = "_gemini_tests_parent_conftest"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, parent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_parent = _load_parent_conftest()

contracts = _parent.contracts
fake_services = _parent.fake_services
client = _parent.client
assert_shape = _parent.assert_shape
load_source_module = _parent.load_source_module
CONTRACT_PATH = _parent.CONTRACT_PATH
contract_entries = _parent.contract_entries
