from typing import Optional

from pydantic import BaseModel

from app.config.llm_import import load_llm_settings

# 默认 active 厂商由设置服务给出；按文件加载，避免被 tests 的假 app.services 包挡住
DEFAULT_PROVIDER = "deepseek"


class LLMProfileUpdate(BaseModel):
    """单个厂商的候选配置；未提交的字段沿用已保存的值。"""

    base_url: Optional[str] = None
    model: Optional[str] = None
    # 只在用户改过输入框时提交明文；不回传掩码值（服务端也拒绝掩码当明文存）
    api_key: Optional[str] = None


class LLMSettingsUpdate(BaseModel):
    # 保存并切换到该厂商
    active: str = DEFAULT_PROVIDER
    # 只提交被改动过的厂商，其余保持原样
    profiles: dict[str, LLMProfileUpdate] = {}


class LLMTestRequest(BaseModel):
    """连通性测试的候选配置：留空的字段用当前生效值补齐。"""

    provider: str = DEFAULT_PROVIDER
    base_url: Optional[str] = None
    model: Optional[str] = None
    # 不回显；为空表示沿用已保存/.env 的 key
    api_key: Optional[str] = None


def provider_names() -> list[str]:
    return list(load_llm_settings().PROVIDER_CATALOG)
