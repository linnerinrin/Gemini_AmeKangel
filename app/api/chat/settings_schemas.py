#"""
#架构作用：模型设置与探测请求结构，供FastAPI解析部分更新草稿
#数据流：可选字段→路由合并生效值→设置服务校验；None表示保留或回退，不表示凭证授权
#"""
from typing import Optional

from pydantic import BaseModel

from app.config.llm_import import load_llm_settings

#默认active厂商在此硬编码为deepseek；按文件加载，避免被tests的假app.services包挡住
DEFAULT_PROVIDER = "deepseek"


#"""
#单厂商补丁：区分未提交与显式修改，密钥仅作为写入输入
#"""
class LLMProfileUpdate(BaseModel):
    """单个厂商的候选配置；未提交的字段沿用已保存的值。"""

    base_url: Optional[str] = None
    model: Optional[str] = None
    #只在用户改过输入框时提交明文；不回传掩码值（服务端也拒绝掩码当明文存）
    api_key: Optional[str] = None


#"""
#保存载荷：active指定切换厂商，profiles仅包含本次修改项
#"""
class LLMSettingsUpdate(BaseModel):
    #保存并切换到该厂商
    active: str = DEFAULT_PROVIDER
    #只提交有变更的厂商档案，其余保持原样
    profiles: dict[str, LLMProfileUpdate] = {}


#"""
#探测草稿可覆盖地址与密钥；安全绑定需在路由/服务实施，模型本身仅做类型解析
#"""
class LLMTestRequest(BaseModel):
    """连通性测试的候选配置：留空的字段用当前生效值补齐。"""

    provider: str = DEFAULT_PROVIDER
    base_url: Optional[str] = None
    model: Optional[str] = None
    #不回显；为空表示沿用已保存/.env的key
    api_key: Optional[str] = None


#"""
#厂商目录读取入口：通过兼容加载器获取设置服务，返回目录键供调用方校验/展示
#"""
def provider_names() -> list[str]:
    return list(load_llm_settings().PROVIDER_CATALOG)
