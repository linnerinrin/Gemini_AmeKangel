#"""
#架构作用：凭据-地址绑定策略模块，CredentialPolicyPort的默认实现
#调用方：适配器在「请求未显式携带api_key」场景下判断是否可复用已存key
#数据流：saved_base_url + candidate_base_url + key_overridden →归一化比对→ True/False
#"""
"""
agent_framework/security/credential_policy.py凭据-地址绑定策略

实现contracts.CredentialPolicyPort，对接settings路由的保存/探测场景：
-请求显式携带api_key（key_overridden=True）时，不携带已存key（返回False）；
-仅当候选地址与已存地址归一化（小写、去尾斜杠）相等时，才允许携带已存key；
-其余情况一律False，调用方必须拒绝携带并提示配置key

归一化逻辑抽出为纯函数normalize_base_url，供本模块与测试共用
"""
from __future__ import annotations

from ..contracts import CredentialPolicyPort


def normalize_base_url(url: str) -> str:
    """base_url归一化：去首尾空白、转小写、去掉末尾全部斜杠。

    纯函数；输入为空串/None时返回空串，比对时空串不等于任何非空地址。
    """
    if not url:
        return ""
    return url.strip().lower().rstrip("/")


class CredentialPolicy(CredentialPolicyPort):
    """CredentialPolicyPort的默认实现：纯函数判定，无状态。"""

    def allow_carry(self, saved_base_url: str, candidate_base_url: str,
                    key_overridden: bool) -> bool:
        """是否允许携带已存key访问候选地址。

        语义说明：返回True表示「允许携带已存key」；
        key_overridden=True（请求已显式携带key）时返回False，
        即不携带已存key，避免覆盖调用方显式意图。
        """
        try:
            if key_overridden:
                return False
            saved = normalize_base_url(saved_base_url)
            candidate = normalize_base_url(candidate_base_url)
            if not saved or not candidate:
                return False
            return saved == candidate
        except Exception:
            #判定失败不得外抛，失败即拒绝携带
            return False
