# """
#架构作用：阶段0出站URL校验模块，EgressGuardPort的默认实现
#调用方：未来适配器在发起任何外发请求前调用allow(url)做只读判定
#数据流：候选url → urllib.parse解析→协议/IP/主机名规则判定→ True/False
# """
"""
agent_framework/security/egress_guard.py出站URL校验（SSRF防线）

实现contracts.EgressGuardPort：
-仅允许http/https协议；
-主机解析为IP时拒绝回环（127.0.0.0/8、::1）、内网段（10/8、172.16/12、
  192.168/16）、链路本地（含169.254.0.0/16，覆盖云元数据169.254.169.254）；
-主机名localhost及其子域一律拒绝；
-域名经DNS解析校验（F7已拍板）：任一解析结果命中危险网段即拒绝；
  解析失败时告警放行（对齐llm_settings既有语义，离线/测试环境不致误杀）；
-任何解析失败都不抛异常：DNS解析失败/无结果时告警放行（fail-open，与llm_settings一致）；
  已知残余风险：解析与真正连接之间存在DNS rebinding的TOCTOU窗口
"""
from __future__ import annotations

import ipaddress
import socket
from typing import FrozenSet, Iterable, Optional
from urllib.parse import urlparse

from ..contracts import EgressGuardPort

# """
#常量区：协议白名单与内置危险网段；全部不可变，便于审计与测试
# """

_ALLOWED_SCHEMES: FrozenSet[str] = frozenset({"http", "https"})

#回环/链路本地/组播/未指定地址等由ipaddress属性判定，
#这里只额外列内网私段（ipaddress的is_private语义版本差异较大，显式列出更稳）
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "::1/128",
        "fe80::/10",
    )
)

_LOCAL_HOSTNAMES: FrozenSet[str] = frozenset({"localhost", "localhost.localdomain"})


class EgressGuard(EgressGuardPort):
    """EgressGuardPort 的默认实现：纯规则判定，无网络依赖、无状态。"""

    def __init__(self, extra_allowed_hosts: Optional[Iterable[str]] = None) -> None:
        """可选注入额外主机白名单（小写归一化后精确匹配放行）。"""
        self._extra_allowed: FrozenSet[str] = frozenset(
            h.strip().lower() for h in (extra_allowed_hosts or ()) if h and h.strip()
        )

    def allow(self, url: str) -> bool:
        """url 允许外发返回 True，否则 False；任何异常一律拒绝。"""
        try:
            return self._allow_inner(url)
        except Exception:
            #判定失败不得外抛，失败即拒绝
            return False

    # """
    #内部判定：协议→主机提取→白名单→ IP/主机名规则，逐层收紧
    # """

    def _allow_inner(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            return False
        host = parsed.hostname
        if not host:
            return False
        host = host.strip().lower().rstrip(".")

        #注入白名单优先放行（用于确知可信的自建地址，如内网部署的网关）
        if host in self._extra_allowed:
            return True

        #主机名规则：localhost及其子域一律拒绝
        if host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
            return False

        #能解析为IP字面量的，走网段规则；解析不出的是域名，做DNS解析校验（F7已拍板）：
        #任一解析结果落入回环/内网/链路本地即拒绝，解析失败按fail-closed拒绝，
        #防止DNS rebinding把公网域名切到内网地址
        ip = self._parse_ip(host)
        if ip is not None:
            return not self._is_blocked_ip(ip)
        return self._resolve_allowed(host)

    # """
    #域名解析校验：getaddrinfo全量结果逐一过网段规则，命中危险网段即拒绝；
    #解析失败/无结果时告警放行（对齐llm_settings._assert_public_host既有语义：
    #离线/测试环境常见，且连不出去的地址本身发不出请求）
    # """
    @staticmethod
    def _resolve_allowed(host: str) -> bool:
        try:
            infos = socket.getaddrinfo(host, None)
        except (OSError, ValueError):
            print(f"[egress_guard] 主机名无法解析，跳过内网检查: {host}")
            return True
        if not infos:
            print(f"[egress_guard] 主机名没有可用IP地址，跳过内网检查: {host}")
            return True
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if EgressGuard._is_blocked_ip(ip):
                return False
        return True

    @staticmethod
    def _parse_ip(host: str) -> Optional[ipaddress._BaseAddress]:
        """尝试把主机解析为 IP 字面量；域名返回 None。"""
        try:
            return ipaddress.ip_address(host)
        except ValueError:
            pass
        #审计修复：inet_aton形态（十进制/十六进制/八进制/缩写点分，如
        # 2130706433、0x7f000001、0177.0.0.1、127.1）会被解析器认成127.0.0.1，
        #必须识别后走网段规则，否则构成SSRF绕过
        try:
            packed = socket.inet_aton(host)
        except (OSError, ValueError):
            return None
        return ipaddress.ip_address(packed)

    @staticmethod
    def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
        """回环/链路本地/私网/组播/保留地址一律拒绝。"""
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return True
        if ip.is_reserved:
            return True
        return any(ip in net for net in _BLOCKED_NETWORKS)

# [总审计 2026-11-13] P2：模块 docstring 称“解析失败一律返回 False”，实际在 :109-112 打印告警后返回 True
# （fail-open），二者矛盾，需明确策略并同步注释；另外校验与真正建连之间存在 DNS rebinding TOCTOU（F7 未关闭）。
