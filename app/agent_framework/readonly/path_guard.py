# """
#架构作用：只读工具的路径安全判定层，PathGuardPort的默认实现
#调用方：fs_read_tools.py的ReadOnlyFileTools（构造注入）；未来适配器可替换实现
#数据流：用户输入路径→ PathGuard.resolve → ResolvedPath(absolute, allowed, reason)
# """
"""
agent_framework/readonly/path_guard.py路径安全判定（PathGuardPort默认实现）

职责：把用户输入路径解析为安全结果，任何越界/黑名单/逃逸都不抛异常，
统一通过ResolvedPath.allowed=False + reason表达（冻结契约见contracts.py）

判定顺序（先静态后解析，尽量不做IO）：
1.空路径拒绝；
2.原始输入静态检查：含".."段、UNC/设备路径（\、//开头）、盘符以外scheme一律拒绝；
3.拼接与realpath：绝对路径直接取，相对路径拼第一个根目录，再os.path.realpath；
4.前缀判定：realpath结果必须落在任一允许根目录前缀内（commonpath防前缀撞名）；
5.符号链接判定：realpath与未解析绝对路径不一致视为symlink，拒绝（防符号链接逃逸）；
6.黑名单：相对根目录命中{".env", "data/llm_settings.json"}，或任意层级命中凭据类
   目录/文件名（.ssh、id_rsa、*.pem、Cookies 等）拒绝（大小写不敏感、统一分隔符）；
   realpath结果先经GetLongPathNameW归一化为长路径，防Windows 8.3短文件名绕过匹配

注意：允许根目录可放宽到用户目录（READONLY_ROOTS），但只读工具仍会拒绝凭据类文件，
避免角色扮演时把 SSH 私钥 / 浏览器 Cookie 读出来发出去。

本模块除realpath外不做任何存在性要求之外的IO；只判定，不读取
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Union

from app.agent_framework.contracts import PathGuardPort, ResolvedPath


def _to_long_path(path: str) -> str:
    """Windows 下用 GetLongPathNameW 把 8.3 短文件名归一化为长路径；失败或非 Windows 原样返回"""
    if os.name != "nt":
        return path
    try:
        import ctypes

        buf_len = ctypes.windll.kernel32.GetLongPathNameW(path, None, 0)
        if buf_len == 0:
            return path
        buf = ctypes.create_unicode_buffer(buf_len)
        if ctypes.windll.kernel32.GetLongPathNameW(path, buf, buf_len) == 0:
            return path
        return buf.value or path
    except Exception:
        return path


# """
#黑名单与静态检查常量
#黑名单以「工作区内相对路径」表示，统一转小写+正斜杠后比较；
# scheme检查用「字母开头后接冒号」匹配，盘符（如C:）先剥离再判
# """
_BLACKLIST = frozenset({".env", "data/llm_settings.json"})

# 放宽读取范围后仍需拦截的凭据类目录/文件名（任意层级）；
# 只读工具没有写入能力，但读出来的私钥/Cookie 会被模型带进回复，等于泄露。
_SENSITIVE_DIRS = frozenset({
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".password-store",
})
_SENSITIVE_NAMES = frozenset({
    ".env", ".git-credentials", ".npmrc", ".pypirc", "llm_settings.json",
    "id_rsa", "id_ed25519", "id_ecdsa", "credentials", "cookies",
    "login data", "login data for account", "key4.db", "logins.json",
})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".jks", ".ppk")

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_DRIVE_RE = re.compile(r"^[a-zA-Z]:")


def _is_sensitive_path(resolved: str) -> bool:
    """任意层级命中凭据类目录/文件名/后缀即拒绝。"""
    parts = [part.lower() for part in re.split(r"[/\\]+", resolved) if part]
    if any(part in _SENSITIVE_DIRS for part in parts):
        return True
    name = parts[-1] if parts else ""
    if name in _SENSITIVE_NAMES or name.startswith(".env."):
        return True
    return name.endswith(_SENSITIVE_SUFFIXES)


class PathGuard(PathGuardPort):
    """PathGuardPort 默认实现：允许根目录前缀 + 黑名单 + 逃逸防护，永不抛异常。

    workspace_root 可以是单个目录，也可以是目录列表；相对路径以第一个根目录为基准。
    """

    def __init__(self, workspace_root: Union[str, Iterable[str]]) -> None:
        """注入允许读取的根目录；构造时即 realpath + 长路径归一化，保证判定基准一致。"""
        if isinstance(workspace_root, (str, bytes, os.PathLike)):
            raw_roots = [str(workspace_root)]
        else:
            raw_roots = [str(item) for item in workspace_root]
        roots = [_to_long_path(os.path.realpath(item)) for item in raw_roots if str(item).strip()]
        if not roots:
            raise ValueError("PathGuard 至少需要一个允许读取的根目录")
        self._roots: List[str] = roots
        self._root = roots[0]

    def resolve(self, path: str) -> ResolvedPath:
        """把用户输入路径解析为安全结果；拒绝不抛异常，allowed=False + reason。"""
        # """
        #第一步：静态检查（解析前完成，不触碰文件系统）
        #空路径、".."段、UNC/设备路径、盘符以外scheme都在这一步拦下
        # """
        try:
            if not path or not str(path).strip():
                return ResolvedPath(reason="路径为空")
            raw = str(path).strip()

            if _is_device_or_unc(raw):
                return ResolvedPath(reason="拒绝 UNC/设备路径")
            if _has_forbidden_scheme(raw):
                return ResolvedPath(reason="拒绝盘符以外的 URL scheme")
            # ".."段按段判定，避免误伤"foo..bar"这类合法名字
            segments = re.split(r"[/\\]+", raw)
            if any(seg == ".." for seg in segments):
                return ResolvedPath(reason='路径含 ".." 段，拒绝逃逸尝试')

            # """
            #第二步：拼接与解析
            #绝对路径直接取，相对路径拼第一个允许根目录；realpath解析符号链接与分隔符
            # """
            candidate = Path(raw) if os.path.isabs(raw) else Path(self._root) / raw
            unresolved = os.path.abspath(str(candidate))
            # realpath后用GetLongPathNameW归一化8.3短文件名，防PROGRA~1之类绕过黑名单
            resolved = _to_long_path(os.path.realpath(unresolved))

            # """
            #第三步：前缀判定（落在任一允许根目录内即可；commonpath防前缀撞名）
            # """
            matched_root = next((root for root in self._roots if _within(resolved, root)), None)
            if matched_root is None:
                return ResolvedPath(
                    reason="路径越出允许读取的范围（可用 READONLY_ROOTS 追加根目录）"
                )

            # """
            #第四步：符号链接判定
            # realpath与未解析绝对路径不一致→存在符号链接跳转，拒绝（防符号链接逃逸）
            # """
            if os.path.normcase(resolved) != os.path.normcase(unresolved):
                return ResolvedPath(reason="路径命中符号链接，拒绝（防符号链接逃逸）")

            # """
            #第五步：黑名单（相对根目录匹配 + 任意层级凭据类路径）
            # """
            relative = _rel_posix(resolved, matched_root)
            if relative.lower() in _BLACKLIST or _is_sensitive_path(resolved):
                return ResolvedPath(reason="路径命中只读黑名单")

            return ResolvedPath(absolute=resolved, allowed=True)
        except Exception as exc:  #永不外抛：任何解析异常都按拒绝处理
            return ResolvedPath(reason=f"路径解析失败：{exc}")


# """
#内部小工具：均为纯函数，不做IO
# """
def _is_device_or_unc(raw: str) -> bool:
    """UNC（\\\\server\\share）、// 开头、Windows 设备路径（\\\\.\\ \\\\?\\）判定。"""
    return raw.startswith("\\\\") or raw.startswith("//")


def _has_forbidden_scheme(raw: str) -> bool:
    """拒绝 http: / file: 等 URL scheme；盘符（C:）单独放行。"""
    match = _SCHEME_RE.match(raw)
    return bool(match) and not _DRIVE_RE.match(raw)


def _within(resolved: str, root: str) -> bool:
    """resolved 是否落在 root 内；commonpath 跨盘符会抛 ValueError，按越界处理。"""
    try:
        return os.path.commonpath([os.path.normcase(resolved),
                                   os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:
        return False


def _rel_posix(resolved: str, root: str) -> str:
    """算工作区内相对路径并统一为正斜杠小写可比较形状。"""
    return os.path.relpath(resolved, root).replace(os.sep, "/")
