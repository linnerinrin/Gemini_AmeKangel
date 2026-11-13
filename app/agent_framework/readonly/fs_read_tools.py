#"""
#架构作用：只读文件工具实现层，对agent暴露read_file / list_dir / glob / grep四种能力
#调用方：tool_specs.py的impl闭包（经mgr.readonly_tools取得本类实例）
#数据流：工具参数→ PathGuardPort.resolve放行→本地只读IO → ReadResult统一返回
#"""
"""
agent_framework/readonly/fs_read_tools.py只读文件工具（ReadOnlyFileTools）

四个同步def方法，全部纯只读：
- read_file(path, offset=1, limit=None)：UTF-8按行读（errors="replace"），1-based offset，
  limit=None表示读到max_bytes上限；截断时truncated=True且total_lines给真实总行数
- list_dir(path)：一层清单，content每行"dir名称" / "file名称"，目录优先排序
- glob_files(pattern)：工作区内通配匹配，结果逐个过guard，越界剔除；每行一个相对路径
- grep(pattern, path=".")：re编译失败返回ok=False；逐行正则搜索文本文件，
  content每行"相对路径:行号:内容"，结果上限200条，超出truncated=True

统一约定：所有方法先过guard；异常一律转成ReadResult(ok=False, reason=...)，不外抛
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from app.agent_framework.contracts import PathGuardPort, ReadResult

#"""
#模块常量
#结果行数上限：防止grep输出撑爆上下文；二进制嗅探读头4KB足够判定
#"""
_GREP_MAX_RESULTS = 200
_BINARY_SNIFF_BYTES = 4096


class ReadOnlyFileTools:
    """只读文件工具集合：构造注入PathGuardPort与max_bytes，全部方法不外抛异常。"""

    def __init__(self, guard: PathGuardPort, max_bytes: int = 200 * 1024) -> None:
        """注入路径判定端口与单文件读取字节上限（默认200KB）。"""
        self._guard = guard
        self._max_bytes = max_bytes

    #"""
    #read_file：按行读取+字节上限截断
    #先读全量统计total_lines（截断时调用方可见原始规模），再按offset/limit切片；
    #字节数按UTF-8编码估算，超max_bytes即截断
    #"""
    def read_file(self, path: str, offset: int = 1,
                  limit: Optional[int] = None) -> ReadResult:
        """按行读取文件；offset 1-based；limit=None表示读到max_bytes上限。"""
        resolved = self._guard.resolve(path)
        if not resolved.allowed:
            return ReadResult(ok=False, reason=resolved.reason)
        try:
            with open(resolved.absolute, "r", encoding="utf-8",
                      errors="replace") as handle:
                lines = handle.read().splitlines()
        except Exception as exc:
            return ReadResult(ok=False, reason=f"读取失败：{exc}")

        total_lines = len(lines)
        start = max(offset - 1, 0)
        if start >= total_lines:
            return ReadResult(ok=True, content="", total_lines=total_lines)

        sliced = lines[start:] if limit is None else lines[start:start + max(limit, 0)]
        content, truncated = _truncate_by_bytes(sliced, self._max_bytes)
        #limit截掉剩余行同样算截断：调用方需知道还有内容未返回
        if start + len(sliced) < total_lines:
            truncated = True
        return ReadResult(ok=True, content=content, total_lines=total_lines,
                          truncated=truncated)

    #"""
    #list_dir：一层清单，目录优先排序（"dir名称"两空格对齐file前缀宽度）
    #"""
    def list_dir(self, path: str) -> ReadResult:
        """列出一层目录清单；content每行"dir名称" / "file名称"，目录优先。"""
        resolved = self._guard.resolve(path)
        if not resolved.allowed:
            return ReadResult(ok=False, reason=resolved.reason)
        try:
            entries = list(os.scandir(resolved.absolute))
        except Exception as exc:
            return ReadResult(ok=False, reason=f"目录读取失败：{exc}")

        dirs = sorted(e.name for e in entries if e.is_dir())
        files = sorted(e.name for e in entries if not e.is_dir())
        content = "\n".join([f"dir  {name}" for name in dirs]
                            + [f"file {name}" for name in files])
        return ReadResult(ok=True, content=content, total_lines=len(entries))

    #"""
    #glob_files：工作区内通配匹配
    #用pathlib glob递归匹配，对每个结果逐个过guard，越界/黑名单剔除；
    #content每行一个工作区内相对路径（正斜杠统一）
    #"""
    def glob_files(self, pattern: str) -> ReadResult:
        """在工作区内按通配匹配；越界结果剔除，content每行一个相对路径。"""
        if not pattern or not pattern.strip():
            return ReadResult(ok=False, reason="匹配模式为空")
        root = self._guard.resolve(".")
        if not root.allowed:
            return ReadResult(ok=False, reason=root.reason)
        try:
            matches = Path(root.absolute).glob(pattern)
            lines = []
            for item in matches:
                check = self._guard.resolve(str(item))
                if not check.allowed:
                    continue
                lines.append(_rel_posix(check.absolute, root.absolute))
        except Exception as exc:
            return ReadResult(ok=False, reason=f"通配匹配失败：{exc}")
        lines.sort()
        return ReadResult(ok=True, content="\n".join(lines), total_lines=len(lines))

    #"""
    #grep：逐行正则搜索
    #re编译失败直接ok=False；遍历path下guard放行的文本文件逐行匹配；
    #结果行数上限_GREP_MAX_RESULTS条，超出truncated=True
    #"""
    def grep(self, pattern: str, path: str = ".") -> ReadResult:
        """按正则逐行搜索文本文件；content每行"相对路径:行号:内容"，上限200条。"""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ReadResult(ok=False, reason=f"正则编译失败：{exc}")

        resolved = self._guard.resolve(path)
        if not resolved.allowed:
            return ReadResult(ok=False, reason=resolved.reason)
        try:
            files = _iter_text_files(resolved.absolute, self._guard)
            root = self._guard.resolve(".").absolute
            lines = []
            truncated = False
            for file_abs in files:
                rel = _rel_posix(file_abs, root)
                for lineno, text in _iter_lines(file_abs):
                    if regex.search(text):
                        lines.append(f"{rel}:{lineno}:{text}")
                        if len(lines) >= _GREP_MAX_RESULTS:
                            truncated = True
                            return ReadResult(ok=True, content="\n".join(lines),
                                              total_lines=len(lines), truncated=True)
            return ReadResult(ok=True, content="\n".join(lines),
                              total_lines=len(lines), truncated=truncated)
        except Exception as exc:
            return ReadResult(ok=False, reason=f"搜索失败：{exc}")


#"""
#内部小工具：文件遍历/文本判定/截断，均不外抛异常
#"""
def _iter_text_files(base: str, guard: PathGuardPort):
    """遍历base下所有文件，yield guard放行且判定为文本的绝对路径。"""
    if os.path.isfile(base):
        if _is_text(base):
            yield base
        return
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            candidate = os.path.join(dirpath, name)
            check = guard.resolve(candidate)
            if check.allowed and _is_text(check.absolute):
                yield check.absolute


def _is_text(path: str) -> bool:
    """嗅探文件头：含NUL字节视为二进制；UTF-8解码失败也按二进制跳过。"""
    try:
        with open(path, "rb") as handle:
            chunk = handle.read(_BINARY_SNIFF_BYTES)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except Exception:
        return False


def _iter_lines(path: str):
    """逐行yield (1-based行号,行内容)；读取异常直接停止该文件。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                yield lineno, line.rstrip("\r\n")
    except Exception:
        return


def _truncate_by_bytes(lines: list, max_bytes: int) -> tuple:
    """按UTF-8字节数截断行列表；返回(文本,是否截断)。"""
    out = []
    used = 0
    truncated = False
    for line in lines:
        cost = len(line.encode("utf-8")) + 1  #+1为换行符
        if used + cost > max_bytes:
            truncated = True
            break
        out.append(line)
        used += cost
    return "\n".join(out), truncated


def _rel_posix(absolute: str, root: str) -> str:
    """工作区内相对路径，统一正斜杠。"""
    return os.path.relpath(absolute, root).replace(os.sep, "/")

#read_file(:49-61)先读整个文件再按offset/limit切片；max_bytes只截输出，超大文件会先占满内存
#grep的结果行无单行长度上限
