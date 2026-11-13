#"""
#架构作用：TokenCounterPort的实现集——默认字符估算+可选tiktoken增强
#调用方：未来适配器在上下文预算环节注入；当前不被原项目运行路径引用
#数据流：文本→ count() → token估算数（int），永不抛异常
#"""
"""
agent_framework/context_budget/counter.py  token计数器

两个实现：
- CharEstimateTokenCounter默认实现：按4字符≈1 token保守估算，
零依赖、任何环境可用；
- TiktokenCounter可选增强：精确分词计数；构造时尝试import tiktoken，
未安装则静默回退到字符估算——契约明确「tiktoken为可选增强，
默认实现不得强依赖它」
"""
from __future__ import annotations

import math
from typing import Any, Optional

from app.agent_framework.contracts import TokenCounterPort


#"""
#默认实现：字符估算，中英混排取ceil(len/4)的保守口径
#"""


class CharEstimateTokenCounter(TokenCounterPort):
    """字符估算计数器：4字符≈1 token（保守上取整），空串返回0。"""

    CHARS_PER_TOKEN: int = 4

    def count(self, text: str) -> int:
        """返回token估算数；任何输入（含None等价空值）都不抛异常。"""
        if not text:
            return 0
        return math.ceil(len(text) / self.CHARS_PER_TOKEN)


#"""
#可选增强：tiktoken精确计数；未安装时静默回退到字符估算
#"""


class TiktokenCounter(TokenCounterPort):
    """tiktoken精确计数器；未安装tiktoken时回退为字符估算。

    回退是静默的：count()的行为与异常面不因此改变，仅精度下降；
    可通过available属性探测当前是否处于精确模式。
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """尝试加载tiktoken编码；失败则回退到CharEstimateTokenCounter。"""
        self._encoding_name = encoding_name
        self._encoding: Optional[Any] = None
        self._fallback = CharEstimateTokenCounter()
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            #tiktoken未安装或编码名非法：回退到字符估算，不向上抛，但提示调用方精度下降
            self._encoding = None
            print(f"tiktoken 不可用，token 计数回退为字符估算：{exc}")

    @property
    def available(self) -> bool:
        """True表示处于tiktoken精确模式，False表示已回退到估算。"""
        return self._encoding is not None

    def count(self, text: str) -> int:
        """精确模式返回len(encode(text))；回退模式按4字符≈1 token估算。"""
        if not text:
            return 0
        if self._encoding is None:
            return self._fallback.count(text)
        try:
            return len(self._encoding.encode(text))
        except Exception:
            #编码异常（如特殊token冲突）同样回退，保证永不抛异常
            return self._fallback.count(text)
