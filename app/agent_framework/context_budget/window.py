#"""
#架构作用：WindowPolicyPort的实现——把「轮数上限」统一换算为「消息条数上限」
#调用方：MemoryManager用TurnWindowPolicy换算deque maxlen（每会话）；reconcile_window供对齐核算
#数据流：turn_limit → message_limit() →消息条数上限；reconcile_window供对齐核算
#"""
"""
agent_framework/context_budget/window.py上下文窗口策略

对接原项目MemoryManager.py:81的轮/消息语义不一致问题：
SQL LIMIT数「轮」、deque maxlen数「消息」，同一参数两种口径
本模块给出唯一换算口径（一轮= user + assistant两条消息），
只提供换算函数与策略实现，不修改原文件——非侵入式对齐
"""
from __future__ import annotations

from app.agent_framework.contracts import WindowPolicyPort


#"""
#策略实现：一轮=两条消息的唯一换算口径
#"""


class TurnWindowPolicy(WindowPolicyPort):
    """轮→消息窗口策略：一轮对话= user + assistant两条消息。"""

    MESSAGES_PER_TURN: int = 2

    def message_limit(self, turn_limit: int) -> int:
        """轮数上限 → 消息条数上限（turn_limit * 2）；负值按0处理。"""
        return max(0, turn_limit) * self.MESSAGES_PER_TURN


#"""
#纯函数：给定轮数上限与当前消息总数，返回建议保留的最近消息条数
#用于对齐MemoryManager.py:81的口径不一致——调用方拿到统一口径后自行裁剪，
#本模块不触碰原文件（非侵入）
#"""


def reconcile_window(turn_limit: int, current_messages: int) -> int:
    """返回建议保留的最近消息条数= min(current_messages, turn_limit * 2)。

    解决原项目MemoryManager.py:81「SQL LIMIT数轮、deque数消息」的口径分歧：
    统一换算后取较小者，保证任何一侧口径都不被突破；本函数只做核算，
    实际裁剪动作由调用方（未来适配器）执行。
    """
    message_cap = max(0, turn_limit) * TurnWindowPolicy.MESSAGES_PER_TURN
    return min(max(0, current_messages), message_cap)
