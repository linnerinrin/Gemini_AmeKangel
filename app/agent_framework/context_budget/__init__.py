# """
#架构作用：agent_framework.context_budget子包边界；承载阶段4的token计数与窗口策略
#调用方：未来适配器经contracts.TokenCounterPort / WindowPolicyPort注入后调用
#数据流：文本→ counter计数；轮数上限→ window换算为消息条数上限
# """
"""
agent_framework.context_budget包：上下文预算（阶段4）

职责：
- counter.py  TokenCounterPort实现：字符估算为默认实现，tiktoken为可选增强；
- window.py   WindowPolicyPort实现：把「轮数上限」统一换算为「消息条数上限」，
并提供reconcile_window对齐原项目轮/消息口径（只提供换算，不改原文件）

边界：
-只依赖agent_framework.contracts与标准库（tiktoken为可选软依赖，缺失时回退）；
-不import原项目任何模块；本__init__仅声明职责，不导出任何类
"""
