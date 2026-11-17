#"""
#架构作用：model_adapter子包的包边界；模型调用适配模块的集合入口说明
#调用方：调用方按需导入具体模块（from .client import OpenAICompatibleChatClient）
#本文件仅说明职责与依赖方向，无独立执行数据流
#"""
"""
agent_framework/model_adapter子包：模型调用适配（半接入：仅非流式路径）

- client.py   OpenAI兼容聊天客户端：OpenAICompatibleChatClient
              （ChatClientPort默认实现，非流式chat/completions）

职责边界：
-统一聊天调用抽象：调用方只认contracts.ChatClientPort，不绑死厂商SDK/HTTP细节；
-错误归一：HTTP 4xx/5xx、网络异常、JSON解析失败、响应缺choices全部包装为
  ModelClientError（可读原因），凭据原文绝不进入错误信息与日志；
-凭据安全：api_key为空时不发请求直接抛ModelClientError，日志只含地址/模型/耗时/用量

依赖方向：各模块只依赖contracts（以及标准库/已装传输库），不导出具体类到包级命名空间，
调用方按需导入，避免包级import带来重依赖；本包不反向依赖任何实现
接入状态（2026-11）：半接入。app/managers/ModelManager.py:19直接import本包，
:187在_generate_via_port内实例化OpenAICompatibleChatClient；
但:163-164的条件（not use_tools and resolved and not injected）决定了只有
「非流式 + 未注入替身」才走本端口，主链路与全部流式调用仍走openai SDK的AsyncOpenAI
"""
