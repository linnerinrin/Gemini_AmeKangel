# """
#架构作用：security子包的包边界；阶段0安全模块的集合入口说明
#调用方：调用方按需导入具体模块（from .egress_guard import EgressGuard等）
#本文件仅说明职责与依赖方向，无独立执行数据流
# """
"""
agent_framework/security子包：阶段0安全模块（纯新增，未接入运行路径）

- egress_guard.py出站URL校验：EgressGuard（EgressGuardPort默认实现）
- credential_policy.py凭据-地址绑定策略：CredentialPolicy（CredentialPolicyPort默认实现）
- session_registry.py会话映射注册表：SessionRegistry（SessionRegistryPort实现）
- audit.py审计落点：InMemoryAuditSink / PrintAuditSink（AuditSinkPort实现）

依赖方向：各模块只依赖contracts（以及标准库），不导出具体类到包级命名空间，
调用方按需导入，避免包级import带来重依赖；本包不反向依赖任何实现
"""
