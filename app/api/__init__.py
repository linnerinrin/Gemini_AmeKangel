#"""
#架构作用：向应用入口统一导出聊天与模型设置路由
#数据流：导入具体路由→绑定别名→ main.app_create挂载；导入链会触发服务单例装配
#"""
from app.api.chat.chat_routes import router as chat_router
from app.api.chat.settings_routes import router as settings_router

__all__ = ["chat_router", "settings_router"]
