"""
chat 核心交互
auth 认证/登录


"""


from app.managers import rag_manager,model_manager,account_manager,memory_manager,functioncall_manager,graph_manager
from app.services.chat_session import ChatSession

chat=ChatSession(model_manager,functioncall_manager,rag_manager,memory_manager,graph_manager)