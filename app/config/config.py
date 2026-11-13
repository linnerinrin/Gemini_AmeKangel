#"""
#架构作用：集中读取环境配置、提示词与推理参数，并提供路径和提示词装配属性
#调用方：服务、模型/检索管理器及工具适配器；模块末尾创建全局配置单例
#数据流：环境变量/.env + prompts文件→ Config →启动时模型与图构造
#"""
"""
config/config.py配置文件
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


#"""
#模块一：配置模型与默认值
#用pydantic-settings把.env /环境变量读成强类型字段；无默认值的字段（INITIAL_MODE、
#SHORT_MEMORY_LEN、BI/CROSS_ENCODER_PATH、MYSQL_USER/PASSWORD等）缺失时导入即失败
#"""
class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT/".env",extra="ignore",env_file_encoding="utf-8",)

    #fastapi配置
    APP_NAME:str="唐人的APP"
    APP_VERSION:str="1.0.0"
    HOST:str="0.0.0.0"
    PORT:int=5577
    DEBUG:bool=False

    #初始状态
    INITIAL_MODE: str
    SHORT_MEMORY_LEN: int
    BASE_PATH: str = str(PROJECT_ROOT)

    #模型路径用于property
    BI_ENCODER_PATH: str
    CROSS_ENCODER_PATH: str

    #原神
    #明文key由控制面板写入data/llm_settings.json；.env只作为回退
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_API_KEY: str = ""

    #推理参数用于property包装
    #0 =不向模型传max_tokens（不截断）；需要限制成本时再设为具体数值
    INFERENCE_MAX_TOKENS: int = 0
    INFERENCE_TEMPERATURE: float = 0.8
    INFERENCE_TOP_P: float = 0.9

    THINK_MAX_TOKENS: int = 0
    THINK_TEMPERATURE: float = 0.3
    THINK_TOP_P: float = 0.85

    REFLECT_MAX_TOKENS: int = 0
    REFLECT_TEMPERATURE: float = 0.3
    REFLECT_TOP_P: float = 0.9

    #graph限制循环次数
    MAX_TOOL_TURNS: int = 5
    MAX_REFLECTION_TURNS: int = 1

    #检索相关
    RAG_RECALL_SIZE: int
    RAG_RERANK_SIZE: int
    RAG_DEVICE: str = "cuda"
    #mode: 0=jine(糖糖) 1=超天酱直播2=堕天酱直播3=糖糖直播
    KNOWLEDGE_PROMPT: dict = {
        "0": "prompts/ame_knowledge.txt",
        "1": "prompts/kangel_knowledge.txt",
        "2": "prompts/kangel_knowledge.txt",
        "3": "prompts/ame_knowledge.txt",
    }

    #记忆相关
    MEMORY_FILE_PATH:str=str(PROJECT_ROOT/"data"/"user_memories.json")
    MYSQL_USER:str
    MYSQL_PASSWORD:str
    MYSQL_DATABASE:str="gemini"

    #状态相关：followers / affection / stress / mental
    DEFAULT_FOLLOWERS: int = 0
    DEFAULT_AFFECTION: int = 50
    DEFAULT_STRESS: int = 0
    DEFAULT_MENTAL: int = 0
    STATUS_CHANGE_PROBABILITY: float = 1.0
    STATUS_CHANGE_MIN_STEP: int = 0
    STATUS_CHANGE_MAX_STEP: int = 5
    #粉丝增长：直播模式每轮生成成功后按概率增加，私聊不增长
    FOLLOWERS_GROW_PROBABILITY: float = 0.5
    FOLLOWERS_GROW_MIN_STEP: int = 1
    FOLLOWERS_GROW_MAX_STEP: int = 30
    SEARCH_MAX_RESULTS: int = 5
    REWRITE_MAX_TOKENS: int = 64
    TOOLS_SCHEMAS: list = []

    #提示词
    #场景轴mode(0-3)选角色；SYS_PROMPT[mode] = base + role/{mode}，完整正文提示词由sys_prompt_for装配
    SYS_PROMPT: dict = {}
    BASE_PROMPT: str = ""
    ROLE_PROMPT: dict = {}
    TASK_PROMPT: str = ""
    REFLECTION_PROMPT: str = ""
    CALL_PROMPT: str = ""



    #"""
    #模块二：提示词文件路径与装配
    #场景轴mode(0-3)决定角色段；装配顺序base -> role -> task不可调换，
    #三个属性分别被图的三个节点读取
    #"""
    #提示词文件路径
    @property
    def base_path(self) -> str: return str(PROJECT_ROOT / "prompts" / "base.txt")
    @property
    def call_prompt_path(self) -> str: return str(PROJECT_ROOT / "prompts" / "call_chat.txt")
    @property
    def reflection_prompt_path(self) -> str: return str(PROJECT_ROOT / "prompts" / "reflection_chat.txt")

    #重写初始化钩子read提示词
    def model_post_init(self, __context):
        def read(path: str) -> str:
            return Path(path).read_text(encoding="utf-8").strip()

        #mode ->角色段文件（与KNOWLEDGE_PROMPT同一套mode语义）
        self.ROLE_PROMPT = {
            "0": read(str(PROJECT_ROOT / "prompts" / "role" / "ame.txt")),
            "1": read(str(PROJECT_ROOT / "prompts" / "role" / "kangel.txt")),
            "2": read(str(PROJECT_ROOT / "prompts" / "role" / "dangel.txt")),
            "3": read(str(PROJECT_ROOT / "prompts" / "role" / "ame_live.txt")),
        }
        self.BASE_PROMPT = read(self.base_path)
        self.TASK_PROMPT = read(str(PROJECT_ROOT / "prompts" / "task" / "chat.txt"))
        self.CALL_PROMPT = read(self.call_prompt_path)
        self.REFLECTION_PROMPT = read(self.reflection_prompt_path)
        #SYS_PROMPT保持"非空字符串"的形状（base + role），完整提示词见sys_prompt_for
        self.SYS_PROMPT = {
            mode: self.join_prompt(self.BASE_PROMPT, role)
            for mode, role in self.ROLE_PROMPT.items()
        }
        #工具定义唯一来源：app/tools/registry.py（经openai适配器转成OpenAI function形状）
        #延迟导入：本模块会被registry反过来依赖，模块级导入会循环
        from app.tools.adapters.openai import to_openai_schemas
        self.TOOLS_SCHEMAS = to_openai_schemas()

    #提示词装配：base -> role -> task的顺序不可调换
    @staticmethod
    def join_prompt(*segments) -> str:
        """按顺序拼接提示词段，跳过空段，段间用空行分隔。"""
        return "\n\n".join(seg.strip() for seg in segments if seg and seg.strip())

    def sys_prompt_for(self, mode) -> str:
        """场景轴mode装配出的完整正文提示词。"""
        mode = str(mode)
        if mode not in self.ROLE_PROMPT:
            raise ValueError(f"未知聊天模式: {mode}")
        return self.join_prompt(self.BASE_PROMPT, self.ROLE_PROMPT[mode], self.TASK_PROMPT)

    def call_prompt_for(self) -> str:
        """决策节点提示词。"""
        return self.CALL_PROMPT

    def reflection_prompt_for(self) -> str:
        """审查节点提示词。"""
        return self.REFLECTION_PROMPT


    #"""
    #模块三：推理参数与工具schema出口
    #max_tokens=0表示不放进请求体（交给模型自行决定长度）；工具schema由注册表经
    #OpenAI适配器派生，是本进程对模型可见的唯一一份
    #"""
    #参数包装
    #max_tokens为0时不放入请求体，交给模型自行决定长度（避免思考链/回答被截断）
    @staticmethod
    def _build_para(max_tokens, temperature, top_p) -> dict:
        para = {"temperature": temperature, "top_p": top_p}
        if max_tokens and max_tokens > 0:
            para["max_tokens"] = max_tokens
        return para

    @property
    def inference_para(self) -> dict:
        return self._build_para(self.INFERENCE_MAX_TOKENS, self.INFERENCE_TEMPERATURE, self.INFERENCE_TOP_P)

    @property
    def think_para(self) -> dict:
        return self._build_para(self.THINK_MAX_TOKENS, self.THINK_TEMPERATURE, self.THINK_TOP_P)

    @property
    def reflect_para(self) -> dict:
        return self._build_para(self.REFLECT_MAX_TOKENS, self.REFLECT_TEMPERATURE, self.REFLECT_TOP_P)

    #rag知识库路径
    @property
    def knowledge_prompt(self) -> dict:
        return {mode: str(Path(self.BASE_PATH) / path)
                for mode, path in self.KNOWLEDGE_PROMPT.items()
                }

    #rag本地模型路径
    @property
    def bi_encoder_dir(self) -> str:
        return str(Path(self.BASE_PATH) / self.BI_ENCODER_PATH)

    @property
    def cross_encoder_dir(self) -> str:
        return str(Path(self.BASE_PATH) / self.CROSS_ENCODER_PATH)


config= Config()

#TASK_PROMPT固定为prompts/task/chat.txt；场景轴由mode(0-3)决定角色段
