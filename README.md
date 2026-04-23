# 双子角色扮演聊天系统

基于大语言模型的双角色扮演聊天系统，支持糖糖（阴暗少女）和超天酱（元气主播）两个性格迥异的角色即时切换，集成RAG知识库检索、查询改写、上下文压缩等完整RAG流程。

## 项目结构

```
project/
├── APIService.py          # FastAPI 服务入口
├── LocalService.py        # 本地命令行交互入口
├── interfaces/            # 接口定义
│   ├── IModel.py          # 模型接口
│   ├── IPre.py            # 预处理接口
│   └── IRAG.py            # RAG接口
├── src/                   # 核心实现
│   ├── ModelManager.py    # 模型管理（LoRA适配器切换）
│   ├── PreManager.py      # 预处理管理（改写+压缩）
│   ├── RAGManager.py      # RAG管理（检索+重排）
│   └── gemini_chat.py     # 聊天会话管理
├── config/                # 配置文件
│   ├── chat_config.yaml   # 角色配置（系统提示、固定开场/结束语）
│   ├── model_config.yaml  # 模型配置（基座模型、LoRA路径）
│   ├── pre_config.yaml    # 预处理配置
│   └── rag_config.yaml    # RAG配置（检索数量、重排模型）
├── static/                # 前端静态文件
│   ├── x.html             # 聊天界面
│   ├── x.css              # 样式文件
│   └── x.js               # 前端逻辑
├── data/                  # 知识库数据
│   ├── ame_knowledge.txt  # 糖糖知识库
│   └── kangel_knowledge.txt # 超天酱知识库
├── models/                # 模型存放目录
│   ├── qwen2.5/           # 基座模型
│   ├── lora_ame/          # 糖糖LoRA适配器
│   ├── lora_kangel/       # 超天酱LoRA适配器
│   ├── all-MiniLM-L6-v2/  # 双编码器（召回）
│   ├── ms-marco-MiniLM-L-6-v2/ # 交叉编码器（重排）
│   └── bart-base-chinese/ # 压缩模型
├── utils/               # 脚本目录
│   ├── download.py        # 模型下载脚本
│   ├── load_train_data.py # 训练数据转换
│   ├── train_ame.py       # 糖糖LoRA训练
│   └── train_kangel.py    # 超天酱LoRA训练
└── requirements.txt       # 依赖清单
```

## 功能特性

- **双角色切换**：糖糖（阴暗抑郁）↔ 超天酱（元气可爱），支持运行时即时切换
- **角色扮演对话**：基于LoRA微调的角色风格输出
- **记忆管理**：保留最近N轮对话历史（可配置）
- **RAG增强**：基于知识库的检索增强生成
- **查询改写**：结合对话历史解决指代不清问题
- **上下文压缩**：压缩对话历史和检索文档，节省Token
- **流式输出**：SSE流式响应，打字机效果
- **Web界面**：深色/浅色主题自动适配，历史记录查看

## 环境要求

- Python 3.10+
- CUDA 11.8+
- 8GB显存（推荐） / 6GB显存（最低）



## 配置说明

### `config/chat_config.yaml`

```yaml
initial_mode: ame          # 初始模式（ame/kangel）
mem_len: 10                # 记忆长度

inference_para:            # 各角色的推理参数
    ame:
        max_new_tokens: 128
        temperature: 0.6
    kangel:
        max_new_tokens: 128
        temperature: 0.6

sys_prompt:                # 系统提示词
    ame: "..."             # 糖糖角色设定
    kangel: "..."          # 超天酱角色设定

fixed_chat:                # 固定开场/结束语
    ame:
        opening: "阿p~"
        closing: "永远爱你~"
    kangel:
        opening: "小天使请安！"
        closing: "升天~"

knowledge_prompt:          # 知识库路径
    ame: data/ame_knowledge.txt
    kangel: data/kangel_knowledge.txt
```

### `config/model_config.yaml`

```yaml
base_model_paths: models/qwen2.5
max_seq_length: 2048
chat_template: qwen-2.5
mode_paths:
    ame: models/lora_ame
    kangel: models/lora_kangel
```

### `config/rag_config.yaml`

```yaml
bi_encoder_paths: models/all-MiniLM-L6-v2    # 召回模型
cross_encoder_paths: models/ms-marco-MiniLM-L-6-v2  # 重排模型
rag_recall_size: 5          # 召回数量
rag_rerank_size: 2          # 重排后保留数量
```

## API接口

### `POST /api/post`

发送消息并获取流式响应

**请求体：**
```json
{
    "content": "用户消息"
}
```

**响应：** SSE流，每帧格式 `data: {内容}\n\n`

### `POST /api/history`

获取聊天历史记录

**响应：**
```json
{
    "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

### `POST /api/change`

切换角色模式（带动画过渡）

**请求体：**
```json
{
    "content": "ame"  // 或 "kangel"
}
```

## 技术架构

```
用户输入
    │
    ▼
┌─────────────────┐
│   Query Rewrite │  ← 结合历史改写查询
│   (PreManager)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Retrieve   │  ← 双编码器召回 + 交叉编码器重排
│   (RAGManager)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Context Compress │  ← 压缩对话历史+检索文档
│   (PreManager)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │  ← LoRA适配器切换 + 流式生成
│  (ModelManager) │
└────────┬────────┘
         │
         ▼
      输出响应
```

## 注意事项

1. **显存要求**：推荐16GB+显存，8GB需调整batch size或使用CPU offload
2. **模型路径**：确保各配置文件中的路径与实际模型位置一致
3. **中文字符**：所有Prompt和知识库建议使用UTF-8编码
4. **配置文件**：修改配置后需重启服务生效

## 扩展开发

### 添加新角色

1. 在 `chat_config.yaml` 中添加新角色的 `sys_prompt`、`fixed_chat`、`knowledge_prompt`
2. 在 `model_config.yaml` 的 `mode_paths` 中添加LoRA路径
3. 训练对应的LoRA适配器
4. 在前端 `x.js` 的主题切换中添加对应样式

### 更换基座模型

修改 `model_config.yaml` 中的 `base_model_paths`，注意调整 `max_seq_length` 和 `chat_template`。
