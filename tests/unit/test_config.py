import ast
from pathlib import Path

import pytest

from app.config.config import config

# 0 = jine(糖糖私聊), 1 = 超天酱直播, 2 = 堕天酱直播, 3 = 糖糖直播
LIVE_MODES = ["0", "1", "2", "3"]


def test_no_hardcoded_api_key_in_source():
    """禁止把真实 API key 写回源码。

    曾经 config.py 里硬编码过一把真实可用的 key（并随源码进了 git 历史），
    后来改为「.env 或控制面板保存的设置」两条来源。这里守住回归：
    key 的类默认值必须是空串，且 config.py 的**字符串字面量**里不得出现 sk- 开头的 key。
    用 AST 取字面量而不是扫原文，否则注释里提到 "sk-" 就会误报。
    """
    field = config.__class__.model_fields["DEEPSEEK_API_KEY"]
    assert field.default == "", f"DEEPSEEK_API_KEY 默认值必须为空串，实际为 {field.default!r}"

    source = (Path(__file__).resolve().parents[2] / "app" / "config" / "config.py").read_text(
        encoding="utf-8"
    )
    literals = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    offenders = [text for text in literals if text.strip().startswith("sk-")]
    assert not offenders, f"config.py 的字面量里出现疑似 API key：{offenders}"


def test_initial_mode_has_prompts():
    assert config.INITIAL_MODE in config.SYS_PROMPT
    assert config.SYS_PROMPT[config.INITIAL_MODE]


def test_every_live_mode_has_prompt_and_knowledge():
    """每个直播主题各自一个 mode，缺少任一 mode 都会让 mode_change 直接 500"""
    for mode in LIVE_MODES:
        assert config.SYS_PROMPT.get(mode), f"mode {mode} 缺少系统提示词"
        knowledge = config.knowledge_prompt.get(mode)
        assert knowledge, f"mode {mode} 缺少知识库路径"
        assert Path(knowledge).exists(), f"mode {mode} 知识库文件不存在: {knowledge}"


def test_generation_parameter_shapes():
    """默认不向模型传 max_tokens：思考链/回答不应再被截断（debug 13）

    注意这里不写死 max_tokens 一定缺席：用户可以为控制成本把 .env 里的上限设回非 0，
    那时应当下发 max_tokens，测试只校验"0 => 不下发 / 非 0 => 原样下发"的语义。
    """
    for max_tokens, params in [
        (config.INFERENCE_MAX_TOKENS, config.inference_para),
        (config.THINK_MAX_TOKENS, config.think_para),
        (config.REFLECT_MAX_TOKENS, config.reflect_para),
    ]:
        assert "temperature" in params
        assert "top_p" in params
        if max_tokens == 0:
            assert "max_tokens" not in params
        else:
            assert params["max_tokens"] == max_tokens

    # 代码里定义的默认值必须是 0（不被本地 .env 影响）
    for field in ("INFERENCE_MAX_TOKENS", "THINK_MAX_TOKENS", "REFLECT_MAX_TOKENS"):
        assert config.__class__.model_fields[field].default == 0

    assert config._build_para(0, 0.5, 0.9) == {"temperature": 0.5, "top_p": 0.9}
    # 需要限制成本时仍然能显式打开
    assert config._build_para(512, 0.5, 0.9)["max_tokens"] == 512


def test_tools_schemas_loaded_as_list():
    assert isinstance(config.TOOLS_SCHEMAS, list)


def test_all_prompts_share_the_same_output_discipline():
    """四个 mode 的输出规范/表情口径必须一致（debug 14、16）

    公共正文边界已拆到 prompts/base.txt，表情与分段口径拆到 prompts/task/chat.txt，
    所以这里断言的是装配后的聊天模式提示词：base + role + task/chat。
    四角色仍共用同一份正文边界，断言意图不变。
    """
    for mode in ("0", "1", "2", "3"):
        prompt = config.sys_prompt_for(mode, "chat")
        assert "输出规范（四种角色使用同一正文边界，各自语气与关系仍有差异）" in prompt
        assert "[理性分析]之类后台标记" in prompt
        assert "允许解释问题" in prompt
        assert "括号本身不是违规" in prompt
        assert "表情代码是协议例外" in prompt
        assert "人物动机与关系" in prompt
        assert "@@@" in prompt, f"mode {mode} 的聊天模式缺少 @@ 分段协议"
        assert "输出时使用尖括号标签" not in prompt, f"mode {mode} 仍要求输出中文标签"
        assert "直接输出下面的代码本身" in prompt, f"mode {mode} 未要求输出表情代码"
        # 示例里不应再出现被尖括号包住的代码或中文标签
        assert "<jine_pien_loveforever>" not in prompt


def test_dual_axis_prompts_assemble_for_every_mode():
    """两轴正交：mode 0-3 × task_mode(chat/work) 共 8 组都要能装配出非空提示词"""
    for mode in ("0", "1", "2", "3"):
        chat = config.sys_prompt_for(mode, "chat")
        work = config.sys_prompt_for(mode, "work")
        assert chat.strip() and work.strip(), f"mode {mode} 装配结果为空"
        # 顺序固定 base -> role -> task
        assert chat.startswith(config.BASE_PROMPT), f"mode {mode} 未以 base 开头"
        assert work.startswith(config.BASE_PROMPT), f"mode {mode} 未以 base 开头"
        assert chat == config.join_prompt(
            config.BASE_PROMPT, config.ROLE_PROMPT[mode], config.TASK_PROMPT["chat"])
        assert work == config.join_prompt(
            config.BASE_PROMPT, config.ROLE_PROMPT[mode], config.TASK_PROMPT["work"])
        # chat 版本带 @@@ 分段协议；work 版本带工作模式特征（允许代码块）且不使用 @@@ 分段
        assert "@@@" in chat
        assert "代码块" in work and "Markdown" in work
        assert "单独占一行输出 @@@" not in work, f"mode {mode} 的工作模式提示词不应出现 @@ 分段协议"
        # 两轴真的不一样，装配不是摆设
        assert chat != work


def test_reflection_work_drops_roleplay_rules_but_keeps_protocol():
    """工作模式审查必须允许代码/Markdown/英文，同时保留审查输出协议（GraphManager 硬依赖）"""
    prompt = config.reflection_prompt_for("work")
    # 仍然拦截私密推理与空内容/离题
    assert "私密推理" in prompt
    assert "空内容" in prompt
    # 角色扮演专属规则不得出现在工作模式审查里
    for roleplay_rule in ("对白以中文为主", "退回动作描写", "中文表情标签"):
        assert roleplay_rule not in prompt, f"工作模式审查仍带角色扮演规则：{roleplay_rule}"
    # 允许并不得退回代码块/Markdown/英文
    assert "代码块" in prompt and "Markdown" in prompt and "英文" in prompt
    # 输出协议字样必须保留
    assert "__ACCEPT__" in prompt
    assert "需要修改：" in prompt
    assert "冒号后必须有实际问题" in prompt
    # chat 版本保持原样，确实还在管中文为主
    assert "对白以中文为主" in config.reflection_prompt_for("chat")


def test_node_prompts_have_chat_and_work_variants():
    """决策/审查两类节点提示词各有 chat/work 两份，且默认属性名仍指 chat 版本"""
    assert config.call_prompt_for("chat") == config.CALL_PROMPT
    assert config.reflection_prompt_for("chat") == config.REFLECTION_PROMPT
    assert config.CALL_WORK_PROMPT.strip() and config.REFLECTION_WORK_PROMPT.strip()
    assert config.CALL_WORK_PROMPT != config.CALL_PROMPT
    assert config.REFLECTION_WORK_PROMPT != config.REFLECTION_PROMPT
    # 决策阶段两份都必须是内部思考，不能承诺不存在的工具轮次
    assert "决策阶段" in config.CALL_WORK_PROMPT
    assert "MAX_TOOL_TURNS" in config.CALL_WORK_PROMPT
    with pytest.raises(ValueError, match="未知对话模式"):
        config.reflection_prompt_for("sleep")
    with pytest.raises(ValueError, match="未知对话模式"):
        config.sys_prompt_for("0", "sleep")
    with pytest.raises(ValueError, match="未知聊天模式"):
        config.sys_prompt_for("9", "chat")


def test_emoji_tables_stay_in_sync_between_backend_and_frontend():
    """后端归一化表与前端别名表必须一致，否则历史数据会出现两种渲染结果"""
    from app.config.emoji_labels import EMOJI_CODES, EMOJI_LABELS

    registry = (Path(__file__).resolve().parents[2] / "static" / "js" / "data" / "registry.js").read_text(encoding="utf-8")
    for code in EMOJI_CODES:
        assert f"'{code}'" in registry
    for label, code in EMOJI_LABELS.items():
        assert f"'{label}': '{code}'" in registry, f"前端缺少别名 {label} -> {code}"
