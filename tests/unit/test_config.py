import ast
from pathlib import Path

import pytest

from app.config.config import config

#0 = jine(糖糖私聊), 1 =超天酱直播, 2 =堕天酱直播, 3 =糖糖直播
LIVE_MODES = ["0", "1", "2", "3"]


def test_no_hardcoded_api_key_in_source():
    """禁止把真实API key写回源码。

    key只来自「.env或控制面板保存的设置」两条来源。这里守住约束：
    key的类默认值必须是空串，且config.py的**字符串字面量**里不得出现sk- 开头的key。
    用AST取字面量而不是扫原文，否则注释里提到"sk-"就会误报。
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
    """每个直播主题各自一个mode，缺少任一mode都会让mode_change直接500"""
    for mode in LIVE_MODES:
        assert config.SYS_PROMPT.get(mode), f"mode {mode} 缺少系统提示词"
        knowledge = config.knowledge_prompt.get(mode)
        assert knowledge, f"mode {mode} 缺少知识库路径"
        assert Path(knowledge).exists(), f"mode {mode} 知识库文件不存在: {knowledge}"


def test_generation_parameter_shapes():
    """默认不向模型传max_tokens：思考链/回答不应再被截断（debug 13）

    这里不写死max_tokens一定缺席：用户可以为控制成本把.env里的上限设回非0，
    那时应当下发max_tokens，测试只校验"0 =>不下发/非0 =>原样下发"的语义。
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

    #代码里定义的默认值必须是0（不被本地.env影响）
    for field in ("INFERENCE_MAX_TOKENS", "THINK_MAX_TOKENS", "REFLECT_MAX_TOKENS"):
        assert config.__class__.model_fields[field].default == 0

    assert config._build_para(0, 0.5, 0.9) == {"temperature": 0.5, "top_p": 0.9}
    #需要限制成本时仍然能显式打开
    assert config._build_para(512, 0.5, 0.9)["max_tokens"] == 512


def test_tools_schemas_loaded_as_list():
    assert isinstance(config.TOOLS_SCHEMAS, list)


def test_all_prompts_share_the_same_output_discipline():
    """四个mode的输出规范/表情口径必须一致（debug 14、16）

    公共正文边界与表情/分段/换行口径都在prompts/base.txt，
    所以这里断言的是装配后的聊天模式提示词：base + role。
    """
    for mode in ("0", "1", "2", "3"):
        prompt = config.sys_prompt_for(mode)
        assert "输出规范（四种角色使用同一正文边界，各自语气与关系仍有差异）" in prompt
        assert "[理性分析]之类后台标记" in prompt
        assert "允许解释问题" in prompt
        assert "括号本身不是违规" in prompt
        assert "表情代码是协议例外" in prompt
        assert "人物动机与关系" in prompt
        assert "@@@" in prompt, f"mode {mode} 的聊天模式缺少 @@ 分段协议"
        assert "输出时使用尖括号标签" not in prompt, f"mode {mode} 仍要求输出中文标签"
        assert "直接输出下面的代码本身" in prompt, f"mode {mode} 未要求输出表情代码"
        #示例里不应再出现被尖括号包住的代码或中文标签
        assert "<jine_pien_loveforever>" not in prompt


def test_prompts_assemble_for_every_mode():
    """mode 0-3都要能装配出非空提示词"""
    for mode in ("0", "1", "2", "3"):
        prompt = config.sys_prompt_for(mode)
        assert prompt.strip(), f"mode {mode} 装配结果为空"
        #顺序固定base -> role
        assert prompt.startswith(config.BASE_PROMPT), f"mode {mode} 未以 base 开头"
        assert prompt == config.join_prompt(
            config.BASE_PROMPT, config.ROLE_PROMPT[mode])
        #带 @@@ 分段协议
        assert "@@@" in prompt


def test_reflection_prompt_keeps_roleplay_rules_and_protocol():
    """审查提示词保留审查输出协议（GraphManager硬依赖）"""
    prompt = config.reflection_prompt_for()
    #仍然拦截私密推理与空内容/离题
    assert "私密推理" in prompt
    assert "空内容" in prompt
    assert "对白以中文为主" in prompt
    #输出协议字样必须保留
    assert "__ACCEPT__" in prompt
    assert "需要修改：" in prompt
    assert "冒号后必须有实际问题" in prompt


def test_node_prompts_and_mode_validation():
    """决策/审查两类节点提示词非空；非法mode要拒绝"""
    assert config.CALL_PROMPT.strip() and config.REFLECTION_PROMPT.strip()
    assert config.call_prompt_for() == config.CALL_PROMPT
    assert config.reflection_prompt_for() == config.REFLECTION_PROMPT
    with pytest.raises(ValueError, match="未知聊天模式"):
        config.sys_prompt_for("9")


def test_emoji_tables_stay_in_sync_between_backend_and_frontend():
    """后端归一化表与前端别名表必须一致，否则历史数据会出现两种渲染结果"""
    from app.config.emoji_labels import EMOJI_CODES, EMOJI_LABELS

    registry = (Path(__file__).resolve().parents[2] / "static" / "js" / "data" / "registry.js").read_text(encoding="utf-8")
    for code in EMOJI_CODES:
        assert f"'{code}'" in registry
    for label, code in EMOJI_LABELS.items():
        assert f"'{label}': '{code}'" in registry, f"前端缺少别名 {label} -> {code}"
