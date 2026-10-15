"""
config/emoji_labels.py  表情标签归一化

提示词里的 <永远爱你> 只是含义说明，模型输出、数据库与前端统一使用 jine_pien_* 代码。
模型偶尔会直接输出中文标签（<永远爱你> / 【永远爱你】），这里统一转成代码，
避免数据库里存中文标签、前端渲染不出表情图片。
"""
import re

#中文标签 -> 代码
EMOJI_LABELS = {
    "你说得对": "jine_pien_compromise",
    "死了": "jine_pien_die",
    "永远爱你": "jine_pien_loveforever",
    "嘤嘤": "jine_pien_nyan",
    "好": "jine_pien_ok",
    "太强了": "jine_pien_praise",
    "抱歉": "jine_pien_sorry",
    "关我屁事": "jine_pien_uncare",
}

EMOJI_CODES = list(EMOJI_LABELS.values())

_LABEL_PATTERN = "|".join(map(re.escape, EMOJI_LABELS))
_WRAPPED = re.compile(
    r"<\s*(?P<angle>" + _LABEL_PATTERN + r")\s*>"
    r"|【\s*(?P<wide>" + _LABEL_PATTERN + r")\s*】"
    r"|\[\s*(?P<square>" + _LABEL_PATTERN + r")\s*\]"
)


def normalize_emoji_text(text):
    """把中文表情标签归一化为 jine_pien_* 代码。

    整条回复就是一个表情时（含带括号写法）返回代码；
    标签夹在文字中间时只替换被括号包裹的标签部分（避免误伤"好"这类单字）。
    """
    if not text:
        return text
    raw = str(text).strip()
    #整条就是表情：带括号的任意标签，或不带括号的多字标签
    if raw in EMOJI_LABELS and len(raw) > 1:
        return EMOJI_LABELS[raw]
    return _WRAPPED.sub(
        lambda match: EMOJI_LABELS[next(value for value in match.groups() if value is not None)],
        str(text),
    )
