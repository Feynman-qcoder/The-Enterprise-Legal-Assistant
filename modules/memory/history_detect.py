# =============================================================================
# -----------------------------------------------------------------------------
# 输入：用户当前问题字符串（可能含标点、口语）。
# 输出：`is_self_history_question` 返回 `bool`：True 表示「像在问自己的聊天记录」。
# 被谁调用：当前主流程 `RagPipeline` **未** import；供 `offline/tests/test_history_detect.py`、
#          文档与将来扩展路由（例如仅自述历史问题时只查库不跑向量）使用。
# =============================================================================
"""
用多条正则覆盖「我一共问了几个」「前 8 个问题」「聊天记录」等口语。

设计取向：宁可误判为自述历史（少检索），也不要漏判导致模型无法列举历史。
"""

from __future__ import annotations

import re  # 编译型正则，多次匹配效率好

# 元组不可变：模块加载时编译一次，供 is_self_history_question 复用
_HISTORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"我一共问|总共问|一共问|问过几个|问了几个|几个问题"),
    re.compile(r"前\s*\d+\s*个\s*(问题|问)"),
    re.compile(r"第\s*\d+\s*个\s*(问题|问)"),
    re.compile(r"最后\s*\d*\s*个\s*(问题|问)|最后\s*(几|多少)\s*(个\s*)?(问题|问)"),
    re.compile(r"我最后"),
    re.compile(r"的\s*\d+\s*个\s*问题"),
    re.compile(r"历史\s*(问题|记录|聊天)|聊天记录|聊天历史"),
    re.compile(r"我(之前|刚才|前面|上次|早前)(问|提到|说过)"),
    re.compile(r"我\s*问\s*过\s*(什么|啥)"),
    re.compile(r"上次\s*问|上回\s*问"),
    re.compile(r"(回顾|汇总|统计).{0,6}(我|问|聊)"),
    re.compile(r"(我|咱|俺).{0,4}(哪些|哪几)(个\s*)?(问题|问)"),
)


def is_self_history_question(text: str) -> bool:
    """
    任一条正则 `search` 命中则 True；过短字符串直接 False 减少噪声。

    入参:
        text: 用户当前自然语言问题。
    返回:
        True 表示句式像在询问自身聊天/历史记录；False 表示不按自述历史处理。
    """
    s = text.strip()  # 去首尾空白
    if len(s) < 2:  # 单字符几乎不可能是完整自述历史问句
        return False
    return any(p.search(s) for p in _HISTORY_PATTERNS)  # 短路：任一匹配即 True
