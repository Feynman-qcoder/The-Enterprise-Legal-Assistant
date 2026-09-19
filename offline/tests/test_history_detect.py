"""记忆意图：判断是否询问自身聊天历史。"""

from modules.memory.history_detect import is_self_history_question


def test_history_question_positive():
    """命中「统计/序数/聊天记录」等模式时应返回 True，允许管线查 his_chat_tab。

    入参:
        无。
    返回:
        无。
    """
    assert is_self_history_question("我一共问了几个问题？") is True  # 「一共问」模式
    assert is_self_history_question("我前8个问题问了什么") is True  # 「前 N 个」模式
    assert is_self_history_question("聊天记录里我说过啥") is True  # 「聊天记录」关键词
    assert is_self_history_question("我最后问题的5个问题是什么？") is True  # 「的 N 个问题」/「我最后」
    assert is_self_history_question("最后5个问题是什么") is True  # 「最后 N 个」


def test_history_question_negative():
    """普通业务咨询与空串不应触发历史检索分支。

    入参:
        无。
    返回:
        无。
    """
    assert is_self_history_question("个人所得税税率是多少") is False  # 正常税法问题
    assert is_self_history_question("") is False  # 长度过短直接 False
