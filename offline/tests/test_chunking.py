"""单元测试：chunking 模块的空白规范化、父块合并与子块滑动窗口（无外部 IO）。"""

from modules.ingestion.chunking import (
    chunk_pdf_pages_to_parents,
    normalize_ws,
    split_children_from_parent,
)


def test_normalize_ws():
    """压缩多余空白后应与预期字符串匹配。

    入参:
        无。
    返回:
        无；失败时 pytest 断言抛错。
    """
    raw = "  a \n\n\n b  "  # 多空格、多重换行
    assert normalize_ws(raw).strip() == "a \n\n b"  # 多余 \\n 被压成最多双换行，首尾 strip 后与预期一致


def test_chunk_parents_merge_pages():
    """两页合成父块时应至少得到一个非空 ParentChunk。

    入参:
        无。
    返回:
        无。
    """
    pages = ["第一章内容" * 50, "第二章" * 10]  # 模拟两页文本，总长触发合并策略
    parents = chunk_pdf_pages_to_parents(pages, max_chars=500)  # 父块目标长度 500 字符
    assert len(parents) >= 1  # 至少形成一个父块
    assert all(p.text for p in parents)  # 每个父块正文非空


def test_split_children_overlap():
    """长文本应切成多块且 overlap 生效（块数>=2）。

    入参:
        无。
    返回:
        无。
    """
    text = "条款" * 200  # 足够长的中文重复串
    kids = split_children_from_parent(text, parent_index=0, child_chars=100, overlap=20)  # 窗口 100、重叠 20
    assert len(kids) >= 2  # 步长 80，总长足够则至少两块
