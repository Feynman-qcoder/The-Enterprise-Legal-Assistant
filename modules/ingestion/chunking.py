# =============================================================================
# -----------------------------------------------------------------------------
# 输入：PDF 每页文本列表 `page_texts`；父块内长字符串 `parent_text`。
# 输出：`ParentChunk` 列表（合并页后的父文档）；`ChildChunk` 列表（滑动窗口子切片）。
# 被谁调用：`mysql_loaders.load_legal_pdfs_to_mysql`（先父后子写入 ORM）。
# =============================================================================
"""
父子二段式：Milvus 只索引子块（短、语义集中）；命中子块后用 parent_id 拉父文档全文做重排与生成。

`normalize_ws` 减少空白噪声，有利于 BM25 与向量稳定性。
"""

from __future__ import annotations

import re  # 正则替换空白
from dataclasses import dataclass  # 轻量不可变数据结构


def normalize_ws(text: str) -> str:
    """
    把各种换行统一为 \\n，压空格，压过多空行，最后 strip。

    入参:
        text: 任意原始字符串（可含 \\r\\n、多余空格与空行）。
    返回:
        规范化后的单行/多行文本字符串；首尾空白已去除。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # Windows / 老 Mac 换行统一到 LF
    text = re.sub(r"[ \t]+", " ", text)  # 连续空格或 Tab 变成一个空格
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3 个以上换行压成两个（保留段落感）
    return text.strip()  # 去掉首尾空白


@dataclass(frozen=True)
class ParentChunk:
    """
    一个父块：展示标题（常含页码范围）、合并正文、起始页号。

    属性（dataclass 字段）:
        title: 父块标题，如单页或跨页页码摘要。
        text: 合并、规范化后的父文档正文。
        page_no: 该父块在原文中的起始页码（1-based）。
    """
    title: str
    text: str
    page_no: int


@dataclass(frozen=True)
class ChildChunk:
    """
    一个子块：属于第几个父块（parent_index）、块内序号、切片文本。

    属性（dataclass 字段）:
        parent_index: 所属父块在父块列表中的下标（从 0 起，与入库 enumerate 对齐）。
        chunk_index: 同一父块内子片的序号（从 0 起）。
        text: 子块切片文本（已做空白规范化）。
    """
    parent_index: int
    chunk_index: int
    text: str


def chunk_pdf_pages_to_parents(page_texts: list[str], max_chars: int = 1800) -> list[ParentChunk]:
    """
    按页往后读，缓冲区内总长不超过 max_chars 就继续合并；超过则 flush 成一个 ParentChunk。

    极端下单个父块仍过长时，再按 2*max_chars 硬切多段（少见）。

    入参:
        page_texts: 按页顺序排列的文本列表，下标 0 对应第 1 页内容。
        max_chars: 单个父块目标最大字符数；合并缓冲超过该值会切分新父块。
    返回:
        `ParentChunk` 列表；若发生二次硬切则返回切分后的列表，否则为合并结果。
    """
    parents: list[ParentChunk] = []  # 输出列表
    buf: list[str] = []  # 当前正在攒的页文本
    buf_pages: list[int] = []  # 与 buf 一一对应的页码（从 1 开始）
    buf_len = 0  # 粗略字符总数，用于快速判断是否超 max_chars

    def flush() -> None:
        """
        把 buf 合成一个 ParentChunk 追加到 parents，然后清空 buf。

        入参:
            无（通过闭包读写外层 parents、buf、buf_pages、buf_len）。
        返回:
            无；副作用为向 parents 追加元素并清空缓冲。
        """
        nonlocal buf, buf_pages, buf_len  # 声明要改外层函数的变量
        if not buf:  # 没有内容不必写
            return
        merged = normalize_ws("\n".join(buf))  # 多页用换行拼接再规范化
        title = f"第{buf_pages[0]}-{buf_pages[-1]}页" if len(set(buf_pages)) > 1 else f"第{buf_pages[0]}页"  # 单页/多页标题不同
        parents.append(ParentChunk(title=title, text=merged, page_no=buf_pages[0]))  # page_no 取该父块起始页
        buf, buf_pages, buf_len = [], [], 0  # 重置缓冲区

    for i, pg in enumerate(page_texts, start=1):  # i 为 1-based 页码
        t = normalize_ws(pg)  # 先规范化单页
        if not t:  # 空白页跳过
            continue
        if buf_len + len(t) > max_chars and buf:  # 加上当前页会超且缓冲区已有内容
            flush()  # 先落盘已有缓冲，再接收新页
        buf.append(t)  # 把当前页文本放入缓冲
        buf_pages.append(i)  # 记录页码
        buf_len += len(t)  # 更新长度计数
    flush()  # 文件末尾把剩余缓冲写出

    split_parents: list[ParentChunk] = []  # 二次处理：超宽父块硬切
    for p in parents:  # 遍历初筛后的每个父块
        if len(p.text) <= max_chars * 2:  # 未超过两倍阈值：认为可接受
            split_parents.append(p)  # 原样保留
            continue  # 下一父块
        start = 0  # 硬切起点字符偏移
        part = 0  # 切片序号，拼进标题区分
        while start < len(p.text):  # 直到覆盖全文
            piece = p.text[start : start + max_chars * 2]  # 取一段固定长度子串
            split_parents.append(
                ParentChunk(title=f"{p.title}-{part}", text=piece, page_no=p.page_no),  # 标题加后缀
            )
            start += max_chars * 2  # 下一段起点
            part += 1  # 序号递增
    return split_parents or parents  # 若 split_parents 为空（理论上不应发生）回退 parents


def split_children_from_parent(
    parent_text: str,
    parent_index: int,
    child_chars: int = 512,
    overlap: int = 128,
) -> list[ChildChunk]:
    """
    滑动窗口：窗口长 child_chars，每次向前移动 step = child_chars - overlap。

    过短父文本不滑窗，整段作为一个子块。

    入参:
        parent_text: 父文档全文（将先经空白规范化）。
        parent_index: 该父块在父块列表中的下标，写入每个子块的 parent_index。
        child_chars: 子块窗口长度（字符数）。
        overlap: 相邻窗口重叠长度；步长为 max(child_chars - overlap, 1)。
    返回:
        `ChildChunk` 列表；父文本为空则返回空列表；极短文本则仅含一个子块。
    """
    t = normalize_ws(parent_text)  # 先规范化
    if not t:  # 空父文本
        return []  # 无子块
    if len(t) <= child_chars:  # 不需要滑动
        return [ChildChunk(parent_index=parent_index, chunk_index=0, text=t)]  # 单子块 index=0

    children: list[ChildChunk] = []  # 输出
    step = max(child_chars - overlap, 1)  # 步长至少 1，否则 while 死循环
    idx = 0  # 子块序号从 0 递增
    pos = 0  # 窗口左端在父串中的位置
    while pos < len(t):  # 直到窗口起点越过末尾
        piece = t[pos : pos + child_chars]  # 切片，可能最后一段不足 child_chars
        if piece.strip():  # 全空白切片不要
            children.append(ChildChunk(parent_index=parent_index, chunk_index=idx, text=piece))  # 记录一块
            idx += 1  # 下一块序号
        pos += step  # 窗口右移
    return children
