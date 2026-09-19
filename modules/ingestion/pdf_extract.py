# =============================================================================
# -----------------------------------------------------------------------------
# 输入：磁盘上 PDF 文件的 `Path`。
# 输出：`list[str]`：第 i 个元素为第 i+1 页的纯文本（解析失败则为空串）。
# 被谁调用：`mysql_loaders.load_legal_pdfs_to_mysql`（遍历每个 PDF 调用本函数）。
# =============================================================================
"""
使用 PyPDF2 的继任者 `pypdf`：只做文本层抽取，扫描件无文字层时得到空串（需 OCR 另方案）。
"""

from __future__ import annotations

from pathlib import Path  # 类型提示与 str(path) 传参

from pypdf import PdfReader  # 轻量依赖，按页遍历 Page 对象


def extract_pages_pdf(path: Path) -> list[str]:
    """
    打开 PDF，顺序遍历每一页，抽取 text；单页异常不中断全书。

    入参:
        path: PDF 文件的本地路径。
    返回:
        长度等于页数的字符串列表；第 i 个元素为第 i+1 页文本，解析失败则为空串。
    """
    reader = PdfReader(str(path))  # pypdf 需要文件路径字符串
    pages: list[str] = []  # 累加每页结果
    for p in reader.pages:  # Page 对象迭代器
        try:  # 单页保护
            pages.append(p.extract_text() or "")  # extract_text 可能返回 None，转空串保持类型 str
        except Exception:  # noqa: BLE001 — 单页损坏、编码问题等
            pages.append("")  # 用空串占位，页码索引仍与物理页对齐
    return pages  # 调用方按页码 1..n 与列表下标 0..n-1 对应
