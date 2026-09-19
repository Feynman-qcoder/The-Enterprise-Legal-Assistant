# =============================================================================
# -----------------------------------------------------------------------------
# 输入：`AsyncSession`（已绑定 MySQL 连接）；磁盘路径 `data/法律问答对.xlsx` 与 `data/*.pdf`。
# 输出：`load_faq_*` 返回导入行数；`load_legal_*` 返回统计 dict；`run_default_file_ingest` 汇总 FAQ+法律。
# 被谁调用：`offline/scripts/run_mysql_ingest.py`、`run_full_offline.py`；间接被学员手工 import 测试。
# =============================================================================
"""
Excel → `faq_tab`；PDF → 抽页 → 父块/子块 → `legal_tab`。

`replace_all` 为 True 时会 DELETE 旧数据，开发环境全量重导常用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from modules.database.models import FaqTab, LegalTab
from modules.ingestion.chunking import (
    chunk_pdf_pages_to_parents,
    split_children_from_parent,
)
from modules.ingestion.document_cleaning import clean_parsed_document
from modules.ingestion.document_parsing import parse_document
from modules.ingestion.pdf_extract import extract_pages_pdf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # modules/ingestion → 上两级到仓库根
DATA_DIR = PROJECT_ROOT / "data"  # 默认 Excel/PDF 所在目录


class FaqWorkbookError(ValueError):
    """Expected FAQ workbook validation failure with a stable API code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _pick_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Excel 表头可能是「问题/答案」或 question/answer 等：本函数返回实际列名字符串二元组。

    入参:
        df: 已读入的 FAQ 表格，列名在 `df.columns` 中。
    返回:
        (问题列名, 答案列名) 二元组，均为 DataFrame 中的真实列名字符串。
        若无法识别则抛出 ValueError。
    """
    cols = {str(c).strip(): c for c in df.columns}  # 原始列名 → 自身，去首尾空格
    lower = {k.lower(): k for k in cols}  # 小写映射，实现大小写不敏感查找

    def find(*names: str) -> str | None:
        """
        按候选名顺序找第一个存在的列名；先精确匹配再小写匹配。

        入参:
            *names: 若干候选列名字符串，按优先级从前到后尝试。
        返回:
            命中的列名字符串；均未命中则返回 None。
        """
        for n in names:  # 遍历候选别名
            if n in cols:  # 精确命中
                return str(cols[n])  # 返回 DataFrame 里真实列名（类型可能是非 str）
            if n.lower() in lower:  # 小写命中
                return str(lower[n.lower()])
        return None  # 都找不到

    q_col = find("问题", "question", "Question", "标题", "问")  # 问题列候选
    a_col = find("答案", "answer", "Answer", "答", "回复")  # 答案列候选
    if not q_col or not a_col:  # 缺一即无法导入
        raise ValueError(
            f"无法识别问答列，请检查 Excel 表头。当前列：{list(df.columns)}",
        )  # 把现有列名打出来方便用户改表头
    return str(q_col), str(a_col)  # 二元组返回


def read_faq_excel_rows(xlsx_path: Path) -> list[tuple[str, str]]:
    """Read the existing first-sheet FAQ schema without writing any storage."""
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as exc:
        raise FaqWorkbookError("PARSE_FAILED", "FAQ XLSX 文件损坏或格式无效") from exc
    try:
        q_col, a_col = _pick_columns(df)
    except ValueError as exc:
        raise FaqWorkbookError("FAQ_SCHEMA_INVALID", "FAQ XLSX 缺少问题/答案列") from exc
    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        question = row.get(q_col)
        answer = row.get(a_col)
        if pd.isna(question) or pd.isna(answer):
            continue
        q_text = str(question).strip()
        a_text = str(answer).strip()
        if q_text and a_text:
            rows.append((q_text, a_text))
    if not rows:
        raise FaqWorkbookError("FAQ_SCHEMA_INVALID", "FAQ XLSX 没有有效问答行")
    return rows


async def load_faq_excel_to_mysql(
    session: AsyncSession,
    xlsx_path: Path,
    *,
    replace_all: bool = True,
) -> int:
    """
    pandas 读 xlsx → 遍历行构造 `FaqTab` → `session.add` → `commit`。

    `replace_all`：先 `DELETE FROM faq_tab` 再插入，保证幂等全量替换。

    入参:
        session: 已打开的异步 SQLAlchemy 会话，用于执行删除与插入。
        xlsx_path: Excel 文件路径（FAQ 表）。
        replace_all: 为 True 时先清空 `faq_tab` 再导入；False 时仅追加。
    返回:
        成功写入（`session.add`）的 FAQ 行数（跳过空行的不计入）。
    """
    rows = read_faq_excel_rows(xlsx_path)
    if replace_all:  # 需要清空旧 FAQ
        await session.execute(delete(FaqTab))  # SQLAlchemy Core：生成 DELETE 语句并执行

    count = 0  # 成功插入行计数
    for question, answer in rows:
        session.add(
            FaqTab(
                question=question,
                answer=answer,
                is_high_frequency=True,  # 默认全部参与后续 Milvus 同步
            ),
        )  # 加入 session 脏队列，尚未 INSERT
        count += 1
    await session.commit()  # 一次性提交事务，真正写入 MySQL
    logger.info("faq rows imported: %s from %s", count, xlsx_path)
    return count


async def load_legal_pdfs_to_mysql(
    session: AsyncSession,
    pdf_dir: Path | None = None,
    *,
    replace_all: bool = True,
) -> dict[str, Any]:
    """
    遍历目录下 PDF：优先文件名含「税」「劳动」等关键字；若无则导入目录内全部 PDF。

    每个 PDF：父行先 `flush` 拿自增 id，再为每个子块写 child 行并外键指向父 id。

    入参:
        session: 异步会话，用于写入 `legal_tab` 并提交。
        pdf_dir: PDF 所在目录；为 None 时使用仓库根下默认 `data/`。
        replace_all: 为 True 时删除 `legal_tab` 全表后再导入（关闭外键检查期间删除）。
    返回:
        统计字典，键含 `files`（处理文件数）、`parents`（父行数）、`children`（子行数）。
    """
    root = pdf_dir or DATA_DIR  # 未传参则用默认 data/
    pdfs = sorted(
        p
        for p in root.glob("*.pdf")  # 只匹配扩展名小写 pdf
        if ("税" in p.name) or ("劳动" in p.name) or ("税法" in p.name) or ("劳动法" in p.name)  # 文件名过滤
    )
    if not pdfs:  # 关键字过滤结果为空
        pdfs = sorted(root.glob("*.pdf"))  # 回退：所有 pdf
        logger.info("未匹配到文件名关键字，回退为导入目录内全部 PDF：%s", [p.name for p in pdfs])
    if not pdfs:  # 目录里根本没有 pdf
        logger.warning("未在 %s 找到任何 PDF，跳过 legal 入库", root)

    if replace_all:  # 清空法律表
        await session.execute(text("SET FOREIGN_KEY_CHECKS=0"))  # MySQL：暂时关闭外键检查，便于任意顺序 DELETE
        await session.execute(text("DELETE FROM legal_tab"))  # 物理删全表（开发环境）
        await session.execute(text("SET FOREIGN_KEY_CHECKS=1"))  # 恢复外键检查

    stats = {"files": 0, "parents": 0, "children": 0}  # 累计统计
    for pdf in pdfs:  # 每个文件单独一批父+子
        # 经由统一 parse → clean，与 Online Incremental Upload 共用同一套 Cleaner
        parsed = parse_document(pdf)
        parsed = clean_parsed_document(parsed)
        pages = list(parsed.segments or (parsed.text,))  # 清洗后按页（PDF）文本
        parents = chunk_pdf_pages_to_parents(pages)  # list[ParentChunk]
        parent_rows: list[LegalTab] = []  # 先收集 ORM 对象，flush 后会有 id
        for pc in parents:  # 每个父块一行 parent
            row = LegalTab(
                source_file=pdf.name,  # 仅文件名
                doc_role="parent",  # 标记为父
                parent_id=None,  # 父行无父
                chunk_index=0,
                title=pc.title,
                content=pc.text,
            )
            session.add(row)  # 挂到 session
            parent_rows.append(row)  # 保留引用以便 flush 后读 row.id
        await session.flush()  # 把 INSERT 发到 DB，但不 commit；自增 id 回填到 row.id

        child_total = 0
        for idx, pr in enumerate(parent_rows):  # idx 与 chunking 里 parent_index 一致（本实现中 enumerate 顺序）
            parent_text = pr.content  # 父正文
            children = split_children_from_parent(parent_text, idx)  # 子块列表
            for ch in children:  # 每个子块一行 child
                session.add(
                    LegalTab(
                        source_file=pdf.name,
                        doc_role="child",
                        parent_id=pr.id,  # flush 后已有数据库主键
                        chunk_index=ch.chunk_index,
                        title=pr.title,
                        content=ch.text,
                    ),
                )
                child_total += 1
        stats["files"] += 1
        stats["parents"] += len(parent_rows)
        stats["children"] += child_total
        logger.info(
            "ingested pdf=%s parents=%s children=%s",
            pdf.name,
            len(parent_rows),
            child_total,
        )

    await session.commit()  # 整个 legal 导入结束提交
    return stats


async def run_default_file_ingest() -> dict[str, Any]:
    """
    离线一键：create_all 建表 + 默认路径 FAQ + 默认目录 PDF。

    在函数内 import 引擎与 Base，避免模块顶层循环 import。

    入参:
        无。
    返回:
        含 `faq_rows` 与法律导入统计（`files`/`parents`/`children` 等）的合并字典。
        若默认 FAQ 文件不存在则抛出 FileNotFoundError。
    """
    from modules.database.models import Base
    from modules.database.session import get_async_engine, get_session_factory

    engine = get_async_engine()  # 异步引擎单例
    async with engine.begin() as conn:  # 自动 begin/commit 或 rollback
        await conn.run_sync(Base.metadata.create_all)  # SQLAlchemy：同步函数包到 run_sync 里在 async 环境执行

    factory = get_session_factory()  # 会话工厂
    async with factory() as session:  # 打开一个会话上下文，退出时 close
        faq_path = DATA_DIR / "法律问答对.xlsx"  # 项目约定默认文件名
        if not faq_path.exists():  # 强依赖：没有就失败
            raise FileNotFoundError(f"缺少 FAQ Excel：{faq_path}")

        faq_n = await load_faq_excel_to_mysql(session, faq_path, replace_all=True)  # 先 FAQ
        legal_stats = await load_legal_pdfs_to_mysql(session, DATA_DIR, replace_all=True)  # 再法律
        return {"faq_rows": faq_n, **legal_stats}  # 合并两个字典返回
