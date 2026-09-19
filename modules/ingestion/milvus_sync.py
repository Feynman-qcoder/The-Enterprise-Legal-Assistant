# =============================================================================
# -----------------------------------------------------------------------------
# 输入：MySQL 中 `faq_tab`（is_high_frequency=True）与 `legal_tab`（child 行）；BGE 向量由本模块现算。
# 输出：Milvus 两个集合被清空重建（recreate=True）并插入向量与标量字段；返回统计 dict。
# 被谁调用：`offline/scripts/run_milvus_sync.py`、`run_full_offline.py`；`backend/app/lifespan` 热更新循环。
# =============================================================================
"""
全量同步：读 MySQL → 本地 Embedding 批量编码 → pymilvus `insert`。

阻塞型 pymilvus 调用放在 `asyncio.to_thread`，避免卡住 FastAPI 的 asyncio 循环。
"""

from __future__ import annotations

import asyncio  # to_thread
import logging
from typing import Any

from pymilvus import Collection
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from modules.database.models import FaqTab, LegalTab
from modules.database.session import get_session_factory
from modules.embeddings.local_embedding import LocalEmbeddingService
from modules.milvus_store.collections import COLLECTION_FAQ, COLLECTION_LEGAL_CHILD, recreate_collections

logger = logging.getLogger(__name__)


async def _fetch_faq_rows(session: AsyncSession) -> list[FaqTab]:
    """
    SELECT 所有「需上向量」的 FAQ 行；`is_high_frequency=False` 的不查。

    入参:
        session: 异步 ORM 会话。
    返回:
        `FaqTab` ORM 对象列表（仅 is_high_frequency 为 True 的行）。
    """
    res = await session.execute(select(FaqTab).where(FaqTab.is_high_frequency.is_(True)))  # 构造异步 ORM 查询并执行
    return list(res.scalars().all())  # Result.scalars() 取 ORM 对象流，all()  materialize 成 list


async def _fetch_legal_children(session: AsyncSession) -> list[LegalTab]:
    """
    SELECT doc_role='child' 的行；parent 行不参与向量集合。

    入参:
        session: 异步 ORM 会话。
    返回:
        `LegalTab` 子块行列表（doc_role 为 child）。
    """
    res = await session.execute(select(LegalTab).where(LegalTab.doc_role == "child"))
    return list(res.scalars().all())


def _insert_faq(col: Collection, rows: list[FaqTab], vectors: list[list[float]]) -> None:
    """
    pymilvus insert 需要「按列拆开」的多个等长列表：顺序 id, question, answer, embedding。

    入参:
        col: 已绑定的 FAQ Milvus 集合对象。
        rows: 与 MySQL 行顺序一致的 FAQ ORM 列表。
        vectors: 与 rows 等长的 embedding 列表，每条为浮点向量。
    返回:
        无；副作用为 insert、flush、load。
    """
    if not rows:  # 无数据
        return  # 不调 insert
    ids = [r.id for r in rows]  # Python int 列表，Milvus 收 INT64
    questions = [r.question[:4000] for r in rows]  # schema VARCHAR 上限防护
    answers = [r.answer[:32000] for r in rows]
    col.insert([ids, questions, answers, vectors])  # 四列并行列表
    col.flush()  # 刷盘/刷段，具体语义随 Milvus 版本
    col.load()  # 加载到内存供检索


def _insert_legal(
    col: Collection,
    rows: list[LegalTab],
    vectors: list[list[float]],
    parent_ids: list[int],
    sources: list[str],
) -> None:
    """
    列顺序：id, parent_id, source_file, text, embedding — 必须与 collections._legal_child_fields 一致。

    入参:
        col: 法律子块 Milvus 集合。
        rows: 子块 ORM 行列表，与向量、parent_ids、sources 下标对齐。
        vectors: 各子块 content 的向量表示。
        parent_ids: 与 rows 对齐的父文档主键列表。
        sources: 与 rows 对齐的 source_file 截断字符串列表。
    返回:
        无；副作用为 insert、flush、load。
    """
    if not rows:
        return
    ids = [r.id for r in rows]
    texts = [r.content[:32000] for r in rows]
    col.insert([ids, parent_ids, sources, texts, vectors])
    col.flush()
    col.load()


def _blocking_milvus_full_write(
    dim: int,
    faqs: list[FaqTab],
    faq_vecs: list[list[float]],
    children: list[LegalTab],
    child_vecs: list[list[float]],
    *,
    recreate: bool,
) -> None:
    """
    在同一线程内串行执行：连接、删建集合、插入；避免多线程同时操作 pymilvus 全局状态。

    必须在 `asyncio.to_thread` 里调用本函数。

    入参:
        dim: 向量维度，与 embedding 模型输出一致。
        faqs: 待写入的 FAQ 行。
        faq_vecs: FAQ 问题向量，与 faqs 等长。
        children: 待写入的法律子块行。
        child_vecs: 子块正文向量，与 children 等长。
        recreate: True 时删集合并按当前 schema 重建后再插入。
    返回:
        无。
    """
    from modules.milvus_store.client import ensure_milvus
    from modules.milvus_store.collections import create_collections_if_not_exist

    ensure_milvus()  # 注册连接
    if recreate:  # 热更新/离线默认：删集合重建
        recreate_collections(dim)
    else:  # 仅开发偶发：不删数据只 ensure
        create_collections_if_not_exist(dim)

    faq_col = Collection(COLLECTION_FAQ)  # 绑定集合名
    legal_col = Collection(COLLECTION_LEGAL_CHILD)
    _insert_faq(faq_col, faqs, faq_vecs)  # 先 FAQ 后法律，便于日志分段排查

    parent_ids: list[int] = []  # 与 children 顺序对齐的 parent_id 列
    sources: list[str] = []  # source_file 列
    for c in children:  # 逐行拆标量
        assert c.parent_id is not None  # 数据完整性断言；违反说明入库有 bug
        parent_ids.append(int(c.parent_id))
        sources.append(c.source_file[:1024])

    _insert_legal(legal_col, children, child_vecs, parent_ids, sources)


def _blocking_incremental_faq_insert(dim: int, rows: list[FaqTab], vectors: list[list[float]]) -> None:
    from modules.milvus_store.client import ensure_milvus
    from modules.milvus_store.collections import create_collections_if_not_exist

    ensure_milvus()
    create_collections_if_not_exist(dim)
    _insert_faq(Collection(COLLECTION_FAQ), rows, vectors)


def _blocking_incremental_legal_insert(
    dim: int,
    rows: list[LegalTab],
    vectors: list[list[float]],
) -> None:
    from modules.milvus_store.client import ensure_milvus
    from modules.milvus_store.collections import create_collections_if_not_exist

    ensure_milvus()
    create_collections_if_not_exist(dim)
    parent_ids = [int(row.parent_id) for row in rows if row.parent_id is not None]
    if len(parent_ids) != len(rows):
        raise ValueError("legal child row missing parent_id")
    sources = [row.source_file[:1024] for row in rows]
    _insert_legal(Collection(COLLECTION_LEGAL_CHILD), rows, vectors, parent_ids, sources)


def _blocking_delete_ids(collection_name: str, ids: list[int]) -> None:
    if not ids:
        return
    from modules.milvus_store.client import ensure_milvus

    ensure_milvus()
    collection = Collection(collection_name)
    collection.delete(f"id in [{','.join(str(int(value)) for value in ids)}]")
    collection.flush()


async def insert_faq_incremental(
    rows: list[FaqTab],
    vectors: list[list[float]],
) -> None:
    if rows:
        await asyncio.to_thread(_blocking_incremental_faq_insert, len(vectors[0]), rows, vectors)


async def insert_legal_incremental(
    rows: list[LegalTab],
    vectors: list[list[float]],
) -> None:
    if rows:
        await asyncio.to_thread(_blocking_incremental_legal_insert, len(vectors[0]), rows, vectors)


async def delete_faq_incremental(ids: list[int]) -> None:
    await asyncio.to_thread(_blocking_delete_ids, COLLECTION_FAQ, ids)


async def delete_legal_incremental(ids: list[int]) -> None:
    await asyncio.to_thread(_blocking_delete_ids, COLLECTION_LEGAL_CHILD, ids)


async def sync_mysql_to_milvus(*, recreate: bool = True) -> dict[str, Any]:
    """
    异步阶段：读库 + embed；同步阶段：to_thread 写 Milvus。

    入参:
        recreate: 是否删除并重建 Milvus 集合后再全量插入。
    返回:
        统计字典，含 `embedding_dim`、`faq_vectors`、`legal_child_vectors`（均为数量或维数说明）。
    """
    emb = LocalEmbeddingService()  # 构造即加载模型（较重）
    dim = len(await emb.embed_query("dimension_probe"))  # 用任意短句探测向量维数

    factory = get_session_factory()
    async with factory() as session:  # 只读查询，同一 session 即可
        faqs = await _fetch_faq_rows(session)
        children = await _fetch_legal_children(session)

    faq_vecs: list[list[float]] = []
    if faqs:  # 有问题文本才编码
        faq_vecs = await emb.embed_documents([f.question for f in faqs])  # 顺序与 faqs 一致

    child_vecs: list[list[float]] = []
    if children:
        child_vecs = await emb.embed_documents([c.content for c in children])

    #pymilvus 的操作是同步阻塞的，如果直接在 FastAPI 的异步循环里执行，会卡住整个服务；
    # 用 asyncio.to_thread 丢到线程池，异步框架就能继续处理其他请求。
    await asyncio.to_thread(
        _blocking_milvus_full_write,
        dim,
        faqs,
        faq_vecs,
        children,
        child_vecs,
        recreate=recreate,
    )  # 把位置参数与关键字参数传给线程池里的同步函数

    return {
        "embedding_dim": dim,
        "faq_vectors": len(faqs),
        "legal_child_vectors": len(children),
    }

#封装供外部调用的函数
async def run_sync_job() -> dict[str, Any]:
    """
    对外稳定入口：固定全量 recreate，供脚本与热更新共用。

    入参:
        无。
    返回:
        `sync_mysql_to_milvus(recreate=True)` 的统计字典；成功时日志打印该 dict。
    """
    stats = await sync_mysql_to_milvus(recreate=True)
    logger.info("milvus sync done: %s", stats)
    return stats
