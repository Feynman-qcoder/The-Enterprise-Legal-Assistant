# =============================================================================
# V2 增量更新写入原语（Task 19 Part A：A1 注册表 + A2 原语）。
# -----------------------------------------------------------------------------
# 输入：V2DocumentPackage（document_identity / source_file / parents[]+children[]）。
# 输出：文档级 delete / insert / replace 的落地副作用 + 统计 dict。
# 被谁调用：offline/scripts/run_v2_doc_sync.py（离线同步编排脚本）。
# 不被谁调用：在线检索链路绝不 import 本模块（写入只走离线脚本）。
# =============================================================================
"""
文档级 replace 语义（顺序契约，不可颠倒）：

delete_document(document_identity)
  ① Milvus: delete(expr='document_identity == ...') + flush   —— 撤可见性
  ② MySQL : 事务删除 map 行 + parent 行                       —— 撤真源
  中间态只会留下孤儿 parent（检索不可见，无害）；
  反序会产生 child 指向已删 parent → 触发在线 fail-closed 契约错误。

insert_document(package)
  ① MySQL : 事务写入 parent 行 + map 行（FK 要求先父后子）
  ② BGE-M3 批量嵌入 children（复用 LocalEmbeddingService.embed_documents）
  ③ Milvus: insert + flush + load                              —— 最后放可见性
  中间态：新 parent 无 child → 检索不可见，无害。

replace_document(package) = delete_document + insert_document + bump_corpus_version
幂等：同一 package 重复执行结果一致。

版本注册表（A1，设计裁决已定）：
  版本真源 = MySQL 单行表 corpus_meta（meta_key='corpus_version'），非 manifest hash。
  manifest 是冻结资产禁止改写；同 identity 的文档内容修改不会改变 manifest，
  hash 方案对该场景失效。任何成功同步以 UPDATE +1 收尾，增/删/改三种操作全部正确翻面。
  本表不注册进 Base.metadata（沿用 V2 parent repo 原则：运行时表不与 create_all 耦合）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from pymilvus import Collection
from sqlalchemy import bindparam, text

from modules.database.session import get_session_factory
from modules.embeddings.local_embedding import LocalEmbeddingService
from modules.milvus_store.client import ensure_milvus

logger = logging.getLogger(__name__)

CORPUS_META_TABLE = "corpus_meta"
CORPUS_VERSION_KEY = "corpus_version"

_CREATE_CORPUS_META_DDL = f"""
CREATE TABLE IF NOT EXISTS `{CORPUS_META_TABLE}` (
  `meta_key` VARCHAR(64) NOT NULL,
  `meta_value` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`meta_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
"""


@dataclass(frozen=True, slots=True)
class V2ChildSpec:
    """子块规格：content 进 Milvus 正文列 + 嵌入源。

    child_id（Task 20 可选）：冻结资产复现灌入时保留原 chunk_id，
    缺省时沿用 Task 19 的 uuid4 生成（行为不变）。
    """

    content: str
    section_path: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    child_id: str | None = None


@dataclass(frozen=True, slots=True)
class V2ParentSpec:
    """父段规格：content 落 MySQL parent 行，children 落 map 行 + Milvus 实体。

    external_parent_id（Task 20 可选）：冻结资产复现灌入时保留原契约 ID，
    缺省时沿用 Task 19 的 sha256 生成（行为不变）。
    """

    section_path: list[str]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    external_parent_id: str | None = None
    children: list[V2ChildSpec] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class V2DocumentPackage:
    """文档级替换包：一篇文档的完整 parent/child 契约。"""

    document_identity: str
    source_file: str
    parents: list[V2ParentSpec]


def _make_external_parent_id(document_identity: str, parent_index: int, section_path: list[str]) -> str:
    """生成 64-hex 父 ID（与既有 external_parent_id CHAR(64) ascii_bin 形态一致）。"""
    joined = "|".join(section_path) if section_path else ""
    return hashlib.sha256(f"{document_identity}|{parent_index}|{joined}".encode("utf-8")).hexdigest()


def _make_child_id() -> str:
    """生成 32-hex 子 ID（与既有 child_id CHAR(32) uuid4().hex 形态一致）。"""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# A1：corpus_meta 版本注册表
# ---------------------------------------------------------------------------

async def ensure_corpus_meta_table() -> None:
    """建表（幂等）并写入初始 version=0（INSERT IGNORE）。DDL 内联，不进 Base.metadata。"""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(text(_CREATE_CORPUS_META_DDL))
        await session.execute(
            text(
                f"INSERT IGNORE INTO `{CORPUS_META_TABLE}` (meta_key, meta_value) "
                "VALUES (:k, '0')"
            ),
            {"k": CORPUS_VERSION_KEY},
        )
        await session.commit()
    logger.info("corpus_meta registry ready (key=%s)", CORPUS_VERSION_KEY)


async def read_corpus_version() -> int:
    """读当前语料版本；表/行不存在返回 0（version 特性惰性生效）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                f"SELECT meta_value FROM `{CORPUS_META_TABLE}` "
                "WHERE meta_key = :k"
            ),
            {"k": CORPUS_VERSION_KEY},
        )
        row = result.first()
    if row is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        logger.error("corpus_version value invalid: %r", row[0])
        return 0


async def bump_corpus_version() -> int:
    """版本 +1 并返回新值；增/删/改任何成功同步都必须以本调用收尾。"""
    await ensure_corpus_meta_table()
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                f"UPDATE `{CORPUS_META_TABLE}` "
                "SET meta_value = CAST(CAST(meta_value AS UNSIGNED) + 1 AS CHAR) "
                "WHERE meta_key = :k"
            ),
            {"k": CORPUS_VERSION_KEY},
        )
        await session.commit()
    new_version = await read_corpus_version()
    logger.info("corpus_version bumped to %d", new_version)
    return new_version


# ---------------------------------------------------------------------------
# A2：写入原语
# ---------------------------------------------------------------------------

async def delete_document(document_identity: str, collection_name: str) -> dict[str, Any]:
    """删除一篇文档：Milvus 先删（撤可见性），MySQL 后删（撤真源）。幂等。

    MySQL 事务内先删 map 行（子）再删 parent 行（父），满足 FK 约束。
    """

    def _milvus_delete() -> int:
        ensure_milvus()  # 先确保 gRPC 连接存在（脚本进程内首次触达 Milvus）
        collection = Collection(collection_name)
        expr = f'document_identity == "{document_identity}"'
        total_deleted = 0
        # Milvus 2.4 已知坑：growing segment 上的 delete delta 可能漏删（G4-⑥ 实测），
        # 故删后以 Strong 一致性验证残留并重试（幂等 expr，重试安全），上限 3 轮。
        for attempt in range(3):
            result = collection.delete(expr)
            collection.flush()
            total_deleted += int(getattr(result, "delete_count", 0) or 0)
            try:
                residual = collection.query(
                    expr=expr,
                    output_fields=["child_id"],
                    consistency_level="Strong",
                )
            except TypeError:  # 旧版 pymilvus 不支持该参数时退化为默认一致性
                residual = collection.query(expr=expr, output_fields=["child_id"])
            if not residual:
                return total_deleted
            logger.warning(
                "milvus delete left %d residual row(s) (attempt=%d), retrying",
                len(residual),
                attempt + 1,
            )
        return total_deleted

    deleted_milvus = await asyncio.to_thread(_milvus_delete)

    deleted_map = 0
    deleted_parents = 0
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            result = await session.execute(
                text(
                    "SELECT external_parent_id FROM legal_parent_contract_v1 "
                    "WHERE document_identity = :ident"
                ),
                {"ident": document_identity},
            )
            parent_ids = [str(row[0]) for row in result.all()]
            if parent_ids:
                stmt_map = text(
                    "DELETE FROM legal_child_parent_map_v1 "
                    "WHERE external_parent_id IN :pids"
                ).bindparams(bindparam("pids", expanding=True))
                res_map = await session.execute(stmt_map, {"pids": parent_ids})
                deleted_map = int(res_map.rowcount or 0)
                res_parent = await session.execute(
                    text(
                        "DELETE FROM legal_parent_contract_v1 "
                        "WHERE document_identity = :ident"
                    ),
                    {"ident": document_identity},
                )
                deleted_parents = int(res_parent.rowcount or 0)

    stats = {
        "operation": "delete",
        "document_identity": document_identity,
        "milvus_deleted": deleted_milvus,
        "mysql_map_deleted": deleted_map,
        "mysql_parent_deleted": deleted_parents,
    }
    logger.info("v2 delete_document stats: %s", stats)
    return stats


async def insert_document(
    package: V2DocumentPackage,
    collection_name: str,
    embedding_service: LocalEmbeddingService | None = None,
) -> dict[str, Any]:
    """插入一篇文档：MySQL 先写（parent+map），BGE-M3 嵌入，Milvus 后插。"""
    if not package.parents:
        raise ValueError("document package must contain at least one parent")

    # ① MySQL：事务写入 parent 行 + map 行（FK：先父后子）
    rows: list[dict[str, Any]] = []  # Milvus 实体行（child 级）
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            for idx, parent in enumerate(package.parents):
                external_parent_id = parent.external_parent_id or _make_external_parent_id(
                    package.document_identity, idx, list(parent.section_path)
                )  # Task 20：冻结资产灌入优先保留原契约 ID（缺省生成行为不变）
                await session.execute(
                    text(
                        "INSERT INTO legal_parent_contract_v1 "
                        "(external_parent_id, document_identity, section_path, content, metadata) "
                        "VALUES (:pid, :ident, :sp, :content, :meta)"
                    ),
                    {
                        "pid": external_parent_id,
                        "ident": package.document_identity,
                        "sp": json.dumps(list(parent.section_path), ensure_ascii=False),
                        "content": parent.content,
                        "meta": json.dumps(parent.metadata, ensure_ascii=False),
                    },
                )
                for child in parent.children:
                    child_id = child.child_id or _make_child_id()  # Task 20：显式 chunk_id 优先
                    await session.execute(
                        text(
                            "INSERT INTO legal_child_parent_map_v1 "
                            "(child_id, external_parent_id) "
                            "VALUES (:cid, :pid)"
                        ),
                        {"cid": child_id, "pid": external_parent_id},
                    )
                    rows.append(
                        {
                            "child_id": child_id,
                            "content": child.content,
                            "external_parent_id": external_parent_id,
                            "document_identity": package.document_identity,
                            "source_file": package.source_file,
                            "section_path": list(child.section_path),
                            "metadata_json": child.metadata,
                        }
                    )

    # ② BGE-M3 批量嵌入 children（顺序与 rows 严格一致）
    if not rows:
        raise ValueError("document package must contain at least one child")
    if embedding_service is None:
        embedding_service = LocalEmbeddingService()
    vectors = await embedding_service.embed_documents([row["content"] for row in rows])

    # ③ Milvus：insert + flush + load（最后放可见性）
    def _milvus_insert() -> int:
        ensure_milvus()  # 先确保 gRPC 连接存在（脚本进程内首次触达 Milvus）
        collection = Collection(collection_name)
        entities = [
            {
                "child_id": row["child_id"],
                "embedding": vec,
                "content": row["content"],
                "external_parent_id": row["external_parent_id"],
                "document_identity": row["document_identity"],
                "source_file": row["source_file"],
                "section_path": row["section_path"],
                "metadata_json": row["metadata_json"],
            }
            for row, vec in zip(rows, vectors, strict=True)
        ]
        result = collection.insert(entities)
        collection.flush()
        collection.load()
        return len(entities)

    inserted_milvus = await asyncio.to_thread(_milvus_insert)

    stats = {
        "operation": "insert",
        "document_identity": package.document_identity,
        "mysql_parent_inserted": len(package.parents),
        "mysql_map_inserted": len(rows),
        "milvus_inserted": inserted_milvus,
    }
    logger.info("v2 insert_document stats: %s", stats)
    return stats


async def replace_document(
    package: V2DocumentPackage,
    collection_name: str,
    embedding_service: LocalEmbeddingService | None = None,
) -> dict[str, Any]:
    """文档级替换：delete + insert + bump_version。幂等：重复执行结果一致。"""
    delete_stats = await delete_document(package.document_identity, collection_name)
    insert_stats = await insert_document(package, collection_name, embedding_service)
    new_version = await bump_corpus_version()
    return {
        "operation": "replace",
        "document_identity": package.document_identity,
        "delete": delete_stats,
        "insert": insert_stats,
        "new_corpus_version": new_version,
    }
