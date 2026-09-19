# =============================================================================
# V2 全量灌入脚本（Task 20 阶段三：开源可复现）。
# -----------------------------------------------------------------------------
# 输入：frozen_assets/ 冻结三资产（chunks / parent_sections / child_parent_mapping）
#       + 可选 embeddings.npy（--precomputed，与生产向量逐位一致）。
# 输出：MySQL 三表（legal_parent_contract_v1 / legal_child_parent_map_v1 / corpus_meta）
#       + Milvus xiaoyi_legal_child_v2（V2 schema + HNSW 索引）+ 终态自检报告。
# 被谁调用：第三方复现者手工执行（README Quickstart 第 2 步）。
# 设计原则：复用 Task 19 写入原语（v2_incremental.replace_document），
#           禁止第二份实现；Milvus 写入沿用 Task 19 加固（Strong 验证 + 重试）。
# =============================================================================
"""
复现语义：
- fresh 环境：自动建表/建集合（幂等 DDL），逐 document_identity（49 个）执行
  replace_document（Task 19 原语：删旧→插新→bump corpus_version），天然幂等。
- 嵌入两条路：
  * 默认：LocalEmbeddingService.embed_documents（BGE-M3 实时嵌入，F3 已确认批量接口）
  * --precomputed：frozen_assets/embeddings.npy（预计算向量，灌入更快且与生产逐位一致；
    通过 duck-typing 适配器实现 embed_documents 接口，Task 19 原语零感知）
- 终态自检（内置硬门禁）：MySQL parent=2217 / identity=49 / map=4134，
  Milvus v2（Strong）=4134，corpus_meta.corpus_version>=1；任一不符 exit(1)。

用法（仓库根目录）：
  python offline/scripts/run_v2_ingest.py                 # 实时嵌入（CPU 慢，GPU 快）
  python offline/scripts/run_v2_ingest.py --precomputed   # 预计算向量（推荐）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根进 sys.path

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility  # noqa: E402

from modules.core.config import get_settings  # noqa: E402
from modules.database.session import get_session_factory  # noqa: E402
from modules.ingestion.v2_incremental import (  # noqa: E402
    V2ChildSpec,
    V2DocumentPackage,
    V2ParentSpec,
    ensure_corpus_meta_table,
    read_corpus_version,
    replace_document,
)
from modules.milvus_store.client import ensure_milvus  # noqa: E402

FROZEN_DIR = Path(__file__).resolve().parents[2] / "frozen_assets"

# 与生产 SHOW CREATE TABLE 逐字对齐（Task 19 G4 探针存证）
_PARENT_DDL = """
CREATE TABLE IF NOT EXISTS `legal_parent_contract_v1` (
  `external_parent_id` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `document_identity` varchar(191) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `section_path` json NOT NULL,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `metadata` json NOT NULL,
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`external_parent_id`),
  KEY `ix_legal_parent_contract_v1_document_identity` (`document_identity`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
"""

_MAP_DDL = """
CREATE TABLE IF NOT EXISTS `legal_child_parent_map_v1` (
  `child_id` char(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `external_parent_id` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  PRIMARY KEY (`child_id`),
  KEY `ix_legal_child_parent_map_v1_parent` (`external_parent_id`),
  CONSTRAINT `fk_legal_child_parent_map_v1_parent` FOREIGN KEY (`external_parent_id`)
    REFERENCES `legal_parent_contract_v1` (`external_parent_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
"""

# 终态校验值（与生产基线一致）
EXPECTED_PARENTS = 2217
EXPECTED_IDENTITIES = 49
EXPECTED_MAP = 4134
EXPECTED_MILVUS = 4134

# parent 行 metadata 列排除字段（这些进独立列，不进 metadata JSON）
_PARENT_META_EXCLUDE = {
    "external_parent_id",
    "document_identity",
    "section_path",
    "content",
}


class PrecomputedEmbeddingService:
    """duck-typing 适配器：实现 embed_documents 接口，向量取自冻结 npy。

    同一文本必然映射到同一向量（Embedding V1 两遍生成 bitwise 一致），
    故 text->vector 字典安全（资产含 49 组重复文本，向量亦相同）。
    """

    def __init__(self, npy_path: Path, manifest_path: Path, chunks_by_id: dict[str, dict[str, Any]]) -> None:
        import numpy as np

        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        vectors = np.load(npy_path)  # (4134, 1024) float32
        self._text_to_vec: dict[str, list[float]] = {}
        for entry in manifest["entries"]:
            child_id = entry["chunk_id"]
            chunk = chunks_by_id.get(child_id)
            if chunk is None:
                raise ValueError(f"embedding manifest references unknown chunk_id: {child_id}")
            self._text_to_vec[chunk["content"]] = [float(x) for x in vectors[entry["vector_index"]]]
        print(f"precomputed vectors loaded: {len(self._text_to_vec)} distinct texts")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: D102
        missing = [t for t in texts if t not in self._text_to_vec]
        if missing:
            raise ValueError(f"precomputed embeddings missing for {len(missing)} text(s), e.g. {missing[0][:60]!r}")
        return [self._text_to_vec[t] for t in texts]


async def ensure_mysql_tables() -> None:
    """幂等建 V2 三表（DDL 与生产逐字对齐；corpus_meta 复用 Task 19 DDL）。"""
    from sqlalchemy import text

    factory = get_session_factory()
    async with factory() as session:
        await session.execute(text(_PARENT_DDL))
        await session.execute(text(_MAP_DDL))
        await session.commit()
    await ensure_corpus_meta_table()  # Task 19 A1：corpus_meta + version=0
    print("[1/5] mysql tables ensured (legal_parent_contract_v1 / legal_child_parent_map_v1 / corpus_meta)")


def ensure_v2_collection(collection_name: str) -> None:
    """幂等建 V2 collection（schema/索引与 retrieval_v2_adapter 契约一致）。"""
    ensure_milvus()
    if utility.has_collection(collection_name):
        col = Collection(collection_name)
        col.load()
        print(f"[2/5] milvus collection exists (loaded): {collection_name}")
        return
    fields = [
        FieldSchema(name="child_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=32768),
        FieldSchema(name="external_parent_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="document_identity", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="section_path", dtype=DataType.JSON),
        FieldSchema(name="metadata_json", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields=fields, description="xiaoyi::frozen_v2", enable_dynamic_field=False)
    col = Collection(name=collection_name, schema=schema)
    col.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.load()
    print(f"[2/5] milvus collection created: {collection_name}")


def load_assets() -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """读三资产：parents 按 identity 分组、mappings 按 parent_id 分组、chunks 按 chunk_id 索引。"""
    with open(FROZEN_DIR / "parent_sections.json", encoding="utf-8") as fh:
        parent_sections = json.load(fh)
    with open(FROZEN_DIR / "child_parent_mapping.json", encoding="utf-8") as fh:
        child_parent_mapping = json.load(fh)
    with open(FROZEN_DIR / "strategy_b_chunks.json", encoding="utf-8") as fh:
        chunks_doc = json.load(fh)

    parents_all = parent_sections["parents"]
    mappings_all = child_parent_mapping["mappings"]
    chunks_all = chunks_doc["chunks"]
    print(
        f"[3/5] assets loaded: parents={len(parents_all)} mappings={len(mappings_all)} chunks={len(chunks_all)}"
    )

    parents_by_identity: dict[str, list[dict]] = {}
    for p in parents_all:
        parents_by_identity.setdefault(p["document_identity"], []).append(p)

    mappings_by_parent: dict[str, list[dict]] = {}
    for m in mappings_all:
        mappings_by_parent.setdefault(m["external_parent_id"], []).append(m)

    # 生产版 chunks.json 每个 chunk 自带 chunk_id/content/document_identity/section_path/source_file
    chunks_by_id = {c["chunk_id"]: c for c in chunks_all}

    # 血缘断言：mapping 的 child 必须在 chunks 中存在
    missing = [m["child_id"] for m in mappings_all if m["child_id"] not in chunks_by_id]
    if missing:
        raise ValueError(f"{len(missing)} mapping child_id(s) missing in chunks.json")
    return parents_by_identity, mappings_by_parent, chunks_by_id, parents_all


def build_packages(
    parents_by_identity: dict[str, list[dict]],
    mappings_by_parent: dict[str, list[dict]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[V2DocumentPackage]:
    """组装 49 个 V2DocumentPackage（显式 ID：复现库与生产逐字节一致）。

    - parent：external_parent_id/section_path/content 来自 parent_sections；
      metadata 列 = 资产行其余字段（title/child_count/document_id/offset_info/source_file/...）
    - child：chunk_id/content/section_path/source_file/metadata(chunk_metadata 全量)
      均来自生产版 chunks.json 自带字段；mapping 仅提供 child→parent 归属
    """
    packages: list[V2DocumentPackage] = []
    for identity in sorted(parents_by_identity):
        parents_spec: list[V2ParentSpec] = []
        source_file_ref = ""
        for p in parents_by_identity[identity]:
            metadata = {k: v for k, v in p.items() if k not in _PARENT_META_EXCLUDE}
            children_spec: list[V2ChildSpec] = []
            for m in mappings_by_parent.get(p["external_parent_id"], []):
                chunk = chunks_by_id[m["child_id"]]
                children_spec.append(
                    V2ChildSpec(
                        content=chunk["content"],
                        section_path=list(chunk["section_path"]),
                        metadata=dict(chunk["chunk_metadata"]),
                        child_id=chunk["chunk_id"],  # 显式 ID：保留冻结资产 chunk_id
                    )
                )
                if not source_file_ref:
                    source_file_ref = chunk.get("source_file") or p.get("source_file") or ""
            parents_spec.append(
                V2ParentSpec(
                    section_path=list(p["section_path"]),
                    content=p["content"],
                    metadata=metadata,
                    external_parent_id=p["external_parent_id"],  # 显式 ID：保留契约 ID
                    children=children_spec,
                )
            )
        packages.append(
            V2DocumentPackage(
                document_identity=identity,
                source_file=source_file_ref,
                parents=parents_spec,
            )
        )
    total_parents = sum(len(pk.parents) for pk in packages)
    total_children = sum(len(ch.children) for pk in packages for ch in pk.parents)
    print(f"[3/5] packages built: {len(packages)} documents / {total_parents} parents / {total_children} children")
    if total_parents != EXPECTED_PARENTS or total_children != EXPECTED_MILVUS or len(packages) != EXPECTED_IDENTITIES:
        raise ValueError(
            f"asset composition mismatch: docs={len(packages)} parents={total_parents} children={total_children}"
        )
    return packages


async def final_check(collection_name: str) -> bool:
    """终态自检（硬门禁）：四值 + version；任一不符返回 False。"""
    from sqlalchemy import text

    factory = get_session_factory()
    async with factory() as session:
        r = await session.execute(text("SELECT COUNT(*) FROM legal_parent_contract_v1"))
        parent_count = r.first()[0]
        r = await session.execute(text("SELECT COUNT(DISTINCT document_identity) FROM legal_parent_contract_v1"))
        identity_count = r.first()[0]
        r = await session.execute(text("SELECT COUNT(*) FROM legal_child_parent_map_v1"))
        map_count = r.first()[0]
    version = await read_corpus_version()

    col = Collection(collection_name)
    try:
        milvus_count = col.query(expr="", output_fields=["count(*)"], consistency_level="Strong")
        milvus_n = int(str(milvus_count[0]).split(":")[-1].rstrip("} ]"))
    except TypeError:
        milvus_count = col.query(expr="", output_fields=["count(*)"])
        milvus_n = int(str(milvus_count[0]).split(":")[-1].rstrip("} ]"))

    checks = [
        ("mysql legal_parent_contract_v1", parent_count, EXPECTED_PARENTS),
        ("mysql distinct document_identity", identity_count, EXPECTED_IDENTITIES),
        ("mysql legal_child_parent_map_v1", map_count, EXPECTED_MILVUS),
        ("milvus xiaoyi_legal_child_v2 (Strong)", milvus_n, EXPECTED_MILVUS),
        ("corpus_meta.corpus_version >= 1", version, None),
    ]
    print()
    print("=== final self-check ===")
    all_ok = True
    for label, actual, expected in checks:
        if expected is None:
            ok = actual >= 1
        else:
            ok = actual == expected
        if not ok:
            all_ok = False
        print(f"  {label}: {actual} (expected={expected if expected is not None else '>=1'}) {'OK' if ok else 'MISMATCH'}")
    print(f"FINAL_CHECK={'PASS' if all_ok else 'FAIL'}")
    return all_ok


async def main() -> None:
    parser = argparse.ArgumentParser(description="V2 full ingestion from frozen assets (reproduction)")
    parser.add_argument("--precomputed", action="store_true", help="use frozen embeddings.npy instead of live BGE-M3")
    parser.add_argument(
        "--confirm-prod",
        action="store_true",
        help="explicitly allow re-ingesting into an already-populated production collection",
    )
    args = parser.parse_args()

    settings = get_settings()
    collection_name = settings.retrieval_v2_collection

    # 防误写保护：目标 collection 已有数据时必须显式确认（防第三方/自己误覆盖生产）
    ensure_milvus()
    if utility.has_collection(collection_name):
        existing = Collection(collection_name)
        try:
            n = int(str(existing.query(expr="", output_fields=["count(*)"])).split(":")[-1].rstrip("} ]"))
        except Exception:  # noqa: BLE001 — 无法计数时放行（由终态自检兜底）
            n = -1
        if n > 0 and not args.confirm_prod:
            print(
                f"ABORT: collection '{collection_name}' already holds {n} entities. "
                "Re-ingestion will replace its content. Pass --confirm-prod to proceed, "
                "or set RETRIEVAL_V2_COLLECTION to a fresh name."
            )
            sys.exit(2)

    await ensure_mysql_tables()
    ensure_v2_collection(collection_name)
    parents_by_identity, mappings_by_parent, chunks_by_id, _ = load_assets()
    packages = build_packages(parents_by_identity, mappings_by_parent, chunks_by_id)

    embedding_service = None
    if args.precomputed:
        embedding_service = PrecomputedEmbeddingService(
            FROZEN_DIR / "embeddings.npy",
            FROZEN_DIR / "embedding_manifest.json",
            chunks_by_id,
        )
        print("[4/5] embedding mode: precomputed (bitwise identical to production)")
    else:
        from modules.embeddings.local_embedding import LocalEmbeddingService

        print("[4/5] embedding mode: live BGE-M3 (LocalEmbeddingService.embed_documents)")

    print(f"[4/5] ingesting {len(packages)} documents via replace_document (Task 19 primitive)...")
    start = time.perf_counter()
    for i, package in enumerate(packages, 1):
        t0 = time.perf_counter()
        stats = await replace_document(package, collection_name, embedding_service)
        inserted = stats["insert"]
        print(
            f"  [{i:02d}/{len(packages)}] {package.document_identity[:44]}: "
            f"parents={inserted['mysql_parent_inserted']} children={inserted['milvus_inserted']} "
            f"v={stats['new_corpus_version']} ({time.perf_counter() - t0:.1f}s)"
        )
    print(f"[4/5] ingestion done in {time.perf_counter() - start:.1f}s")

    ok = await final_check(collection_name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
