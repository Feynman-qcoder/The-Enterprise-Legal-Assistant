# -*- coding: utf-8 -*-
"""诊断 C 失败：LAW_003(2025修正版) 子块 parent_id 在 MySQL/Milvus 的映射是否完整。只读。"""
import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pymilvus import Collection  # noqa: E402
from sqlalchemy import select  # noqa: E402

from modules.database.models import LegalTab  # noqa: E402
from modules.database.session import get_session_factory  # noqa: E402
from modules.embeddings.local_embedding import LocalEmbeddingService  # noqa: E402
from modules.milvus_store.client import ensure_milvus  # noqa: E402
from modules.milvus_store.collections import COLLECTION_LEGAL_CHILD  # noqa: E402
from modules.rag.corpus_scope import get_canonical_scope  # noqa: E402
from modules.rag.pipeline import _parse_milvus_hits  # noqa: E402

Q_C = "2025年修正后的网络安全法从什么时候开始施行？现行有效的版本是哪一次修正？"
SF_2025 = "LEGAL_DATA_003_中华人民共和国网络安全法_2025修正版.md"


async def main() -> None:
    scope = get_canonical_scope()
    emb = LocalEmbeddingService()
    ensure_milvus()
    col = Collection(COLLECTION_LEGAL_CHILD)
    col.load()

    def search(vec, limit, expr):
        return col.search(
            data=[vec], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=limit, expr=expr or "", output_fields=["source_file", "parent_id", "text"],
        )

    vec = await emb.embed_query(Q_C)
    hits = _parse_milvus_hits(await asyncio.to_thread(search, vec, 60, scope.build_source_file_expr()))
    pids = []
    for _, _, e in hits:
        p = e.get("parent_id")
        pids.append(int(p) if p is not None else -1)
    print(f"dense hits={len(hits)} distinct_pids={sorted(set(pids))}")

    # 各 source_file 分布
    from collections import Counter
    dist = Counter(str(e.get('source_file')) for _, _, e in hits)
    for k, v in dist.most_common():
        print(f"  {v:2d}  {k}")

    # MySQL：这些 pid 是否存在
    factory = get_session_factory()
    async with factory() as session:
        res = await session.execute(select(LegalTab.id).where(LegalTab.id.in_(sorted(set(pids)))))
        existing = {r[0] for r in res.all()}
        res2 = await session.execute(select(LegalTab.id).where(LegalTab.source_file == SF_2025))
        mysql_2025 = [r[0] for r in res2.all()]
    missing = sorted(set(pids) - existing)
    print(f"mysql existing pids={len(existing)} missing={missing}")
    print(f"mysql parents for {SF_2025}: n={len(mysql_2025)} ids={mysql_2025[:20]}")

    # Milvus：2025修正版全量子块的 parent_id 集合
    def q_file():
        return [
            (e.to_dict() if hasattr(e, "to_dict") else dict(e))
            for e in col.query(expr=f'source_file == "{SF_2025}"', output_fields=["parent_id"], limit=1000)
        ]
    rows = await asyncio.to_thread(q_file)
    milvus_pids = sorted({int(r["parent_id"]) for r in rows if r.get("parent_id") is not None})
    print(f"milvus children(2025)={len(rows)} distinct_parent_ids={milvus_pids}")
    print(f"milvus_pids in mysql: {sorted(set(milvus_pids) & existing) if existing else []}")
    print(f"mysql(2025) vs milvus(2025) diff: mysql-only={sorted(set(mysql_2025) - set(milvus_pids))} milvus-only={sorted(set(milvus_pids) - set(mysql_2025))}")


if __name__ == "__main__":
    asyncio.run(main())
