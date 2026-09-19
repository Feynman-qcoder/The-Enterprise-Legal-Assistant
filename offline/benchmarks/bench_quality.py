# -*- coding: utf-8 -*-
"""Phase 2 + 3a(single-user) — Quality conservation gate & latency for configs B/C.

Env-driven (experiment switches from Phase 1):
  RERANK_POOL_TOP_N=12  -> truncate rerank pool by RRF order (B & C)
  RERANK_DEVICE=cuda    -> GPU + FP16 reranker (C)
Config A (CPU + full pool): cite Final Regression V1 P30 numbers (not re-run).

Gate (per spec): Final Top-5 target recall == 44/44.
Also records per-stage latency (serves as Phase 3a single-user numbers).

Usage:
  python bench_quality.py <tag> [--gold PATH] [--out PATH]
    # tag in {B, C}; reads env set by caller; 需在仓库根目录运行（modules.* 导入）
    # Task 22 迁移：--gold/--out 默认相对本脚本目录（offline/benchmarks/）
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import pathlib
import statistics
import sys
import time
import warnings
from typing import Any

from pymilvus import Collection
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # 仓库根（Task 22 迁移适配：任意 cwd 可运行）
from modules.core.config import get_settings
from modules.database.session import get_async_engine
from modules.database.v2_parent_repository import V2ParentRepository
from modules.embeddings.local_embedding import LocalEmbeddingService
from modules.milvus_store.client import ensure_milvus
from modules.rag.hybrid_rrf import reciprocal_rank_fusion
from modules.rag.pipeline import _tokenize
from modules.rag.retrieval_v2_adapter import V2DenseRetrievalAdapter
from modules.rerank.local_rerank import LocalRerankService

warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse  # noqa: E402 — Task 22 F8 参数化

_parser = argparse.ArgumentParser(description="Quality conservation gate & single-user latency")
_parser.add_argument("tag", nargs="?", default="B")
_parser.add_argument("--gold", default=str(pathlib.Path(__file__).parent / "aligned_goldset_v2.jsonl"))
_parser.add_argument("--out", default=None)
_args = _parser.parse_args()
TAG = _args.tag
GOLD_PATH = pathlib.Path(_args.gold)
OUT = pathlib.Path(_args.out) if _args.out else pathlib.Path(__file__).parent / f"bench_quality_{TAG.lower()}.json"

P = {"dense_k": 60, "bm25_k": 60, "pool": 30}  # P30 (production-equivalent)
_WS = __import__("re").compile(r"\s+")
norm = lambda s: _WS.sub("", s)  # noqa: E731


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3),
        "p99": round(ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)], 3),
        "max": round(ordered[-1], 3),
    }


async def main() -> None:
    pool_top_n = int(os.environ.get("RERANK_POOL_TOP_N", "0") or 0)
    device = os.environ.get("RERANK_DEVICE", "cpu")
    print(f"CONFIG {TAG}: RERANK_POOL_TOP_N={pool_top_n or 'unset(full)'} RERANK_DEVICE={device}", flush=True)
    assert os.environ.get("RETRIEVAL_CONTRACT") == "v2"

    settings = get_settings()
    gold = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(gold) == 44

    ensure_milvus()
    legacy_before = {
        "legal": int(Collection("xiaoyi_legal_child").num_entities),
        "faq": int(Collection("xiaoyi_faq_highfreq").num_entities),
        "v2": int(Collection(settings.retrieval_v2_collection).num_entities),
    }

    repository = V2ParentRepository()
    identities = await repository.fetch_document_identities()
    adapter = V2DenseRetrievalAdapter(settings.retrieval_v2_collection)
    embedding = LocalEmbeddingService()
    reranker = LocalRerankService()  # loads per RERANK_DEVICE

    wv = await embedding.embed_query("合同争议解决条款")
    wh = await adapter.search(wv, limit=1, document_identities=identities)
    wp = await repository.fetch_parents([str(wh[0].parent_reference)])
    await reranker.rank("合同争议解决条款", [next(iter(wp.values())).content])
    print("WARMUP complete", flush=True)

    vectors = {g["query_id"]: await embedding.embed_query(g["query"]) for g in gold}

    stage_ms: dict[str, list[float]] = {"dense": [], "bm25_rrf": [], "parent": [], "rerank": [], "total": []}
    rows: list[dict[str, Any]] = []
    for idx, g in enumerate(gold, start=1):
        qid, query, prefix = g["query_id"], g["query"], g["target_identity_prefix"]
        phrase = norm(g["primary_phrase"])

        t0 = time.perf_counter()
        hits = await adapter.search(vectors[qid], limit=P["dense_k"], document_identities=identities)
        dense_ms = (time.perf_counter() - t0) * 1000
        child_ids = [h.chunk_id for h in hits]
        hit_by_id = {h.chunk_id: h for h in hits}

        t0 = time.perf_counter()
        corpus = [_tokenize(h.content) for h in hits]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        bm25_order = [child_ids[i] for i in sorted(range(len(child_ids)), key=lambda k: scores[k], reverse=True)][:P["bm25_k"]]
        fused = reciprocal_rank_fusion([child_ids, bm25_order], k=settings.hybrid_rrf_k)
        fused_ids = [c for c, _ in fused[:P["pool"]]]
        bm25_rrf_ms = (time.perf_counter() - t0) * 1000

        parent_ids: list[str] = []
        seen: set[str] = set()
        for c in fused_ids:
            pid = str(hit_by_id[c].parent_reference)
            if pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
        # 实验开关：按 RRF 顺序截断 rerank 池（等价 pipeline.py 的 RERANK_POOL_TOP_N 位置）
        if pool_top_n > 0 and len(parent_ids) > pool_top_n:
            parent_ids = parent_ids[:pool_top_n]

        t0 = time.perf_counter()
        parents = await repository.fetch_parents(parent_ids)
        parent_ms = (time.perf_counter() - t0) * 1000
        orphans = len(set(parent_ids) - set(parents))

        passages = [parents[pid].content for pid in parent_ids]
        t0 = time.perf_counter()
        ce = await reranker.rank(query, passages)
        rerank_ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(passages)), key=lambda i: ce[i], reverse=True)
        top5_ids = [parent_ids[i] for i in order[: settings.legal_rerank_top_n]]

        final_target = any(parents[pid].document_identity.startswith(prefix) for pid in top5_ids)
        evidence_hit = any(phrase in norm(parents[pid].content) for pid in top5_ids)
        gt_hit = any(pid in set(g["gt_parent_ids"]) for pid in top5_ids)

        total_ms = dense_ms + bm25_rrf_ms + parent_ms + rerank_ms
        for k, v in (("dense", dense_ms), ("bm25_rrf", bm25_rrf_ms), ("parent", parent_ms), ("rerank", rerank_ms), ("total", total_ms)):
            stage_ms[k].append(v)
        rows.append({
            "query_id": qid, "pool_size": len(parent_ids), "orphans": orphans,
            "final_top5_target": final_target, "gt_parent_hit": gt_hit, "evidence_hit": evidence_hit,
            "latency_ms": {k: round(v, 1) for k, v in (("dense", dense_ms), ("bm25_rrf", bm25_rrf_ms), ("parent", parent_ms), ("rerank", rerank_ms), ("total", total_ms))},
        })
        if idx % 10 == 0 or not final_target:
            print(f"  {TAG} {idx}/44 {qid} pool={len(parent_ids)} final={final_target} ev={evidence_hit} total={total_ms:.0f}ms", flush=True)

    legacy_after = {
        "legal": int(Collection("xiaoyi_legal_child").num_entities),
        "faq": int(Collection("xiaoyi_faq_highfreq").num_entities),
        "v2": int(Collection(settings.retrieval_v2_collection).num_entities),
    }
    await get_async_engine().dispose()

    result = {
        "tag": TAG,
        "config": {"pool_top_n": pool_top_n or None, "rerank_device": device, "profile": "P30"},
        "quality": {
            "final_top5_target_recall": sum(r["final_top5_target"] for r in rows),
            "gt_parent_hit": sum(r["gt_parent_hit"] for r in rows),
            "evidence_hit": sum(r["evidence_hit"] for r in rows),
            "orphan_total": sum(r["orphans"] for r in rows),
            "gate_pass": sum(r["final_top5_target"] for r in rows) == 44,
        },
        "single_user_latency_ms": {k: summarize(v) for k, v in stage_ms.items()},
        "pool_size_mean": round(statistics.fmean([r["pool_size"] for r in rows]), 2),
        "legacy_before": legacy_before,
        "legacy_after": legacy_after,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT {TAG}: final={result['quality']['final_top5_target_recall']}/44 ev={result['quality']['evidence_hit']}/44 "
          f"orphan={result['quality']['orphan_total']} GATE={'PASS' if result['quality']['gate_pass'] else 'FAIL'}")
    print(f"TOTAL latency: {result['single_user_latency_ms']['total']}")
    print(f"RERANK latency: {result['single_user_latency_ms']['rerank']}")
    print(f"OUT {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
