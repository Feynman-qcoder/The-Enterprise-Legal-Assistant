# -*- coding: utf-8 -*-
"""Phase 3a — Retrieval-chain concurrency benchmark (in-process, no LLM).

Config via env (A: none / B: RERANK_POOL_TOP_N=12 / C: +RERANK_DEVICE=cuda).
VU level & duration via argv. Workers loop round-robin over 44 gold queries,
running the full retrieval chain per request. Records per-request latency,
QPS, failure rate.

Usage: python bench_concurrency.py <tag> <vu> <duration_s> [--gold PATH] [--out PATH]
  Task 22 迁移：--gold/--out 默认相对本脚本目录（offline/benchmarks/）
"""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import statistics
import sys
import time
import warnings

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # 仓库根（Task 22 迁移适配：任意 cwd 可运行）
from modules.core.config import get_settings
from modules.database.v2_parent_repository import V2ParentRepository
from modules.embeddings.local_embedding import LocalEmbeddingService
from modules.milvus_store.client import ensure_milvus
from modules.rag.hybrid_rrf import reciprocal_rank_fusion
from modules.rag.pipeline import _tokenize
from modules.rag.retrieval_v2_adapter import V2DenseRetrievalAdapter
from modules.rerank.local_rerank import LocalRerankService

warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse  # noqa: E402 — Task 22 F8 参数化

_parser = argparse.ArgumentParser(description="Retrieval-chain concurrency benchmark (in-process)")
_parser.add_argument("tag")
_parser.add_argument("vu", type=int)
_parser.add_argument("duration_s", type=float)
_parser.add_argument("--gold", default=str(pathlib.Path(__file__).parent / "aligned_goldset_v2.jsonl"))
_parser.add_argument("--out", default=None)
_args = _parser.parse_args()
TAG = _args.tag
VU = _args.vu
DURATION = _args.duration_s
OUT = pathlib.Path(_args.out) if _args.out else pathlib.Path(__file__).parent / f"bench_conc_{TAG.lower()}_vu{VU}.json"

GOLD = [json.loads(l) for l in pathlib.Path(_args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]

P = {"dense_k": 60, "bm25_k": 60, "pool": 30}
import os  # noqa: E402

POOL_TOP_N = int(os.environ.get("RERANK_POOL_TOP_N", "0") or 0)
DEVICE = os.environ.get("RERANK_DEVICE", "cpu")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 1)


async def main() -> None:
    assert os.environ.get("RETRIEVAL_CONTRACT") == "v2"
    settings = get_settings()
    repository = V2ParentRepository()
    identities = await repository.fetch_document_identities()
    adapter = V2DenseRetrievalAdapter(settings.retrieval_v2_collection)
    embedding = LocalEmbeddingService()
    reranker = LocalRerankService()

    # warm-up once
    wv = await embedding.embed_query("合同争议解决条款")
    wh = await adapter.search(wv, limit=1, document_identities=identities)
    wp = await repository.fetch_parents([str(wh[0].parent_reference)])
    await reranker.rank("合同争议解决条款", [next(iter(wp.values())).content])
    print(f"WARMUP done ({TAG} vu={VU} pool={POOL_TOP_N or 'full'} dev={DEVICE})", flush=True)

    latencies: list[float] = []
    failures = 0
    counter = 0
    lock = asyncio.Lock()
    stop_at = time.perf_counter() + DURATION

    async def worker(wid: int) -> None:
        nonlocal counter, failures
        i = wid
        while time.perf_counter() < stop_at:
            g = GOLD[i % len(GOLD)]
            i += VU
            t0 = time.perf_counter()
            try:
                qv = await embedding.embed_query(g["query"])
                hits = await adapter.search(qv, limit=P["dense_k"], document_identities=identities)
                child_ids = [h.chunk_id for h in hits]
                corpus = [_tokenize(h.content) for h in hits]
                bm25 = BM25Okapi(corpus)
                scores = bm25.get_scores(_tokenize(g["query"]))
                bm25_order = [child_ids[k] for k in sorted(range(len(child_ids)), key=lambda x: scores[x], reverse=True)][:P["bm25_k"]]
                fused = reciprocal_rank_fusion([child_ids, bm25_order], k=settings.hybrid_rrf_k)
                fused_ids = [c for c, _ in fused[:P["pool"]]]
                parent_ids, seen = [], set()
                for c in fused_ids:
                    pid = str(next(h for h in hits if h.chunk_id == c).parent_reference)
                    if pid not in seen:
                        seen.add(pid)
                        parent_ids.append(pid)
                if POOL_TOP_N > 0 and len(parent_ids) > POOL_TOP_N:
                    parent_ids = parent_ids[:POOL_TOP_N]
                parents = await repository.fetch_parents(parent_ids)
                passages = [parents[pid].content for pid in parent_ids]
                ce = await reranker.rank(g["query"], passages)
                top5 = sorted(range(len(passages)), key=lambda k: ce[k], reverse=True)[: settings.legal_rerank_top_n]
                assert top5
                ms = (time.perf_counter() - t0) * 1000
                async with lock:
                    latencies.append(ms)
                    counter += 1
            except Exception:  # noqa: BLE001
                async with lock:
                    failures += 1
                    counter += 1

    print(f"RUN {TAG} vu={VU} for {DURATION:.0f}s ...", flush=True)
    await asyncio.gather(*[worker(w) for w in range(VU)])

    ok = [x for x in latencies]
    result = {
        "tag": TAG, "vu": VU, "duration_s": round(DURATION, 1),
        "config": {"pool_top_n": POOL_TOP_N or None, "rerank_device": DEVICE, "profile": "P30"},
        "completed": len(latencies) + failures,
        "success": len(latencies),
        "failures": failures,
        "failure_rate": round(failures / max(1, len(latencies) + failures), 4),
        "qps": round(len(latencies) / DURATION, 4),
        "latency_ms": {
            "mean": round(statistics.fmean(ok), 1) if ok else None,
            "p50": pct(ok, 0.50) if ok else None,
            "p95": pct(ok, 0.95) if ok else None,
            "p99": pct(ok, 0.99) if ok else None,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)

    from modules.database.session import get_async_engine

    await get_async_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
