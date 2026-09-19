# -*- coding: utf-8 -*-
"""Phase 3b — E2E SSE benchmark (with LLM) against the live backend.

Mix per spec: 70% new LQ (unique user each -> real RAG) / 20% repeat (fixed user,
warm question -> cache) / 10% FAQ (fast path).
Usage: python bench_e2e.py <tag> <vu> <duration_s> [--gold PATH] [--out PATH]
  Task 22 迁移：--gold/--out 默认相对本脚本目录（offline/benchmarks/），仓库内自洽
"""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import httpx

import argparse  # noqa: E402 — Task 22 F8 参数化

_parser = argparse.ArgumentParser(description="E2E SSE benchmark (with LLM)")
_parser.add_argument("tag")
_parser.add_argument("vu", type=int)
_parser.add_argument("duration_s", type=float)
_parser.add_argument("--gold", default=str(pathlib.Path(__file__).parent / "aligned_goldset_v2.jsonl"))
_parser.add_argument("--out", default=None)
_args = _parser.parse_args()
TAG = _args.tag
VU = _args.vu
DURATION = _args.duration_s
OUT = pathlib.Path(_args.out) if _args.out else pathlib.Path(__file__).parent / f"bench_e2e_{TAG.lower()}_vu{VU}.json"
BASE = "http://127.0.0.1:8000"

GOLD = [json.loads(l) for l in pathlib.Path(_args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
WARM_USER = f"bench-e2e-warm-{TAG}"
WARM_Q = "处理敏感个人信息应当取得个人什么样的同意？"  # LQ-014（较短链路，预热快）
FAQ_Q = "转让不动产需查验什么？"

random.seed(42)


def pct(values: list[float], q: float) -> float:
    if not values:
        return None  # type: ignore[return-value]
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 1)


async def one(client: httpx.AsyncClient, message: str, user: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    ttfb = None
    chunks = 0
    error = None
    done = False
    try:
        async with client.stream(
            "POST", f"{BASE}/api/chat/stream",
            json={"message": message, "user_external_id": user},
            timeout=300.0,
        ) as resp:
            if resp.status_code != 200:
                return {"kind": "http_error", "status": resp.status_code, "total_s": time.perf_counter() - t0}
            buf = ""
            async for piece in resp.aiter_text():
                buf += piece
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        done = True
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "chunk" in obj:
                        if ttfb is None:
                            ttfb = time.perf_counter() - t0
                        chunks += 1
                    if "error" in obj:
                        error = obj["error"]
    except Exception as exc:  # noqa: BLE001
        return {"kind": "exc", "error": str(exc)[:100], "total_s": time.perf_counter() - t0}
    return {
        "kind": "ok", "ttfb_s": ttfb, "total_s": time.perf_counter() - t0,
        "chunks": chunks, "done": done, "error": error,
    }


async def main() -> None:
    # 预热：填充缓存 + 模型
    async with httpx.AsyncClient() as client:
        w = await one(client, WARM_Q, WARM_USER)
        assert w.get("kind") == "ok", w
        print("WARM done", flush=True)
        f = await one(client, FAQ_Q, f"bench-faq-{TAG}")
        assert f.get("kind") == "ok", f
        print("FAQ check done", flush=True)

    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    stop_at = time.perf_counter() + DURATION
    lq_idx = 0

    async def worker(wid: int) -> None:
        nonlocal lq_idx
        rnd = random.Random(1000 + wid)
        while time.perf_counter() < stop_at:
            roll = rnd.random()
            async with httpx.AsyncClient() as client:
                if roll < 0.70:
                    async with lock:
                        g = GOLD[lq_idx % len(GOLD)]
                        lq_idx += 1
                    r = await one(client, g["query"], f"bench-{TAG}-u{time.time_ns()}")
                    r["mix"] = "new_lq"
                elif roll < 0.90:
                    r = await one(client, WARM_Q, WARM_USER)
                    r["mix"] = "repeat_cache"
                else:
                    r = await one(client, FAQ_Q, f"bench-faq-{TAG}-{time.time_ns()}")
                    r["mix"] = "faq"
            async with lock:
                results.append(r)

    print(f"RUN {TAG} vu={VU} {DURATION:.0f}s mix 70/20/10 ...", flush=True)
    await asyncio.gather(*[worker(w) for w in range(VU)])

    new_lq = [r for r in results if r.get("mix") == "new_lq" and r.get("kind") == "ok" and r.get("ttfb_s") is not None]
    rep = [r for r in results if r.get("mix") == "repeat_cache" and r.get("kind") == "ok" and r.get("ttfb_s") is not None]
    faq = [r for r in results if r.get("mix") == "faq" and r.get("kind") == "ok" and r.get("ttfb_s") is not None]
    failures = [r for r in results if r.get("kind") != "ok" or r.get("error") or not r.get("done") or r.get("ttfb_s") is None]
    # 原始结果保护：无论如何先落盘
    pathlib.Path(str(OUT).replace(".json", "_raw.json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # cache 命中判定：repeat 且 ttfb<1s
    cache_hits = [r for r in rep if (r.get("ttfb_s") or 99) < 1.0]

    summary = {
        "tag": TAG, "vu": VU, "duration_s": round(DURATION, 1),
        "total_requests": len(results),
        "mix_counts": {"new_lq": len(new_lq), "repeat_cache": len(rep), "faq": len(faq)},
        "failures": len(failures),
        "failure_rate": round(len(failures) / max(1, len(results)), 4),
        "new_lq": {
            "count": len(new_lq),
            "ttfb_s": {"p50": pct([r["ttfb_s"] for r in new_lq], 0.5), "p95": pct([r["ttfb_s"] for r in new_lq], 0.95), "mean": round(statistics.fmean([r["ttfb_s"] for r in new_lq]), 2) if new_lq else None},
            "total_s": {"p50": pct([r["total_s"] for r in new_lq], 0.5), "p95": pct([r["total_s"] for r in new_lq], 0.95)},
        },
        "repeat_cache": {
            "count": len(rep), "cache_hits": len(cache_hits),
            "hit_rate": round(len(cache_hits) / max(1, len(rep)), 4),
            "hit_ttfb_ms_mean": round(statistics.fmean([r["ttfb_s"] for r in cache_hits]) * 1000, 1) if cache_hits else None,
        },
        "faq": {
            "count": len(faq),
            "ttfb_ms_mean": round(statistics.fmean([r["ttfb_s"] for r in faq]) * 1000, 1) if faq else None,
            "qps": round(len(faq) / DURATION, 3),
        },
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
