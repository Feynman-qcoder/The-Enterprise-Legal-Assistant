# -*- coding: utf-8 -*-
"""Phase 3b add-on — Fast-path QPS: FAQ direct + cache hit (no LLM, no reranker).

Run against live backend at 20 concurrency for 30s each path.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
FAQ_Q = "转让不动产需查验什么？"
CACHE_USER = "bench-fastpath-cache-user"
CACHE_Q = "处理敏感个人信息应当取得个人什么样的同意？"
VU = 20
DURATION = 30.0
import argparse  # noqa: E402 — Task 22 F8 参数化

_ap = argparse.ArgumentParser(description="Fast-path QPS benchmark")
_ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "bench_fastpath.json"))
OUT = pathlib.Path(_ap.parse_args().out)


async def one(client, message, user) -> tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{BASE}/api/chat/stream",
            json={"message": message, "user_external_id": user}, timeout=60.0,
        ) as resp:
            if resp.status_code != 200:
                return False, time.perf_counter() - t0
            ok_done = False
            async for piece in resp.aiter_text():
                if "[DONE]" in piece:
                    ok_done = True
                    break
            return ok_done, time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        return False, time.perf_counter() - t0


async def run_path(tag: str, message: str, user_fn) -> dict:
    done = 0
    fail = 0
    lat: list[float] = []
    lock = asyncio.Lock()
    stop_at = time.perf_counter() + DURATION

    async def worker(wid: int) -> None:
        nonlocal done, fail
        i = 0
        while time.perf_counter() < stop_at:
            async with httpx.AsyncClient() as client:
                ok, dt = await one(client, message, user_fn(wid, i))
            async with lock:
                if ok:
                    done += 1
                    lat.append(dt)
                else:
                    fail += 1
            i += 1

    await asyncio.gather(*[worker(w) for w in range(VU)])
    return {
        "path": tag, "vu": VU, "duration_s": DURATION,
        "success": done, "fail": fail,
        "qps": round(done / DURATION, 2),
        "latency_ms_mean": round(sum(lat) / max(1, len(lat)) * 1000, 1),
    }


async def main() -> None:
    # 预热缓存
    async with httpx.AsyncClient() as client:
        ok, _ = await one(client, CACHE_Q, CACHE_USER)
        assert ok
    print("warm cache filled", flush=True)

    faq = await run_path("faq_direct", FAQ_Q, lambda w, i: f"fastpath-faq-{w}-{i}")
    print(json.dumps(faq, ensure_ascii=False), flush=True)
    cache = await run_path("cache_hit", CACHE_Q, lambda w, i: CACHE_USER)
    print(json.dumps(cache, ensure_ascii=False), flush=True)

    OUT.write_text(json.dumps({"faq_direct": faq, "cache_hit": cache}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OUT {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
