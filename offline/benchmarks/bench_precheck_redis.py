# -*- coding: utf-8 -*-
"""Phase 0.3 — Pre-benchmark redis idle gate (60s group only)."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根（Task 22 迁移适配：原 D 盘绝对路径）

from modules.cache.redis_client import get_redis  # noqa: E402

KEY = "xiaoyi:diag:bench-precheck"


async def main() -> None:
    r = get_redis()
    await r.set(KEY, "warm")
    await asyncio.sleep(60)
    t0 = time.perf_counter()
    try:
        await r.set(KEY, "after-idle")
        v = await r.get(KEY)
        ok = v == "after-idle"
        print(f"IDLE60_SET_GET {'PASS' if ok else 'FAIL'} first_op={round((time.perf_counter()-t0)*1000,1)}ms")
    except Exception as exc:  # noqa: BLE001
        print(f"IDLE60_SET_GET FAIL {type(exc).__name__}: {exc}")
    await r.delete(KEY)
    try:
        await r.aclose()
    except Exception:  # noqa: BLE001
        pass


asyncio.run(main())
