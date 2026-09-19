# =============================================================================
# 教学说明：本文件在整体链路中的位置
# -----------------------------------------------------------------------------
# 输入：命令行无参数；要求 MySQL 已有数据；`.env` 中 Milvus 与 Embedding 路径可用。
# 输出：Milvus 集合重建并插入向量；控制台打印统计 dict。
# 被谁调用：人工执行 `python offline/scripts/run_milvus_sync.py`；逻辑等同热更新里的 `run_sync_job`。
# =============================================================================
"""
不跑 Excel/PDF 入库，只把当前 MySQL 中的高频 FAQ 与法律子块同步到 Milvus。

适合：已单独导完库，仅需重建向量索引的场景。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.ingestion.milvus_sync import run_sync_job  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_sync")


async def _main() -> None:
    """
    单入口协程：仅触发 Milvus 全量同步任务。

    入参:
        无。
    返回:
        无；通过日志输出 `run_sync_job` 返回的统计 dict。
    """
    stats = await run_sync_job()  # recreate=True 全量删表重建（见 milvus_sync 实现）
    logger.info("result: %s", stats)


if __name__ == "__main__":
    asyncio.run(_main())
