# =============================================================================
# 教学说明：本文件在整体链路中的位置
# -----------------------------------------------------------------------------
# 输入：命令行无参数；依赖项目根下 `data/法律问答对.xlsx` 与 PDF；依赖 `.env` 中 MySQL。
# 输出：控制台日志 + 进程退出码；数据库内 `faq_tab`/`legal_tab` 被写入或清空重导。
# 被谁调用：学员/运维在仓库根执行 `python offline/scripts/run_mysql_ingest.py`（非 import）。
# 说明：本文件为「薄封装」——真正逻辑在 `modules.ingestion.mysql_loaders.run_default_file_ingest`。
# =============================================================================
"""
仅跑 MySQL 入库：不把向量写入 Milvus。

运行前须 `cd` 到仓库根或任意目录均可，脚本会把根目录插入 `sys.path`。
"""

from __future__ import annotations

import asyncio  # asyncio.run 启动事件循环
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # 本文件 offline/scripts/xxx.py → parents[2] = 仓库根
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # 让 Python 能找到顶层包 `modules`、`backend`

from modules.ingestion.mysql_loaders import run_default_file_ingest  # noqa: E402 — 须在 path 修正后导入

logging.basicConfig(level=logging.INFO)  # 配置根日志：子 logger 默认 INFO 也会输出
logger = logging.getLogger("mysql_ingest")  # 命名 logger，便于过滤日志


async def _main() -> None:
    """
    单入口协程：只调用一处业务函数。

    入参:
        无；依赖仓库根 `data/` 与 `.env` 中 MySQL 配置。
    返回:
        无；向日志打印含 `faq_rows`、法律导入计数的统计 dict。
    """
    stats = await run_default_file_ingest()  # 内部：create_all + FAQ + legal
    logger.info("done: %s", stats)  # 打印 dict，含 faq_rows、files、parents、children


if __name__ == "__main__":
    asyncio.run(_main())  # Python 3.7+：创建 loop、跑 _main、关闭 loop
