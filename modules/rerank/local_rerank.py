 # =============================================================================
# -----------------------------------------------------------------------------
# 输入：用户问题字符串 `query`；父文档全文列表 `passages`（来自 MySQL 拉取）。
# 输出：与 passages 等长的浮点分数列表，分数越高表示该父文档越适合回答该问题。
# 被谁调用：`RagPipeline` 在法律混合检索分支末尾（懒加载 `_reranker()`）。
# =============================================================================
"""
CrossEncoder（如 bge-reranker-large）：对 (query, passage) 对逐对打分，比双塔向量「点积」更准但更慢。

`sentence_transformers.CrossEncoder.predict` 是同步的，必须用 `asyncio.to_thread` 避免卡死事件循环。

GPU 并发闸门（Task 19 B2）：RTX 3050 4GB 显存下 ≥3 路并发 rank 会触发 CUDA OOM，
故以进程级 `asyncio.Semaphore(2)` 限流——超并发请求排队等待，不拒绝不报错。
信号量在首次 rank 调用时于运行中的事件循环内懒创建（禁止模块顶层实例化，
避免绑定错误的 loop）。
"""

from __future__ import annotations

import asyncio  # to_thread
import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from sentence_transformers import CrossEncoder  # 官方 CrossEncoder 封装

from modules.core.config import get_settings

logger = logging.getLogger(__name__)

# B2：GPU 并发上限。RTX 3050 4GB 实测 ≥3 并发 OOM，2 为安全水位。
RERANK_MAX_CONCURRENCY = 2


@lru_cache
def _make_cross_encoder() -> CrossEncoder:
    """
    从本地目录加载 CrossEncoder 权重；设备与 embedding 共用 EMBEDDING_DEVICE。

    入参:
        无；路径与设备来自 `get_settings()`。
    返回:
        可用于 `predict` 的 `CrossEncoder` 实例。
    """
    settings = get_settings()
    root = Path(__file__).resolve().parents[2]  # 仓库根
    path = root / settings.rerank_model_path  # models/bge-reranker-large
    if not path.exists():
        logger.warning("Rerank 模型路径不存在：%s", path)
    device = settings.rerank_device  # Settings 化（Config C 转正 V2）：默认 cpu，.env 的 RERANK_DEVICE 可配
    cross_encoder = CrossEncoder(str(path), device=device)  # device 如 cpu、cuda
    if device.startswith("cuda"):
        cross_encoder.model.half()  # GPU FP16：显存减半、吞吐提升（质量守恒 gate 由 Benchmark 独立验证）
    return cross_encoder


class LocalRerankService:
    """
    仅暴露 async `rank`，内部转线程池。
    """

    def __init__(self) -> None:
        """
        构造重排服务并加载 CrossEncoder 模型。

        入参:
            无。
        返回:
            无。
        """
        self._model = _make_cross_encoder()  # 初始化即加载模型
        self._semaphore: asyncio.Semaphore | None = None  # B2：懒创建，首次 rank 时在运行中的 loop 构造

    def _gpu_semaphore(self) -> asyncio.Semaphore:
        """
        获取（必要时创建）进程级 GPU 并发信号量。

        入参:
            无。
        返回:
            已构造的 `asyncio.Semaphore` 实例（上限 RERANK_MAX_CONCURRENCY）。
        """
        if self._semaphore is None:  # 首次调用发生在已运行的事件循环内
            self._semaphore = asyncio.Semaphore(RERANK_MAX_CONCURRENCY)  # 懒创建，绑定当前 loop
            logger.info("rerank GPU semaphore initialized (max_concurrency=%d)", RERANK_MAX_CONCURRENCY)
        return self._semaphore

    async def rank(self, query: str, passages: Sequence[str]) -> list[float]:
        """
        返回每个 passage 的相关性分数；passages 为空则返回空列表。

        并发语义（B2）：超并发请求在信号量处排队等待，不拒绝不报错。

        入参:
            query: 用户问题。
            passages: 待打分的父文档全文等段落序列。
        返回:
            与 `passages` 等长的浮点分列表，分数越高越相关；空输入时返回 []。
        """
        if not passages:  # 无父文档可排
            return []  # 避免 predict 收到空输入
        pairs = [(query, p) for p in passages]  # CrossEncoder 输入：N 个二元组列表

        def _run() -> list[float]:
            """
            在线程池中执行的同步打分闭包。

            入参:
                无（使用外层 `pairs` 与 `self._model`）。
            返回:
                各 passage 的 Python float 分数列表。
            """
            scores = self._model.predict(list(pairs))  # numpy 或 tensor 转成的数组-like
            return [float(s) for s in scores]  # 统一为 Python float，便于 sorted / zip

        async with self._gpu_semaphore():  # B2：GPU 并发闸门，超并发排队等待
            return await asyncio.to_thread(_run)  # 在默认线程池执行 _run，释放事件循环
