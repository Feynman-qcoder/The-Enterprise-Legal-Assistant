# =============================================================================
# -----------------------------------------------------------------------------
# 输入：多路「已按相关性排序的文档 id 列表」`ranked_lists`；平滑系数 `k`（通常与配置 HYBRID_RRF_K 一致）。
# 输出：`(doc_id, 融合分数)` 元组列表，按分数降序；供后续取 Top 子块或父文档。
# 被谁调用：`modules/rag/pipeline.py` 法律分支：一路 Milvus 稠密序 + 一路 BM25 序（可选）。
# =============================================================================
"""
RRF（Reciprocal Rank Fusion）：不依赖各路的绝对分数，只利用「名次」融合多路排序。

公式：对每个文档累加 1/(k+rank)，rank 从 1 开始；同一文档在多路都靠前则累加更高。
"""

from __future__ import annotations

from collections import defaultdict  # 当 key 不存在时默认值为 float 累加器 0.0
from typing import Hashable  # doc_id 可能是 int（Milvus 主键）等可哈希类型


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hashable]],
    k: int = 60,
) -> list[tuple[Hashable, float]]:
    """
    `ranked_lists` 例如 `[[3,1,5],[5,3,7]]`：第一路认为 3 最相关，第二路认为 5 最相关。

    返回列表按融合分从高到低排序，便于 `fused[:30]` 截断。

    入参:
        ranked_lists: 多路检索结果，每路为按相关性从高到低排列的文档 id 列表。
        k: RRF 公式中的平滑常数（越大高名次权重衰减越慢）。
    返回:
        `(文档 id, 融合得分)` 元组列表，得分降序排列。
    """
    scores: dict[Hashable, float] = defaultdict(float)  # doc_id -> 当前 RRF 累积分
    for lst in ranked_lists:  # 遍历稠密路、BM25 路等
        for rank, doc_id in enumerate(lst, start=1):  # enumerate(..., 1)：第一名 rank=1
            scores[doc_id] += 1.0 / (k + rank)  # 该路该名次对 doc_id 的贡献
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 按分数降序；item 为 (doc, score)
