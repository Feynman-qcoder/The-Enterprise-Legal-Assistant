# =============================================================================
# -----------------------------------------------------------------------------
# Task 23 语义缓存阈值离线标定（证据驱动，复用现成离线资产）。
#
# 输入（均相对仓库根）：
# - offline/retrieval_eval_v1/offline_embedding_index/query_embeddings.npz
#   （F7 预核 0.36MB：vectors (100, 1024) float32 + query_ids (100,)）
# - offline/benchmarks/aligned_goldset_v2.jsonl（44 条 gold query 的 query_id 清单）
#
# 输出（stdout）：
# - 44 条 gold query 两两余弦相似度矩阵的非对角线 max + 分位数直方图
# - 推荐阈值 = min(max_offdiag + 0.03 安全余度, 0.97 上限)
# - 裁决：推荐值 <= 0.92 → 维持默认 0.92；否则上调 Settings 默认值并记录依据
#
# 运行：cd <仓库根> && python offline/scripts/calibrate_semantic_threshold.py
# =============================================================================

"""语义缓存命中阈值的离线标定：gold 问题集内最大两两相似度 + 安全余度。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # offline/scripts/ -> 仓库根
NPZ_PATH = ROOT / "offline" / "retrieval_eval_v1" / "offline_embedding_index" / "query_embeddings.npz"
GOLD_PATH = ROOT / "offline" / "benchmarks" / "aligned_goldset_v2.jsonl"

SAFETY_MARGIN = 0.03  # 推荐阈值 = 最强负样本相似度 + 安全余度
CAP = 0.97  # 上限：法律领域再相似的问题也可能措辞级等价，留出 0.03 语义空间
DEFAULT_THRESHOLD = 0.97  # modules/core/config.py 当前默认值（Task 23 标定后更新）
QUANTILES = (0.50, 0.90, 0.95, 0.99, 1.00)


def main() -> None:
    gold_ids = {
        json.loads(line)["query_id"]
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    data = np.load(NPZ_PATH, allow_pickle=True)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    ids = [str(x) for x in data["query_ids"]]

    picked = [i for i, qid in enumerate(ids) if qid in gold_ids]
    missing = gold_ids - {ids[i] for i in picked}
    if missing:
        raise SystemExit(f"gold query 缺失于 npz：{sorted(missing)}")

    matrix = vectors[picked]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / norms
    sims = unit @ unit.T  # (N, N) 余弦相似度矩阵

    n = len(picked)
    offdiag = sims[~np.eye(n, dtype=bool)]  # 展平全部非对角线元素（N*(N-1) 对）
    max_offdiag = float(offdiag.max())
    worst_pair_idx = np.unravel_index(np.argmax(np.where(np.eye(n), -1.0, sims)), sims.shape)
    worst_a, worst_b = ids[picked[worst_pair_idx[0]]], ids[picked[worst_pair_idx[1]]]

    recommended = min(round(max_offdiag + SAFETY_MARGIN, 4), CAP)
    final = DEFAULT_THRESHOLD if DEFAULT_THRESHOLD >= recommended else recommended

    print("=" * 62)
    print("Task 23 语义缓存阈值标定报告")
    print("=" * 62)
    print(f"gold query 数量：{n}（来源 {GOLD_PATH.name}）")
    print(f"embedding 资产：{NPZ_PATH.name}（{vectors.shape[1]} 维，共 {len(ids)} 条）")
    print(f"两两相似度对数：{offdiag.size}")
    print(f"max 非对角线相似度：{max_offdiag:.4f}（{worst_a} vs {worst_b}）")
    for q in QUANTILES:
        print(f"  P{int(q * 100):<3d} = {float(np.quantile(offdiag, q)):.4f}")
    print(f"推荐阈值（max+{SAFETY_MARGIN}，上限 {CAP}）：{recommended:.4f}")
    print(f"最终采用阈值：{final:.4f}")
    if final == DEFAULT_THRESHOLD:
        print(f"裁决：维持 Settings 默认 {DEFAULT_THRESHOLD}（默认值已覆盖推荐值，无需改动）。")
    else:
        print(f"裁决：需上调 modules/core/config.py 默认值至 {final}（依据见上方 max_offdiag）。")
    print("=" * 62)


if __name__ == "__main__":
    main()
