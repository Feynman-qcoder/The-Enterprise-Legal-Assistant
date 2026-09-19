# =============================================================================
# -----------------------------------------------------------------------------
# 这是两级缓存的「第二级」：exact 缓存的向量泛化。
# exact 缓存回答「字面相同的问题」；语义缓存回答「意思相同的问题」。
#
# 输入：查询向量（list[float]，BGE-M3 1024 维）+ 语料版本号。
# 输出：lookup 返回 (answer, sim, route) 三元组或 None；store 无返回值。
# 被谁调用：RagPipeline（exact miss 后 embed 提前、语义命中直出回填 exact；
#           全链路完整成功后写回语义缓存，兜底/中断路径禁写）。
#
# 设计决策（Task 23）：
# - 存储结构：Redis Hash，key = xiaoyi:rag:sem:{contract}:v{version}，
#   field = sha1(question.strip())[:16]（禁止内置 hash()：进程盐化，多 worker 不一致）。
#   version 段进入 key 复用 A6 epoch 失效：corpus_version+1 后旧条目整体自然 miss。
# - 向量编码：float32 字节 + base64（1024 维约 5.5KB）；禁止 JSON 数组（约 11KB，体积翻倍）。
# - 相似度计算：HGETALL 全量 → 逐条解析 → numpy 行归一化 → 矩阵 @ 查询 → argmax
#   （条目 <1000，暴力打分足够；FAISS/向量索引明确不引入）。
# - 法律近义陷阱：解除≠终止劳动合同 / 第二章≠第二节，默认阈值 0.97（离线标定：
#   gold 集两两 max_offdiag=0.9868 + 0.03 余度被 cap 至 0.97；只读 Settings 不散落魔法数字）；
#   命中必须留审计日志（相似度分数 + 原 route）。
# - B1 降级：Redis 故障 = 读视为 miss / 写静默跳过，主链路任何情况不抛 500。
# - 维度防护：维度 != len(qvec) 的条目直接跳过（防换模型后旧向量污染打分）。
# =============================================================================

"""语义缓存（两级缓存 L2）：向量相似度命中直出，B1 降级 + 容量驱逐 + 维度防护。"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from typing import Any

import numpy as np  # 矩阵化相似度打分（向量编解码 + argmax）
import redis.asyncio as redis  # 仅用其 RedisError 异常类型做 B1 兜底

from modules.core.config import Settings  # 阈值 / 容量 / TTL 全部走统一配置

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_PREFIX = "xiaoyi:rag:sem"  # 与 exact 层 xiaoyi:rag:qa 前缀区分


def encode_vector(v: list[float]) -> str:
    """float32 字节 → base64 ASCII 字符串（体积比 JSON 数组减半）。"""
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode("ascii")


def decode_vector(s: str) -> np.ndarray:
    """base64 字符串 → float32 一维 numpy 数组；损坏输入抛 ValueError 由调用方跳过。"""
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


class SemanticCache:
    """Redis Hash 实现的语义缓存：查询向量与已缓存问题向量做余弦相似度匹配。"""

    def __init__(self, client: redis.Redis, settings: Settings) -> None:
        """
        入参:
            client: 异步 Redis 客户端（decode_responses=True，HGETALL 返回 str dict）。
            settings: 全局配置（读 threshold / max_entries / cache_ttl_seconds / contract）。
        返回:
            无。
        """
        self._r = client
        self._settings = settings

    def _key(self, version: int) -> str:
        """语料版本进入 key：复用 A6 epoch 失效，翻面后旧条目整体不可见。"""
        return f"{SEMANTIC_CACHE_PREFIX}:{self._settings.retrieval_contract}:v{version}"

    @staticmethod
    def _field(question: str) -> str:
        """sha1 而非内置 hash()：跨进程稳定，多 worker 部署下 field 一致。"""
        return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:16]

    async def lookup(
        self, qvec: list[float], version: int
    ) -> tuple[str, float, str] | None:
        """
        在语义缓存中寻找与查询向量最相似的已缓存条目。

        入参:
            qvec: 当前问题的 embedding（list[float]）。
            version: 当前语料版本（进入 Redis key）。
        返回:
            (answer, sim, route) 最优条目且 sim >= 阈值；否则 None（含 Redis 故障 B1 降级）。
        """
        try:  # B1：读失败 = miss
            entries: dict[Any, Any] = await self._r.hgetall(self._key(version))
        except (redis.RedisError, OSError) as exc:
            logger.warning("semantic cache lookup degraded: %s", exc)
            return None
        if not entries:
            return None

        q = np.asarray(qvec, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)  # 零向量兜底避免除零 NaN
        q_len = q.shape[0]

        rows: list[np.ndarray] = []
        metas: list[tuple[str, str]] = []  # 与 rows 对齐的 (answer, route)
        for raw in entries.values():
            try:
                item = json.loads(raw)
                vec = decode_vector(item["v"])
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                continue  # 条目损坏：跳过，不影响其余条目打分
            if vec.ndim != 1 or vec.shape[0] != q_len:
                continue  # 维度防护：模型变更后的旧条目不参与打分
            answer = str(item.get("a", ""))
            if not answer:
                continue  # 空答案条目无意义
            rows.append(vec)
            metas.append((answer, str(item.get("r", "unknown"))))
        if not rows:
            return None

        matrix = np.vstack(rows)  # (N, d)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0  # 零向量兜底
        sims = (matrix / norms[:, None]) @ q_norm  # 行归一化后矩阵乘 = 余弦相似度
        idx = int(np.argmax(sims))
        sim = float(sims[idx])

        if sim < self._settings.semantic_cache_threshold:
            return None  # 低于阈值一律 miss（法律近义陷阱防线）
        answer, route = metas[idx]
        return answer, sim, route

    async def store(
        self,
        question: str,
        qvec: list[float],
        answer: str,
        route: str,
        version: int,
    ) -> None:
        """
        写入一条语义缓存条目（仅「完整成功」的答案允许调用此方法）。

        入参:
            question: 原始问题文本（sha1 后作为 Hash field）。
            qvec: 问题向量（与 answer 同时落库，供后续 lookup 打分）。
            answer: 完整答案文本。
            route: 答案来源路由（faq_direct / rag_llm），命中审计时回显。
            version: 当前语料版本（进入 Redis key）。
        返回:
            无；Redis 写失败静默（B1）。
        """
        if not answer:
            return
        key = self._key(version)
        payload = json.dumps(
            {
                "q": question,
                "v": encode_vector(qvec),
                "a": answer,
                "r": route,
                "t": int(time.time()),  # 驱逐排序依据：unix 秒
            },
            ensure_ascii=False,
        )
        try:  # B1：写失败静默跳过
            await self._r.hset(key, self._field(question), payload)
            await self._r.expire(key, self._settings.cache_ttl_seconds)  # 每次写刷新整表 TTL
            length = await self._r.hlen(key)
        except (redis.RedisError, OSError) as exc:
            logger.warning("semantic cache store degraded: %s", exc)
            return

        # 容量驱逐：超出 max_entries 时按 t 从旧到新删除差值条数
        excess = length - self._settings.semantic_cache_max_entries
        if excess <= 0:
            return
        try:
            entries: dict[Any, Any] = await self._r.hgetall(key)
        except (redis.RedisError, OSError) as exc:
            logger.warning("semantic cache evict scan degraded: %s", exc)
            return
        stamped: list[tuple[int, str]] = []
        for field, raw in entries.items():
            try:
                stamped.append((int(json.loads(raw).get("t", 0)), str(field)))
            except (json.JSONDecodeError, ValueError, TypeError):
                stamped.append((0, str(field)))  # 损坏条目视作最旧，优先清除
        stamped.sort()
        for _, field in stamped[:excess]:
            try:
                await self._r.hdel(key, field)
            except (redis.RedisError, OSError) as exc:
                logger.warning("semantic cache evict hdel degraded: %s", exc)
                return
