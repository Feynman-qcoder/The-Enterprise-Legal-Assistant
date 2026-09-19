# =============================================================================
# -----------------------------------------------------------------------------
# 语义缓存（Task 23 两级缓存 L2）单元测试：T1-T6。
#
# 覆盖维度：
# T1 向量编解码 roundtrip（b64 float32 → 还原 → 余弦=1.0）
# T2 store 后 lookup 命中（同向量，route 透传）
# T3 相似度低于阈值 → miss（正交向量 sim=0）
# T4 超容量驱逐：max_entries=3 存 5 条 → HLEN=3 且按写入时间最旧两条被删
# T5 Redis 异常 → lookup 返回 None / store 不抛（B1 降级语义）
# T6 维度不匹配条目被跳过（防模型变更污染）
#
# 运行：pytest tests/unit/test_semantic_cache.py -q
# =============================================================================

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import redis.asyncio as redis

from modules.cache.semantic_cache import (
    SemanticCache,
    decode_vector,
    encode_vector,
)
from modules.core.config import Settings

DIM = 64  # 测试用小维度即可（语义缓存逻辑与维度无关，真实为 1024）
KEY = "xiaoyi:rag:sem:legacy_v1:v0"  # contract=legacy_v1 + version=0 的默认 key


class FakeRedis:
    """内存 dict 实现的异步 Redis 替身：只实现 SemanticCache 用到的 5 个 Hash 命令。"""

    def __init__(self, fail: bool = False) -> None:
        self.data: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail  # True：所有命令抛 RedisError（T5 用）

    async def hset(self, key: str, field: str, value: str) -> None:
        if self.fail:
            raise redis.RedisError("hset boom")
        self.data.setdefault(key, {})[field] = value

    async def hgetall(self, key: str) -> dict[str, str]:
        if self.fail:
            raise redis.RedisError("hgetall boom")
        return dict(self.data.get(key, {}))

    async def expire(self, key: str, seconds: int) -> None:
        if self.fail:
            raise redis.RedisError("expire boom")
        self.ttls[key] = seconds

    async def hlen(self, key: str) -> int:
        if self.fail:
            raise redis.RedisError("hlen boom")
        return len(self.data.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> int:
        if self.fail:
            raise redis.RedisError("hdel boom")
        bucket = self.data.get(key, {})
        removed = 0
        for f in fields:
            if bucket.pop(f, None) is not None:
                removed += 1
        return removed


class FakeTime:
    """store 内 time.time() 替身：每次调用自增 1s，使各条目 t 严格递增（T4 驱逐排序用）。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def time(self) -> float:
        self.now += 1.0
        return self.now


def make_settings(**kwargs: object) -> Settings:
    """构造隔离于 .env / 环境变量的 Settings（用 alias 传参覆盖语义缓存三项）。"""
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


def unit_vec(seed: int) -> list[float]:
    """确定性伪随机单位向量：同 seed 同向量，不同 seed 近似正交（维度 64 时内积≈0）。"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


# ---------------------------------------------------------------- T1 编解码
def test_t1_vector_codec_roundtrip() -> None:
    v = unit_vec(7)
    encoded = encode_vector(v)
    assert isinstance(encoded, str)
    decoded = decode_vector(encoded)
    assert decoded.shape == (DIM,)
    assert decoded.dtype == np.float32
    assert cosine(list(map(float, decoded)), v) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- T2 命中
async def test_t2_store_then_lookup_hit() -> None:
    settings = make_settings()
    fake = FakeRedis()
    cache = SemanticCache(fake, settings)
    qv = unit_vec(1)

    await cache.store("劳动合同解除的条件是什么？", qv, "答案甲", "rag_llm", 0)

    hit = await cache.lookup(qv, 0)
    assert hit is not None
    answer, sim, route = hit
    assert answer == "答案甲"
    assert route == "rag_llm"
    assert sim == pytest.approx(1.0, abs=1e-5)
    # TTL 刷新副作用：写 store 后整表 key 应带 TTL
    assert fake.ttls.get(KEY) == settings.cache_ttl_seconds


# ---------------------------------------------------------------- T3 低于阈值
async def test_t3_below_threshold_miss() -> None:
    settings = make_settings(SEMANTIC_CACHE_THRESHOLD=0.92)
    fake = FakeRedis()
    cache = SemanticCache(fake, settings)

    await cache.store("问题一", unit_vec(1), "答案一", "rag_llm", 0)

    # 不同 seed 向量近似正交：sim≈0 << 0.92 → 必须 miss
    hit = await cache.lookup(unit_vec(2), 0)
    assert hit is None


# ---------------------------------------------------------------- T4 容量驱逐
async def test_t4_eviction_beyond_max_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(SEMANTIC_CACHE_MAX_ENTRIES=3)
    fake = FakeRedis()
    cache = SemanticCache(fake, settings)
    monkeypatch.setattr("modules.cache.semantic_cache.time", FakeTime())  # 各条目 t 严格递增

    for i in range(5):  # 依次写入 q0..q4（t 递增 → q0 最旧）
        await cache.store(f"question-{i}", unit_vec(100 + i), f"answer-{i}", "rag_llm", 0)

    assert await fake.hlen(KEY) == 3  # 容量收敛到 max_entries
    fields = set((await fake.hgetall(KEY)).keys())
    assert SemanticCache._field("question-0") not in fields  # 最旧两条被驱逐
    assert SemanticCache._field("question-1") not in fields
    assert SemanticCache._field("question-2") in fields  # 最近三条保留
    assert SemanticCache._field("question-3") in fields
    assert SemanticCache._field("question-4") in fields

    # 幸存条目仍可正常命中
    hit = await cache.lookup(unit_vec(104), 0)
    assert hit is not None and hit[0] == "answer-4"


# ---------------------------------------------------------------- T5 B1 降级
async def test_t5_redis_failure_degrades_to_miss_and_silent_write() -> None:
    settings = make_settings()
    fake = FakeRedis(fail=True)  # 所有命令抛 RedisError
    cache = SemanticCache(fake, settings)

    assert await cache.lookup(unit_vec(1), 0) is None  # 读失败 = miss
    await cache.store("问题", unit_vec(1), "答案", "rag_llm", 0)  # 写失败静默，不抛异常
    assert fake.data == {}  # 确认没有任何写入落地


# ---------------------------------------------------------------- T6 维度防护
async def test_t6_dimension_mismatch_entry_skipped() -> None:
    settings = make_settings()
    fake = FakeRedis()
    cache = SemanticCache(fake, settings)

    # 预置一条 32 维旧条目（模拟换模型后的残留数据）
    rng = np.random.default_rng(9)
    stale = (rng.standard_normal(32) / np.sqrt(32)).tolist()
    fake.data[KEY] = {
        SemanticCache._field("旧模型问题"): json.dumps(
            {"q": "旧模型问题", "v": encode_vector(stale), "a": "旧答案", "r": "rag_llm", "t": 1},
            ensure_ascii=False,
        )
    }

    # 64 维查询：维度不一致 → 条目被跳过 → miss（而不是崩溃或错配）
    assert await cache.lookup(unit_vec(1), 0) is None


# ---------------------------------------------------------------- 配置校验（附）
def test_settings_validators_reject_illegal_values() -> None:
    with pytest.raises(ValueError):  # 阈值必须 ∈ (0, 1]
        make_settings(SEMANTIC_CACHE_THRESHOLD=0.0)
    with pytest.raises(ValueError):
        make_settings(SEMANTIC_CACHE_THRESHOLD=1.5)
    with pytest.raises(ValueError):  # 容量必须 >= 1
        make_settings(SEMANTIC_CACHE_MAX_ENTRIES=0)
    ok = make_settings(SEMANTIC_CACHE_THRESHOLD=0.95, SEMANTIC_CACHE_MAX_ENTRIES=1)
    assert ok.semantic_cache_threshold == 0.95
    assert ok.semantic_cache_max_entries == 1
    assert ok.semantic_cache_enabled is False  # 默认关闭（转正前红线）


# ---------------------------------------------------------------- 数学辅助自检
def test_orthogonal_helper_vectors_are_actually_far() -> None:
    # 保护 T3 的前提：不同 seed 的辅助向量真实相似度必须远低于默认阈值
    assert cosine(unit_vec(1), unit_vec(2)) < 0.5
    assert math.isfinite(cosine(unit_vec(1), unit_vec(2)))
