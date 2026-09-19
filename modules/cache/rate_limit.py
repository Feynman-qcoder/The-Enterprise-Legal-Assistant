# =============================================================================
# Redis ZSET 滑动窗口限流器（Task 21 Part 2）。
# -----------------------------------------------------------------------------
# 输入：调用方身份字符串（user_external_id 优先，缺失回落客户端 IP）。
# 输出：(allowed: bool, retry_after_seconds: int)。
# 被谁调用：backend/app/deps.py 的 FastAPI dependency（挂载在 chat 路由上）。
# =============================================================================
"""
滑动窗口算法（pipeline 保证原子性）：
  ZREMRANGEBYSCORE  清除窗口外的旧记录
  ZADD              记录本次请求（member=时间戳，score=时间戳）
  ZCARD             统计窗口内请求数（含本次）
  EXPIRE            兜底 TTL，防孤儿 key

fail-open 哲学（红线）：Redis 任何异常 → 放行 + warning 日志，
不得因限流组件故障拒绝服务（与 B1 可用性优先一致）。

默认值依据：E2E Config C @VU5 单用户自然速率 ~7 req/min；
12/min 只拦截滥用突发，不误伤正常用户。
"""

from __future__ import annotations

import logging
import time

import redis.asyncio as redis

from modules.cache.redis_client import get_redis
from modules.core.config import get_settings

logger = logging.getLogger(__name__)

_RATE_LIMIT_KEY_PREFIX = "xiaoyi:ratelimit"


async def check_rate_limit(identity: str) -> tuple[bool, int]:
    """
    检查身份是否超出滑动窗口限额；本次请求会被计入窗口。

    入参:
        identity: 已归一化的身份串（如 "user:abc" / "ip:1.2.3.4"）。
    返回:
        (allowed, retry_after_seconds)：
        - allowed=True：放行（含 Redis 故障降级放行），retry_after=0
        - allowed=False：超限，retry_after 为建议等待秒数（>=1）
    """
    settings = get_settings()
    if not settings.rate_limit_enabled:  # 总开关关闭：直接放行
        return True, 0

    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_per_minute
    now = time.time()
    key = f"{_RATE_LIMIT_KEY_PREFIX}:{identity}"

    client = get_redis()
    try:
        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)  # 清窗口外旧记录
        pipe.zadd(key, {str(now): now})  # 记录本次（member 唯一化用时间戳字符串）
        pipe.zcard(key)  # 窗口内计数（含本次）
        pipe.expire(key, window)  # 兜底 TTL
        results = await pipe.execute()
        count = int(results[2])
    except (redis.RedisError, OSError) as exc:
        # fail-open（红线）：限流组件故障绝不拒绝服务
        logger.warning("rate limit degraded, fail-open (identity=%s): %s", identity, exc)
        return True, 0

    if count <= limit:
        return True, 0

    # 超限：计算 Retry-After（窗口剩余时间，按窗口内最旧请求推算）
    retry_after = window
    try:
        oldest = await client.zrange(key, 0, 0, withscores=True)
        if oldest:
            retry_after = max(1, int(window - (now - oldest[0][1])))
    except (redis.RedisError, OSError):
        pass  # 取不到最旧记录就用整窗兜底
    logger.warning(
        "rate limit exceeded: identity=%s count=%d limit=%d window=%ds",
        identity,
        count,
        limit,
        window,
    )
    return False, retry_after
