# =============================================================================
# 教学说明：本文件在整体链路中的位置
# -----------------------------------------------------------------------------
# 输入：`.env` 的 REDIS_URL；业务侧传入的缓存 key 或「用户+问题」拼成的 scope 字符串。
# 输出：`RedisCache` 实例（get/set JSON）；`cache_key_for_query` 返回稳定短 key。
# 被谁调用：`RagPipeline`（读缓存命中直出、写 FAQ/RAG 结果）、其它需复用 Redis 的模块通过 `get_redis()`。
# =============================================================================
"""
使用 `redis.asyncio`：所有方法都是 async def，可在 FastAPI 路由里 await。

缓存值存 JSON 字符串，便于存 dict（含 answer、route 等字段）。

业务层降级（Task 19 B1）：get_json/set_json 的全部 Redis IO 均以
`redis.RedisError + OSError` 兜底——读失败视为缓存 miss（返回 None），
写失败静默跳过并打 warning。主链路任何情况下不因 Redis 故障抛 500。
"""

from __future__ import annotations

import json  # 把 Python 对象序列化为 str 存入 Redis
import logging  # JSON 损坏/Redis 降级时打 warning
from functools import lru_cache  # Redis 客户端单例
from typing import Any  # get_json 返回值可能是 dict/list/None

import redis.asyncio as redis  # 官方异步客户端，与 asyncio 协作
from redis.asyncio.retry import Retry  # redis-py 原生重试器（检出死连接后丢弃重建再重试）
from redis.backoff import ExponentialBackoff  # 重试间隔指数退避，避免打爆服务端

from modules.core.config import get_settings  # 取 REDIS_URL

logger = logging.getLogger(__name__)


class RedisCache:
    """
    封装 Redis 的 GET/SET 操作，自动处理 JSON 编解码
    薄封装：固定用 JSON 编解码，调用方不用自己 dumps/loads。
    """

    def __init__(self, client: redis.Redis) -> None:
        """
        入参:
            client: 已创建的异步 Redis 客户端（建议 decode_responses=True）。
        返回:
            无。
        """
        self._r = client  # 保存底层客户端引用；decode_responses=True 时 value 已是 str

    async def get_json(self, key: str) -> Any | None:
        """
        从缓存读数据，如果缓存不存在，则返回 None
        await GET key；不存在返回 None；JSON 非法返回 None 并打日志。

        降级语义（B1）：Redis 不可达/超时等 IO 异常 → warning + 返回 None（视为 miss），
        调用方继续走完整 RAG 链路，主请求不受影响。

        入参:
            key: Redis 键名。
        返回:
            反序列化后的 Python 对象（通常为 dict）；键不存在、JSON 损坏或 Redis 故障时返回 None。
        """
        try:  # B1：包住全部 Redis 读 IO
            raw = await self._r.get(key)  # 异步 IO：等待 Redis 响应
        except (redis.RedisError, OSError) as exc:  # 连接断开/超时/DNS 等，不区分一律降级
            logger.warning("redis get degraded (key=%s): %s", key, exc)  # 运维可见，但不打断主链路
            return None  # 视为缓存 miss
        if raw is None:  # 键不存在
            return None  # 表示未命中缓存
        try:  # 尝试解析
            return json.loads(raw)  # 解析成功，str → Python 对象（通常是 dict）
        except json.JSONDecodeError:  # 值被手工改坏或版本不兼容
            logger.warning("cache corrupt for key=%s", key)  # 便于排查
            return None  # 当作未命中，避免抛异常打断主链路,直接走rag逻辑

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        """
        往缓存存数据，并设置过期时间
        SET key value EX ttl；中文用 ensure_ascii=False 保持可读，不转义，避免中文乱码。

        降级语义（B1）：Redis 写失败 → warning + 静默跳过（本条结果不缓存，
        下次同问题重新生成，正确性不受影响）。

        入参:
            key: Redis 键名。
            value: 可 JSON 序列化的 Python 对象。
            ttl_seconds: 过期时间（秒），传给 Redis EX。
        返回:
            无。
        """
        payload = json.dumps(value, ensure_ascii=False)  # 序列化不属于 IO，失败不该被吞
        try:  # B1：只包 Redis 写 IO
            await self._r.set(key, payload, ex=ttl_seconds)  # ex= 过期秒数
        except (redis.RedisError, OSError) as exc:  # 连接断开/超时等
            logger.warning("redis set degraded (key=%s): %s", key, exc)  # 静默跳过本条写入


@lru_cache
def _build_client() -> redis.Redis:
    """
    从 URL 解析出连接参数并创建连接池客户端；进程内只执行一次。

    入参:
        无；连接串取自 `get_settings().redis_url`。
    返回:
        异步 Redis 客户端单例。
    """
    settings = get_settings()  # 读配置
    #从 URL 解析出连接参数并创建连接池客户端；进程内只执行一次，创建 Redis 连接池和客户端
    return redis.from_url(
        settings.redis_url,
        decode_responses=True,  # True：bytes 自动 decode 成 str，JSON 友好
        health_check_interval=25,  # 空闲连接使用前 PING 自检，检出 NAT 断开的死连接
        socket_keepalive=True,  # TCP 层 keepalive
        retry=Retry(ExponentialBackoff(), 3),  # 死连接/超时后丢弃重建并指数退避重试 3 次
        retry_on_error=[ConnectionError, TimeoutError],  # 仅对连接类错误重试
    )


def get_redis() -> redis.Redis:
    """
    对外暴露单例：多处 `RedisCache(get_redis())` 共享同一连接池。所有需要 Redis 客户端的地方，都通过这个函数获取，共享连接池

    入参:
        无。
    返回:
        与 `_build_client()` 相同的缓存客户端实例。
    """
    return _build_client()  # 返回缓存的客户端实例


def cache_key_for_query(q: str) -> str:
    """
    把任意长度问题映射为固定前缀 + 数字哈希，避免 key 过长。

    管线传入的 `q` 实为 `user_external_id:question`，使不同用户同问题不共用一个缓存桶。

    入参:
        q: 经 `strip()` 前后可能变化的原始 scope 字符串（用户 id 与问题拼接）。
    返回:
        形如 `xiaoyi:rag:qa:<hash>` 的稳定短键字符串。
    """
    return f"xiaoyi:rag:qa:{hash(q.strip())}"  # Python 内置 hash（进程生命周期内稳定；注意多进程不共享）


"""
具体用法举例：
# 1. 创建缓存实例（全项目共享一个客户端）
cache = RedisCache(get_redis())

# 2. 生成缓存键（用户ID+问题）
user_id = "user123"
question = "什么是Python"
scope = f"{user_id}:{question}"  # 拼接成 "user123:什么是Python"
cache_key = cache_key_for_query(scope)  # 生成短键：xiaoyi:rag:qa:<hash>

# 3. 先查缓存
cached_result = await cache.get_json(cache_key)
if cached_result:
    # 缓存命中：直接返回结果给用户
    return {"answer": cached_result["answer"], "source": "cache"}

# 4. 缓存未命中：走RAG流程生成回答
rag_answer = await rag_pipeline.run(question)  # 假设这是流式生成回答的逻辑

# 5. 把结果存入缓存（过期时间1小时）
await cache.set_json(cache_key, {"answer": rag_answer, "route": "rag"}, ttl_seconds=3600)

# 6. 返回结果
return {"answer": rag_answer, "source": "rag"}
"""
