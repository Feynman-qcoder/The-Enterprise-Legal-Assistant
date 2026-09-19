# =============================================================================
# -----------------------------------------------------------------------------
# 输入：无（`get_pipeline` 无参）；依赖进程内首次调用时的构造副作用（加载大模型）。
# 输出：单例 `RagPipeline` 实例，供 FastAPI `Depends` 注入到路由处理函数参数。
# 被谁调用：`backend/app/api/chat.py` 中 `pipeline: RagPipeline = Depends(get_pipeline)`。
# =============================================================================
"""
FastAPI 依赖注入：把「重对象」与「请求处理函数参数」解耦。

`lru_cache` 保证全进程一个 `RagPipeline`，避免每请求 new 一次导致显存爆炸。

Task 21 Part 2 新增 `enforce_rate_limit`：chat 路由的限流 dependency
（user_external_id 优先、IP 回落；Redis 故障 fail-open 放行）。
"""

from __future__ import annotations

import logging
from functools import lru_cache  # 标准库单例装饰器

from fastapi import HTTPException, Request, status  # 限流 dependency 需要

from modules.cache.rate_limit import check_rate_limit  # 滑动窗口限流器
from modules.core.config import get_settings  # 限流开关与阈值
from modules.rag.pipeline import RagPipeline  # 管线定义在 modules 层，backend 只组装

logger = logging.getLogger(__name__)


@lru_cache  # 无括号等价 maxsize=None：缓存任意多次调用中「唯一」无参调用结果
def get_pipeline() -> RagPipeline:
    """
    FastAPI 解析 Depends(get_pipeline) 时：第一次请求调用本函数，之后直接返回缓存实例。

    入参:
        无。
    返回:
        进程内单例 `RagPipeline`。
    """
    return RagPipeline()  # 构造：内部会 new LocalEmbeddingService 等


async def _resolve_rate_limit_identity(request: Request) -> str:
    """
    限流身份归一化：请求体 user_external_id 优先，缺失回落客户端 IP。

    入参:
        request: 当前请求对象（读 body 与客户端地址）。
    返回:
        形如 "user:<id>" 或 "ip:<host>" 的身份串。
    """
    identity: str | None = None
    try:
        body = await request.json()  # JSON body 会被 Starlette 缓存，路由处理函数可正常重读
        if isinstance(body, dict):
            identity = body.get("user_external_id")
    except Exception:  # noqa: BLE001 — body 非法/为空时回落 IP（schema 校验随后自行 422）
        identity = None
    if isinstance(identity, str) and identity.strip():
        return f"user:{identity.strip()}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


async def enforce_rate_limit(request: Request) -> None:
    """
    chat 路由限流 dependency：超限抛 429（带 Retry-After 头），正常/降级放行。

    挂载点：`@router.post("/stream", dependencies=[Depends(enforce_rate_limit)])`。
    /health 不挂本 dependency，天然豁免。

    入参:
        request: 当前请求对象。
    返回:
        无（通过即放行；超限抛 HTTPException 429）。
    """
    if not get_settings().rate_limit_enabled:
        return
    identity = await _resolve_rate_limit_identity(request)
    allowed, retry_after = await check_rate_limit(identity)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试。",
            headers={"Retry-After": str(retry_after)},
        )
