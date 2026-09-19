# =============================================================================
# 这是 FastAPI 框架的「生命周期管理文件」，负责应用启动 / 关闭时的核心操作；
# 启动时要做 3 件事：检查 / 创建 MySQL 表、连接 Milvus（向量数据库）、可选启动「MySQL→Milvus 数据同步的后台任务」；
# 关闭时要优雅停止这个后台同步任务；
# 被谁调用：`main.py` 中 `FastAPI(..., lifespan=app_lifespan)`，由 ASGI 服务器在启动/关闭时驱动。
# =============================================================================
"""
启动顺序：MySQL 表结构 ensure → Milvus connect →（可选）后台循环全量同步 MySQL→Milvus。

热更新适合开发环境「改库后自动刷新向量」；生产需评估全量 recreate 成本。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from modules.core.config import get_settings
from modules.database.models import Base
from modules.database.session import get_async_engine
from modules.ingestion.milvus_sync import run_sync_job
from modules.milvus_store.client import ensure_milvus
from modules.rag.dashscope_http import close_dashscope_clients
from modules.rag.intent import clear_intent_llm_cache

logger = logging.getLogger(__name__)


async def _ensure_tables() -> None:
    """
    核心作用：创建数据库表
    `Base.metadata.create_all`：已存在的表不会删数据，只补缺失表（开发友好）。

    入参:
        无。
    返回:
        无。
    """
    engine = get_async_engine()
    async with engine.begin() as conn:  # begin 自动事务
        await conn.run_sync(Base.metadata.create_all)  # SQLAlchemy 同步 API 包到 run_sync


async def _ensure_milvus() -> None:
    """
    核心作用：确保 Milvus 连接
    `ensure_milvus` 为同步阻塞：用 to_thread 避免阻塞 asyncio 启动阶段其它协程。

    入参:
        无。
    返回:
        无。
    """
    await asyncio.to_thread(ensure_milvus)


async def _hot_update_loop(stop: asyncio.Event) -> None:
    """
    核心作用：定时全量同步 MySQL 到 Milvus

    入参:
        stop: 应用关闭时 set 的事件，用于打断循环。
    返回:
        无；内部异常会记录日志不向外抛。
    """
    settings = get_settings()
    interval = float(settings.effective_hot_update_interval_seconds)
    logger.info("milvus hot-update worker started, interval=%ss", interval)
    while not stop.is_set():  # 主循环直到 stop 被 set
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)  # 等 stop 或超时
        except asyncio.TimeoutError:
            pass  # 正常定时唤醒
        else:
            break  # stop.wait() 返回说明 Event 已触发，退出 while
        if not settings.hot_update_enabled:  # 如果热更新未启用，则跳过本次循环
            continue
        try:
            await run_sync_job()  # 异步全量同步（内部 to_thread 写 Milvus）
        except Exception as exc:  # noqa: BLE001 — 任意异常转成 JSON 事件给前端展示
            logger.exception("hot milvus sync failed: %s", exc)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    核心作用：启动时确保 MySQL 表结构和 Milvus 连接，并启动定时全量同步任务。
    关闭时取消定时全量同步任务。


    入参:
        app: FastAPI 应用实例（当前实现未直接使用，保留签名兼容框架）。
    返回:
        异步上下文管理器；`yield` 后进入请求处理阶段，`finally` 中取消热更新任务。
    """
    await _ensure_tables()
    await _ensure_milvus()
    settings = get_settings()
    stop = asyncio.Event()  # 创建一个事件对象，用于通知后台循环结束
    task: asyncio.Task | None = None # 创建一个异步任务，用于定时全量同步 MySQL 到 Milvus的后台任务
    if settings.hot_update_enabled: # 如果热更新启用，则启动定时全量同步任务
        task = asyncio.create_task(_hot_update_loop(stop))
    try:
        yield  # 此处放行：开始接受 HTTP 请求
    finally:
        stop.set()  # 通知后台循环结束
        if task:
            task.cancel()  # 向 Task 注入 CancelledError，强制取消任务
            try:
                await task  # 等待任务真正结束
            except asyncio.CancelledError: # 如果任务被取消，则忽略异常
                pass  # 取消是预期路径
        clear_intent_llm_cache()
        try:
            await close_dashscope_clients()
        finally:
            # SQLAlchemy 必须在当前 event loop 关闭前释放底层 aiomysql 连接池。
            await get_async_engine().dispose()

"""
yield: FastAPI 固定写法
yield 之前 = 启动 :安全开启后台循环任务（自动同步数据 / 定时刷新）
yield 之后 = 运行服务（接收请求）：后台任务和 HTTP 请求同时跑（互不干扰）
yield之后是 关闭 （在 finally 语义里）:优雅杀死后台任务（不崩、不残留、不卡死程序）。
"""
