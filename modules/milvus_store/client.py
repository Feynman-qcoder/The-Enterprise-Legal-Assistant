# =============================================================================
# 这个文件负责 Milvus 连接。  
# 它解决“谁来 connect、何时 connect、是否重复 connect”的问题。
# “后面你看到任何模块拿 Milvus，都是 `ensure_milvus()`，这就是统一连接入口。”

# -----------------------------------------------------------------------------
# 输入：`.env` 中的 MILVUS_URI（非空时优先，支持 Milvus Lite 本地文件）；
#      为空则回退 MILVUS_HOST / PORT / USER / PASSWORD（经 `get_settings()`）。
# 输出：副作用——在 pymilvus 全局注册名为 `default` 的连接；函数返回别名字符串。
# 被谁调用：`collections.py` 的 `ensure_milvus()`、`pipeline._milvus_search` 闭包、
#          `milvus_sync._blocking_milvus_full_write`、`lifespan._ensure_milvus`。
# =============================================================================
"""
PyMilvus 使用全局 `connections.connect`：同一进程内应用只连一次即可，后续 `Collection` 默认用该连接。
"""

from __future__ import annotations

import logging  # 连接成功后打一条 INFO 日志

from functools import lru_cache  # `ensure_milvus` 多次调用只真正 connect 一次

from pymilvus import connections  # Milvus 官方 Python SDK 的连接入口

from modules.core.config import get_settings  # 读主机端口账号

logger = logging.getLogger(__name__)  # 模块级 logger，日志名一般为 modules.milvus_store.client

_ALIAS = "default"  # 连接别名；后续 `Collection(name)` 未显式指定 alias 时即用 default


@lru_cache
def ensure_milvus() -> str:
    """
    若当前进程尚未 connect，则建立 TCP（或 gRPC）连接；已连接则立即返回。

    返回值为别名，便于单元测试或日志拼接（本项目中多数调用方忽略返回值）。

    入参:
        无；主机端口等来自 `get_settings()`。
    返回:
        连接别名字符串（常量为 `default`）。
    """
    s = get_settings()  # 配置单例

    if s.milvus_uri:  # 【新增】uri 模式：Milvus Lite 本地文件 / 完整 URI
        connections.connect(
            alias=_ALIAS,  # 注册到该名字下
            uri=s.milvus_uri,  # 如 "/root/autodl-tmp/milvus.db"
        )
        logger.info("milvus connected by uri: %s", s.milvus_uri)  # 运维可见连接目标
        return _ALIAS  # 把别名返回给调用方

    # ↓↓↓ 以下为原有 host/port 模式：MILVUS_URI 为空时行为与改造前逐字节一致 ↓↓↓
    addr = f"{s.milvus_host}:{s.milvus_port}"  # 人类可读地址串，仅用于日志
    connections.connect(
        alias=_ALIAS,  # 注册到该名字下
        host=s.milvus_host,  # 字符串主机名或 IP
        port=str(s.milvus_port),  # pymilvus 此处要求字符串端口
        user=s.milvus_user or None,  # 空串转 Python None，表示不传用户
        password=s.milvus_password or None,  # 空串同理
    )  # 同步阻塞调用；在线服务里应放在 `asyncio.to_thread` 中（见 lifespan / pipeline）
    logger.info("milvus connected: %s", addr)  # 运维可见连接目标
    return _ALIAS  # 把别名返回给调用方


def close_milvus() -> None:
    """
    主动断开 default 连接；单元测试 teardown 或进程退出前可选用。

    若从未 connect，`disconnect` 可能抛异常，故用 try/except 吞掉。

    入参:
        无。
    返回:
        无；失败时静默忽略。
    """
    try:  # 尝试断开
        connections.disconnect(_ALIAS)  # 释放与 Milvus 的 socket
    except Exception:  # noqa: BLE001 — 断开失败不向上抛，避免影响主流程退出
        pass  # 静默忽略
