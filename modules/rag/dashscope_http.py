# =============================================================================
# -----------------------------------------------------------------------------
# 输入：无参数工厂；隐式使用全局无状态配置（超时等写死在代码里）。
# 输出：`httpx.Client` / `httpx.AsyncClient` 单例，供 LangChain `ChatOpenAI` 注入。
# 被谁调用：`modules/rag/intent.py`、`_intent_llm`；`modules/rag/pipeline.py` 的 `_llm()`。
# =============================================================================
"""
`trust_env=True`：httpx 读取系统环境变量里的代理设置；当前部署访问 Provider 依赖该网络路径。

`lru_cache(maxsize=1)`：全进程复用同一 TCP 连接池，减少 TLS 握手开销。
"""

from __future__ import annotations

from functools import lru_cache  # 缓存工厂函数返回值

import httpx  # 底层 HTTP 库，LangChain OpenAI 集成会调用其 request 接口


@lru_cache(maxsize=1)
def get_dashscope_sync_client() -> httpx.Client:
    """
    同步 Client：LangChain 部分代码路径仍可能阻塞式调用。

    入参:
        无。
    返回:
        配置了 `trust_env=True` 与超时 120s 的 `httpx.Client` 单例。
    """
    return httpx.Client(trust_env=True, timeout=120.0)  # 120s：生成可能较慢；代理策略见模块头说明


@lru_cache(maxsize=1)
def get_dashscope_async_client() -> httpx.AsyncClient:
    """
    异步 Client：`llm.astream` 等主路径使用。

    入参:
        无。
    返回:
        配置了 `trust_env=True` 与超时 120s 的 `httpx.AsyncClient` 单例。
    """
    return httpx.AsyncClient(trust_env=True, timeout=120.0)  # 与同步版超时一致


async def close_dashscope_clients() -> None:
    """关闭已经创建的共享 httpx 客户端，并清空缓存供应用再次启动。"""
    if get_dashscope_async_client.cache_info().currsize:
        await get_dashscope_async_client().aclose()
        get_dashscope_async_client.cache_clear()
    if get_dashscope_sync_client.cache_info().currsize:
        get_dashscope_sync_client().close()
        get_dashscope_sync_client.cache_clear()
