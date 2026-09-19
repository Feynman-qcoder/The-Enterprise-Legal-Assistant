# =============================================================================
# -----------------------------------------------------------------------------
# 输入：纯文本字符串（单条 query 或多条 documents）；`.env` 中模型路径与设备。
# 输出：`list[float]` 单条向量，或 `list[list[float]]` 批量向量；`dimension` 属性为整数维数。
# 被谁调用：`RagPipeline`（在线对用户问题编码）、`milvus_sync`（离线批量编码 FAQ/子块）。
# =============================================================================
"""
LangChain `HuggingFaceEmbeddings` 封装 BGE-M3：`encode_kwargs.normalize_embeddings=True` 使向量单位长度，配合 COSINE。

同步 `embed_query` 用 `asyncio.to_thread` 包一层，避免阻塞 FastAPI 事件循环。
"""

from __future__ import annotations

import asyncio  # 提供 to_thread 把同步 CPU 计算丢到线程池
import logging  # 模型路径不存在时 warning
from functools import lru_cache  # 全进程只加载一次 LangChain Embeddings 对象
from pathlib import Path  # 把相对路径拼成绝对路径
from typing import Sequence  # 批量接口参数类型：str 序列

from modules.core.config import get_settings  # 设备、embedding_model_path

logger = logging.getLogger(__name__)


@lru_cache
def _make_lc_embeddings():
    """
    内部工厂：首次调用时才 import langchain_huggingface，加快「只跑 DB 脚本」的冷启动。

    返回 LangChain 的 Embeddings 实例（带本地模型目录）。

    入参:
        无。
    返回:
        配置好本地路径与设备的 `HuggingFaceEmbeddings` 实例。
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # 延迟导入：避免无 torch 环境 import 失败扩散

    settings = get_settings()  # 读路径与 device
    root = Path(__file__).resolve().parents[2]  # 本文件在 modules/embeddings/，parents[2] = 仓库根 Legal_System
    path = root / settings.embedding_model_path  # Path 拼接：根 / models/bge-m3
    if not path.exists():  # 目录不存在
        logger.warning(
            "Embedding 模型路径不存在：%s（请将模型下载到该目录）",
            path,
        )  # 仍继续构造，可能在首次 encode 时才硬失败
    return HuggingFaceEmbeddings(
        model_name=str(path),  # 传入字符串路径，HF 从本地文件夹加载
        model_kwargs={"device": settings.embedding_device, "trust_remote_code": True},  # device: cpu/cuda；trust_remote_code：部分国产模型需要
        encode_kwargs={"normalize_embeddings": True},  # L2 归一化，COSINE 等价于点积且数值稳定
    )


class LocalEmbeddingService:
    """
    对外只暴露 async 方法，便于与 async pipeline 统一风格。
    """

    def __init__(self) -> None:
        """
        构造本地嵌入服务，内部加载句向量模型。

        入参:
            无。
        返回:
            无。
        """
        self._emb = _make_lc_embeddings()  # 构造时加载模型权重到内存/GPU（较重）
    @property
    def dimension(self) -> int:
        """
        对任意短句编码一次，用返回向量长度作为 Milvus dim；避免手写 1024/768 与模型不一致。

        入参:
            无。
        返回:
            整数，表示模型输出的向量维度。
        """
        v = self._emb.embed_query("ping")  # 同步调用；仅启动建表时用，频率低可接受
        return len(v)  # 向量维度 = 列表长度

    async def embed_query(self, text: str) -> list[float]:
        """
        单条文本 → 向量；在线主路径每个用户问题调用一次。

        入参:
            text: 待编码的单条字符串。
        返回:
            与模型维度一致的 `float` 列表（句向量）。
        """
        return await asyncio.to_thread(self._emb.embed_query, text)  # 在线程池执行同步方法，主线程继续调度其它协程

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        多条文本 → 向量列表；顺序与输入 texts 严格一致，供 Milvus insert 对齐。

        入参:
            texts: 待批量编码的文本序列；空序列时直接返回空列表。
        返回:
            与 `texts` 等长的向量列表，每项为 `list[float]`。
        """
        if not texts:  # 空列表：避免底层收到 [] 行为未定义
            return []  # 约定返回空列表
        return await asyncio.to_thread(self._emb.embed_documents, list(texts))  # LangChain 接口需要 list
