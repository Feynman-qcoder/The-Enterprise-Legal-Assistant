"""Milvus 客户端与集合：对外统一导出常用符号（便于脚本 `from modules.milvus_store import …`）。"""

from modules.milvus_store.client import close_milvus, ensure_milvus
from modules.milvus_store.collections import (
    COLLECTION_FAQ,
    COLLECTION_LEGAL_CHILD,
    create_collections_if_not_exist,
)

__all__ = [
    "COLLECTION_FAQ",
    "COLLECTION_LEGAL_CHILD",
    "close_milvus",
    "create_collections_if_not_exist",
    "ensure_milvus",
]
