# =============================================================================
# 解决两个在线链路问题：
# 1) Canonical Online Isolation：生产 Milvus dense search 与 Legacy 数据同集合共存，
#    本模块从冻结的 ingest manifest 构造 source_file 白名单 expr，让 dense 检索从源头
#    只召回 canonical 子块；后续 BM25/RRF/父文档回溯/重排自然继承该候选集。
# 2) Retrieved Context Metadata Header：提供 source_file -> manifest 元数据的只读 lookup，
#    在父文档正文送入 GLM 前拼接轻量 [Source Metadata] 头（仅含 manifest 中实际存在的字段）。
#
# 输入：ingest_manifest_v1.jsonl（53 条 canonical 记录，含 canonical_file/title/
#       logical_document_id/source_org/version/effective_date 等字段）。
# 输出：CanonicalScope 对象（进程内缓存）：白名单集合、Milvus expr、元数据 header。
# 被谁调用：modules/rag/pipeline.py（dense search 过滤 + context header 注入）。
# 约束：只读；不修改 Milvus/MySQL Schema；manifest 更新需重启进程生效（冻结语料下可接受）。
# =============================================================================

"""Canonical corpus scope：manifest 白名单 + 元数据 lookup（只读、进程内缓存）。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from functools import lru_cache

from modules.core.config import get_settings

logger = logging.getLogger(__name__)

# header 中允许出现的字段及顺序（只输出 manifest 中实际存在且非空的字段，绝不编造）
HEADER_FIELDS: tuple[str, ...] = (
    "title",
    "logical_document_id",
    "source_org",
    "version",
    "effective_date",
    "source_file",
)


class CanonicalScope:
    """冻结语料范围对象：白名单集合 + Milvus expr + source_file 元数据 lookup。"""

    def __init__(self, source_files: frozenset[str], metadata: dict[str, dict[str, str]]) -> None:
        self.source_files = source_files  # 53 个 canonical_file 文件名
        self.metadata = metadata  # canonical_file -> {字段: 值}（仅保留实际存在的字段）

    def build_source_file_expr(self) -> str:
        """
        构造 Milvus 标量过滤表达式：source_file in ["...", ...]。

        入参:
            无。
        返回:
            可直接传给 Collection.search(expr=...) 的字符串。
        """
        quoted = ",".join(json.dumps(f, ensure_ascii=False) for f in sorted(self.source_files))
        return f"source_file in [{quoted}]"

    def is_canonical(self, source_file: str | None) -> bool:
        """判断某个 source_file 是否属于 canonical 白名单。"""
        return bool(source_file) and source_file in self.source_files

    def format_header(self, source_file: str | None) -> str | None:
        """
        生成某篇父文档的 [Source Metadata] 轻量头。

        入参:
            - source_file: 该父文档对应的 canonical 文件名（来自子块实体）。
        返回:
            - 多行 header 字符串；lookup 失败（不在 manifest / 为空）返回 None，由调用方保持裸文本。
        """
        if not source_file:
            return None
        meta = self.metadata.get(source_file)
        if not meta:
            return None
        lines = ["[Source Metadata]"]
        for key in HEADER_FIELDS:
            if key == "source_file":
                continue  # manifest 记录本身无 source_file 字段（其 key 为 canonical_file），改用入参真实检索值
            value = meta.get(key)
            if value:  # 只加入实际存在的字段
                lines.append(f"{key}: {value}")
        lines.append(f"source_file: {source_file}")  # 真实 retrieval record 的 source_file（= manifest 命中键，非生成字段）
        return "\n".join(lines)


@lru_cache
def get_canonical_scope() -> CanonicalScope:
    """
    读取并解析 ingest manifest，构建进程内单例 CanonicalScope（fail-closed）。

    入参:
        无。
    返回:
        CanonicalScope 单例。
    异常:
        RuntimeError：manifest 缺失 / 无 canonical 记录 —— 隔离无法保证时宁可启动失败，
        也不允许静默退化为「不过滤」（否则 Legacy 将重新进入生产检索）。
    """
    raw_path = get_settings().canonical_manifest_path
    # Task 20：相对路径锚定仓库根（modules/rag/ 上两级），与启动工作目录解耦
    path = raw_path if os.path.isabs(raw_path) else str(
        Path(__file__).resolve().parents[2] / raw_path
    )
    if not os.path.isfile(path):
        raise RuntimeError(f"canonical manifest not found: {path}")  # fail-closed
    source_files: set[str] = set()
    metadata: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"manifest line {line_no} invalid JSON: {exc}") from exc
            if not rec.get("canonical", False):
                continue  # 只收 canonical 记录（同 manifest 中不含非 canonical，防御式判断）
            sf = rec.get("canonical_file")
            if not sf:
                continue
            source_files.add(sf)
            metadata[sf] = {k: rec[k] for k in HEADER_FIELDS if rec.get(k)}
    if not source_files:
        raise RuntimeError(f"manifest has no canonical files: {path}")  # fail-closed
    logger.info("canonical scope loaded: %d files from %s", len(source_files), path)
    return CanonicalScope(frozenset(source_files), metadata)
