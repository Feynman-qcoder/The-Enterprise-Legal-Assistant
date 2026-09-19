"""Read-only repository for frozen SectionSpan parents used by Retrieval V2."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import bindparam, text

from modules.database.session import get_session_factory
from modules.rag.retrieval_contract import RetrievalContractError

logger = logging.getLogger(__name__)

EXPECTED_V2_DOCUMENT_COUNT = 49
MAX_V2_PARENT_CONTENT_CHARS = 8000
EXPECTED_V2_DOCUMENT_IDENTITY_SET_SHA256 = (
    "53403dac09904f39d65c34d5fe7a18d48f5fe4753845c9c876f25ff3aa821770"
)

# Task 19 A5：identities / corpus_version 统一 60s TTL 缓存。
# 事实核查 F1 裁决：原实现为进程级永久缓存 + 冻结校验（非每请求直读），
# 按「实例缓存 → 60s TTL」策略改造，使增量更新后的新 identity 可见，
# 生效延迟上限 60s（月更场景无感）。
_CACHE_TTL_SECONDS = 60.0

CORPUS_META_TABLE = "corpus_meta"
CORPUS_VERSION_KEY = "corpus_version"


def _normalize_json(value: Any, *, field: str, parent_id: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RetrievalContractError(
                f"V2 parent {parent_id!r} contains invalid JSON in {field}",
            ) from exc
    return value


@dataclass(frozen=True, slots=True)
class V2ParentRecord:
    external_parent_id: str
    document_identity: str
    section_path: list[str]
    content: str
    metadata: dict[str, Any]


class V2ParentRepository:
    """Provides fail-closed, batched reads without registering schema in create_all."""

    def __init__(self) -> None:
        self._document_identities: frozenset[str] | None = None
        self._identity_cached_at: float = 0.0
        self._identity_lock = asyncio.Lock()
        self._corpus_version: int | None = None
        self._version_cached_at: float = 0.0
        self._version_lock = asyncio.Lock()
        self._version_error_logged = False  # 表不存在等错误只 log 一次

    async def fetch_document_identities(self) -> frozenset[str]:
        """V2 文档身份白名单（60s TTL 缓存）。

        计数/SHA256 冻结校验降级为 WARNING（Task 19 A5，与 A4 裁决同源）：
        文档集合变化 = 增量更新的正常预期；校验漂移仅告警审计，不再 fail-closed。
        """
        now = time.monotonic()
        if (
            self._document_identities is not None
            and now - self._identity_cached_at < _CACHE_TTL_SECONDS
        ):
            return self._document_identities
        async with self._identity_lock:
            now = time.monotonic()
            if (
                self._document_identities is not None
                and now - self._identity_cached_at < _CACHE_TTL_SECONDS
            ):
                return self._document_identities
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    text("SELECT DISTINCT document_identity FROM legal_parent_contract_v1"),
                )
                identities = frozenset(str(row[0]) for row in result.all() if row[0])
            serialized = json.dumps(
                sorted(identities),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            identity_hash = hashlib.sha256(serialized).hexdigest()
            if len(identities) != EXPECTED_V2_DOCUMENT_COUNT:
                # 增量更新后文档数偏离冻结基线：告警放行（A5 裁决，原 fail-closed）。
                logger.warning(
                    "V2 parent document scope count drift (incremental-update expected): expected=%d actual=%d",
                    EXPECTED_V2_DOCUMENT_COUNT,
                    len(identities),
                )
            if identity_hash != EXPECTED_V2_DOCUMENT_IDENTITY_SET_SHA256:
                # 文档集合指纹漂移：告警放行（A5 裁决，原 fail-closed）。
                logger.warning(
                    "V2 parent document identity scope hash drift (incremental-update expected)"
                )
            self._document_identities = identities
            self._identity_cached_at = time.monotonic()
            return identities

    async def fetch_corpus_version(self) -> int:
        """读取语料版本号（60s TTL 缓存，Task 19 A5）。

        版本真源 = MySQL 单行表 corpus_meta（meta_key='corpus_version'）。
        容错：表不存在 → log error 一次，返回 0（version 特性惰性生效，
        首次同步建表后自愈；返回 0 时缓存 key 与现状等价，零风险）。
        """
        now = time.monotonic()
        if self._corpus_version is not None and now - self._version_cached_at < _CACHE_TTL_SECONDS:
            return self._corpus_version
        async with self._version_lock:
            now = time.monotonic()
            if (
                self._corpus_version is not None
                and now - self._version_cached_at < _CACHE_TTL_SECONDS
            ):
                return self._corpus_version
            version = 0
            try:
                factory = get_session_factory()
                async with factory() as session:
                    result = await session.execute(
                        text(
                            f"SELECT meta_value FROM `{CORPUS_META_TABLE}` "
                            f"WHERE meta_key = :k"
                        ),
                        {"k": CORPUS_VERSION_KEY},
                    )
                    row = result.first()
                if row is not None:
                    version = int(str(row[0]))
            except Exception as exc:  # noqa: BLE001 — 版本特性惰性生效，任何读失败降级为 0
                if not self._version_error_logged:
                    logger.error(
                        "corpus_meta read failed (version feature inert until first sync): %s",
                        exc,
                    )
                    self._version_error_logged = True
                version = 0
            self._corpus_version = version
            self._version_cached_at = time.monotonic()
            return version

    async def fetch_parents(
        self,
        external_parent_ids: Sequence[str],
    ) -> dict[str, V2ParentRecord]:
        ordered_unique = list(dict.fromkeys(external_parent_ids))
        if not ordered_unique:
            return {}
        if any(not isinstance(value, str) or not value for value in ordered_unique):
            raise RetrievalContractError("V2 parent references must be non-empty strings")
        statement = text(
            "SELECT external_parent_id, document_identity, section_path, content, metadata "
            "FROM legal_parent_contract_v1 "
            "WHERE external_parent_id IN :parent_ids"
        ).bindparams(bindparam("parent_ids", expanding=True))
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(statement, {"parent_ids": ordered_unique})
            rows = result.mappings().all()

        records: dict[str, V2ParentRecord] = {}
        for row in rows:
            parent_id = str(row["external_parent_id"])
            section_path = _normalize_json(
                row["section_path"],
                field="section_path",
                parent_id=parent_id,
            )
            metadata = _normalize_json(
                row["metadata"],
                field="metadata",
                parent_id=parent_id,
            )
            content = row["content"]
            document_identity = row["document_identity"]
            if not isinstance(section_path, list):
                raise RetrievalContractError(f"V2 parent {parent_id!r} has invalid section_path")
            if not isinstance(metadata, dict):
                raise RetrievalContractError(f"V2 parent {parent_id!r} has invalid metadata")
            if not isinstance(content, str) or not content.strip():
                raise RetrievalContractError(f"V2 parent {parent_id!r} has empty content")
            if len(content) > MAX_V2_PARENT_CONTENT_CHARS:
                raise RetrievalContractError(
                    f"V2 parent {parent_id!r} exceeds the 8000-character contract",
                )
            if not isinstance(document_identity, str) or not document_identity:
                raise RetrievalContractError(
                    f"V2 parent {parent_id!r} has invalid document_identity",
                )
            records[parent_id] = V2ParentRecord(
                external_parent_id=parent_id,
                document_identity=document_identity,
                section_path=section_path,
                content=content,
                metadata=metadata,
            )
        missing = [parent_id for parent_id in ordered_unique if parent_id not in records]
        if missing:
            raise RetrievalContractError(f"V2 parent recall missing {len(missing)} parent(s)")
        return records
