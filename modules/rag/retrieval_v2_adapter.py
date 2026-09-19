"""Read-only Dense Retrieval adapter for ``xiaoyi_legal_child_v2``."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from pymilvus import Collection, DataType

from modules.milvus_store.client import ensure_milvus
from modules.rag.retrieval_contract import LegalRetrievalHit, RetrievalContractError

logger = logging.getLogger(__name__)

EXPECTED_V2_ENTITY_COUNT = 4134
EXPECTED_V2_VECTOR_DIM = 1024
V2_OUTPUT_FIELDS = [
    "content",
    "external_parent_id",
    "document_identity",
    "source_file",
    "section_path",
    "metadata_json",
]

_EXPECTED_FIELDS = [
    ("child_id", DataType.VARCHAR, True, {"max_length": 64}),
    ("embedding", DataType.FLOAT_VECTOR, False, {"dim": EXPECTED_V2_VECTOR_DIM}),
    ("content", DataType.VARCHAR, False, {"max_length": 32768}),
    ("external_parent_id", DataType.VARCHAR, False, {"max_length": 64}),
    ("document_identity", DataType.VARCHAR, False, {"max_length": 256}),
    ("source_file", DataType.VARCHAR, False, {"max_length": 1024}),
    ("section_path", DataType.JSON, False, {}),
    ("metadata_json", DataType.JSON, False, {}),
]


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    if entity is None:
        return {}
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "to_dict"):
        value = entity.to_dict()
        return value if isinstance(value, dict) else {}
    try:
        return dict(entity)
    except (TypeError, ValueError):
        return {}


def _require_non_empty_string(entity: dict[str, Any], field: str, child_id: str) -> str:
    value = entity.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RetrievalContractError(f"V2 child {child_id!r} missing non-empty {field}")
    return value


def parse_v2_legal_hits(raw_hits: Any) -> list[LegalRetrievalHit]:
    """Normalize PyMilvus V2 hits without coercing VARCHAR child IDs to integers."""

    if not raw_hits or not raw_hits[0]:
        return []
    parsed: list[LegalRetrievalHit] = []
    seen: set[str] = set()
    for hit in raw_hits[0]:
        child_id = getattr(hit, "id", None)
        if not isinstance(child_id, str) or not child_id:
            raise RetrievalContractError("V2 Milvus hit has invalid VARCHAR child_id")
        if child_id in seen:
            raise RetrievalContractError(f"duplicate V2 child_id in search result: {child_id}")
        seen.add(child_id)
        entity = _entity_to_dict(getattr(hit, "entity", None))
        content = _require_non_empty_string(entity, "content", child_id)
        external_parent_id = _require_non_empty_string(entity, "external_parent_id", child_id)
        document_identity = _require_non_empty_string(entity, "document_identity", child_id)
        source_file = _require_non_empty_string(entity, "source_file", child_id)
        section_path = entity.get("section_path")
        metadata_json = entity.get("metadata_json")
        if not isinstance(section_path, list):
            raise RetrievalContractError(f"V2 child {child_id!r} has invalid section_path")
        if not isinstance(metadata_json, dict):
            raise RetrievalContractError(f"V2 child {child_id!r} has invalid metadata_json")
        similarity = float(getattr(hit, "distance"))
        if not math.isfinite(similarity):
            raise RetrievalContractError(f"V2 child {child_id!r} has non-finite similarity")
        parsed.append(
            LegalRetrievalHit(
                chunk_id=child_id,
                content=content,
                parent_reference=external_parent_id,
                metadata={
                    "retrieval_contract": "v2",
                    "document_identity": document_identity,
                    "source_file": source_file,
                    "section_path": section_path,
                    "metadata_json": metadata_json,
                },
                similarity=similarity,
            ),
        )
    return parsed


def build_document_identity_expr(document_identities: Iterable[str]) -> str:
    values = sorted(set(document_identities))
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise RetrievalContractError("V2 document identity scope is empty or invalid")
    quoted = ",".join(json.dumps(value, ensure_ascii=False) for value in values)
    return f"document_identity in [{quoted}]"


@lru_cache(maxsize=4)
def validate_v2_collection_contract(collection_name: str) -> None:
    """Fail closed on schema/index drift; never creates or modifies a collection.

    实体数校验（Task 19 A4）：精确校验降级为 WARNING 后放行。
    理由：schema/索引错 = 契约破坏必须拒；实体数变化 = 增量更新的正常预期。
    fail-closed 部分保持不动：schema 8 字段 / VARCHAR PK / dim=1024 / HNSW/COSINE/M=16/efC=200。
    """

    ensure_milvus()
    collection = Collection(collection_name)
    if collection.schema.enable_dynamic_field:
        raise RetrievalContractError("V2 collection dynamic fields must be disabled")
    fields = collection.schema.fields
    if len(fields) != len(_EXPECTED_FIELDS):
        raise RetrievalContractError("V2 collection field count mismatch")
    for actual, expected in zip(fields, _EXPECTED_FIELDS, strict=True):
        name, dtype, primary, params = expected
        if actual.name != name or actual.dtype != dtype or bool(actual.is_primary) != primary:
            raise RetrievalContractError(f"V2 collection schema mismatch at field {name}")
        if bool(actual.auto_id):
            raise RetrievalContractError(f"V2 collection field {name} unexpectedly uses auto_id")
        for key, value in params.items():
            if int(actual.params.get(key, -1)) != value:
                raise RetrievalContractError(f"V2 collection schema mismatch at {name}.{key}")
    actual_entities = int(collection.num_entities)
    if actual_entities != EXPECTED_V2_ENTITY_COUNT:
        # A4：增量更新后实体数偏离冻结基线属于正常预期，仅告警放行（fail-closed → WARNING）。
        logger.warning(
            "V2 collection entity count drift (incremental-update expected): expected=%d actual=%d",
            EXPECTED_V2_ENTITY_COUNT,
            actual_entities,
        )
    indexes = list(collection.indexes)
    if len(indexes) != 1:
        raise RetrievalContractError("V2 collection must have exactly one vector index")
    index = indexes[0]
    params = dict(index.params)
    nested = dict(params.get("params", {}))
    if (
        index.field_name != "embedding"
        or str(params.get("index_type", "")).upper() != "HNSW"
        or str(params.get("metric_type", "")).upper() != "COSINE"
        or int(nested.get("M", -1)) != 16
        or int(nested.get("efConstruction", -1)) != 200
    ):
        raise RetrievalContractError("V2 collection index contract mismatch")


class V2DenseRetrievalAdapter:
    """Executes read-only V2 dense search and returns unified legal hits."""

    def __init__(self, collection_name: str) -> None:
        if not collection_name.strip():
            raise RetrievalContractError("V2 collection name must not be empty")
        self.collection_name = collection_name.strip()

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_identities: Iterable[str],
    ) -> list[LegalRetrievalHit]:
        expr = build_document_identity_expr(document_identities)

        def _run() -> Any:
            validate_v2_collection_contract(self.collection_name)
            collection = Collection(self.collection_name)
            collection.load()
            return collection.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=limit,
                expr=expr,
                output_fields=V2_OUTPUT_FIELDS,
            )

        return parse_v2_legal_hits(await asyncio.to_thread(_run))
