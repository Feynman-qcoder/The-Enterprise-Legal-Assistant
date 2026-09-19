from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest
from pymilvus import Collection, connections

from modules.core.config import get_settings
from modules.database.session import get_async_engine
from modules.database.v2_parent_repository import V2ParentRepository
from modules.rag.retrieval_v2_adapter import V2DenseRetrievalAdapter


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_XIAOYI_V2_LIVE") != "1",
    reason="set RUN_XIAOYI_V2_LIVE=1 for the read-only local integration smoke",
)

ROOT = pathlib.Path(r"F:\All_APP_projects\小易RAG\项目复盘")
CHUNKS_PATH = ROOT / "strategy_b_production_v2" / "chunks.json"
EMBEDDINGS_PATH = ROOT / "embedding_v1" / "embeddings.npy"


def _schema_snapshot(collection: Collection) -> list[tuple]:
    return [
        (
            field.name,
            field.dtype.name,
            bool(field.is_primary),
            bool(field.auto_id),
            tuple(sorted(dict(field.params).items())),
        )
        for field in collection.schema.fields
    ]


@pytest.mark.asyncio
async def test_v2_dense_parent_recall_and_legacy_protection_live() -> None:
    required_python = pathlib.Path(r"D:\AI\Anaconda3\envs\xiaoyi_rag\python.exe")
    assert pathlib.Path(sys.executable).resolve() == required_python.resolve()
    settings = get_settings()
    assert settings.hot_update_enabled is False
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        user=settings.milvus_user or None,
        password=settings.milvus_password or None,
    )
    legacy_legal = Collection("xiaoyi_legal_child")
    legacy_faq = Collection("xiaoyi_faq_highfreq")
    before = {
        "legal_count": int(legacy_legal.num_entities),
        "faq_count": int(legacy_faq.num_entities),
        "legal_schema": _schema_snapshot(legacy_legal),
        "faq_schema": _schema_snapshot(legacy_faq),
    }
    try:
        with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
            chunks = json.load(handle)["chunks"]
        vectors = np.load(EMBEDDINGS_PATH, mmap_mode="r", allow_pickle=False)
        vector_index = 1688
        repository = V2ParentRepository()
        identities = await repository.fetch_document_identities()
        adapter = V2DenseRetrievalAdapter(settings.retrieval_v2_collection)

        hits = await adapter.search(
            vectors[vector_index].tolist(),
            limit=5,
            document_identities=identities,
        )

        assert len(hits) == 5
        assert hits[0].chunk_id == chunks[vector_index]["chunk_id"]
        assert hits[0].similarity >= 0.9999
        assert all(isinstance(hit.chunk_id, str) and hit.content for hit in hits)
        assert all(isinstance(hit.parent_reference, str) for hit in hits)
        parents = await repository.fetch_parents(
            [str(hit.parent_reference) for hit in hits],
        )
        assert all(str(hit.parent_reference) in parents for hit in hits)
        assert all(parents[str(hit.parent_reference)].content for hit in hits)

        assert int(legacy_legal.num_entities) == before["legal_count"] == 2125
        assert int(legacy_faq.num_entities) == before["faq_count"] == 30
        assert _schema_snapshot(legacy_legal) == before["legal_schema"]
        assert _schema_snapshot(legacy_faq) == before["faq_schema"]
    finally:
        await get_async_engine().dispose()
        connections.disconnect("default")
