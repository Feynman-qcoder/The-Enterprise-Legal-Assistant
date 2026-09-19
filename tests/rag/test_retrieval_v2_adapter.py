from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import modules.database.v2_parent_repository as parent_repository_module
import modules.rag.pipeline as pipeline_module
from modules.core.config import Settings
from modules.database.v2_parent_repository import V2ParentRepository
from modules.milvus_store.collections import COLLECTION_FAQ, COLLECTION_LEGAL_CHILD
from modules.rag.hybrid_rrf import reciprocal_rank_fusion
from modules.rag.pipeline import RagPipeline, _parse_legacy_legal_hits
from modules.rag.retrieval_contract import RetrievalContractError
from modules.rag.retrieval_contract import LegalRetrievalHit
from modules.rag.retrieval_v2_adapter import (
    build_document_identity_expr,
    parse_v2_legal_hits,
)


class FakeHit:
    def __init__(self, hit_id, distance: float, entity: dict) -> None:
        self.id = hit_id
        self.distance = distance
        self.entity = entity


def test_retrieval_contract_defaults_to_legacy() -> None:
    settings = Settings(_env_file=None)
    assert settings.retrieval_contract == "legacy_v1"
    assert settings.retrieval_v2_collection == "xiaoyi_legal_child_v2"


def test_legacy_collection_names_are_unchanged() -> None:
    assert COLLECTION_LEGAL_CHILD == "xiaoyi_legal_child"
    assert COLLECTION_FAQ == "xiaoyi_faq_highfreq"


def test_legacy_retrieval_regression_adapts_int_contract() -> None:
    raw = [[FakeHit(17, 0.91, {"text": "法律正文", "parent_id": 3, "source_file": "a.pdf"})]]

    hits = _parse_legacy_legal_hits(raw)

    assert len(hits) == 1
    assert hits[0].chunk_id == 17
    assert hits[0].content == "法律正文"
    assert hits[0].parent_reference == 3
    assert hits[0].source_file == "a.pdf"
    assert hits[0].metadata["retrieval_contract"] == "legacy_v1"


def test_v2_dense_retrieval_adapts_varchar_contract() -> None:
    raw = [[FakeHit(
        "child-01",
        0.98,
        {
            "content": "电子商务法正文",
            "external_parent_id": "p" * 64,
            "document_identity": "LAW_001@sha256:abc",
            "source_file": r"D:\corpus\law.pdf",
            "section_path": ["第一章", "第一条"],
            "metadata_json": {"chunk_metadata_version": "v1"},
        },
    )]]

    hits = parse_v2_legal_hits(raw)

    assert len(hits) == 1
    assert hits[0].chunk_id == "child-01"
    assert hits[0].content == "电子商务法正文"
    assert hits[0].parent_reference == "p" * 64
    assert hits[0].metadata["section_path"] == ["第一章", "第一条"]
    assert hits[0].metadata["retrieval_contract"] == "v2"


def test_v2_dense_retrieval_fails_closed_on_missing_parent() -> None:
    raw = [[FakeHit(
        "child-01",
        0.98,
        {
            "content": "正文",
            "external_parent_id": "",
            "document_identity": "LAW_001@sha256:abc",
            "source_file": "law.pdf",
            "section_path": [],
            "metadata_json": {},
        },
    )]]

    with pytest.raises(RetrievalContractError, match="external_parent_id"):
        parse_v2_legal_hits(raw)


def test_v2_document_identity_expr_is_stable_and_non_empty() -> None:
    expr = build_document_identity_expr(["doc-b", "doc-a", "doc-a"])
    assert expr == 'document_identity in ["doc-a","doc-b"]'
    with pytest.raises(RetrievalContractError):
        build_document_identity_expr([])


def test_rrf_accepts_string_chunk_ids_without_algorithm_change() -> None:
    fused = reciprocal_rank_fusion(
        [["child-a", "child-b"], ["child-b", "child-c"]],
        k=60,
    )
    assert fused[0][0] == "child-b"
    assert {item[0] for item in fused} == {"child-a", "child-b", "child-c"}


class _FakeVersionRepo:
    """A6 适配：_cache_scope 现为 async 且读取 corpus version。"""

    def __init__(self, version: int) -> None:
        self._version = version

    async def fetch_corpus_version(self) -> int:
        return self._version


@pytest.mark.asyncio
async def test_cache_scope_isolated_by_retrieval_contract() -> None:
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline.settings = SimpleNamespace(retrieval_contract="legacy_v1")
    pipeline._v2_parents = _FakeVersionRepo(version=7)  # legacy 分支不读 version，固定 v0
    legacy_scope, legacy_version = await pipeline._cache_scope("问题", "user-1")
    pipeline.settings = SimpleNamespace(retrieval_contract="v2")
    pipeline._v2_parents = _FakeVersionRepo(version=7)  # v2 分支读 version=7 进 key
    v2_scope, v2_version = await pipeline._cache_scope("问题", "user-1")
    assert legacy_scope == "legacy_v1:v0:user-1:问题"
    assert legacy_version == 0  # Task 23：legacy 固定 version=0（元组第二位）
    assert v2_scope == "v2:v7:user-1:问题"
    assert v2_version == 7  # Task 23：version 随 scope 一并返回（语义缓存 key 用）
    assert legacy_scope != v2_scope


@pytest.mark.asyncio
async def test_cache_scope_version_flip_changes_key() -> None:
    """Task 19 A6：同 contract 下 version 翻面必须改变缓存 key（增量更新后旧缓存整体失效）。"""
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline.settings = SimpleNamespace(retrieval_contract="v2")
    pipeline._v2_parents = _FakeVersionRepo(version=1)
    scope_v1, version_v1 = await pipeline._cache_scope("问题", "user-1")
    pipeline._v2_parents = _FakeVersionRepo(version=2)
    scope_v2, version_v2 = await pipeline._cache_scope("问题", "user-1")
    assert scope_v1 == "v2:v1:user-1:问题"
    assert scope_v2 == "v2:v2:user-1:问题"
    assert scope_v1 != scope_v2
    assert version_v1 != version_v2  # Task 23：version 翻面同时体现在元组第二位


class FakeMappings:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def all(self) -> list[dict]:
        return self._rows


class FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows

    def mappings(self) -> FakeMappings:
        return FakeMappings(self._rows)


class FakeSession:
    def __init__(self, results: list[FakeResult]) -> None:
        self._results = iter(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def execute(self, *_args, **_kwargs) -> FakeResult:
        return next(self._results)


@pytest.mark.asyncio
async def test_v2_parent_repository_returns_content_metadata_and_section_path(monkeypatch) -> None:
    identities = ["doc-a", "doc-b"]
    identity_hash = hashlib.sha256(
        json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    parent_row = {
        "external_parent_id": "p1",
        "document_identity": "doc-a",
        "section_path": '["第一章"]',
        "content": "父级正文",
        "metadata": '{"parent_contract":"v1.1"}',
    }
    sessions = iter([
        FakeSession([FakeResult([(value,) for value in identities])]),
        FakeSession([FakeResult([parent_row])]),
    ])
    monkeypatch.setattr(parent_repository_module, "get_session_factory", lambda: lambda: next(sessions))
    monkeypatch.setattr(parent_repository_module, "EXPECTED_V2_DOCUMENT_COUNT", 2)
    monkeypatch.setattr(
        parent_repository_module,
        "EXPECTED_V2_DOCUMENT_IDENTITY_SET_SHA256",
        identity_hash,
    )
    repository = V2ParentRepository()

    assert await repository.fetch_document_identities() == frozenset(identities)
    parents = await repository.fetch_parents(["p1"])

    assert parents["p1"].content == "父级正文"
    assert parents["p1"].section_path == ["第一章"]
    assert parents["p1"].metadata == {"parent_contract": "v1.1"}


@pytest.mark.asyncio
async def test_v2_parent_repository_fails_closed_on_orphan(monkeypatch) -> None:
    monkeypatch.setattr(
        parent_repository_module,
        "get_session_factory",
        lambda: lambda: FakeSession([FakeResult([])]),
    )
    repository = V2ParentRepository()

    with pytest.raises(RetrievalContractError, match="missing 1 parent"):
        await repository.fetch_parents(["missing-parent"])


@pytest.mark.asyncio
async def test_v2_pipeline_boundary_keeps_bm25_rrf_parent_and_reranker(monkeypatch) -> None:
    class FakeCache:
        async def get_json(self, _key):
            return None

    class FakeEmbedding:
        async def embed_query(self, _text):
            return [0.0] * 1024

    class FakeDense:
        async def search(self, _vector, *, limit, document_identities):
            assert limit == 3
            assert document_identities == frozenset({"doc-a"})
            return [
                LegalRetrievalHit("c1", "电子 商务 绿色发展", "p1", {"source_file": r"D:\x\a.pdf"}, 0.9),
                LegalRetrievalHit("c2", "电子商务 产业融合", "p2", {"source_file": r"D:\x\b.pdf"}, 0.8),
                LegalRetrievalHit("c3", "无关文本", "p1", {"source_file": r"D:\x\a.pdf"}, 0.7),
            ]

    class FakeParents:
        async def fetch_document_identities(self):
            return frozenset({"doc-a"})

        async def fetch_corpus_version(self):  # Task 19 A5/A6 适配
            return 0

        async def fetch_parents(self, parent_ids):
            assert set(parent_ids) == {"p1", "p2"}
            return {
                "p1": SimpleNamespace(content="父文档一"),
                "p2": SimpleNamespace(content="父文档二"),
            }

    class FakeReranker:
        async def rank(self, _question, passages):
            assert set(passages) == {"父文档一", "父文档二"}
            return [0.8, 0.9]

    class FakeScope:
        def format_header(self, source_file):
            return f"[Source Metadata]\nsource_file: {source_file}"

    async def professional(_question):
        return True

    monkeypatch.setattr(pipeline_module, "is_professional_query", professional)
    monkeypatch.setattr(pipeline_module, "get_canonical_scope", lambda: FakeScope())
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline.settings = SimpleNamespace(
        retrieval_contract="v2",
        hybrid_dense_candidate_k=3,
        legal_hybrid_bm25_enabled=True,
        hybrid_bm25_candidate_k=3,
        hybrid_rrf_k=60,
        legal_rerank_top_n=2,
        rerank_pool_top_n=None,  # Config C 转正后的 Settings 字段（None = 全量父文档进 rerank）
        rerank_timeout_seconds=30.0,  # Task 21 Part 3：rerank 超时降级字段（FakeReranker 秒回不触发）
        faq_direct_distance_threshold=0.01,
        faq_llm_distance_threshold=0.15,
        faq_top_k_for_llm=3,
        semantic_cache_enabled=False,  # Task 23：语义缓存总开关（关闭时 stream_chat 跳过语义检查块）
    )
    pipeline._cache = FakeCache()
    pipeline._emb = FakeEmbedding()
    pipeline._v2_dense = FakeDense()
    pipeline._v2_parents = FakeParents()
    pipeline._reranker = lambda: FakeReranker()

    async def no_faq(*_args, **_kwargs):
        return []

    captured: dict[str, list[str]] = {}

    async def fake_rag_stream(*, question, contexts, **_kwargs):
        assert question == "电子商务如何绿色发展"
        captured["contexts"] = contexts
        yield "ok"

    pipeline._milvus_search = no_faq
    pipeline._rag_stream_llm = fake_rag_stream

    output = [piece async for piece in pipeline.stream_chat("电子商务如何绿色发展")]

    assert output == ["ok"]
    assert len(captured["contexts"]) == 2
    assert any("source_file: a.pdf" in context for context in captured["contexts"])
    assert any("source_file: b.pdf" in context for context in captured["contexts"])
