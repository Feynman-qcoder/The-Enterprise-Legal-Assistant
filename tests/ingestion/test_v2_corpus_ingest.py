from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from offline.scripts import run_v2_corpus_ingest as batch
from offline.scripts import run_v2_file_ingest as file_ingest


@dataclass(frozen=True)
class Layout:
    corpus: Path
    baseline: Path
    live: Path
    metadata: Path


def _entry(logical_id: str, canonical_file: str) -> dict:
    return {
        "logical_document_id": logical_id,
        "title": f"标题-{logical_id}",
        "canonical_file": canonical_file,
        "format": Path(canonical_file).suffix.lstrip("."),
        "source_org": "测试机构",
        "cleaning_status": "PASS",
        "canonical": True,
        "normalized_sha256": ("a" if logical_id.endswith("1") else "b") * 64,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    corpus = tmp_path / "data_corpus"
    corpus.mkdir()
    (corpus / "DOC_001.md").write_text("# 第一份文档\n\n有效正文。", encoding="utf-8")
    (corpus / "DOC_002.md").write_text("# 第二份文档\n\n有效正文。", encoding="utf-8")
    baseline = tmp_path / "frozen" / "manifest.jsonl"
    live = tmp_path / "data" / "manifests" / "live.jsonl"
    metadata = tmp_path / "data" / "manifests" / "batch.jsonl"
    _write_jsonl(baseline, [_entry("DOC_001", "DOC_001.md")])
    _write_jsonl(
        metadata,
        [
            {
                "canonical_file": "DOC_002.md",
                "logical_document_id": "DOC_002",
                "title": "第二份文档",
                "source_org": "测试机构",
            }
        ],
    )
    return Layout(corpus, baseline, live, metadata)


def _common_args(layout: Layout) -> list[str]:
    return [
        "--corpus-dir",
        str(layout.corpus),
        "--manifest",
        str(layout.live),
        "--baseline-manifest",
        str(layout.baseline),
        "--metadata-file",
        str(layout.metadata),
    ]


def _report(args, *, version: int = 1) -> dict:
    logical_id = args.doc or args.logical_document_id
    canonical_file = (
        f"{logical_id}.md" if args.doc is not None else Path(args.file).name
    )
    if not args.apply:
        return {
            "mode": "dry-run",
            "document_identity": f"{logical_id}@sha256:{'c' * 64}",
            "parent_count": 1,
            "child_count": 2,
            "ingestion_status": "DRY_RUN",
        }
    return {
        "mode": "apply",
        "document_identity": f"{logical_id}@sha256:{'c' * 64}",
        "canonical_file": canonical_file,
        "parent_count": 1,
        "child_count": 2,
        "mysql_parent_written": 1,
        "mysql_mapping_written": 2,
        "milvus_written": 2,
        "orphan_count": 0,
        "vector_dimension": 1024,
        "corpus_version_before": version,
        "corpus_version_after": version + 1,
        "legacy_collection_unchanged": True,
        "backend_restart_required": args.doc is None,
        "ingestion_status": "PASS",
    }


@pytest.mark.asyncio
async def test_default_dry_run_scans_registered_and_metadata_files(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    calls = []

    async def fake_run_locked(args, _manifest):
        calls.append(args)
        return _report(args)

    monkeypatch.setattr(file_ingest, "_run_locked", fake_run_locked)
    result = await batch.run(_common_args(layout))

    assert result["batch_ingestion_status"] == "DRY_RUN"
    assert result["total_files"] == result["prevalidated"] == 2
    assert result["database_writes"] == result["milvus_writes"] == 0
    assert len(calls) == 2
    assert all(not args.apply and args.dry_run for args in calls)
    registered = next(args for args in calls if args.doc)
    unregistered = next(args for args in calls if args.file)
    assert registered.doc == "DOC_001"
    assert unregistered.logical_document_id == "DOC_002"
    assert unregistered.title == "第二份文档"
    assert unregistered.source_org == "测试机构"


@pytest.mark.asyncio
async def test_apply_prevalidates_every_file_before_any_write(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    calls = []

    async def fake_run_locked(args, _manifest):
        calls.append(args)
        if not args.apply and (args.doc or args.logical_document_id) == "DOC_002":
            raise file_ingest.IngestionFailure("chunk_contract", "invalid chunk")
        return _report(args)

    monkeypatch.setattr(file_ingest, "_run_locked", fake_run_locked)
    with pytest.raises(file_ingest.IngestionFailure) as caught:
        await batch.run(["--apply", *_common_args(layout)])

    assert caught.value.stage == "batch_prevalidate"
    assert len(calls) == 2
    assert all(not args.apply for args in calls)


@pytest.mark.asyncio
async def test_apply_aggregates_verified_single_file_reports(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    calls = []
    version = 10

    async def fake_run_locked(args, _manifest):
        nonlocal version
        calls.append(args)
        if not args.apply:
            return _report(args)
        result = _report(args, version=version)
        version += 1
        return result

    monkeypatch.setattr(file_ingest, "_run_locked", fake_run_locked)
    result = await batch.run(["--apply", *_common_args(layout)])

    assert [args.apply for args in calls] == [False, False, True, True]
    assert result["batch_ingestion_status"] == "PASS"
    assert result["succeeded"] == 2
    assert result["parent_count"] == 2
    assert result["child_count"] == 4
    assert result["mysql_parent_written"] == 2
    assert result["mysql_mapping_written"] == 4
    assert result["milvus_written"] == 4
    assert result["orphan_count"] == 0
    assert result["vector_dimensions"] == [1024]
    assert result["corpus_version_before"] == 10
    assert result["corpus_version_after"] == 12
    assert result["legacy_collection_unchanged"] is True


@pytest.mark.asyncio
async def test_unregistered_file_without_metadata_fails_before_delegation(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    layout.metadata.unlink()

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("single-file wrapper must not run")

    monkeypatch.setattr(file_ingest, "_run_locked", must_not_run)
    with pytest.raises(file_ingest.IngestionFailure) as caught:
        await batch.run(_common_args(layout))
    assert caught.value.stage == "batch_metadata"
    assert "DOC_002.md" in str(caught.value)


@pytest.mark.asyncio
async def test_unsupported_corpus_file_fails_closed(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    (layout.corpus / "notes.xlsx").write_bytes(b"not a supported document")
    with pytest.raises(file_ingest.IngestionFailure) as caught:
        await batch.run(_common_args(layout))
    assert caught.value.stage == "batch_scan"
    assert "notes.xlsx" in str(caught.value)


def test_duplicate_metadata_identity_is_rejected(tmp_path: Path) -> None:
    metadata = tmp_path / "batch.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "canonical_file": "ONE.md",
                "logical_document_id": "SAME_001",
                "title": "一",
            },
            {
                "canonical_file": "TWO.md",
                "logical_document_id": "SAME_001",
                "title": "二",
            },
        ],
    )
    with pytest.raises(file_ingest.IngestionFailure) as caught:
        batch.load_batch_metadata(metadata)
    assert caught.value.stage == "batch_metadata"
    assert "repeat logical_document_id" in str(caught.value)


@pytest.mark.asyncio
async def test_apply_failure_reports_partial_commit_and_stops(
    layout: Layout,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(file_ingest, "validate_runtime", lambda: None)
    applied = 0

    async def fake_run_locked(args, _manifest):
        nonlocal applied
        if not args.apply:
            return _report(args)
        applied += 1
        if applied == 2:
            raise file_ingest.IngestionFailure("milvus_preflight", "unavailable")
        return _report(args, version=20)

    monkeypatch.setattr(file_ingest, "_run_locked", fake_run_locked)
    with pytest.raises(file_ingest.IngestionFailure) as caught:
        await batch.run(["--apply", *_common_args(layout)])
    output = capsys.readouterr().out

    assert caught.value.stage == "batch_apply"
    assert applied == 2
    assert '"partial_commit": true' in output
    assert "BATCH_INGESTION_STATUS=FAIL" in output
    assert "BATCH_INGESTION_STATUS=PASS" not in output


def test_apply_authorization_flags_are_mutually_exclusive(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = batch.main(["--dry-run", "--apply"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "FAILED_STAGE=cli" in captured.err
    assert "BATCH_INGESTION_STATUS=FAIL" in captured.err
    assert "BATCH_INGESTION_STATUS=PASS" not in captured.out + captured.err
