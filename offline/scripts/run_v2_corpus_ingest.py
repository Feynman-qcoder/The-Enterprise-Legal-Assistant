"""One-command, whole-corpus V2 ingestion orchestration.

This wrapper owns no parser, cleaner, chunker, embedding, MySQL persistence,
or Milvus insertion logic.  Every document is delegated to
``run_v2_file_ingest.py``.  In apply mode the complete corpus is dry-run
validated first, so a parser/cleaner/contract failure prevents all writes.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_FILE = REPO / "data" / "manifests" / "ingest_batch_v2.jsonl"

sys.path.insert(0, str(REPO))

from offline.scripts import run_v2_file_ingest as file_ingest  # noqa: E402


@dataclass(frozen=True)
class BatchItem:
    """One deterministic corpus item and the metadata needed by the file wrapper."""

    source_path: Path
    logical_document_id: str
    title: str
    source_org: str | None
    registered: bool

    @property
    def canonical_file(self) -> str:
        return self.source_path.name


def create_parser() -> argparse.ArgumentParser:
    parser = file_ingest.IngestionArgumentParser(
        description=(
            "Validate every supported file in data_corpus, then ingest each one "
            "through run_v2_file_ingest.py"
        )
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every document; never connect to or write MySQL/Milvus (default)",
    )
    authorization.add_argument(
        "--apply",
        action="store_true",
        help="after every document passes dry-run, authorize sequential writes",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help=(
            "JSONL metadata for files not already registered in the live/baseline "
            "Manifest"
        ),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="explicitly authorize changed identities; forwarded to the file wrapper",
    )
    parser.add_argument(
        "--manifest", type=Path, default=file_ingest.DEFAULT_LIVE_MANIFEST
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=file_ingest.DEFAULT_BASELINE_MANIFEST,
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=file_ingest.DEFAULT_CORPUS_DIR
    )
    return parser


def _require_metadata_text(record: dict[str, Any], key: str, line_no: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise file_ingest.IngestionFailure(
            "batch_metadata", f"metadata line {line_no}: {key} must be a non-empty string"
        )
    return value.strip()


def load_batch_metadata(path: Path) -> list[dict[str, str]]:
    """Read optional new-file identity metadata without inventing document IDs."""

    if not path.exists():
        return []
    if not path.is_file():
        raise file_ingest.IngestionFailure(
            "batch_metadata", f"metadata path is not a file: {path}"
        )

    records: list[dict[str, str]] = []
    canonical_seen: dict[str, int] = {}
    logical_seen: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise file_ingest.IngestionFailure(
            "batch_metadata", f"cannot read metadata file {path}: {exc}"
        ) from exc

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise file_ingest.IngestionFailure(
                "batch_metadata", f"metadata line {line_no}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(raw, dict):
            raise file_ingest.IngestionFailure(
                "batch_metadata", f"metadata line {line_no}: expected a JSON object"
            )

        canonical_file = _require_metadata_text(raw, "canonical_file", line_no)
        logical_id = _require_metadata_text(raw, "logical_document_id", line_no)
        title = _require_metadata_text(raw, "title", line_no)
        if Path(canonical_file).name != canonical_file:
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata line {line_no}: canonical_file must be a filename, not a path",
            )
        if Path(canonical_file).suffix.lower() not in file_ingest.SUPPORTED_SUFFIXES:
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata line {line_no}: unsupported canonical_file format",
            )
        if not file_ingest._LOGICAL_ID_RE.fullmatch(logical_id):
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata line {line_no}: unsafe logical_document_id {logical_id!r}",
            )

        source_org_raw = raw.get("source_org")
        if source_org_raw is not None and (
            not isinstance(source_org_raw, str) or not source_org_raw.strip()
        ):
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata line {line_no}: source_org must be a non-empty string",
            )
        source_org = source_org_raw.strip() if source_org_raw is not None else None

        canonical_key = canonical_file.casefold()
        logical_key = logical_id.casefold()
        if canonical_key in canonical_seen:
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata lines {canonical_seen[canonical_key]} and {line_no} repeat "
                f"canonical_file {canonical_file!r}",
            )
        if logical_key in logical_seen:
            raise file_ingest.IngestionFailure(
                "batch_metadata",
                f"metadata lines {logical_seen[logical_key]} and {line_no} repeat "
                f"logical_document_id {logical_id!r}",
            )
        canonical_seen[canonical_key] = line_no
        logical_seen[logical_key] = line_no
        record = {
            "canonical_file": canonical_file,
            "logical_document_id": logical_id,
            "title": title,
        }
        if source_org is not None:
            record["source_org"] = source_org
        records.append(record)
    return records


def discover_batch_items(
    *,
    corpus_dir: Path,
    manifest_entries: Sequence[dict[str, Any]],
    metadata_records: Sequence[dict[str, str]],
    replace_existing: bool,
) -> list[BatchItem]:
    """Build a complete, collision-free batch before any parser or storage work."""

    corpus = corpus_dir.resolve()
    if not corpus.is_dir():
        raise file_ingest.IngestionFailure(
            "batch_scan", f"data_corpus directory not found: {corpus}"
        )

    all_files = sorted(
        (path.resolve() for path in corpus.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(corpus)).casefold(),
    )
    unsupported = [
        str(path.relative_to(corpus))
        for path in all_files
        if path.suffix.lower() not in file_ingest.SUPPORTED_SUFFIXES
    ]
    if unsupported:
        preview = ", ".join(unsupported[:10])
        more = f" (+{len(unsupported) - 10} more)" if len(unsupported) > 10 else ""
        raise file_ingest.IngestionFailure(
            "batch_scan",
            "data_corpus contains unsupported files; remove them before batch ingestion: "
            f"{preview}{more}",
        )
    if not all_files:
        raise file_ingest.IngestionFailure("batch_scan", f"no documents found in {corpus}")

    file_by_name: dict[str, Path] = {}
    for source in all_files:
        key = source.name.casefold()
        if key in file_by_name:
            raise file_ingest.IngestionFailure(
                "batch_scan",
                "canonical filename is not unique in data_corpus: "
                f"{file_by_name[key]} and {source}",
            )
        file_by_name[key] = source

    manifest_by_file = {
        str(entry["canonical_file"]).casefold(): entry for entry in manifest_entries
    }
    manifest_by_id = {
        str(entry["logical_document_id"]).casefold(): entry
        for entry in manifest_entries
    }
    metadata_by_file = {
        record["canonical_file"].casefold(): record for record in metadata_records
    }

    stale_metadata = [
        record["canonical_file"]
        for record in metadata_records
        if record["canonical_file"].casefold() not in file_by_name
    ]
    if stale_metadata:
        raise file_ingest.IngestionFailure(
            "batch_metadata",
            "metadata references files not present in data_corpus: "
            + ", ".join(stale_metadata),
        )

    redundant_metadata = [
        record["canonical_file"]
        for record in metadata_records
        if record["canonical_file"].casefold() in manifest_by_file
    ]
    if redundant_metadata:
        raise file_ingest.IngestionFailure(
            "batch_metadata",
            "metadata must describe only unregistered files; already registered: "
            + ", ".join(redundant_metadata),
        )

    unregistered = [
        source.name
        for source in all_files
        if source.name.casefold() not in manifest_by_file
        and source.name.casefold() not in metadata_by_file
    ]
    if unregistered:
        example = {
            "canonical_file": unregistered[0],
            "logical_document_id": "UNIQUE_DOCUMENT_ID",
            "title": "文档标题",
            "source_org": "来源机构",
        }
        raise file_ingest.IngestionFailure(
            "batch_metadata",
            "unregistered files require metadata records in --metadata-file: "
            + ", ".join(unregistered)
            + "; example: "
            + json.dumps(example, ensure_ascii=False),
        )

    items: list[BatchItem] = []
    batch_ids: dict[str, str] = {}
    for source in all_files:
        file_key = source.name.casefold()
        manifest_entry = manifest_by_file.get(file_key)
        if manifest_entry is not None:
            logical_id = str(manifest_entry["logical_document_id"])
            items.append(
                BatchItem(
                    source_path=source,
                    logical_document_id=logical_id,
                    title=str(manifest_entry["title"]),
                    source_org=manifest_entry.get("source_org"),
                    registered=True,
                )
            )
            batch_ids[logical_id.casefold()] = source.name
            continue

        metadata = metadata_by_file[file_key]
        logical_id = metadata["logical_document_id"]
        logical_key = logical_id.casefold()
        old_entry = manifest_by_id.get(logical_key)
        if old_entry is not None:
            old_file = str(old_entry["canonical_file"])
            if old_file.casefold() in file_by_name:
                raise file_ingest.IngestionFailure(
                    "batch_identity",
                    f"logical_document_id {logical_id!r} would select both {old_file!r} "
                    f"and {source.name!r}; remove the obsolete source before replacement",
                )
            if not replace_existing:
                raise file_ingest.IngestionFailure(
                    "batch_identity",
                    f"logical_document_id {logical_id!r} already belongs to {old_file!r}; "
                    "pass --replace-existing for an explicit replacement",
                )
        if logical_key in batch_ids:
            raise file_ingest.IngestionFailure(
                "batch_identity",
                f"logical_document_id {logical_id!r} is selected by both "
                f"{batch_ids[logical_key]!r} and {source.name!r}",
            )
        batch_ids[logical_key] = source.name
        items.append(
            BatchItem(
                source_path=source,
                logical_document_id=logical_id,
                title=metadata["title"],
                source_org=metadata.get("source_org"),
                registered=False,
            )
        )
    return items


def _file_namespace(
    item: BatchItem, batch_args: argparse.Namespace, *, apply: bool
) -> argparse.Namespace:
    argv: list[str] = []
    if item.registered:
        argv.extend(["--doc", item.logical_document_id])
    else:
        argv.extend(
            [
                "--file",
                str(item.source_path),
                "--logical-document-id",
                item.logical_document_id,
                "--title",
                item.title,
            ]
        )
        if item.source_org is not None:
            argv.extend(["--source-org", item.source_org])
    if batch_args.replace_existing:
        argv.append("--replace-existing")
    argv.append("--apply" if apply else "--dry-run")
    argv.extend(["--manifest", str(batch_args.manifest.resolve())])
    argv.extend(["--baseline-manifest", str(batch_args.baseline_manifest.resolve())])
    argv.extend(["--corpus-dir", str(batch_args.corpus_dir.resolve())])
    parsed = file_ingest.create_parser().parse_args(argv)
    file_ingest._validate_args(parsed)
    return parsed


async def _run_file_locked(
    item: BatchItem,
    batch_args: argparse.Namespace,
    *,
    apply: bool,
) -> dict[str, Any]:
    """Delegate one item while the batch owns the Manifest lock."""

    parsed = _file_namespace(item, batch_args, apply=apply)
    captured = io.StringIO()
    with redirect_stdout(captured):
        return await file_ingest._run_locked(parsed, batch_args.manifest.resolve())


def _failure_row(item: BatchItem, exc: BaseException) -> dict[str, str]:
    stage = exc.stage if isinstance(exc, file_ingest.IngestionFailure) else "unexpected"
    return {
        "logical_document_id": item.logical_document_id,
        "canonical_file": item.canonical_file,
        "failed_stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _emit_summary(summary: dict[str, Any], status: str) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"BATCH_INGESTION_STATUS={status}")


async def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = create_parser().parse_args(argv)
    file_ingest.validate_runtime()
    live_manifest = args.manifest.resolve()

    with file_ingest.manifest_run_lock(live_manifest):
        entries = file_ingest.ensure_live_manifest(
            live_manifest, args.baseline_manifest.resolve()
        )
        metadata = load_batch_metadata(args.metadata_file.resolve())
        items = discover_batch_items(
            corpus_dir=args.corpus_dir,
            manifest_entries=entries,
            metadata_records=metadata,
            replace_existing=args.replace_existing,
        )

        validated: list[dict[str, Any]] = []
        validation_failures: list[dict[str, str]] = []
        for item in items:
            try:
                report = await _run_file_locked(item, args, apply=False)
                validated.append(
                    {
                        "logical_document_id": item.logical_document_id,
                        "canonical_file": item.canonical_file,
                        "document_identity": report["document_identity"],
                        "parent_count": report["parent_count"],
                        "child_count": report["child_count"],
                    }
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                validation_failures.append(_failure_row(item, exc))

        if validation_failures:
            summary = {
                "mode": "apply" if args.apply else "dry-run",
                "total_files": len(items),
                "prevalidated": len(validated),
                "storage_writes_started": False,
                "succeeded": 0,
                "failed": validation_failures,
                "batch_ingestion_status": "FAIL",
            }
            _emit_summary(summary, "FAIL")
            raise file_ingest.IngestionFailure(
                "batch_prevalidate",
                f"{len(validation_failures)} document(s) failed; no storage writes started",
            )

        if not args.apply:
            summary = {
                "mode": "dry-run",
                "total_files": len(items),
                "prevalidated": len(validated),
                "storage_writes_started": False,
                "database_writes": 0,
                "milvus_writes": 0,
                "documents": validated,
                "batch_ingestion_status": "DRY_RUN",
            }
            _emit_summary(summary, "DRY_RUN")
            return summary

        applied: list[dict[str, Any]] = []
        for item in items:
            try:
                applied.append(await _run_file_locked(item, args, apply=True))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                failure = _failure_row(item, exc)
                summary = {
                    "mode": "apply",
                    "total_files": len(items),
                    "prevalidated": len(validated),
                    "storage_writes_started": True,
                    "succeeded": len(applied),
                    "partial_commit": bool(applied),
                    "failed": [failure],
                    "batch_ingestion_status": "FAIL",
                }
                _emit_summary(summary, "FAIL")
                raise file_ingest.IngestionFailure(
                    "batch_apply",
                    f"apply failed for {item.canonical_file}; "
                    f"{len(applied)} prior document(s) remain committed",
                ) from exc

        dimensions = sorted({int(report["vector_dimension"]) for report in applied})
        summary = {
            "mode": "apply",
            "total_files": len(items),
            "prevalidated": len(validated),
            "storage_writes_started": True,
            "succeeded": len(applied),
            "failed": [],
            "parent_count": sum(int(report["parent_count"]) for report in applied),
            "child_count": sum(int(report["child_count"]) for report in applied),
            "mysql_parent_written": sum(
                int(report["mysql_parent_written"]) for report in applied
            ),
            "mysql_mapping_written": sum(
                int(report["mysql_mapping_written"]) for report in applied
            ),
            "milvus_written": sum(int(report["milvus_written"]) for report in applied),
            "orphan_count": sum(int(report["orphan_count"]) for report in applied),
            "vector_dimensions": dimensions,
            "corpus_version_before": applied[0]["corpus_version_before"],
            "corpus_version_after": applied[-1]["corpus_version_after"],
            "legacy_collection_unchanged": all(
                bool(report["legacy_collection_unchanged"]) for report in applied
            ),
            "backend_restart_required": any(
                bool(report["backend_restart_required"]) for report in applied
            ),
            "documents": [
                {
                    "document_identity": report["document_identity"],
                    "parent_count": report["parent_count"],
                    "child_count": report["child_count"],
                }
                for report in applied
            ],
            "batch_ingestion_status": "PASS",
        }
        _emit_summary(summary, "PASS")
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        asyncio.run(run(argv))
    except file_ingest.IngestionFailure as exc:
        print(f"FAILED_STAGE={exc.stage}", file=sys.stderr)
        print(f"BATCH_INGESTION_STATUS=FAIL: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("FAILED_STAGE=interrupted", file=sys.stderr)
        print("BATCH_INGESTION_STATUS=FAIL: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # defensive CLI boundary; never emit a false PASS
        print("FAILED_STAGE=unexpected", file=sys.stderr)
        print(
            f"BATCH_INGESTION_STATUS=FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
