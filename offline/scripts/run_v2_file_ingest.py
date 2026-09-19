"""One-click, single-file V2 ingestion orchestration.

This module intentionally owns no parser, cleaner, chunker, embedding, MySQL
persistence, or Milvus insertion algorithm.  It coordinates the existing
``build_doc_package.py`` and ``run_v2_doc_sync.py`` entry points, adds safety
gates, and performs read-after-write verification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path(r"D:\AI\Anaconda3\envs\xiaoyi_rag\python.exe")
DEFAULT_CORPUS_DIR = REPO / "data_corpus"
DEFAULT_BASELINE_MANIFEST = REPO / "frozen_assets" / "ingest_manifest_v1.jsonl"
DEFAULT_LIVE_MANIFEST = REPO / "data" / "manifests" / "ingest_manifest_v2_live.jsonl"
BUILD_SCRIPT = REPO / "offline" / "scripts" / "build_doc_package.py"
V2_COLLECTION = "xiaoyi_legal_child_v2"
LEGACY_COLLECTION = "xiaoyi_legal_child"
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".md", ".txt"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# MySQL legal_parent_contract_v1.document_identity is VARCHAR(191).  The
# identity suffix ("@sha256:" + 64 hex characters) consumes 72 characters.
_LOGICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,118}$")
_MAX_MILVUS_QUERY_ROWS = 16384

sys.path.insert(0, str(REPO))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from offline.scripts.run_v2_doc_sync import (  # noqa: E402
    parse_package,
    sync_delete,
    sync_package,
)


class IngestionFailure(RuntimeError):
    """Expected fail-fast error with an explicit pipeline stage."""

    def __init__(self, stage: str, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.stage = stage
        self.exit_code = exit_code


class IngestionArgumentParser(argparse.ArgumentParser):
    """Turn CLI usage errors into the same staged, non-PASS failure contract."""

    def error(self, message: str) -> None:
        raise IngestionFailure("cli", message)


@dataclass(frozen=True)
class Selection:
    source_path: Path
    source_fingerprint: "SourceFingerprint"
    entry: dict[str, Any]
    entries_after: list[dict[str, Any]]
    manifest_changed: bool
    previous_identity: str | None


@dataclass(frozen=True)
class ApplyPreflight:
    version_before: int
    mysql_identities: frozenset[str]
    milvus_identities: frozenset[str]
    legacy_count_before: int | None
    vector_dimension: int
    embedding_service: Any


@dataclass(frozen=True)
class WriteVerification:
    parent_count: int
    mapping_count: int
    milvus_count: int
    orphan_count: int
    version_after: int
    legacy_count_after: int | None


@dataclass(frozen=True)
class SourceFingerprint:
    size: int
    modified_ns: int
    raw_sha256: str


def create_parser() -> argparse.ArgumentParser:
    parser = IngestionArgumentParser(
        description="Raw file -> validated V2 package -> MySQL/Milvus, with read-after-write checks"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--doc", help="logical_document_id or canonical_file already in live Manifest"
    )
    source.add_argument("--file", type=Path, help="new/replacement file inside data_corpus")
    parser.add_argument("--logical-document-id")
    parser.add_argument("--title")
    parser.add_argument("--source-org")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="explicitly replace an existing logical document when its SHA/file metadata changes",
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="build and validate only")
    authorization.add_argument("--apply", action="store_true", help="authorize MySQL/Milvus writes")
    parser.add_argument(
        "--package-out", type=Path, help="optionally retain the validated package JSON"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_LIVE_MANIFEST)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    return parser


def _norm_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def fingerprint_source(path: Path) -> SourceFingerprint:
    """Hash raw bytes only as a TOCTOU guard, never as normalized_sha256."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise IngestionFailure("source_changed", f"source changed while hashing: {path}")
    return SourceFingerprint(after.st_size, after.st_mtime_ns, digest.hexdigest())


def require_source_unchanged(path: Path, expected: SourceFingerprint) -> None:
    actual = fingerprint_source(path)
    if actual != expected:
        raise IngestionFailure(
            "source_changed",
            "source file changed during ingestion; discard the package and retry: "
            f"{path}",
        )


@contextmanager
def manifest_run_lock(manifest_path: Path):
    """Hold a process-level lock so concurrent runs cannot lose Manifest updates."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production runtime is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise IngestionFailure(
                "manifest_lock",
                f"another V2 ingestion is already using {manifest_path}",
            ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - production runtime is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_runtime(executable: str | Path | None = None) -> None:
    actual = Path(executable or sys.executable)
    if _norm_path(actual) != _norm_path(EXPECTED_PYTHON):
        raise IngestionFailure(
            "runtime",
            f"must use {EXPECTED_PYTHON}; current interpreter is {actual.resolve()}",
        )
    if actual.resolve().parent.name.casefold() != "xiaoyi_rag":
        raise IngestionFailure("runtime", "Python interpreter is not inside xiaoyi_rag")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.lower()))


def _manifest_identity(entry: dict[str, Any]) -> str:
    return f"{entry['logical_document_id']}@sha256:{entry['normalized_sha256'].lower()}"


def _validate_manifest_entry(
    entry: Any,
    *,
    line_no: int,
    require_hash: bool = True,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise IngestionFailure("manifest", f"line {line_no} must be a JSON object")
    for field in ("logical_document_id", "title", "canonical_file", "format"):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise IngestionFailure("manifest", f"line {line_no} missing non-empty {field}")
    logical_id = entry["logical_document_id"]
    if not _LOGICAL_ID_RE.fullmatch(logical_id):
        raise IngestionFailure("manifest", f"line {line_no} has unsafe logical_document_id")
    fmt = entry["format"].lower().lstrip(".")
    if f".{fmt}" not in SUPPORTED_SUFFIXES:
        raise IngestionFailure("manifest", f"line {line_no} has unsupported format: {fmt}")
    if Path(entry["canonical_file"]).suffix.lower() != f".{fmt}":
        raise IngestionFailure("manifest", f"line {line_no} format does not match canonical_file")
    if entry.get("canonical") is not True:
        raise IngestionFailure("manifest", f"line {line_no} must have canonical=true")
    if entry.get("cleaning_status") != "PASS":
        raise IngestionFailure("manifest", f"line {line_no} must have cleaning_status=PASS")
    if require_hash and not _is_sha256(entry.get("normalized_sha256")):
        raise IngestionFailure("manifest", f"line {line_no} lacks verified normalized_sha256")
    return dict(entry)


def load_manifest(path: Path, *, require_hash: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        raise IngestionFailure("manifest", f"Manifest not found: {path}")
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IngestionFailure(
                        "manifest", f"invalid JSON at {path}:{line_no}: {exc}"
                    ) from exc
                entries.append(
                    _validate_manifest_entry(raw, line_no=line_no, require_hash=require_hash)
                )
    except OSError as exc:
        raise IngestionFailure("manifest", f"cannot read Manifest {path}: {exc}") from exc
    if not entries:
        raise IngestionFailure("manifest", f"Manifest has no records: {path}")

    logical_ids: dict[str, int] = {}
    canonical_files: dict[str, int] = {}
    identities: dict[str, int] = {}
    for line_no, entry in enumerate(entries, 1):
        logical_key = entry["logical_document_id"].casefold()
        file_key = entry["canonical_file"].casefold()
        identity_key = _manifest_identity(entry).casefold() if require_hash else ""
        for label, key, seen in (
            ("logical_document_id", logical_key, logical_ids),
            ("canonical_file", file_key, canonical_files),
            ("document_identity", identity_key, identities),
        ):
            if not key:
                continue
            if key in seen:
                raise IngestionFailure(
                    "manifest",
                    f"duplicate {label} at lines {seen[key]} and {line_no}",
                )
            seen[key] = line_no
    return entries


def _write_manifest_temp(entries: Sequence[dict[str, Any]], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def ensure_live_manifest(live_path: Path, baseline_path: Path) -> list[dict[str, Any]]:
    if live_path.exists():
        return load_manifest(live_path)
    baseline_entries = load_manifest(baseline_path)
    temp_path: Path | None = None
    try:
        temp_path = _write_manifest_temp(baseline_entries, live_path)
        os.replace(temp_path, live_path)
        temp_path = None
    except OSError as exc:
        raise IngestionFailure("manifest", f"cannot initialize live Manifest: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return load_manifest(live_path)


def _resolve_corpus_file(corpus_dir: Path, canonical_file: str) -> Path:
    matches = [
        path.resolve()
        for path in corpus_dir.rglob(Path(canonical_file).name)
        if path.is_file()
    ]
    if not matches:
        raise IngestionFailure("file", f"canonical file not found in data_corpus: {canonical_file}")
    if len(matches) > 1:
        raise IngestionFailure(
            "file", f"ambiguous canonical filename in data_corpus: {canonical_file}"
        )
    return validate_source_file(matches[0], corpus_dir)


def validate_source_file(path: Path, corpus_dir: Path) -> Path:
    corpus = corpus_dir.resolve()
    source = path.resolve()
    if not source.is_file():
        raise IngestionFailure("file", f"source file not found: {source}")
    try:
        source.relative_to(corpus)
    except ValueError as exc:
        raise IngestionFailure("file", f"source file must be inside {corpus}") from exc
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise IngestionFailure("file", f"unsupported format {source.suffix}; allowed: {allowed}")
    same_name = [item.resolve() for item in corpus.rglob(source.name) if item.is_file()]
    if same_name != [source]:
        raise IngestionFailure(
            "file", f"canonical filename is not unique in data_corpus: {source.name}"
        )
    return source


def compute_normalized_sha256(source: Path) -> str:
    """Reuse the established parser/cleaner hash contract used by the frozen Manifest."""
    from modules.ingestion.document_cleaning import clean_parsed_document
    from modules.ingestion.document_parsing import DocumentParseError, parse_document

    try:
        parsed = parse_document(source)
    except DocumentParseError as exc:
        raise IngestionFailure("parser", f"{exc.code}: {exc}") from exc
    except Exception as exc:
        raise IngestionFailure("parser", f"{type(exc).__name__}: {exc}") from exc
    try:
        cleaned = clean_parsed_document(parsed)
    except Exception as exc:
        raise IngestionFailure(
            "cleaner", f"Cleaner hash pass failed: {type(exc).__name__}: {exc}"
        ) from exc
    cleaning = cleaned.metadata.get("cleaning")
    if not isinstance(cleaning, dict):
        raise IngestionFailure("cleaner", "Cleaner did not return cleaning metadata")
    if cleaning.get("quality_status") != "PASS":
        reasons = cleaning.get("quality_reasons") or ["Cleaner did not return PASS"]
        raise IngestionFailure("cleaner", "; ".join(str(reason) for reason in reasons))
    normalized_sha = cleaning.get("normalized_sha256")
    if not _is_sha256(normalized_sha):
        raise IngestionFailure("cleaner", "Cleaner did not return a valid normalized_sha256")
    return str(normalized_sha).lower()


def select_document(args: argparse.Namespace, entries: list[dict[str, Any]]) -> Selection:
    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise IngestionFailure("file", f"data_corpus directory not found: {corpus_dir}")

    if args.doc:
        if args.logical_document_id or args.title or args.source_org or args.file:
            raise IngestionFailure("cli", "--doc cannot be combined with new-file metadata")
        matches = [
            (index, entry)
            for index, entry in enumerate(entries)
            if entry["logical_document_id"] == args.doc
            or Path(entry["canonical_file"]).name == args.doc
        ]
        if len(matches) != 1:
            raise IngestionFailure(
                "manifest", f"Manifest document not found or ambiguous: {args.doc}"
            )
        index, old_entry = matches[0]
        source = _resolve_corpus_file(corpus_dir, old_entry["canonical_file"])
        source_fingerprint = fingerprint_source(source)
        normalized_sha = compute_normalized_sha256(source)
        require_source_unchanged(source, source_fingerprint)
        previous_identity = _manifest_identity(old_entry)
        if normalized_sha != old_entry["normalized_sha256"].lower() and not args.replace_existing:
            raise IngestionFailure(
                "document_identity",
                "source content changed; pass --replace-existing to replace old identity "
                f"{previous_identity}",
            )
        entry = dict(old_entry)
        entry["normalized_sha256"] = normalized_sha
        entry["cleaning_status"] = "PASS"
        changed = entry != old_entry
        entries_after = [dict(item) for item in entries]
        entries_after[index] = entry
        return Selection(
            source_path=source,
            source_fingerprint=source_fingerprint,
            entry=entry,
            entries_after=entries_after,
            manifest_changed=changed,
            previous_identity=previous_identity if changed else None,
        )

    if not args.logical_document_id or not args.title:
        raise IngestionFailure("cli", "--file requires --logical-document-id and --title")
    if not _LOGICAL_ID_RE.fullmatch(args.logical_document_id):
        raise IngestionFailure(
            "cli",
            "--logical-document-id must use 1-119 ASCII letters, digits, dots, "
            "underscores, or hyphens",
        )
    source = validate_source_file(args.file, corpus_dir)
    source_fingerprint = fingerprint_source(source)
    normalized_sha = compute_normalized_sha256(source)
    require_source_unchanged(source, source_fingerprint)
    id_match = next(
        (
            (index, entry)
            for index, entry in enumerate(entries)
            if entry["logical_document_id"].casefold() == args.logical_document_id.casefold()
        ),
        None,
    )
    file_match = next(
        (
            (index, entry)
            for index, entry in enumerate(entries)
            if entry["canonical_file"].casefold() == source.name.casefold()
        ),
        None,
    )
    if id_match and file_match and id_match[0] != file_match[0]:
        raise IngestionFailure(
            "manifest_conflict",
            "logical_document_id and canonical_file belong to different Manifest records",
        )
    if file_match and not id_match:
        raise IngestionFailure(
            "manifest_conflict",
            f"canonical_file already belongs to {file_match[1]['logical_document_id']}",
        )
    if id_match and not args.replace_existing:
        raise IngestionFailure(
            "duplicate_document",
            "logical_document_id already exists; use --doc for an exact retry or "
            f"--replace-existing to replace {_manifest_identity(id_match[1])}",
        )

    previous_identity: str | None = None
    if id_match:
        index, old_entry = id_match
        entry = dict(old_entry)
        previous_identity = _manifest_identity(old_entry)
    else:
        index = len(entries)
        entry = {}
    entry.update(
        {
            "logical_document_id": args.logical_document_id,
            "title": args.title.strip(),
            "canonical_file": source.name,
            "format": source.suffix.lower().lstrip("."),
            "cleaning_status": "PASS",
            "canonical": True,
            "normalized_sha256": normalized_sha,
        }
    )
    if args.source_org is not None:
        if not args.source_org.strip():
            raise IngestionFailure("cli", "--source-org must not be blank")
        entry["source_org"] = args.source_org.strip()
    entry = _validate_manifest_entry(entry, line_no=index + 1)
    entries_after = [dict(item) for item in entries]
    if id_match:
        entries_after[index] = entry
    else:
        entries_after.append(entry)
    return Selection(
        source_path=source,
        source_fingerprint=source_fingerprint,
        entry=entry,
        entries_after=entries_after,
        manifest_changed=True,
        previous_identity=previous_identity,
    )


def _write_json_atomic(payload: dict[str, Any], target: Path) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def run_package_builder(
    *,
    entry: dict[str, Any],
    manifest_path: Path,
    corpus_dir: Path,
    package_out: Path | None = None,
) -> tuple[dict[str, Any], Any]:
    with tempfile.TemporaryDirectory(prefix="xiaoyi-v2-package-") as raw_temp_dir:
        output_path = Path(raw_temp_dir) / "document_package.json"
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        command = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--doc",
            entry["logical_document_id"],
            "--manifest",
            str(manifest_path),
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown build failure").strip()
            if "stage=cleaner" in detail or "CLEAN_FAIL" in detail:
                stage = "cleaner"
            elif completed.returncode == 3 or "Chunk Contract" in detail:
                stage = "chunk_contract"
            elif "PARSE_ERROR" in detail or "解析" in detail:
                stage = "parser"
            else:
                stage = "package_build"
            raise IngestionFailure(stage, detail, exit_code=completed.returncode or 2)
        if not output_path.is_file():
            raise IngestionFailure("package_build", "builder succeeded without a package file")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestionFailure(
                "package_contract", f"cannot read generated package: {exc}"
            ) from exc

    expected_identity = _manifest_identity(entry)
    if raw.get("document_identity") != expected_identity:
        raise IngestionFailure(
            "package_contract",
            "document_identity mismatch: "
            f"expected {expected_identity}, got {raw.get('document_identity')}",
        )
    if raw.get("source_file") != entry["canonical_file"]:
        raise IngestionFailure("package_contract", "package source_file does not match Manifest")
    if raw.get("identity_sha256_kind") != "normalized_sha256":
        raise IngestionFailure("package_contract", "builder did not use verified normalized_sha256")
    try:
        package = parse_package(raw)
    except (TypeError, ValueError) as exc:
        raise IngestionFailure("package_contract", str(exc)) from exc
    child_ids = [child.child_id for parent in package.parents for child in parent.children]
    if not child_ids or any(child_id is None for child_id in child_ids):
        raise IngestionFailure("package_contract", "every one-click V2 child must retain child_id")
    if len(set(child_ids)) != len(child_ids):
        raise IngestionFailure("package_contract", "generated package contains duplicate child_id")
    if package_out is not None:
        try:
            _write_json_atomic(raw, package_out)
        except OSError as exc:
            raise IngestionFailure("package_output", f"cannot write --package-out: {exc}") from exc
    return raw, package


def _resolve_setting_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


async def run_apply_preflight(
    *,
    logical_document_id: str,
    live_manifest: Path,
) -> ApplyPreflight:
    from sqlalchemy import text

    from modules.core.config import get_settings
    from modules.database.session import get_session_factory
    from modules.embeddings.local_embedding import LocalEmbeddingService

    settings = get_settings()
    if settings.retrieval_contract != "v2":
        raise IngestionFailure("config", "RETRIEVAL_CONTRACT must be v2")
    if settings.retrieval_v2_collection != V2_COLLECTION:
        raise IngestionFailure(
            "config", f"RETRIEVAL_V2_COLLECTION must be {V2_COLLECTION}"
        )
    configured_manifest = _norm_path(_resolve_setting_path(settings.canonical_manifest_path))
    if configured_manifest != _norm_path(live_manifest):
        raise IngestionFailure(
            "config",
            f"CANONICAL_MANIFEST_PATH must point to {live_manifest.resolve()}",
        )
    if settings.hot_update_enabled:
        raise IngestionFailure("config", "HOT_UPDATE_ENABLED must be false for V2 ingestion")

    model_path = Path(settings.embedding_model_path)
    if not model_path.is_absolute():
        model_path = REPO / model_path
    model_path = model_path.resolve()
    required_model_files = (model_path / "config.json", model_path / "tokenizer.json")
    if not model_path.is_dir() or any(not path.is_file() for path in required_model_files):
        raise IngestionFailure("embedding_model", f"BGE-M3 model is incomplete: {model_path}")
    try:
        for path in required_model_files:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError as exc:
        raise IngestionFailure("embedding_model", f"BGE-M3 model is not readable: {exc}") from exc

    prefix = f"{logical_document_id}@sha256:"
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
            await session.execute(text("SELECT 1 FROM legal_parent_contract_v1 LIMIT 0"))
            await session.execute(text("SELECT 1 FROM legal_child_parent_map_v1 LIMIT 0"))
            result = await session.execute(
                text("SELECT DISTINCT document_identity FROM legal_parent_contract_v1")
            )
            mysql_identities = frozenset(
                str(row[0]) for row in result.all() if str(row[0]).startswith(prefix)
            )
            result = await session.execute(
                text(
                    "SELECT meta_value FROM corpus_meta "
                    "WHERE meta_key = 'corpus_version'"
                )
            )
            row = result.first()
            if row is None:
                raise ValueError("corpus_meta.corpus_version is missing")
            version_before = int(str(row[0]))
            if version_before < 0:
                raise ValueError("corpus_version must be non-negative")
    except IngestionFailure:
        raise
    except Exception as exc:
        raise IngestionFailure("mysql_preflight", f"{type(exc).__name__}: {exc}") from exc

    def _milvus_preflight() -> tuple[frozenset[str], int | None]:
        from pymilvus import Collection, utility

        from modules.rag.retrieval_v2_adapter import validate_v2_collection_contract

        validate_v2_collection_contract.cache_clear()
        validate_v2_collection_contract(V2_COLLECTION)
        collection = Collection(V2_COLLECTION)
        collection.load()
        expr = f"document_identity like {json.dumps(prefix + '%', ensure_ascii=False)}"
        try:
            rows = collection.query(
                expr=expr,
                output_fields=["document_identity"],
                limit=_MAX_MILVUS_QUERY_ROWS,
                consistency_level="Strong",
            )
        except TypeError:
            rows = collection.query(
                expr=expr,
                output_fields=["document_identity"],
                limit=_MAX_MILVUS_QUERY_ROWS,
            )
        identities = frozenset(str(row["document_identity"]) for row in rows)
        legacy_count = (
            int(Collection(LEGACY_COLLECTION).num_entities)
            if utility.has_collection(LEGACY_COLLECTION)
            else None
        )
        return identities, legacy_count

    try:
        milvus_identities, legacy_count = await asyncio.to_thread(_milvus_preflight)
    except Exception as exc:
        raise IngestionFailure("milvus_preflight", f"{type(exc).__name__}: {exc}") from exc

    try:
        embedding_service = LocalEmbeddingService()
        vector_dimension = await asyncio.to_thread(lambda: embedding_service.dimension)
    except Exception as exc:
        raise IngestionFailure("embedding_model", f"{type(exc).__name__}: {exc}") from exc
    if vector_dimension != 1024:
        raise IngestionFailure(
            "embedding_model", f"BGE-M3 vector dimension must be 1024, got {vector_dimension}"
        )
    return ApplyPreflight(
        version_before=version_before,
        mysql_identities=mysql_identities,
        milvus_identities=milvus_identities,
        legacy_count_before=legacy_count,
        vector_dimension=vector_dimension,
        embedding_service=embedding_service,
    )


def determine_old_identities(
    *,
    new_identity: str,
    preflight: ApplyPreflight,
    replace_existing: bool,
) -> list[str]:
    if preflight.mysql_identities != preflight.milvus_identities:
        raise IngestionFailure(
            "document_identity",
            "MySQL/Milvus document_identity sets disagree: "
            f"mysql={sorted(preflight.mysql_identities)} "
            f"milvus={sorted(preflight.milvus_identities)}",
        )
    old_identities = sorted(preflight.mysql_identities - {new_identity})
    if old_identities and not replace_existing:
        raise IngestionFailure(
            "document_identity",
            "new SHA would leave old version(s); pass --replace-existing. Old identities: "
            + ", ".join(old_identities),
        )
    return old_identities


def _decode_json_column(value: Any, field_name: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise IngestionFailure("post_verify", f"invalid MySQL {field_name} JSON") from exc
    raise IngestionFailure("post_verify", f"invalid MySQL {field_name} value")


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def verify_write(
    *,
    package: Any,
    old_identities: Sequence[str],
    preflight: ApplyPreflight,
) -> WriteVerification:
    from sqlalchemy import bindparam, text

    from modules.database.session import get_session_factory

    expected_child_ids = {
        child.child_id for parent in package.parents for child in parent.children
    }
    if None in expected_child_ids:
        raise IngestionFailure("post_verify", "package lost a child_id before verification")
    logical_id, marker, _sha = package.document_identity.partition("@sha256:")
    if not marker or not logical_id:
        raise IngestionFailure("post_verify", "package has an invalid document_identity")
    identity_prefix = f"{logical_id}@sha256:"
    expected_parent_rows = Counter(
        (
            parent.content,
            _stable_json(list(parent.section_path)),
            _stable_json(parent.metadata),
        )
        for parent in package.parents
    )
    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                text(
                    "SELECT external_parent_id, content, section_path, metadata "
                    "FROM legal_parent_contract_v1 WHERE document_identity = :ident"
                ),
                {"ident": package.document_identity},
            )
            parent_rows = result.all()
            mysql_parent_ids: set[str] = set()
            actual_parent_rows: Counter[tuple[str, str, str]] = Counter()
            for row in parent_rows:
                parent_id, content, section_path, metadata = row
                if not isinstance(parent_id, str) or len(parent_id) != 64:
                    raise IngestionFailure("post_verify", "invalid MySQL external_parent_id")
                if not isinstance(content, str) or not content.strip():
                    raise IngestionFailure("post_verify", f"empty MySQL parent {parent_id}")
                decoded_section = _decode_json_column(section_path, "section_path")
                if not isinstance(decoded_section, list):
                    raise IngestionFailure("post_verify", f"invalid section_path for {parent_id}")
                decoded_metadata = _decode_json_column(metadata, "metadata")
                if not isinstance(decoded_metadata, dict):
                    raise IngestionFailure("post_verify", f"invalid metadata for {parent_id}")
                mysql_parent_ids.add(parent_id)
                actual_parent_rows[
                    (content, _stable_json(decoded_section), _stable_json(decoded_metadata))
                ] += 1
            result = await session.execute(
                text(
                    "SELECT m.child_id, m.external_parent_id "
                    "FROM legal_child_parent_map_v1 AS m "
                    "JOIN legal_parent_contract_v1 AS p "
                    "ON p.external_parent_id = m.external_parent_id "
                    "WHERE p.document_identity = :ident"
                ),
                {"ident": package.document_identity},
            )
            mysql_mapping = {str(row[0]): str(row[1]) for row in result.all()}
            result = await session.execute(
                text("SELECT DISTINCT document_identity FROM legal_parent_contract_v1")
            )
            mysql_logical_identities = {
                str(row[0])
                for row in result.all()
                if str(row[0]).startswith(identity_prefix)
            }
            if old_identities:
                statement = text(
                    "SELECT COUNT(*) FROM legal_parent_contract_v1 "
                    "WHERE document_identity IN :identities"
                ).bindparams(bindparam("identities", expanding=True))
                result = await session.execute(statement, {"identities": list(old_identities)})
                if int(result.first()[0]) != 0:
                    raise IngestionFailure(
                        "post_verify", "old MySQL document_identity still exists"
                    )
            result = await session.execute(
                text(
                    "SELECT meta_value FROM corpus_meta "
                    "WHERE meta_key = 'corpus_version'"
                )
            )
            version_row = result.first()
            if version_row is None:
                raise IngestionFailure("post_verify", "corpus_version disappeared")
            version_after = int(str(version_row[0]))
    except IngestionFailure:
        raise
    except Exception as exc:
        raise IngestionFailure("post_verify", f"MySQL verification failed: {exc}") from exc

    def _milvus_verify() -> tuple[list[dict[str, Any]], set[str], int | None]:
        from pymilvus import Collection, utility

        collection = Collection(V2_COLLECTION)
        expr = f"document_identity == {json.dumps(package.document_identity, ensure_ascii=False)}"
        try:
            rows = collection.query(
                expr=expr,
                output_fields=[
                    "child_id",
                    "external_parent_id",
                    "document_identity",
                    "source_file",
                ],
                limit=_MAX_MILVUS_QUERY_ROWS,
                consistency_level="Strong",
            )
        except TypeError:
            rows = collection.query(
                expr=expr,
                output_fields=[
                    "child_id",
                    "external_parent_id",
                    "document_identity",
                    "source_file",
                ],
                limit=_MAX_MILVUS_QUERY_ROWS,
            )
        logical_expr = (
            f"document_identity like "
            f"{json.dumps(identity_prefix + '%', ensure_ascii=False)}"
        )
        try:
            identity_rows = collection.query(
                expr=logical_expr,
                output_fields=["document_identity"],
                limit=_MAX_MILVUS_QUERY_ROWS,
                consistency_level="Strong",
            )
        except TypeError:
            identity_rows = collection.query(
                expr=logical_expr,
                output_fields=["document_identity"],
                limit=_MAX_MILVUS_QUERY_ROWS,
            )
        for identity in old_identities:
            old_expr = f"document_identity == {json.dumps(identity, ensure_ascii=False)}"
            try:
                residual = collection.query(
                    expr=old_expr,
                    output_fields=["child_id"],
                    limit=1,
                    consistency_level="Strong",
                )
            except TypeError:
                residual = collection.query(expr=old_expr, output_fields=["child_id"], limit=1)
            if residual:
                raise IngestionFailure(
                    "post_verify", f"old Milvus document_identity still exists: {identity}"
                )
        legacy_count = (
            int(Collection(LEGACY_COLLECTION).num_entities)
            if utility.has_collection(LEGACY_COLLECTION)
            else None
        )
        logical_identities = {
            str(row["document_identity"])
            for row in identity_rows
            if str(row["document_identity"]).startswith(identity_prefix)
        }
        return list(rows), logical_identities, legacy_count

    try:
        milvus_rows, milvus_logical_identities, legacy_count_after = await asyncio.to_thread(
            _milvus_verify
        )
    except IngestionFailure:
        raise
    except Exception as exc:
        raise IngestionFailure("post_verify", f"Milvus verification failed: {exc}") from exc

    milvus_child_ids: set[str] = set()
    orphan_count = 0
    for row in milvus_rows:
        child_id = str(row.get("child_id", ""))
        parent_id = str(row.get("external_parent_id", ""))
        if row.get("document_identity") != package.document_identity:
            raise IngestionFailure("post_verify", f"Milvus identity mismatch for child {child_id}")
        if row.get("source_file") != package.source_file:
            raise IngestionFailure(
                "post_verify", f"Milvus source_file mismatch for child {child_id}"
            )
        if parent_id not in mysql_parent_ids:
            orphan_count += 1
        if mysql_mapping.get(child_id) != parent_id:
            raise IngestionFailure("post_verify", f"mapping mismatch for child {child_id}")
        milvus_child_ids.add(child_id)

    expected_parent_count = len(package.parents)
    if len(parent_rows) != expected_parent_count:
        raise IngestionFailure(
            "post_verify",
            "MySQL parent count mismatch: "
            f"expected {expected_parent_count}, got {len(parent_rows)}",
        )
    if actual_parent_rows != expected_parent_rows:
        raise IngestionFailure("post_verify", "MySQL parent content/metadata differs from package")
    if set(mysql_mapping) != expected_child_ids:
        raise IngestionFailure("post_verify", "MySQL child mapping set differs from package")
    if milvus_child_ids != expected_child_ids:
        raise IngestionFailure("post_verify", "Milvus child set differs from package")
    if orphan_count:
        raise IngestionFailure("post_verify", f"Milvus orphan count is {orphan_count}")
    expected_logical_identities = {package.document_identity}
    if mysql_logical_identities != expected_logical_identities:
        raise IngestionFailure(
            "post_verify",
            "MySQL logical document identity set is not exact: "
            f"{sorted(mysql_logical_identities)}",
        )
    if milvus_logical_identities != expected_logical_identities:
        raise IngestionFailure(
            "post_verify",
            "Milvus logical document identity set is not exact: "
            f"{sorted(milvus_logical_identities)}",
        )
    if version_after <= preflight.version_before:
        raise IngestionFailure(
            "post_verify",
            f"corpus_version did not advance: {preflight.version_before} -> {version_after}",
        )
    if legacy_count_after != preflight.legacy_count_before:
        raise IngestionFailure(
            "post_verify",
            "Legacy collection changed: "
            f"{preflight.legacy_count_before} -> {legacy_count_after}",
        )
    return WriteVerification(
        parent_count=len(parent_rows),
        mapping_count=len(mysql_mapping),
        milvus_count=len(milvus_rows),
        orphan_count=orphan_count,
        version_after=version_after,
        legacy_count_after=legacy_count_after,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.doc is None and (args.logical_document_id is None or args.title is None):
        raise IngestionFailure("cli", "--file requires --logical-document-id and --title")
    if args.doc is not None and (args.logical_document_id or args.title or args.source_org):
        raise IngestionFailure("cli", "--doc cannot be combined with new-file metadata")


async def _run_locked(args: argparse.Namespace, live_manifest: Path) -> dict[str, Any]:
    entries = ensure_live_manifest(live_manifest, args.baseline_manifest.resolve())
    selection = select_document(args, entries)

    staged_manifest: Path | None = None
    pending_live_manifest: Path | None = None
    try:
        staged_manifest = _write_manifest_temp(selection.entries_after, live_manifest)
        raw_package, package = run_package_builder(
            entry=selection.entry,
            manifest_path=staged_manifest,
            corpus_dir=args.corpus_dir.resolve(),
            package_out=None,
        )
        require_source_unchanged(selection.source_path, selection.source_fingerprint)
        if args.package_out is not None:
            try:
                _write_json_atomic(raw_package, args.package_out)
            except OSError as exc:
                raise IngestionFailure(
                    "package_output", f"cannot write --package-out: {exc}"
                ) from exc
        parent_count = len(package.parents)
        child_count = sum(len(parent.children) for parent in package.parents)
        if not args.apply:
            report = {
                "mode": "dry-run",
                "document_identity": package.document_identity,
                "parent_count": parent_count,
                "child_count": child_count,
                "normalized_sha256": selection.entry["normalized_sha256"],
                "chunk_metadata_contract": "PASS",
                "manifest_updated": False,
                "database_writes": 0,
                "milvus_writes": 0,
                "ingestion_status": "DRY_RUN",
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("INGESTION_STATUS=DRY_RUN")
            return report

        if selection.manifest_changed:
            pending_live_manifest = _write_manifest_temp(selection.entries_after, live_manifest)
        preflight = await run_apply_preflight(
            logical_document_id=selection.entry["logical_document_id"],
            live_manifest=live_manifest,
        )
        old_identities = determine_old_identities(
            new_identity=package.document_identity,
            preflight=preflight,
            replace_existing=args.replace_existing,
        )
        require_source_unchanged(selection.source_path, selection.source_fingerprint)
        try:
            sync_stats = await sync_package(
                package,
                V2_COLLECTION,
                embedding_service=preflight.embedding_service,
            )
        except Exception as exc:
            raise IngestionFailure(
                "write", f"V2 document sync failed: {type(exc).__name__}: {exc}"
            ) from exc
        insert_stats = sync_stats.get("insert") or {}
        if (
            insert_stats.get("mysql_parent_inserted") != parent_count
            or insert_stats.get("mysql_map_inserted") != child_count
            or insert_stats.get("milvus_inserted") != child_count
        ):
            raise IngestionFailure("write", f"sync statistics mismatch: {sync_stats}")
        for old_identity in old_identities:
            try:
                await sync_delete(old_identity, V2_COLLECTION)
            except Exception as exc:
                raise IngestionFailure(
                    "replace_old_identity",
                    f"failed to delete {old_identity}: {type(exc).__name__}: {exc}",
                ) from exc
        verification = await verify_write(
            package=package,
            old_identities=old_identities,
            preflight=preflight,
        )
        require_source_unchanged(selection.source_path, selection.source_fingerprint)
        if pending_live_manifest is not None:
            try:
                os.replace(pending_live_manifest, live_manifest)
                pending_live_manifest = None
            except OSError as exc:
                raise IngestionFailure(
                    "manifest_commit", f"cannot commit live Manifest: {exc}"
                ) from exc
        report = {
            "mode": "apply",
            "document_identity": package.document_identity,
            "parent_count": parent_count,
            "child_count": child_count,
            "mysql_parent_written": verification.parent_count,
            "mysql_mapping_written": verification.mapping_count,
            "milvus_written": verification.milvus_count,
            "orphan_count": verification.orphan_count,
            "mysql_parent_complete": True,
            "mysql_child_mapping_complete": True,
            "external_parent_lookup": "PASS",
            "vector_dimension": preflight.vector_dimension,
            "corpus_version_before": preflight.version_before,
            "corpus_version_after": verification.version_after,
            "old_identities_removed": old_identities,
            "legacy_collection": LEGACY_COLLECTION,
            "legacy_count_before": preflight.legacy_count_before,
            "legacy_count_after": verification.legacy_count_after,
            "legacy_collection_unchanged": True,
            "chunk_metadata_contract": "PASS",
            "manifest_updated": selection.manifest_changed,
            "manifest_path": str(live_manifest),
            "backend_restart_required": selection.manifest_changed,
            "ingestion_status": "PASS",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("INGESTION_STATUS=PASS")
        return report
    finally:
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)
        if pending_live_manifest is not None:
            pending_live_manifest.unlink(missing_ok=True)


async def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = create_parser().parse_args(argv)
    _validate_args(args)
    validate_runtime()
    live_manifest = args.manifest.resolve()
    with manifest_run_lock(live_manifest):
        return await _run_locked(args, live_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        asyncio.run(run(argv))
    except IngestionFailure as exc:
        print(f"FAILED_STAGE={exc.stage}", file=sys.stderr)
        print(f"INGESTION_STATUS=FAIL: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("FAILED_STAGE=interrupted", file=sys.stderr)
        print("INGESTION_STATUS=FAIL: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # defensive CLI boundary; never emit a false PASS
        print("FAILED_STAGE=unexpected", file=sys.stderr)
        print(f"INGESTION_STATUS=FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
