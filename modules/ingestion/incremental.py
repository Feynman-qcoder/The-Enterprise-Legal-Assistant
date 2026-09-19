"""Online, per-file incremental ingestion; offline full rebuild remains separate."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modules.database.models import FaqTab, LegalTab
from modules.database.session import get_session_factory
from modules.embeddings.local_embedding import LocalEmbeddingService
from modules.ingestion.chunking import chunk_pdf_pages_to_parents, split_children_from_parent
from modules.ingestion.document_cleaning import clean_parsed_document
from modules.ingestion.document_parsing import DocumentParseError, parse_document
from modules.ingestion.milvus_sync import (
    delete_faq_incremental,
    delete_legal_incremental,
    insert_faq_incremental,
    insert_legal_incremental,
)
from modules.ingestion.mysql_loaders import FaqWorkbookError, read_faq_excel_rows

logger = logging.getLogger(__name__)

LEGAL_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".docx"})
FAQ_EXTENSIONS = frozenset({".xlsx"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class IngestionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class IngestionResult:
    filename: str
    status: str
    knowledge_type: str
    parents: int = 0
    chunks: int = 0
    rows: int = 0
    parent_ids: list[int] = field(default_factory=list)
    record_ids: list[int] = field(default_factory=list)
    error_code: str | None = None
    message: str | None = None
    size_bytes: int = 0
    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    embedding_ms: float = 0.0
    mysql_ms: float = 0.0
    milvus_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_filename(filename: str) -> str:
    safe = Path(filename or "").name.strip()
    if not safe or safe in {".", ".."}:
        raise IngestionError("INVALID_FILENAME", "文件名无效")
    return safe


def _validate_upload(filename: str, content: bytes, allowed: frozenset[str]) -> tuple[str, str]:
    safe = _safe_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in allowed:
        raise IngestionError("UNSUPPORTED_FILE_TYPE", f"该接口不支持文件类型：{suffix or '<none>'}")
    if not content:
        raise IngestionError("EMPTY_FILE", "上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise IngestionError("FILE_TOO_LARGE", f"单文件不得超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return safe, suffix


def _failed_result(
    filename: str,
    knowledge_type: str,
    started: float,
    exc: Exception,
    *,
    size_bytes: int,
) -> IngestionResult:
    if isinstance(exc, (IngestionError, DocumentParseError, FaqWorkbookError)):
        code = exc.code
        message = str(exc)
    else:
        code = "INGESTION_FAILED"
        message = "文件入库失败"
        logger.exception("incremental ingestion failed for %s", filename)
    return IngestionResult(
        filename=filename or "<unnamed>",
        status="failed",
        knowledge_type=knowledge_type,
        error_code=code,
        message=message,
        size_bytes=size_bytes,
        total_ms=round((time.perf_counter() - started) * 1000, 2),
    )


async def ingest_legal_document(filename: str, content: bytes) -> IngestionResult:
    started = time.perf_counter()
    safe = filename or "<unnamed>"
    try:
        safe, _suffix = _validate_upload(filename, content, LEGAL_EXTENSIONS)
        with TemporaryDirectory(prefix="xiaoyi-upload-") as temp_dir:
            path = Path(temp_dir) / safe
            path.write_bytes(content)
            parse_started = time.perf_counter()
            parsed = await asyncio.to_thread(parse_document, path)
            parse_ms = (time.perf_counter() - parse_started) * 1000

            # Cleaning V1：确定性去噪，保留 legal 语义；返回清洗后 ParsedDocument（复用同一 dataclass）
            parsed = clean_parsed_document(parsed)
            cleaning = parsed.metadata.get("cleaning", {})
            if cleaning.get("quality_status") != "PASS":
                reasons = cleaning.get("quality_reasons") or []
                detail = "; ".join(str(reason) for reason in reasons) or "Cleaner 未返回 PASS"
                raise IngestionError("CLEANING_QUALITY_GATE_FAILED", detail)

        chunk_started = time.perf_counter()
        parents = chunk_pdf_pages_to_parents(list(parsed.segments or (parsed.text,)))
        children_by_parent = [split_children_from_parent(parent.text, index) for index, parent in enumerate(parents)]
        child_count = sum(len(children) for children in children_by_parent)
        if not parents or not child_count:
            raise IngestionError("EMPTY_DOCUMENT", "文档没有产生可入库 Chunk")
        chunk_ms = (time.perf_counter() - chunk_started) * 1000

        embedding = LocalEmbeddingService()
        child_texts = [child.text for children in children_by_parent for child in children]
        embedding_started = time.perf_counter()
        vectors = await embedding.embed_documents(child_texts)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        factory = get_session_factory()
        milvus_ids: list[int] = []
        async with factory() as session:
            try:
                mysql_started = time.perf_counter()
                parent_rows = [
                    LegalTab(
                        source_file=safe,
                        doc_role="parent",
                        parent_id=None,
                        chunk_index=0,
                        title=parent.title,
                        content=parent.text,
                    )
                    for parent in parents
                ]
                session.add_all(parent_rows)
                await session.flush()
                child_rows: list[LegalTab] = []
                for parent_index, chunks in enumerate(children_by_parent):
                    parent_row = parent_rows[parent_index]
                    child_rows.extend(
                        LegalTab(
                            source_file=safe,
                            doc_role="child",
                            parent_id=parent_row.id,
                            chunk_index=child.chunk_index,
                            title=parent_row.title,
                            content=child.text,
                        )
                        for child in chunks
                    )
                session.add_all(child_rows)
                await session.flush()
                mysql_ms = (time.perf_counter() - mysql_started) * 1000
                milvus_ids = [int(row.id) for row in child_rows]

                milvus_started = time.perf_counter()
                await insert_legal_incremental(child_rows, vectors)
                milvus_ms = (time.perf_counter() - milvus_started) * 1000
                commit_started = time.perf_counter()
                await session.commit()
                mysql_ms += (time.perf_counter() - commit_started) * 1000
            except Exception:
                await session.rollback()
                if milvus_ids:
                    try:
                        await delete_legal_incremental(milvus_ids)
                    except Exception:
                        logger.exception("legal Milvus compensation failed: ids=%s", milvus_ids)
                raise

        return IngestionResult(
            filename=safe,
            status="success",
            knowledge_type="legal_document",
            parents=len(parent_rows),
            chunks=len(child_rows),
            parent_ids=[int(row.id) for row in parent_rows],
            record_ids=milvus_ids,
            size_bytes=len(content),
            parse_ms=round(parse_ms, 2),
            chunk_ms=round(chunk_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            mysql_ms=round(mysql_ms, 2),
            milvus_ms=round(milvus_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return _failed_result(safe, "legal_document", started, exc, size_bytes=len(content))


async def ingest_faq_workbook(filename: str, content: bytes) -> IngestionResult:
    started = time.perf_counter()
    safe = filename or "<unnamed>"
    try:
        safe, _suffix = _validate_upload(filename, content, FAQ_EXTENSIONS)
        with TemporaryDirectory(prefix="xiaoyi-faq-") as temp_dir:
            path = Path(temp_dir) / safe
            path.write_bytes(content)
            parse_started = time.perf_counter()
            faq_pairs = await asyncio.to_thread(read_faq_excel_rows, path)
            parse_ms = (time.perf_counter() - parse_started) * 1000

        embedding = LocalEmbeddingService()
        embedding_started = time.perf_counter()
        vectors = await embedding.embed_documents([question for question, _answer in faq_pairs])
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        factory = get_session_factory()
        milvus_ids: list[int] = []
        async with factory() as session:
            try:
                mysql_started = time.perf_counter()
                rows = [
                    FaqTab(question=question, answer=answer, is_high_frequency=True)
                    for question, answer in faq_pairs
                ]
                session.add_all(rows)
                await session.flush()
                mysql_ms = (time.perf_counter() - mysql_started) * 1000
                milvus_ids = [int(row.id) for row in rows]

                milvus_started = time.perf_counter()
                await insert_faq_incremental(rows, vectors)
                milvus_ms = (time.perf_counter() - milvus_started) * 1000
                commit_started = time.perf_counter()
                await session.commit()
                mysql_ms += (time.perf_counter() - commit_started) * 1000
            except Exception:
                await session.rollback()
                if milvus_ids:
                    try:
                        await delete_faq_incremental(milvus_ids)
                    except Exception:
                        logger.exception("FAQ Milvus compensation failed: ids=%s", milvus_ids)
                raise

        return IngestionResult(
            filename=safe,
            status="success",
            knowledge_type="faq",
            rows=len(rows),
            chunks=len(rows),
            record_ids=milvus_ids,
            size_bytes=len(content),
            parse_ms=round(parse_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            mysql_ms=round(mysql_ms, 2),
            milvus_ms=round(milvus_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return _failed_result(safe, "faq", started, exc, size_bytes=len(content))
