"""Serial multi-file upload endpoints for online incremental knowledge ingestion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas import IngestionBatchResponse, IngestionFileResponse
from modules.ingestion.incremental import (
    MAX_UPLOAD_BYTES,
    IngestionResult,
    ingest_faq_workbook,
    ingest_legal_document,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
MAX_FILES_PER_REQUEST = 20


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, MAX_UPLOAD_BYTES + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            break
    await upload.close()
    return b"".join(chunks)


async def _process_batch(
    files: list[UploadFile],
    ingest: Callable[[str, bytes], Awaitable[IngestionResult]],
) -> IngestionBatchResponse:
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_FILES_PER_REQUEST} 个文件")
    results: list[IngestionFileResponse] = []
    for upload in files:
        content = await _read_limited(upload)
        result = await ingest(upload.filename or "", content)
        results.append(IngestionFileResponse.model_validate(result.to_dict()))
    succeeded = sum(result.status == "success" for result in results)
    return IngestionBatchResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        files=results,
    )


@router.post("/documents/upload", response_model=IngestionBatchResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> IngestionBatchResponse:
    """Upload PDF/TXT/MD/DOCX legal documents; unsupported files fail independently."""
    return await _process_batch(files, ingest_legal_document)


@router.post("/faqs/upload", response_model=IngestionBatchResponse)
async def upload_faqs(files: list[UploadFile] = File(...)) -> IngestionBatchResponse:
    """Upload existing-schema FAQ XLSX workbooks; generic XLSX is rejected by schema validation."""
    return await _process_batch(files, ingest_faq_workbook)
