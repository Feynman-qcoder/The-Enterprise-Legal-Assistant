from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import knowledge
from modules.ingestion.incremental import IngestionResult


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(knowledge.router, prefix="/api")
    return app


def test_documents_batch_keeps_partial_failure(monkeypatch) -> None:
    async def fake_ingest(filename: str, content: bytes) -> IngestionResult:
        if filename == "broken.docx":
            return IngestionResult(
                filename=filename,
                status="failed",
                knowledge_type="legal_document",
                error_code="PARSE_FAILED",
                message="DOCX 文件损坏",
            )
        return IngestionResult(
            filename=filename,
            status="success",
            knowledge_type="legal_document",
            parents=1,
            chunks=1,
        )

    monkeypatch.setattr(knowledge, "ingest_legal_document", fake_ingest)
    with TestClient(_app()) as client:
        response = client.post(
            "/api/knowledge/documents/upload",
            files=[
                ("files", ("good.pdf", b"pdf", "application/pdf")),
                ("files", ("broken.docx", b"broken", "application/octet-stream")),
                ("files", ("good.md", b"markdown", "text/markdown")),
            ],
        )
    body = response.json()
    assert response.status_code == 200
    assert (body["total"], body["succeeded"], body["failed"]) == (3, 2, 1)
    assert [item["status"] for item in body["files"]] == ["success", "failed", "success"]


def test_faq_route_is_separate(monkeypatch) -> None:
    async def fake_ingest(filename: str, content: bytes) -> IngestionResult:
        return IngestionResult(filename=filename, status="success", knowledge_type="faq", rows=1, chunks=1)

    monkeypatch.setattr(knowledge, "ingest_faq_workbook", fake_ingest)
    with TestClient(_app()) as client:
        response = client.post(
            "/api/knowledge/faqs/upload",
            files=[("files", ("faq.xlsx", b"xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))],
        )
    assert response.status_code == 200
    assert response.json()["files"][0]["knowledge_type"] == "faq"
