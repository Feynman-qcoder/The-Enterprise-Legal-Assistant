from pathlib import Path

import pytest

from modules.ingestion.document_parsing import (
    DocumentParseError,
    get_parser_capabilities,
    parse_document,
)


def test_txt_utf8_sig_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "policy.TXT"
    path.write_text("\ufeff华辰科技采购合同管理制度", encoding="utf-8")
    parsed = parse_document(path)
    assert parsed.text == "华辰科技采购合同管理制度"
    assert parsed.metadata["extension"] == ".txt"


def test_pdf_dispatch_uses_existing_fixture() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "中华人民共和国劳动法.pdf"
    parsed = parse_document(path)
    assert parsed.metadata["extension"] == ".pdf"
    assert parsed.text
    assert parsed.segments


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "macro.docm"
    path.write_bytes(b"not executed")
    with pytest.raises(DocumentParseError) as raised:
        parse_document(path)
    assert raised.value.code == "UNSUPPORTED_FILE_TYPE"


def test_capabilities_keep_doc_optional() -> None:
    capabilities = get_parser_capabilities()
    assert all(capabilities[ext]["available"] for ext in (".pdf", ".txt", ".md", ".docx"))
    assert capabilities[".doc"] == {
        "available": False,
        "reason": "LibreOffice unavailable",
    }
