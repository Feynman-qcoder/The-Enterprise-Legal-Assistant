from pathlib import Path

import pytest

from modules.ingestion.document_parsing import DocumentParseError, parse_document


REAL_BINARY_DOC = (
    Path(__file__).resolve().parents[2]
    / "项目用到的软件及工具"
    / "mysql软件"
    / "mac系统"
    / "mysql安装步骤(mac版本).doc"
)


def test_real_binary_doc_reports_libreoffice_gate() -> None:
    assert REAL_BINARY_DOC.read_bytes()[:8] == bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1")
    with pytest.raises(DocumentParseError) as raised:
        parse_document(REAL_BINARY_DOC)
    assert raised.value.code == "PARSER_UNAVAILABLE"
