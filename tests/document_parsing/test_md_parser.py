from pathlib import Path

from modules.ingestion.document_parsing import parse_document


def test_markdown_front_matter_is_metadata_not_body() -> None:
    path = Path(__file__).with_name("fixtures") / "policy.md"

    parsed = parse_document(path)

    assert parsed.metadata["doc_id"] == "POLICY-001"
    assert parsed.metadata["title"] == "华辰科技采购合同管理制度"
    assert "ignored_field" not in parsed.metadata
    assert "doc_id:" not in parsed.text
    assert "# 审批规则" in parsed.text
    assert "| >500万 | 总经理 |" in parsed.text


def test_markdown_accepts_utf8_sig(tmp_path: Path) -> None:
    path = tmp_path / "bom.md"
    path.write_text("\ufeff# 中文标题\n\n正文", encoding="utf-8")
    assert parse_document(path).text.startswith("# 中文标题")
