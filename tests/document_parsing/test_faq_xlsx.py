from pathlib import Path

import pytest
from openpyxl import Workbook

from modules.ingestion.mysql_loaders import FaqWorkbookError, read_faq_excel_rows


FIXTURE = Path(__file__).with_name("fixtures") / "faq_test.xlsx"


def test_existing_faq_xlsx_schema_passes() -> None:
    rows = read_faq_excel_rows(FIXTURE)
    assert rows == [
        (
            "FAQ_TEST_001 什么情况下采购合同需要法务审核？",
            "FAQ_TEST_001 按照测试 FAQ，金额超过一百万元需要法务审核。",
        )
    ]


def _save_workbook(path: Path, headers: tuple[str, str], row: tuple[str, str] | None) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(list(headers))
    if row:
        sheet.append(list(row))
    workbook.save(path)


def test_wrong_faq_columns_are_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "generic.xlsx"
    _save_workbook(path, ("合同金额", "审批人"), (">500万", "总经理"))
    with pytest.raises(FaqWorkbookError) as raised:
        read_faq_excel_rows(path)
    assert raised.value.code == "FAQ_SCHEMA_INVALID"


def test_empty_faq_is_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.xlsx"
    _save_workbook(path, ("问题", "答案"), None)
    with pytest.raises(FaqWorkbookError) as raised:
        read_faq_excel_rows(path)
    assert raised.value.code == "FAQ_SCHEMA_INVALID"


def test_corrupt_faq_is_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"not an xlsx")
    with pytest.raises(FaqWorkbookError) as raised:
        read_faq_excel_rows(path)
    assert raised.value.code == "PARSE_FAILED"
