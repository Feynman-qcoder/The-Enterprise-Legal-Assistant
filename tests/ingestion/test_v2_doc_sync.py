from __future__ import annotations

import copy

import pytest

from offline.scripts.run_v2_doc_sync import parse_package


def _raw_package(*, child_id: str | None = "a" * 32) -> dict:
    child = {
        "content": "第一条 本合同用于测试稳定的子块身份。",
        "section_path": ["第一条"],
        "metadata": {"chunk_index": 0},
    }
    if child_id is not None:
        child["child_id"] = child_id
    return {
        "document_identity": f"TEST_001@sha256:{'b' * 64}",
        "source_file": "TEST_001.md",
        "parents": [
            {
                "section_path": ["第一条"],
                "content": "第一条 本合同用于测试稳定的子块身份。",
                "metadata": {},
                "children": [child],
            }
        ],
    }


def test_parse_package_preserves_explicit_child_id() -> None:
    package = parse_package(_raw_package(child_id="a" * 32))
    assert package.parents[0].children[0].child_id == "a" * 32


def test_parse_package_keeps_legacy_missing_child_id_compatible() -> None:
    package = parse_package(_raw_package(child_id=None))
    assert package.parents[0].children[0].child_id is None


def test_same_package_retry_has_same_child_identity() -> None:
    raw = _raw_package()
    first = parse_package(copy.deepcopy(raw))
    second = parse_package(copy.deepcopy(raw))
    assert first.parents[0].children[0].child_id == second.parents[0].children[0].child_id


@pytest.mark.parametrize("child_id", ["short", "z" * 32, "A" * 32, "a" * 33, ""])
def test_parse_package_rejects_invalid_explicit_child_id(child_id: str) -> None:
    with pytest.raises(ValueError, match="child_id"):
        parse_package(_raw_package(child_id=child_id))
