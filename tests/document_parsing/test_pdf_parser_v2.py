# =============================================================================
# Production Tests: PDF Parser V2 Implementation V1 (D-Project Deployed Copy)
# =============================================================================
# Deploy target:
#   D:\xiaoyi\Legal_System\tests\document_parsing\test_pdf_parser_v2.py
#
# Run:
#   $env:PYTHONNOUSERSITE=1
#   Set-Location D:\xiaoyi\Legal_System
#   D:\AI\Anaconda3\envs\xiaoyi_rag\python.exe -s -B -m pytest tests\document_parsing\test_pdf_parser_v2.py -v --tb=line -p no:cacheprovider
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# --- Path bootstrap ---------------------------------------------------------
_HERE = Path(__file__).resolve()
_D_ROOT = Path(r"D:\xiaoyi\Legal_System")
if str(_D_ROOT) not in sys.path:
    sys.path.insert(0, str(_D_ROOT))
for _p in list(sys.path):
    if _p.replace("/", "\\").startswith(r"F:\DataBase"):
        sys.path.remove(_p)

from modules.ingestion.parsed_document_v2 import (  # noqa: E402
    Block,
    BlockType,
    LegacyAdapter,
    LegacyParsedDocument,
    ParsedDocument,
)
from modules.ingestion.chunking import (  # noqa: E402
    ChildChunk,
    ParentChunk,
    chunk_pdf_pages_to_parents,
    split_children_from_parent,
)

from modules.ingestion.pdf_parser_v2 import (  # noqa: E402 — D-project package import (§9/10)
    PARSER_NAME,
    PARSER_VERSION,
    PdfParserV2Config,
    W_MERGED_CELL,
    W_SCANNED,
    W_TABLE_AMBIG,
    classify_table_candidate,
    detect_table_candidates,
    enforce_table_ownership,
    extract_page_geometry,
    normalize_layout_elements,
    parse_pdf_v2,
    parse_pdf_v2_to_legacy,
    reconstruct_reading_order,
    table_to_tabledata,
)

assert PARSER_NAME == "pdf_parser_v2", f"Deploy parser PARSER_NAME mismatch: {PARSER_NAME}"
import modules.ingestion.pdf_parser_v2 as _parser_mod  # noqa: E402
_IMPORTER_PATH = str(Path(_parser_mod.__file__).resolve()).replace("/", "\\")
assert _IMPORTER_PATH.startswith(r"D:\xiaoyi\Legal_System"), (
    f"§9/10 FAIL: parser loaded from outside D-project: {_parser_mod.__file__}"
)
del _IMPORTER_PATH, _parser_mod

import pymupdf  # noqa: E402

FIXTURES: Dict[str, Path] = {
    "FIX-001": Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_011_农副产品买卖合同（市场监管总局2025版）.pdf"),
    "FIX-006": Path(r"D:\xiaoyi\data_source\01_public_legal\raw\pdf\LEGAL_LAW_001_中华人民共和国民法典.pdf"),
    "FIX-008": Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_037_建设工程施工合同（住房城乡建设部、国家工商总局2017版）.pdf"),
    "ADD-002": Path(r"D:\xiaoyi\data_source\01_public_legal\raw\pdf\LEGAL_LAW_002_中华人民共和国公司法.pdf"),
}
CRIT_ANCHORS_FIX006: Tuple[str, ...] = ("主席令", "中华人民共和国民法典", "第一编", "第一条")
CRIT_ANCHORS_ADD002: Tuple[str, ...] = ("中华人民共和国公司法", "第一条", "公司", "股东")


def _skip_missing() -> None:
    missing = [n for n, p in FIXTURES.items() if not p.exists()]
    if missing:
        pytest.skip(f"Missing dev fixtures (in D:\\xiaoyi\\data_source): {missing}")


# =============================================================================
# 1. Frozen Contract / BlockType / Order / Provenance
# =============================================================================

class TestFrozenContractOnParser:
    def test_runtime_pymupdf_path_and_version(self) -> None:
        assert "xiaoyi_rag" in pymupdf.__file__ or (
            Path(pymupdf.__file__).resolve().drive == "d:" and "Anaconda3" in pymupdf.__file__
        ), f"PyMuPDF path mismatch: {pymupdf.__file__}"
        assert pymupdf.__version__.startswith("1.28.")

    def test_only_6_blocktypes_emitted(self) -> None:
        _skip_missing()
        ALLOWED = {BlockType.HEADING, BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.TABLE, BlockType.METADATA, BlockType.UNKNOWN}
        for name, p in FIXTURES.items():
            doc = parse_pdf_v2(str(p))
            emitted = {b.type for b in doc.blocks}
            assert emitted.issubset(ALLOWED), f"{name} emits non-frozen types: {emitted - ALLOWED}"

    def test_doc_validate_and_triple_equality_invariant(self) -> None:
        _skip_missing()
        for name, p in FIXTURES.items():
            doc = parse_pdf_v2(str(p))
            doc.validate()
            for i, b in enumerate(doc.blocks):
                assert b.order == i, f"{name} blocks[{i}].order={b.order} != {i}"
                assert b.block_id.endswith(f"_b{i:04d}"), f"{name} block_id suffix mismatch"
                assert b.provenance_stored is not None, f"{name} provenance missing @ i={i}"
                assert b.provenance_stored.block_order == i
                assert b.provenance_stored.source_file.endswith(Path(doc.source_file).name)
                if b.page is not None:
                    assert b.page >= 1
                    assert b.provenance_stored.page == b.page

    def test_parser_identity_stable(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["ADD-002"]))
        assert doc.parser_name == PARSER_NAME == "pdf_parser_v2"
        assert doc.parser_version == PARSER_VERSION == "2.0.0"
        assert doc.metadata["parser_name"] == PARSER_NAME
        assert doc.metadata["parser_version"] == PARSER_VERSION


# =============================================================================
# 2. Determinism Gate
# =============================================================================

def _doc_stable_signature(doc: ParsedDocument) -> dict:
    return {
        "block_count": len(doc.blocks),
        "type_sequence": [b.type.value for b in doc.blocks],
        "text_sequence": [b.text for b in doc.blocks],
        "order_sequence": [b.order for b in doc.blocks],
        "provenance_sequence": [
            None if b.provenance_stored is None else (b.provenance_stored.page, b.provenance_stored.block_order)
            for b in doc.blocks
        ],
        "table_shapes": [(len(b.table_data.headers), len(b.table_data.rows)) for b in doc.blocks if b.table_data is not None],
        "title": doc.title,
    }


class TestDeterminism:
    @pytest.mark.parametrize("fixture_name", ["FIX-001", "FIX-006", "ADD-002"])
    def test_parsed_twice_equal_all_7_dimensions(self, fixture_name: str) -> None:
        _skip_missing()
        p = FIXTURES[fixture_name]
        doc_a = parse_pdf_v2(str(p))
        doc_b = parse_pdf_v2(str(p))
        sa = _doc_stable_signature(doc_a)
        sb = _doc_stable_signature(doc_b)
        for k in sa:
            assert sa[k] == sb[k], f"{fixture_name}: determinism failure @ key={k}"


# =============================================================================
# 3. Reading Order
# =============================================================================

class TestReadingOrder:
    def test_add002_single_column_order_natural(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["ADD-002"]))
        texts = " ".join(b.text for b in doc.blocks if b.type in {BlockType.PARAGRAPH, BlockType.HEADING})
        for anchor in CRIT_ANCHORS_ADD002:
            assert anchor in texts, f"ADD-002 missing {anchor}"
        last_page = 0
        for b in doc.blocks:
            p = b.page or 0
            assert p >= last_page
            last_page = p

    def test_fix006_critical_anchors_in_natural_sequence(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-006"]))
        positions = {}
        BODY_TYPES = {BlockType.HEADING, BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.TABLE}
        for i, b in enumerate(doc.blocks):
            if b.type not in BODY_TYPES:
                continue
            for a in CRIT_ANCHORS_FIX006:
                if a in b.text and a not in positions:
                    positions[a] = i
        keys = ["主席令", "中华人民共和国民法典", "第一编", "第一条"]
        for a in keys:
            assert a in positions, f"FIX-006 missing critical anchor: {a}"
        seq = [positions[k] for k in keys]
        assert seq == sorted(seq), f"FIX-006 reading order regression: positions {seq}"

    def test_no_cross_column_paragraph_merge(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-006"]))
        blocks_page_10 = [b for b in doc.blocks if b.page == 10 and b.type in {BlockType.PARAGRAPH, BlockType.LIST_ITEM}]
        assert blocks_page_10, "No body blocks on p.10"
        assert len(blocks_page_10) >= 2


# =============================================================================
# 4. Table Router
# =============================================================================

class TestTableRouterAndGate:
    def test_RULE1_default_strategy_is_lines_strict_by_evidence_fix001(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-001"]))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
        assert tables, "FIX-001 product table expected"
        for t in tables:
            md = t.metadata or {}
            strat = md.get("table_strategy")
            assert strat == "lines_strict", f"RULE1 broken: strategy={strat}"
        shapes = [(len(t.table_data.headers), 1 + len(t.table_data.rows)) for t in tables]
        assert any(c >= 9 and r >= 6 for c, r in shapes), f"FIX-001 missing 9x6 table, got shapes {shapes}"

    def test_RULE2_text_strategy_disabled_even_if_config_flipped(self) -> None:
        _skip_missing()
        import pymupdf as _pm
        _orig = _pm.Page.find_tables

        def _spy(self, *args, **kwargs):
            strat = kwargs.get("strategy", None)
            if args:
                strat = args[0]
            if strat == "text":
                raise AssertionError("RULE2 VIOLATION: find_tables strategy=text CALLED.")
            return _orig(self, *args, **kwargs)

        cfg = PdfParserV2Config()
        cfg.text_strategy_enable = True
        try:
            _pm.Page.find_tables = _spy
            doc = parse_pdf_v2(str(FIXTURES["ADD-002"]), cfg=cfg)
            assert doc.blocks, "Blocks expected"
            assert doc.parser_name == "pdf_parser_v2"
        finally:
            _pm.Page.find_tables = _orig

    def test_RULE3_lines_fallback_toggle(self) -> None:
        _skip_missing()
        import pymupdf as _pm2
        doc_h = _pm2.open(str(FIXTURES["ADD-002"]))
        try:
            page = doc_h[2]
            geom = extract_page_geometry(page, 2, PdfParserV2Config())
            geom.lines = normalize_layout_elements(geom, PdfParserV2Config())
            cfg_no_fb = PdfParserV2Config(enable_lines_fallback=False)
            _, info_no = detect_table_candidates(page, geom, cfg_no_fb)
            assert info_no["stage2_run"] is False
            cfg_yes_fb = PdfParserV2Config(enable_lines_fallback=True)
            _, info_yes = detect_table_candidates(page, geom, cfg_yes_fb)
            assert info_no["stage2_run"] is False
            assert info_no["text_strategy_disabled_RULE2"] is True
            assert info_yes["text_strategy_disabled_RULE2"] is True
        finally:
            doc_h.close()

    def test_RULE4_signature_layout_rejection_fix008_page_88(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-008"]))
        blocks_88 = [b for b in doc.blocks if b.page == 88]
        tables_on_88 = [b for b in blocks_88 if b.type is BlockType.TABLE]
        assert tables_on_88 == [], (
            f"RULE4 FAIL: signature layout p.88 emitted {len(tables_on_88)} TABLE blocks. "
            f"Block types observed: {[b.type.value for b in blocks_88]}"
        )
        big_text_88 = "\n".join(b.text for b in blocks_88)
        assert any(s in big_text_88 for s in ("签字", "盖章", "开户银行", "邮政编码", "法定代表人")), "p.88 signature signal missing"


# =============================================================================
# 5. TableData / Ownership
# =============================================================================

class TestTableDataAndOwnership:
    def test_fix001_first_column_1_through_5_rows(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-001"]))
        tables = [b.table_data for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        product_table = None
        for td in tables:
            headers = [h.text for h in td.headers]
            joined_h = "".join(headers)
            if any(k in joined_h for k in ("序号", "产品", "规格", "数量")) and len(td.headers) >= 9:
                product_table = td
                break
        assert product_table is not None, f"FIX-001 product info table not found; tables={[len(td.headers) for td in tables]}"
        first_col = [row[0].text.strip() for row in product_table.rows]
        numeric_vals = [int(c) for c in first_col if c.isdigit()]
        assert len(numeric_vals) >= 5, f"FIX-001 first column numeric rows = {numeric_vals}"
        for i in range(1, 6):
            assert i in numeric_vals, f"FIX-001 table first column missing row {i}"

    def test_ownership_no_duplicate_paragraph_repeat(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-001"]))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        product_table_block = None
        for tb in tables:
            if len(tb.table_data.headers) >= 9:
                product_table_block = tb
                break
        assert product_table_block, "Product table not found"
        header_tokens = [h.text.strip() for h in product_table_block.table_data.headers if h.text.strip()]
        assert len(header_tokens) >= 5
        table_header_signature = "| " + " | ".join(
            product_table_block.table_data.headers[i].display_text()
            for i in range(min(6, len(product_table_block.table_data.headers)))
        ) + " |"
        for b in doc.blocks:
            if b.type is BlockType.TABLE:
                continue
            assert table_header_signature not in b.text, (
                "Ownership RULE5 FAIL: table content duplicated in non-TABLE block "
                f"type={b.type.value} order={b.order}"
            )

    def test_no_merge_guess_RULE6_merges_default_1_1(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-001"]))
        for b in doc.blocks:
            if b.table_data is None:
                continue
            for c in b.table_data.headers:
                assert c.colspan == 1 and c.rowspan == 1, "RULE6 FAIL: merge guess on header"
            for row in b.table_data.rows:
                for c in row:
                    assert c.colspan == 1 and c.rowspan == 1, "RULE6 FAIL: merge guess on data cell"
        assert any(W_MERGED_CELL in w for w in doc.warnings), "MERGED_CELL_NOT_RECOVERED warning missing"


# =============================================================================
# 6. 4 Fixtures Mandatory Acceptance
# =============================================================================

class TestFourFixturesAcceptance:
    def test_FIX_001_product_info_table_9cols_6rows(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-001"]))
        tblocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        found = False
        for t in tblocks:
            td = t.table_data
            rows_total = 1 + len(td.rows)
            cols = len(td.headers)
            if cols >= 9 and rows_total >= 6:
                first_col = [r[0].text.strip() for r in td.rows]
                nums = [int(x) for x in first_col if x.isdigit()]
                if set(range(1, 6)).issubset(set(nums)):
                    found = True
                    break
        assert found, "FIX-001: Product info table (9x6, first col 1..5) NOT found"

    def test_FIX_006_critical_content_loss_zero(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-006"]))
        full = " ".join(b.text for b in doc.blocks)
        for anchor in CRIT_ANCHORS_FIX006:
            assert anchor in full, f"FIX-006 CRITICAL LOSS: anchor '{anchor}' missing"

    def test_FIX_008_no_systemic_FP_and_true_tables(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["FIX-008"]))
        t86 = [b for b in doc.blocks if b.page == 86 and b.type is BlockType.TABLE]
        t90 = [b for b in doc.blocks if b.page == 90 and b.type is BlockType.TABLE]
        assert len(t86) >= 1, "FIX-008 p.86 true attachment table missing"
        assert len(t90) >= 1, "FIX-008 p.90 true attachment table missing"
        for pg in (26, 28, 31, 88):
            on_pg = [b for b in doc.blocks if b.page == pg and b.type is BlockType.TABLE]
            assert len(on_pg) == 0, f"FIX-008 p.{pg} FORM/SIGNATURE FP TABLE count={len(on_pg)}"

    def test_ADD_002_single_column_baseline_ok(self) -> None:
        _skip_missing()
        doc = parse_pdf_v2(str(FIXTURES["ADD-002"]))
        full = " ".join(b.text for b in doc.blocks)
        for anchor in CRIT_ANCHORS_ADD002:
            assert anchor in full, f"ADD-002 anchor loss: {anchor}"
        body_blocks = [b for b in doc.blocks if b.type is not BlockType.METADATA]
        pages_set = {b.page for b in body_blocks if b.page is not None}
        assert len(pages_set) >= 5, f"ADD-002 page coverage too low: {pages_set}"
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
        for t in tables:
            assert t.metadata.get("table_strategy") in ("lines_strict", "lines")


# =============================================================================
# 7. LegacyAdapter + Current Chunker Compatibility
# =============================================================================

class TestLegacyChunkerCompatibility:
    def test_linkage_no_crash_critical_anchors_intact(self) -> None:
        _skip_missing()
        for name in ("FIX-006", "ADD-002"):
            v2_doc = parse_pdf_v2(str(FIXTURES[name]))
            legacy: LegacyParsedDocument = LegacyAdapter.to_legacy_parsed_document(v2_doc)
            assert isinstance(legacy, LegacyParsedDocument)
            assert isinstance(legacy.text, str) and len(legacy.text) > 0
            assert isinstance(legacy.segments, tuple) and len(legacy.segments) >= 1
            max_page = v2_doc.metadata.get("page_count", 0)
            if max_page:
                assert len(legacy.segments) == max_page, (
                    f"{name}: Legacy segments len={len(legacy.segments)} != page_count={max_page}"
                )
            anchors = CRIT_ANCHORS_FIX006 if name == "FIX-006" else CRIT_ANCHORS_ADD002
            for a in anchors[:2]:
                assert a in legacy.text, f"{name}: Legacy text lost anchor '{a}'"
            page_texts: List[str] = list(legacy.segments)
            parents: List[ParentChunk] = chunk_pdf_pages_to_parents(page_texts)
            assert len(parents) >= 1, f"{name}: Current Chunker produced 0 parents"
            anchors_parents = " ".join(p.text for p in parents)
            for a in anchors[:2]:
                assert a in anchors_parents, f"{name}: Parent chunks lost anchor '{a}'"
            children: List[ChildChunk] = split_children_from_parent(parents[0].text, parent_index=0)
            assert len(children) >= 1, f"{name}: No children chunks from first parent"
            for p in parents:
                assert p.page_no >= 1, f"{name}: Parent page_no wrong: {p.page_no}"


# =============================================================================
# 8. Error Handling
# =============================================================================

class TestErrorHandling:
    def test_nonexistent_pdf_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_pdf_v2(str(tmp_path / "definitely_not_real_12345.pdf"))

    def test_corrupt_bytes_pdf_raises_value_error_not_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"ThisIsNotARealPDFFile\xc0\xff\x00\x01\x02")
        with pytest.raises((ValueError,)):
            parse_pdf_v2(str(bad))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
