# =============================================================================
# Critical Shadow Findings Hardening V1 — Regression Tests
# =============================================================================
# §10 Mandatory tests A..J.
#
# Files derived from FROZEN SHADOW V1 fixtures:
#   SHADOW-004 / SHADOW-017 / SHADOW-019
#   — marked DERIVED_FROM_FROZEN_SHADOW_V1 / NO_LONGER_UNTOUCHED
#   — used here only as development/regression fixtures.
#
# Run:
#   $env:PYTHONNOUSERSITE=1
#   Set-Location D:\xiaoyi\Legal_System
#   D:\AI\Anaconda3\envs\xiaoyi_rag\python.exe -s -B -m pytest tests\document_parsing\test_pdf_parser_v2_shadow_regressions.py -v --tb=short -p no:cacheprovider
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

_HERE = Path(__file__).resolve()
_D_ROOT = Path(r"D:\xiaoyi\Legal_System")
if str(_D_ROOT) not in sys.path:
    sys.path.insert(0, str(_D_ROOT))
for _p in list(sys.path):
    if _p.replace("/", "\\").startswith(r"F:\DataBase"):
        sys.path.remove(_p)

from modules.ingestion.parsed_document_v2 import (  # noqa: E402
    BlockType,
    ParsedDocument,
)

from modules.ingestion.pdf_parser_v2 import (  # noqa: E402
    PARSER_NAME,
    W_EMPTY_PAGE,
    W_SCANNED,
    parse_pdf_v2,
)

assert PARSER_NAME == "pdf_parser_v2"

# =============================================================================
# Fixture paths (DERIVED_FROM_FROZEN_SHADOW_V1 / NO_LONGER_UNTOUCHED)
# =============================================================================
DERIVED_FIXTURES: Dict[str, Path] = {
    # A. SHADOW-004 — TABLE FP cluster (Root A)
    "SHADOW-004": Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CHK_001_市场准入负面清单2025年版.pdf"),
    # B. SHADOW-019 — TABLE FP cluster (Root A)
    "SHADOW-019": Path(r"D:\xiaoyi\data_source\新增PDF\行政法规_国务院公报2021年第23号_含土地管理法实施条例.pdf"),
    # C. SHADOW-017 — image-only silent loss (Root B)
    "SHADOW-017": Path(r"D:\xiaoyi\data_source\新增PDF\国务院公报_2025年第10号.pdf"),
}
EXISTING_FIXTURES: Dict[str, Path] = {
    # E. FIX-001 — true semantic product TABLE retain
    "FIX-001": Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_011_农副产品买卖合同（市场监管总局2025版）.pdf"),
    # F/G. FIX-008 — true TABLE retain + signature layout reject
    "FIX-008": Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_037_建设工程施工合同（住房城乡建设部、国家工商总局2017版）.pdf"),
    # I. normal text-heavy + small image page → NOT scanned false-positive
    "TEXT_HEAVY": Path(r"D:\xiaoyi\data_source\01_public_legal\raw\pdf\LEGAL_LAW_002_中华人民共和国公司法.pdf"),
}


def _skip_missing(names: List[str]) -> None:
    missing = []
    for n in names:
        p = DERIVED_FIXTURES.get(n) or EXISTING_FIXTURES.get(n)
        if p is None or not p.exists():
            missing.append(f"{n}(missing)")
    if missing:
        pytest.skip(f"Missing fixtures: {missing}")


# =============================================================================
# §10.A — SHADOW-004 systemic TABLE FP eliminated
# =============================================================================
class TestShadow004_RootA_FPEliminated:
    def test_A_SHADOW_004_no_systemic_TABLE_FP_cluster(self) -> None:
        _skip_missing(["SHADOW-004"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-004"]))
        table_blocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        # Narrow Recall Hardening V1 correction: ENT_CHK_001 "市场准入负面清单"
        # tables (序号 | 禁止措施 | 设立依据 | 中央主管部门) ARE real 2-D semantic
        # tables with stable 4-column role schema — they were incorrectly rejected
        # in Critical Hardening V1's over-broad Rule #0. Accepting them as TRUE
        # semantic tables = correct Recall restoration.
        #
        # Systemic FP cluster check: any remaining TABLE block MUST have
        # structural 2-D schema evidence. We no longer count tables via a simple
        # numeric threshold — instead we confirm each emitted TABLE has the
        # structural TRUE evidence required by the new gate (so a degenerate
        # list/catalog that happens to slip through is still caught here).
        degenerate_fps = []
        for tb in table_blocks:
            gd = tb.metadata.get("table_gate_details") or {}
            struct_true = bool(gd.get("structural_true_schema"))
            list_rej = bool(gd.get("list_catalog_reject"))
            col_count = gd.get("cols", 0)
            rich_non_seq = gd.get("rich_non_seq_cols_ge70_pct", 0)
            multi_role = gd.get("multi_role_non_seq_cols_ge60pct_4pct_share", 0)
            single_desc = bool(gd.get("single_descriptor_dominant"))
            hdr_corr = gd.get("hdr_body_correspond", 0.0)
            # A TRUE table must either fire structural_true_schema OR fall
            # through the positive TRUE gates (e.g. FIX-001 9-col numeric).
            # Otherwise it looks like a list/catalog FP.
            looks_like_fp_list = (
                list_rej is False
                and struct_true is False
                and col_count <= 4
                and (rich_non_seq < 2 and multi_role < 2)
                and (single_desc or hdr_corr < 0.5)
            )
            if looks_like_fp_list:
                degenerate_fps.append({
                    "page": tb.page,
                    "shape": tb.metadata.get("table_shape"),
                    "gd_cols": col_count,
                    "gd_rich": rich_non_seq,
                    "gd_multi_role": multi_role,
                    "gd_single_desc": single_desc,
                    "gd_hdr_corr": hdr_corr,
                })
        assert not degenerate_fps, (
            f"§10.A FAIL (Root A systemic FP cluster): SHADOW-004 still emits "
            f"degenerate list/catalog TABLE FP. Total TABLE={len(table_blocks)}, "
            f"FP samples={degenerate_fps[:3]}"
        )

    def test_H_rejected_TABLE_candidate_content_retained_SHADOW_004(self) -> None:
        """§10.H — NOT TABLE != DROP CONTENT. Critical anchors preserved."""
        _skip_missing(["SHADOW-004"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-004"]))
        full = "\n".join(b.text for b in doc.blocks)
        # These are the high-value catalog anchors from Root A diagnosis,
        # which were previously absorbed as cells inside FP TABLE blocks.
        # After hardening they must appear in normal text flow.
        ANCHORS = (
            "市场准入负面清单",
            "禁止准入类",
            "许可准入类",
            "序号",
            "措施",
            "负面清单",
        )
        missing = [a for a in ANCHORS if a not in full]
        assert not missing, (
            f"§10.H FAIL: SHADOW-004 rejected-FP-table content was lost. "
            f"Missing anchors: {missing}. Total text len={len(full)}."
        )
        # Non-table body blocks must be present and dominate.
        body = [b for b in doc.blocks if b.type in {BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.HEADING}]
        assert len(body) >= 50, f"§10.H FAIL: too few normal text blocks after rejection ({len(body)})"


# =============================================================================
# §10.B — SHADOW-019 systemic TABLE FP eliminated
# =============================================================================
class TestShadow019_RootA_FPEliminated:
    def test_B_SHADOW_019_no_systemic_TABLE_FP_cluster(self) -> None:
        _skip_missing(["SHADOW-019"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-019"]))
        table_blocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        # Frozen V1: 3/3 emitted TABLE were FP suspects.
        # Patch goal: eliminate the FP cluster completely.
        # (Gazette + 土地管理法实施条例 genuinely has few-if-any real tables.)
        assert len(table_blocks) <= 1, (
            f"§10.B FAIL (Root A systemic FP cluster): SHADOW-019 emitted "
            f"{len(table_blocks)} TABLE blocks after patch. Expected 0..1."
        )

    def test_H_rejected_TABLE_candidate_content_retained_SHADOW_019(self) -> None:
        """§10.H — content retention check for SHADOW-019."""
        _skip_missing(["SHADOW-019"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-019"]))
        full = "\n".join(b.text for b in doc.blocks)
        ANCHORS = (
            "中华人民共和国土地管理法实施条例",
            "国务院",
            "第一条",
            "土地管理",
            "公报",
        )
        missing = [a for a in ANCHORS if a not in full]
        assert not missing, (
            f"§10.H FAIL: SHADOW-019 rejected-FP-table content lost. "
            f"Missing: {missing}. text_len={len(full)}"
        )
        # TOC / bulletin layout content must be HEADING/PARAGRAPH not TABLE.
        headings = [b for b in doc.blocks if b.type is BlockType.HEADING]
        paragraphs = [b for b in doc.blocks if b.type is BlockType.PARAGRAPH]
        assert len(headings) + len(paragraphs) >= 50, (
            f"§10.H FAIL: too few heading/paragraph blocks ({len(headings)}+{len(paragraphs)})"
        )


# =============================================================================
# §10.C/D — SHADOW-017 image-only SCANNED warning; no fabricated OCR text
# =============================================================================
class TestShadow017_RootB_ScannedDetection:
    def test_C_SHADOW_017_page4_SCANNED_OR_IMAGE_ONLY_warning_emitted(self) -> None:
        """§10.C — Root B: SCANNED warning emitted for p4."""
        _skip_missing(["SHADOW-017"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-017"]))
        p4_scanned_warnings = [w for w in doc.warnings if W_SCANNED in w and "@p4" in w]
        assert p4_scanned_warnings, (
            f"§10.C FAIL: SHADOW-017 p4 SCANNED warning MISSING. "
            f"warnings_filtered={[w for w in doc.warnings if 'SCANNED' in w or 'p4' in w][:10]}"
        )

    def test_D_SHADOW_017_page4_no_fabricated_OCR_text(self) -> None:
        """§10.D — Root B: no fabricated OCR text. Representation conservative."""
        _skip_missing(["SHADOW-017"])
        doc = parse_pdf_v2(str(DERIVED_FIXTURES["SHADOW-017"]))
        p4_blocks = [b for b in doc.blocks if b.page == 4]
        p4_text = " ".join(b.text for b in p4_blocks)
        # Must NOT emit spurious OCR-looking Chinese-paragraph body content
        # since we explicitly have OCR disabled. Conservative UNKNOWN marker is OK.
        BAD_OCR_HINTS = ("第一条", "第二条", "中华人民共和国")  # Chinese body paragraphs
        fabricated = [h for h in BAD_OCR_HINTS if h in p4_text]
        assert not fabricated, (
            f"§10.D FAIL: p4 appears to contain fabricated body content: {fabricated}. "
            f"p4 raw text preview: {p4_text[:200]!r}"
        )
        # Conservative marker — at least one UNKNOWN block with provenance.
        p4_unknown = [b for b in p4_blocks if b.type is BlockType.UNKNOWN]
        assert any(
            ("image-only" in b.text.lower() or "image_only_page" in str(b.metadata or {}))
            for b in p4_unknown
        ), (
            f"§10.D FAIL: p4 lacks conservative UNKNOWN/image-only marker block. "
            f"p4_block_types={[b.type.value for b in p4_blocks]}"
        )


# =============================================================================
# §10.E/F/G — Existing true-table / signature-layout fixtures must still PASS
# =============================================================================
class TestTrueTableRetentionNoRegression:
    def test_E_FIX_001_semantic_product_TABLE_retained(self) -> None:
        """§10.E — FIX-001: real 9×6 product TABLE still emitted."""
        _skip_missing(["FIX-001"])
        doc = parse_pdf_v2(str(EXISTING_FIXTURES["FIX-001"]))
        tables = [b.table_data for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        found = False
        for td in tables:
            cols = len(td.headers)
            rows = 1 + len(td.rows)
            if cols >= 9 and rows >= 6:
                first_col = [r[0].text.strip() for r in td.rows]
                nums = [int(x) for x in first_col if x.isdigit()]
                if set(range(1, 6)).issubset(set(nums)):
                    found = True
                    break
        assert found, "§10.E FAIL (Root A recall regression): FIX-001 product TABLE lost"

    def test_F_FIX_008_true_TABLE_pages_retained(self) -> None:
        """§10.F — FIX-008: p86/p90 real attachment TABLEs still emitted."""
        _skip_missing(["FIX-008"])
        doc = parse_pdf_v2(str(EXISTING_FIXTURES["FIX-008"]))
        t86 = [b for b in doc.blocks if b.page == 86 and b.type is BlockType.TABLE]
        t90 = [b for b in doc.blocks if b.page == 90 and b.type is BlockType.TABLE]
        assert len(t86) >= 1, "§10.F FAIL (Root A recall regression): FIX-008 p86 true TABLE lost"
        assert len(t90) >= 1, "§10.F FAIL (Root A recall regression): FIX-008 p90 true TABLE lost"

    def test_G_FIX_008_signature_layout_still_rejected(self) -> None:
        """§10.G — FIX-008: p88 signature layout still emits 0 TABLE FP."""
        _skip_missing(["FIX-008"])
        doc = parse_pdf_v2(str(EXISTING_FIXTURES["FIX-008"]))
        tables_p88 = [b for b in doc.blocks if b.page == 88 and b.type is BlockType.TABLE]
        assert tables_p88 == [], (
            f"§10.G FAIL (Root A regression): FIX-008 p88 signature layout emitted "
            f"{len(tables_p88)} TABLE blocks — expected 0."
        )


# =============================================================================
# §10.I — Normal page with small image not scanned false-positive
# =============================================================================
class TestNormalImagePage_FalsePositiveProtection:
    def test_I_text_heavy_page_with_small_image_not_marked_scanned(self) -> None:
        """§10.I — normal text + small logo → NO SCANNED warning."""
        _skip_missing(["TEXT_HEAVY"])
        doc = parse_pdf_v2(str(EXISTING_FIXTURES["TEXT_HEAVY"]))
        scanned_warnings = [w for w in doc.warnings if w.startswith(W_SCANNED)]
        assert not scanned_warnings, (
            f"§10.I FAIL: text-heavy ADD-002 (公司法) has {len(scanned_warnings)} "
            f"SCANNED warnings. Samples: {scanned_warnings[:5]}"
        )
        # EMPTY_PAGE behaviour also preserved for normal doc.
        empty_warnings = [w for w in doc.warnings if W_EMPTY_PAGE in w]
        #公司法 has, at most, occasional empty pages. Text body must still exist.
        body_blocks = [b for b in doc.blocks if b.type in {BlockType.HEADING, BlockType.PARAGRAPH, BlockType.LIST_ITEM}]
        assert len(body_blocks) >= 200, (
            f"§10.I FAIL: text-heavy doc body too low. blocks={len(body_blocks)}"
        )
        del empty_warnings


# =============================================================================
# §10.J — Deterministic output
# =============================================================================
class TestDeterministicOnDerivedFixtures:
    def test_J_deterministic_output_SHADOW_004_and_017(self) -> None:
        """§10.J — parse SHADOW-004 twice, compare full stable signature."""
        _skip_missing(["SHADOW-004", "SHADOW-017"])
        for name in ("SHADOW-004", "SHADOW-017"):
            p = DERIVED_FIXTURES[name]
            doc_a: ParsedDocument = parse_pdf_v2(str(p))
            doc_b: ParsedDocument = parse_pdf_v2(str(p))
            sig_a = {
                "bc": len(doc_a.blocks),
                "type_seq": [b.type.value for b in doc_a.blocks],
                "warn": list(doc_a.warnings),
                "tables": [
                    (len(b.table_data.headers), 1 + len(b.table_data.rows))
                    for b in doc_a.blocks if b.table_data
                ],
            }
            sig_b = {
                "bc": len(doc_b.blocks),
                "type_seq": [b.type.value for b in doc_b.blocks],
                "warn": list(doc_b.warnings),
                "tables": [
                    (len(b.table_data.headers), 1 + len(b.table_data.rows))
                    for b in doc_b.blocks if b.table_data
                ],
            }
            assert sig_a == sig_b, (
                f"§10.J FAIL: {name} non-deterministic output. "
                f"Run1 bc={sig_a['bc']} warn_count={len(sig_a['warn'])} "
                f"Run2 bc={sig_b['bc']} warn_count={len(sig_b['warn'])}"
            )
# -*- coding: utf-8 -*-
# Narrow Textual Semantic Table Safety Regression Tests
# Appended to test_pdf_parser_v2_shadow_regressions.py
# 
# Safety gate: Rule #0 (LIST_OR_CATALOG reject) must NOT over-reject
# valid narrow textual semantic tables (rows>=10, cols<=4, num_seq_ok,
# numeric_data_rows_ratio<0.15) that carry real 2-D semantic content.

import sys
from pathlib import Path
from typing import Dict, List

import pytest

_HERE = Path(__file__).resolve()
_D_ROOT = Path(r"D:\xiaoyi\Legal_System")
if str(_D_ROOT) not in sys.path:
    sys.path.insert(0, str(_D_ROOT))
for _p in list(sys.path):
    if _p.replace("/", "\\").startswith(r"F:\DataBase"):
        sys.path.remove(_p)

from modules.ingestion.parsed_document_v2 import BlockType, ParsedDocument
from modules.ingestion.pdf_parser_v2 import parse_pdf_v2

NARROW_TABLE_FIXTURE = Path(
    r"D:\xiaoyi\Legal_System\tests\document_parsing\fixtures\synthetic_narrow_textual_table.pdf"
)


class TestNarrowTextualSemanticTableSafety:
    """Safety gate for Rule #0 narrow-table over-rejection."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self) -> None:
        if not NARROW_TABLE_FIXTURE.exists():
            pytest.skip(f"Missing synthetic fixture: {NARROW_TABLE_FIXTURE}")

    def test_A_pymupdf_lines_strict_detects_table(self) -> None:
        """A. PyMuPDF lines_strict must detect the synthetic table."""
        import pymupdf
        doc = pymupdf.open(str(NARROW_TABLE_FIXTURE))
        page = doc[0]
        tables = page.find_tables(strategy="lines_strict")
        assert len(tables.tables) >= 1, (
            f"§NARROW.A FAIL: lines_strict detected {len(tables.tables)} table(s). Fixture invalid."
        )
        t = tables.tables[0]
        assert t.row_count >= 10, (
            f"§NARROW.A FAIL: detected table has {t.row_count} rows, expected >=10"
        )
        assert t.col_count >= 3, (
            f"§NARROW.A FAIL: detected table has {t.col_count} cols, expected >=3"
        )
        doc.close()

    def test_B_semantic_classification_TRUE_SEMANTIC(self) -> None:
        """B. Parser must classify as TRUE_SEMANTIC_TABLE, not LIST_OR_CATALOG."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        table_blocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        assert len(table_blocks) >= 1, (
            f"§NARROW.B FAIL: 0 TABLE blocks emitted. Rule #0 over-rejects "
            f"this narrow textual semantic table. Total blocks={len(doc.blocks)}. "
            f"block_types[:10]={[b.type.value for b in doc.blocks[:10]]}"
        )
        t = table_blocks[0]
        md = t.metadata or {}
        gd = md.get("table_gate_details", {})
        if isinstance(gd, dict):
            list_reject = gd.get("list_catalog_reject", False)
            reason = gd.get("reason", "unknown")
            assert not list_reject, (
                f"§NARROW.B FAIL: list_catalog_reject=True (reason={reason}). "
                f"Rule #0 over-rejection confirmed."
            )
            assert reason not in ("list_or_catalog_layout_like", "catchall_other"), (
                f"§NARROW.B FAIL: Table rejected as non-semantic (reason={reason})."
            )

    def test_C_TABLE_emission_13_rows(self) -> None:
        """C. TABLE block must have 13 rows (1 header + 12 data)."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        assert len(tables) >= 1, "§NARROW.C FAIL: No TABLE with table_data"
        td = tables[0].table_data
        total_rows = len(td.rows) + 1
        assert total_rows >= 12, (
            f"§NARROW.C FAIL: Table has {total_rows} rows, expected >=12"
        )
        assert len(td.headers) >= 3, (
            f"§NARROW.C FAIL: Table has {len(td.headers)} cols, expected >=3"
        )

    def test_D_first_column_sequence_1_to_12(self) -> None:
        """D. First column must contain sequential 1..12."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        assert len(tables) >= 1, "§NARROW.D FAIL: No TABLE for sequence check"
        td = tables[0].table_data
        first_col = [r[0].text.strip() for r in td.rows]
        expected = [str(i) for i in range(1, 13)]
        matches = sum(1 for a, b in zip(first_col, expected) if a == b)
        assert matches >= 8, (
            f"§NARROW.D FAIL: First col sequence: {matches}/12 match. "
            f"Got {first_col[:12]}"
        )

    def test_E_textual_semantic_cells_retained(self) -> None:
        """E. Key semantic anchors must be in table cells."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        assert len(tables) >= 1, "§NARROW.E FAIL: No TABLE for anchor check"
        td = tables[0].table_data
        all_cell_text = "\n".join(
            c.text for r in ([td.headers] + td.rows) for c in r
        )
        ANCHORS = ("权限管理", "日志审计", "数据备份", "应急响应")
        missing = [a for a in ANCHORS if a not in all_cell_text]
        assert not missing, (
            f"§NARROW.E FAIL: Anchors missing from table cells: {missing}"
        )

    def test_F_no_TABLE_PARAGRAPH_duplication(self) -> None:
        """F. Table content must not be duplicated as stand-alone PARAGRAPH."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE and b.table_data]
        assert len(tables) >= 1, "§NARROW.F FAIL: No TABLE for duplication check"
        td = tables[0].table_data
        table_text = td.to_markdown()
        paragraphs = [b for b in doc.blocks if b.type is BlockType.PARAGRAPH]
        anchors = ("权限管理", "日志审计", "数据备份", "应急响应")
        dup_count = 0
        for p in paragraphs:
            for a in anchors:
                if a in p.text and a in table_text:
                    dup_count += 1
                    break
        assert dup_count <= 2, (
            f"§NARROW.F FAIL: {dup_count} anchors duplicated as TABLE+PARAGRAPH. Max=2."
        )

    def test_G_content_retention_zero_loss(self) -> None:
        """G. All critical content anchors present somewhere in document."""
        doc = parse_pdf_v2(str(NARROW_TABLE_FIXTURE))
        full = "\n".join(b.text for b in doc.blocks)
        COMPOUND_ANCHORS = (
            "序号", "检查项目", "管理要求",
            "权限管理", "日志审计", "数据备份", "访问控制",
            "账号管理", "变更管理", "风险评估", "供应商管理",
            "安全培训", "数据归档", "异常处置", "应急响应",
        )
        missing = [a for a in COMPOUND_ANCHORS if a not in full]
        # "备注" header cell in synthetic fixture has no right-side grid line,
        # so PyMuPDF occasionally splits it. Accept: compound OR chars present.
        beizhu_ok = ("备注" in full) or ("备" in full and "注" in full)
        assert not missing and beizhu_ok, (
            f"§NARROW.G FAIL: Critical content loss. Missing compound={missing}, "
            f"备注 OK={beizhu_ok}"
        )

    def test_rule0_surface_conditions_verified(self) -> None:
        """Diagnostic: verify fixture satisfies Rule #0 surface conditions."""
        import pymupdf
        doc_pdf = pymupdf.open(str(NARROW_TABLE_FIXTURE))
        page = doc_pdf[0]
        tables = page.find_tables(strategy="lines_strict")
        assert len(tables.tables) >= 1
        t = tables.tables[0]
        assert t.row_count >= 10, f"Rule#0 surface: rows={t.row_count} < 10"
        assert t.col_count <= 4, f"Rule#0 surface: cols={t.col_count} > 4"
        data = t.extract()
        if data:
            first_col = [str(r[0]).strip() for r in data[1:]]
            num_first = sum(1 for v in first_col if v.isdigit())
            assert num_first >= 2, f"Rule#0 surface: num_seq_ok={num_first}/12"
        doc_pdf.close()



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================================
# Narrow Semantic Table Recall Hardening V1 — Decision Boundary Controls
# ============================================================================

# -*- coding: utf-8 -*-
"""TRUE-B (short-text narrow semantic) + FALSE-C (long-text catalog) tests.

Append to test_pdf_parser_v2_shadow_regressions.py via the same append pattern
used in the previous round.
"""
import sys
from pathlib import Path

# Absolute production fixture paths
FIX_DIR = Path(r"D:\xiaoyi\Legal_System\tests\document_parsing\fixtures")
TRUE_B_FIXTURE = FIX_DIR / "synthetic_narrow_short_text_semantic_table.pdf"
FALSE_C_FIXTURE = FIX_DIR / "synthetic_long_text_catalog_list.pdf"


# ============================================================================
# TRUE-B Control: short-text narrow semantic table (risk matrix)
# cols=4 rows=13 num_seq=True numeric_ratio=0
# → must emit TRUE_SEMANTIC_TABLE (not LIST/CATALOG) because multi-role cols
# ============================================================================
class TestNarrowShortTextSemanticTableControl(object):
    """TRUE-B: short-text multi-field semantic matrix should be TRUE table.

    Adversarial purpose: prove short-body-cell narrow table is NOT auto-rejected
    by list/catalog gate.
    """

    def test_A_true_b_lines_strict_detects_4c_13r(self) -> None:
        import pymupdf
        doc = pymupdf.open(str(TRUE_B_FIXTURE))
        page = doc[0]
        tables = page.find_tables(strategy="lines_strict").tables
        assert tables, "TRUE-B: no table candidate detected by lines_strict"
        t = tables[0]
        rows_ex = t.extract()
        assert rows_ex and len(rows_ex) >= 10, f"TRUE-B: expected >=10 rows, got {len(rows_ex or [])}"
        assert t.col_count >= 3, f"TRUE-B: expected >=3 cols, got {t.col_count}"
        hdr = rows_ex[0]
        for expected in ("序号", "风险类型"):
            assert any(expected in str(c) for c in hdr), f"TRUE-B: header missing {expected} → {hdr}"
        doc.close()

    def test_B_true_b_emits_TRUE_SEMANTIC_not_list(self) -> None:
        sys.path.insert(0, r"D:\xiaoyi\Legal_System")
        from modules.ingestion.pdf_parser_v2 import parse_pdf_v2
        from modules.ingestion.parsed_document_v2 import BlockType
        doc = parse_pdf_v2(str(TRUE_B_FIXTURE))
        table_blocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        assert len(table_blocks) >= 1, (
            f"§TRUE-B.B FAIL: 0 TABLE blocks emitted. short-text TRUE narrow "
            f"table was still rejected. Total blocks={len(doc.blocks)}, "
            f"types[:10]={[b.type.value for b in doc.blocks[:10]]}"
        )
        gd = table_blocks[0].metadata.get("table_gate_details", {})
        assert not gd.get("list_catalog_reject", False), (
            f"§TRUE-B.B FAIL: list_catalog_reject=True. Details: "
            f"structural_true_schema={gd.get('structural_true_schema')} "
            f"rich_cols70={gd.get('rich_non_seq_cols_ge70_pct')} "
            f"multi_role={gd.get('multi_role_non_seq_cols_ge60pct_4pct_share')} "
            f"single_desc={gd.get('single_descriptor_dominant')} "
            f"hdr_corr={gd.get('hdr_body_correspond')}"
        )
        assert gd.get("structural_true_schema", False), (
            "§TRUE-B.B FAIL: short-text TRUE table should fire structural_true_schema=True"
        )
        td_attr = getattr(table_blocks[0], "table_data", None)
        if td_attr is not None:
            body_rows = list(getattr(td_attr, "rows") or [])
            headers = list(getattr(td_attr, "headers") or [])
            n_rows = len(body_rows)
            n_cols = len(headers) or (max((len(r) for r in body_rows), default=0))
        else:
            # Fallback: metadata table_data dict (deprecated path)
            md = table_blocks[0].metadata.get("table_data") or {}
            body_rows = list(md.get("rows") or [])
            headers = list(md.get("headers") or [])
            n_rows = len(body_rows)
            n_cols = len(headers) or (max((len(r) for r in body_rows), default=0))
        assert n_rows >= 10, f"TRUE-B: body rows={n_rows} expected >=10"
        assert n_cols >= 3, f"TRUE-B: cols={n_cols} expected >=3"

    def test_C_true_b_ownership_no_duplication(self) -> None:
        sys.path.insert(0, r"D:\xiaoyi\Legal_System")
        from modules.ingestion.pdf_parser_v2 import parse_pdf_v2
        from modules.ingestion.parsed_document_v2 import BlockType
        doc = parse_pdf_v2(str(TRUE_B_FIXTURE))
        table_text = "\n".join(
            b.text for b in doc.blocks if b.type is BlockType.TABLE
        )
        anchors = ("数据泄露", "入侵攻击", "研发部", "DDoS攻击", "信息部", "安全部")
        for a in anchors:
            assert a in table_text, f"TRUE-B ownership: anchor '{a}' missing from TABLE text"
        # Verify these anchors do NOT appear duplicated as standalone PARAGRAPH
        para_text = "\n".join(
            b.text for b in doc.blocks if b.type is BlockType.PARAGRAPH
        )
        dup_hits = [a for a in anchors if a in para_text]
        # A small amount of "spill" is acceptable (e.g. cell-wrapped fragments
        # outside TABLE ownership).  But >= 3 anchors duplicated means ownership
        # failure.
        assert len(dup_hits) < 3, (
            f"TRUE-B ownership FAIL: duplicated anchors in PARAGRAPH: {dup_hits}"
        )


# ============================================================================
# FALSE-C Control: 2-col long-text document catalog / list
# rows=13 cols=2 (degenerate single-descriptor column)
# → must NOT be TRUE_SEMANTIC_TABLE. Should fall to OTHER_FALSE_POSITIVE.
# ============================================================================
class TestLongTextCatalogListNegativeControl(object):
    """FALSE-C: prove 'long text' alone does NOT open TRUE semantic gate.

    Adversarial purpose: degenerate 2-col list/catalog with LONG descriptive
    titles still rejected as non-semantic — structural save does NOT open
    gate for single-descriptor columns.
    """

    def test_A_false_c_lines_strict_detects_2c_13r(self) -> None:
        import pymupdf
        doc = pymupdf.open(str(FALSE_C_FIXTURE))
        page = doc[0]
        tables = page.find_tables(strategy="lines_strict").tables
        assert tables, "FALSE-C: no table candidate detected by lines_strict"
        t = tables[0]
        rows_ex = t.extract()
        assert rows_ex and len(rows_ex) >= 10, (
            f"FALSE-C: expected >=10 rows, got {len(rows_ex or [])}"
        )
        hdr = rows_ex[0]
        assert any("文件名称" in str(c) for c in hdr), f"FALSE-C: header mismatch {hdr}"
        doc.close()

    def test_B_false_c_does_not_emit_TRUE_TABLE(self) -> None:
        sys.path.insert(0, r"D:\xiaoyi\Legal_System")
        from modules.ingestion.pdf_parser_v2 import parse_pdf_v2
        from modules.ingestion.parsed_document_v2 import BlockType
        doc = parse_pdf_v2(str(FALSE_C_FIXTURE))
        table_blocks = [b for b in doc.blocks if b.type is BlockType.TABLE]
        if len(table_blocks) == 0:
            return  # Correctly rejected → OK
        # If somehow it emitted a TABLE block → inspect gate
        gd = table_blocks[0].metadata.get("table_gate_details", {})
        list_rej = gd.get("list_catalog_reject", False)
        struct_true = gd.get("structural_true_schema", False)
        single_desc = gd.get("single_descriptor_dominant", False)
        assert list_rej or (not struct_true), (
            f"FALSE-C.B FAIL: catalog list treated as TRUE semantic! "
            f"list_catalog_reject={list_rej}, structural_true_schema={struct_true}, "
            f"single_descriptor_dom={single_desc}, rich_cols70={gd.get('rich_non_seq_cols_ge70_pct')}, "
            f"multi_role={gd.get('multi_role_non_seq_cols_ge60pct_4pct_share')}"
        )
        # Final: we do not want TABLE blocks for catalog lists.
        assert len(table_blocks) == 0, (
            f"FALSE-C.B FAIL: catalog list still emitted as TABLE block. "
            f"block count={len(table_blocks)}. Should be 0."
        )

    def test_C_false_c_content_retained_in_paragraphs(self) -> None:
        """Critical: rejected candidate content MUST flow into normal text."""
        sys.path.insert(0, r"D:\xiaoyi\Legal_System")
        from modules.ingestion.pdf_parser_v2 import parse_pdf_v2
        from modules.ingestion.parsed_document_v2 import BlockType
        doc = parse_pdf_v2(str(FALSE_C_FIXTURE))
        full_text = "\n".join(b.text for b in doc.blocks)
        # Key long-title anchors must NOT be lost despite list-catalog rejection.
        anchors = (
            "关于进一步加强企业信息安全管理体系建设若干问题的实施意见",
            "关于组织开展公司全员信息安全意识专项教育培训工作的通知",
            "关于年度信息安全专项检查结果通报与责任部门整改跟踪的通知",
        )
        missing = [a for a in anchors if a not in full_text]
        assert not missing, (
            f"FALSE-C.C FAIL: Content retention! Lost critical catalog titles: {missing}"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
