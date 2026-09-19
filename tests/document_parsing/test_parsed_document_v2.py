"""
Unit tests for ParsedDocument V2 Contract (POST Freeze-Patch).

Pure stdlib + pytest. NO DB / Milvus / Redis / network / re-ingestion.

Covered targets:
  * Original 42 tests (Heading/Paragraph/List/Table/Metadata exclusion/
    Unknown preservation / Reading order / to_legacy_text / invalid heading
    / invalid table / Provenance / Enum / Identity invariants)
  * FREEZE PATCH new tests:
      - wrong block.order no implicit repair → ContractViolation (#1)
      - Block.order != Provenance.block_order → ContractViolation (#2)
      - correct: index == order == provenance.block_order → PASS (#3)
      - explicit renumber_blocks utility → sync order + provenance (#4)
  * REAL CURRENT CHUNKER compatibility smoke (PURE IN-MEMORY, actual chunker):
      ParsedDocumentV2 → LegacyAdapter → LegacyParsedDocument →
        chunk_pdf_pages_to_parents() → split_children_from_parent() → chunks
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import production Contract module (actual deployed path)
from modules.ingestion.parsed_document_v2 import (  # noqa: E402
    Block,
    BlockType,
    ContractViolation,
    LegacyAdapter,
    LegacyParsedDocument,
    ParsedDocument,
    Provenance,
    TableCell,
    TableData,
)

# Import REAL CURRENT Chunker (NOT mocked, actual production module)
from modules.ingestion.chunking import (  # noqa: E402
    ChildChunk,
    ParentChunk,
    chunk_pdf_pages_to_parents,
    split_children_from_parent,
)


# =============================================================================
# Helpers (POST FREEZE: Constructor STRICT; so build blocks WITH correct order
#          via ParsedDocument.prepare_blocks() when using _doc()).
# =============================================================================

def _h(text, level=1, *, page=None, meta=None, order=0):
    return Block(
        block_id="__placeholder__",
        type=BlockType.HEADING,
        text=text,
        order=order,
        level=level,
        page=page,
        metadata=dict(meta or {}),
    )


def _p(text, *, page=None, meta=None, order=0):
    return Block(
        block_id="__placeholder__",
        type=BlockType.PARAGRAPH,
        text=text,
        order=order,
        page=page,
        metadata=dict(meta or {}),
    )


def _li(text, level=0, *, page=None, order=0):
    return Block(
        block_id="__placeholder__",
        type=BlockType.LIST_ITEM,
        text=text,
        order=order,
        page=page,
        metadata={"list_level": level},
    )


def _meta(text, *, page=None, order=0):
    return Block(
        block_id="__placeholder__",
        type=BlockType.METADATA,
        text=text,
        order=order,
        page=page,
    )


def _unknown(text, *, page=None, order=0):
    return Block(
        block_id="__placeholder__",
        type=BlockType.UNKNOWN,
        text=text,
        order=order,
        page=page,
    )


def _table_block(headers, rows, *, caption=None, page=None, order=0):
    h_cells = [TableCell(h) if isinstance(h, str) else h for h in headers]
    r_cells = [[TableCell(c) if isinstance(c, str) else c for c in row] for row in rows]
    td = TableData(headers=h_cells, rows=r_cells, caption=caption)
    return Block(
        block_id="__placeholder__",
        type=BlockType.TABLE,
        text="",
        order=order,
        table_data=td,
        page=page,
    )


def _doc(blocks, *, doc_id="DOC001", source="law.md", title="T",
         parser="v2_test", ver="2.0.0", meta=None):
    """
    Build a ParsedDocument the CORRECT, post-freeze way:
    1) Create raw blocks (individual `order` values may be placeholder 0)
    2) EXPLICITLY call ParsedDocument.prepare_blocks(doc_id, blocks)
    3) Then construct ParsedDocument(...)
    """
    ParsedDocument.prepare_blocks(doc_id, blocks)
    return ParsedDocument(
        document_id=doc_id,
        source_file=source,
        title=title,
        metadata=dict(meta or {}),
        blocks=blocks,
        parser_name=parser,
        parser_version=ver,
    )


# =============================================================================
# 1. Heading (valid)
# =============================================================================

class TestHeading:
    def test_valid_heading_levels_1_to_9(self):
        for lvl in range(1, 10):
            b = _h(f"标题{lvl}", level=lvl)
            ParsedDocument.prepare_blocks("D", [b])
            assert b.type is BlockType.HEADING
            assert b.level == lvl

    def test_heading_level_and_page_provenance(self):
        b = _h("总则", level=2, page=5)
        ParsedDocument.prepare_blocks("D", [b])
        prov = b.build_provenance("law.md")
        assert prov.source_file == "law.md"
        assert prov.page == 5

    def test_heading_in_non_heading_type_level_must_be_none(self):
        b = _p("正文")
        assert b.level is None


# =============================================================================
# 2. Invalid heading level
# =============================================================================

class TestInvalidHeading:
    def test_heading_level_0_raises(self):
        with pytest.raises(ContractViolation, match="HEADING level must be in 1..9"):
            _h("H", level=0)

    def test_heading_level_10_raises(self):
        with pytest.raises(ContractViolation, match="HEADING level must be in 1..9"):
            _h("H", level=10)

    def test_heading_level_none_raises(self):
        with pytest.raises(ContractViolation, match="HEADING block requires level"):
            Block(block_id="x", type=BlockType.HEADING, text="X", order=0, level=None)

    def test_paragraph_accidental_level_raises(self):
        with pytest.raises(ContractViolation, match="level must be None"):
            Block(block_id="x", type=BlockType.PARAGRAPH, text="X", order=0, level=2)


# =============================================================================
# 3. Paragraph
# =============================================================================

class TestParagraph:
    def test_paragraph_plain_text(self):
        p = _p("第一条 为了规范个人信息处理活动。")
        assert p.type is BlockType.PARAGRAPH
        assert "个人信息" in p.text

    def test_paragraph_order_invariant(self):
        with pytest.raises(ContractViolation, match="Block.order must be >=0"):
            Block(block_id="x", type=BlockType.PARAGRAPH, text="X", order=-1)


# =============================================================================
# 4. List Item
# =============================================================================

class TestListItem:
    def test_plain_list_item(self):
        li = _li("申请材料", level=0)
        assert li.metadata.get("list_level") == 0

    def test_nested_list_item(self):
        li = _li("加盖公章复印件", level=1)
        assert li.metadata.get("list_level") == 1


# =============================================================================
# 5. Table
# =============================================================================

class TestTable:
    def test_table_to_markdown_basic(self):
        tb = _table_block(["序号", "品名", "数量"], [["1", "苹果", "10"], ["2", "橘子", "20"]])
        md = tb.table_data.to_markdown()
        assert md.startswith("| 序号 | 品名 | 数量 |")
        assert "| 1 | 苹果 | 10 |" in md
        assert tb.text == md

    def test_table_with_caption(self):
        tb = _table_block(["品名", "规格"], [["A4 纸", "70g"]],
                          caption="表 3-1 办公用品明细")
        md = tb.table_data.to_markdown()
        assert md.startswith("**表 3-1 办公用品明细**")

    def test_table_cell_escape_pipe_and_newline(self):
        c = TableCell("带|管\n第二行")
        assert c.display_text() == "带\\|管<br>第二行"

    def test_table_invariant_text_must_equal_markdown(self):
        td = TableData(headers=[TableCell("A"), TableCell("B")],
                       rows=[[TableCell("1"), TableCell("2")]])
        correct_text = td.to_markdown()
        Block(block_id="x", type=BlockType.TABLE, text=correct_text, order=0, table_data=td)
        with pytest.raises(ContractViolation, match="must equal table_data.to_markdown"):
            Block(block_id="x", type=BlockType.TABLE, text="WRONG", order=0, table_data=td)

    def test_tablecell_invalid_colspan(self):
        with pytest.raises(ContractViolation, match="colspan must be >= 1"):
            TableCell("x", colspan=0)


# =============================================================================
# 6. Invalid table
# =============================================================================

class TestInvalidTable:
    def test_table_data_row_length_mismatch(self):
        with pytest.raises(ContractViolation, match="cells, headers have"):
            TableData(headers=[TableCell("A"), TableCell("B"), TableCell("C")],
                      rows=[[TableCell("1"), TableCell("2")]])

    def test_table_data_empty_headers(self):
        with pytest.raises(ContractViolation, match="headers cannot be empty"):
            TableData(headers=[], rows=[])

    def test_table_block_missing_table_data(self):
        with pytest.raises(ContractViolation, match="TABLE block requires table_data"):
            Block(block_id="x", type=BlockType.TABLE, text="anything", order=0, table_data=None)

    def test_paragraph_accidental_table_data_raises(self):
        td = TableData(headers=[TableCell("A")], rows=[[TableCell("1")]])
        with pytest.raises(ContractViolation, match="table_data must be None"):
            Block(block_id="x", type=BlockType.PARAGRAPH, text="X", order=0, table_data=td)


# =============================================================================
# 7. Metadata exclusion from legacy evidence body
# =============================================================================

class TestMetadataExclusion:
    def test_metadata_excluded_by_default(self):
        doc = _doc([
            _h("总则", level=1),
            _p("本合同由甲乙双方订立。"),
            _meta("文章来源：法规局"),
            _meta("发布时间：2023-06-28"),
        ])
        txt = LegacyAdapter.to_legacy_text(doc)
        assert "总则" in txt
        assert "甲乙双方" in txt
        assert "文章来源" not in txt
        assert "发布时间" not in txt

    def test_metadata_included_when_flagged(self):
        doc = _doc([_h("办法", level=1), _meta("来源：法规局")])
        txt = LegacyAdapter.to_legacy_text(doc, include_metadata=True)
        assert "来源：法规局" in txt
        assert "<!-- METADATA" in txt


# =============================================================================
# 8. Unknown preservation (HTML comment marker, no silent drop)
# =============================================================================

class TestUnknownPreservation:
    def test_unknown_is_marked_not_silently_dropped(self):
        doc = _doc([
            _p("合法正文段落"),
            _unknown("（此处 Parser 无法识别的扫描件残片 cid:123）"),
        ])
        txt = LegacyAdapter.to_legacy_text(doc)
        assert "合法正文段落" in txt
        assert "cid:123" in txt
        assert "<!-- UNKNOWN BLOCK" in txt


# =============================================================================
# 9. Reading order + FREEZE PATCH new tests
# =============================================================================

class TestReadingOrderAndFreezePatch:
    # ----- original: proper order passes -----
    def test_doc_blocks_order_matches_index(self):
        blocks = [_h("标题", 1), _h("第一章", 2), _p("第一条 ..."), _p("第二条 ...")]
        doc = _doc(blocks)
        for i, b in enumerate(doc.blocks):
            assert b.order == i

    def test_doc_validate_passes_for_proper_order(self):
        doc = _doc([_p("A"), _p("B"), _p("C")])
        doc.validate()

    def test_doc_empty_source_file_fails(self):
        with pytest.raises(ContractViolation, match="source_file cannot be empty"):
            ParsedDocument(document_id="DOC", source_file="", title="T", metadata={},
                           blocks=[], parser_name="v2", parser_version="2.0.0")

    # ----- FREEZE PATCH §6 Test 1: wrong order → NO implicit repair, CV raised -----
    def test_wrong_block_order_no_implicit_repair_raises(self):
        """
        blocks: index 0 → order 0
                index 1 → order 5  (WRONG)
                index 2 → order 2
        Must fail visibly. Constructor shall NOT auto-renumber.
        """
        b0 = Block(block_id="D_b0000", type=BlockType.PARAGRAPH, text="a", order=0)
        b1 = Block(block_id="D_b0001", type=BlockType.PARAGRAPH, text="b", order=5)  # WRONG
        b2 = Block(block_id="D_b0002", type=BlockType.PARAGRAPH, text="c", order=2)
        with pytest.raises(ContractViolation, match="blocks\\[1\\]\\.order == 5, expected 1"):
            ParsedDocument(
                document_id="D",
                source_file="x.md",
                title="T",
                metadata={},
                blocks=[b0, b1, b2],
                parser_name="v2",
                parser_version="2.0.0",
            )

    # ----- FREEZE PATCH §6 Test 2: Block.order != provenance_stored.block_order -----
    def test_order_provenance_block_order_mismatch_raises(self):
        b = Block(block_id="x", type=BlockType.PARAGRAPH, text="x", order=7)
        # Provenance with block_order=3 (conflict)
        bad_prov = Provenance(source_file="x.md", page=None, block_order=3)
        object.__setattr__(b, "provenance_stored", bad_prov)
        # Provenance consistency must fail at block __post_init__ check too?
        # Already stored after __post_init__, so validate at doc level:
        ParsedDocument.prepare_blocks("D", [b])
        # After prepare_blocks both order & provenance become 0 (fixed).
        # Now let's recreate with correct block but manually inject bad provenance AFTER
        # prepare_blocks to confirm doc constructor catches it:
        good = Block(block_id="D_b0000", type=BlockType.PARAGRAPH, text="x", order=0)
        wrong_prov = Provenance(source_file="x.md", page=None, block_order=123)
        object.__setattr__(good, "provenance_stored", wrong_prov)
        with pytest.raises(ContractViolation, match="provenance_stored\\.block_order == 123"):
            ParsedDocument(document_id="D", source_file="x.md", title="T", metadata={},
                           blocks=[good], parser_name="v2", parser_version="2.0.0")

    # ----- FREEZE PATCH §6 Test 3: correct triple equality (index == order == prov) -----
    def test_index_order_provenance_triple_consistency_passes(self):
        blocks = [
            Block(block_id="__", type=BlockType.PARAGRAPH, text="A", order=99),
            Block(block_id="__", type=BlockType.PARAGRAPH, text="B", order=99),
            Block(block_id="__", type=BlockType.PARAGRAPH, text="C", order=99),
        ]
        # Attach provenance EARLY (dirty block orders) before prepare_blocks.
        # Use attach_provenance so block-level consistency holds (order==prov.block_order)
        for b in blocks:
            b.attach_provenance("contract.md")
        # After attach, each block has prov.block_order == 99 (== each block.order at that moment).
        # Now EXPLICITLY run prepare_blocks → should ALSO update provenance.
        ParsedDocument.prepare_blocks("DOC", blocks)

        # Now triple equality:
        for i, b in enumerate(blocks):
            assert b.order == i, f"block[{i}].order={b.order} expected {i}"
            assert b.provenance_stored is not None
            assert b.provenance_stored.block_order == i, \
                f"block[{i}] prov.block_order={b.provenance_stored.block_order} expected {i}"
        # Doc construction must succeed:
        doc = ParsedDocument(document_id="DOC", source_file="contract.md", title="T",
                             metadata={}, blocks=blocks,
                             parser_name="v2", parser_version="2.0.0")
        doc.validate()

    # ----- FREEZE PATCH §6 Test 4: explicit renumber_blocks on existing doc -----
    def test_explicit_renumber_syncs_order_and_provenance(self):
        # Build doc normally (order 0,1,2)
        blocks = [_p("A"), _p("B"), _p("C")]
        for b in blocks:
            b.attach_provenance("x.docx")  # order==provenance_block_order each
        doc = _doc(blocks)

        # UNSAFELY damage state (simulate Parser-side append/insert bug):
        # blocks are [A,B,C] → we insert new block so indices 0,1,2,3 but orders remain old
        new_d = _p("D")
        new_d.attach_provenance("x.docx")
        doc.blocks.insert(2, new_d)  # now [A,B,D,C]
        # Also damage provenance for block[0] via replacement
        object.__setattr__(doc.blocks[0], "order", 999)
        object.__setattr__(
            doc.blocks[0],
            "provenance_stored",
            doc.blocks[0].provenance_stored.with_block_order(999),
        )

        # validate() must FAIL now
        with pytest.raises(ContractViolation):
            doc.validate()

        # EXPLICIT renumber
        doc.renumber_blocks()

        # Now should be consistent: index == order == provenance.block_order
        doc.validate()
        for i, b in enumerate(doc.blocks):
            assert b.order == i
            assert b.provenance_stored is not None
            assert b.provenance_stored.block_order == i


# =============================================================================
# 10. Legacy adapter rendering
# =============================================================================

class TestToLegacyText:
    def test_heading_renders_markdown_prefix(self):
        doc = _doc([_h("一级标题", level=1), _h("四级标题", level=4)])
        out = LegacyAdapter.to_legacy_text(doc)
        assert "# 一级标题" in out
        assert "#### 四级标题" in out

    def test_list_item_renders_with_indent(self):
        doc = _doc([_li("一级项", level=0), _li("二级项", level=1)])
        out = LegacyAdapter.to_legacy_text(doc)
        assert out.startswith("- 一级项") or "- 一级项" in out
        assert "\n  - 二级项" in out

    def test_table_in_legacy_output_is_markdown(self):
        doc = _doc([
            _h("对比表", level=2),
            _table_block(["条款", "内容"], [["第一条", "守法"], ["第二条", "保密"]],
                         caption="条款表"),
        ])
        out = LegacyAdapter.to_legacy_text(doc)
        assert "## 对比表" in out
        assert "**条款表**" in out
        assert "| 条款 | 内容 |" in out
        assert "| 第一条 | 守法 |" in out

    def test_legacy_parsed_document_structure_matches_v1(self):
        doc = _doc([_h("A", 1), _p("B")], meta={"title": "A", "extension": ".md"})
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)
        assert isinstance(legacy, LegacyParsedDocument)
        assert isinstance(legacy.text, str)
        assert isinstance(legacy.metadata, dict)
        assert isinstance(legacy.segments, tuple)
        assert len(legacy.segments) == 1

    def test_legacy_per_page_segments_when_page_info_present(self):
        doc = _doc([
            _h("封面", level=1, page=1),
            _p("目录...", page=1),
            _h("第一章", level=2, page=2),
            _p("正文", page=2),
            _p("附录", page=4),
        ])
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)
        assert len(legacy.segments) == 4
        assert "封面" in legacy.segments[0]
        assert "第一章" in legacy.segments[1]
        assert legacy.segments[2] == ""  # page 3 gap
        assert "附录" in legacy.segments[3]


# =============================================================================
# 11. Provenance + block_id factory
# =============================================================================

class TestProvenanceAndBlockId:
    def test_provenance_minimal_fields(self):
        p = Provenance(source_file="c.pdf", page=3, block_order=12)
        assert p.source_file == "c.pdf" and p.page == 3 and p.block_order == 12

    def test_provenance_page_none_ok(self):
        assert Provenance(source_file="x.md", page=None, block_order=0).page is None

    def test_provenance_invalid_block_order(self):
        with pytest.raises(ContractViolation, match="block_order must be >=0"):
            Provenance(source_file="x.md", page=None, block_order=-1)

    def test_provenance_empty_source_fails(self):
        with pytest.raises(ContractViolation, match="source_file cannot be empty"):
            Provenance(source_file="", page=None, block_order=0)

    def test_block_id_format(self):
        assert Block.make_block_id("ENT_CONTRACT_011", 7) == "ENT_CONTRACT_011_b0007"


# =============================================================================
# 12. Doc identity invariants
# =============================================================================

class TestDocIdentityInvariants:
    def test_empty_document_id_fails(self):
        with pytest.raises(ContractViolation, match="document_id cannot be empty"):
            ParsedDocument(document_id="", source_file="x.md", title="T", metadata={},
                           blocks=[], parser_name="v2", parser_version="2.0.0")

    def test_empty_parser_name_or_version(self):
        with pytest.raises(ContractViolation, match="parser_name / parser_version"):
            ParsedDocument(document_id="D", source_file="x.md", title="T", metadata={},
                           blocks=[], parser_name="", parser_version="")


# =============================================================================
# 13. BlockType enum coverage
# =============================================================================

class TestBlockTypeEnum:
    def test_block_type_has_exactly_six(self):
        names = sorted(b.value for b in BlockType)
        assert names == sorted(["HEADING", "PARAGRAPH", "LIST_ITEM",
                                 "TABLE", "METADATA", "UNKNOWN"])

    def test_unknown_block_type_string_rejected(self):
        with pytest.raises(ContractViolation, match="not in BlockType enum"):
            Block(block_id="x", type="FOOTER", text="y", order=0)  # type: ignore[arg-type]

    def test_block_type_string_case_insensitive(self):
        b = Block(block_id="x", type="heading", text="T", order=0, level=1)  # type: ignore[arg-type]
        assert b.type is BlockType.HEADING


# =============================================================================
# 14. FREEZE PATCH §3 — REAL CURRENT CHUNKER SMOKE
#     Pure in-memory. No DB/Milvus/Redis/network.
#     Uses ACTUAL modules.ingestion.chunking.chunk_pdf_pages_to_parents()
#          AND ACTUAL modules.ingestion.chunking.split_children_from_parent().
# =============================================================================

class TestRealCurrentChunkerCompatibility:
    """
    Actual public entrypoints (discovered by reading chunking.py):
        1) chunk_pdf_pages_to_parents(page_texts: list[str], max_chars=1800)
              -> list[ParentChunk]
        2) split_children_from_parent(parent_text, parent_index, child_chars=512, overlap=128)
              -> list[ChildChunk]

    Both are 100% duck typed: chunking module imports re + dataclasses ONLY.
    No isinstance() check against V1 ParsedDocument. No nominal dependency.
    """

    CHUNKER_PUBLIC_ENTRYPOINTS = (chunk_pdf_pages_to_parents, split_children_from_parent)

    # --- helpers ---
    @staticmethod
    def _make_v2_doc():
        """
        Covers: HEADING + PARAGRAPH + LIST_ITEM + TABLE + METADATA + UNKNOWN
        Minimum sensible sizes to allow chunks to form.
        """
        para = (
            "为规范企业法律纠纷案件管理，健全依法维权和化解纠纷机制，"
            "维护企业合法权益，根据《中华人民共和国公司法》《中华人民共和国企业国有资产法》"
            "等法律法规，制定本办法。本办法所称法律纠纷案件，是指中央企业作为当事人"
            "的各类民商事案件、行政案件、刑事案件以及可能引发上述案件的纠纷事件。"
        )
        para2 = (
            "中央企业应当依法独立处理本企业及所属子企业发生的各类法律纠纷案件，"
            "落实企业主要负责人法治建设第一责任人职责，完善案件管理制度，"
            "建立重大案件风险防控机制，保障案件处理所需经费，加强法务队伍建设。"
        )
        para3 = (
            "中央企业应当建立法律纠纷案件备案制度，对重大法律纠纷案件实行备案管理。"
            "中央企业发生重大法律纠纷案件，应当自案件发生之日起一定时限内报国资委备案。"
            "备案内容包括案件基本情况、企业处理方案、企业法务机构意见等。"
        )
        blocks = [
            _h("中央企业法律纠纷案件管理办法", level=1),
            _h("第一章 总则", level=2),
            _p(para),
            _li("（一）明确处理标准", level=0),
            _li("1. 先法律评估后决策", level=1),
            _li("（二）强化重大案件管控", level=0),
            _table_block(
                ["类别", "标准", "报送时限"],
                [
                    ["重大案件", "标的额 ≥ 5000 万元", "30 日"],
                    ["特别重大案件", "标的额 ≥ 1 亿元", "15 日"],
                ],
                caption="表 1 案件分级标准",
            ),
            _p(para2),
            _p(para3),
            # Metadata (should NOT appear in evidence body)
            _meta("文章来源：法规局"),
            _meta("发布时间：2023-06-28"),
            # Unknown: synthetic debug marker, preserved but not trusted as evidence
            _unknown("（Parser V2 fallback: CMS div stripped; raw fragment retained）"),
        ]
        return _doc(blocks, doc_id="ENT_POLICY_002_CHUNKER", source="ent_policy_002.md",
                    title="中央企业法律纠纷案件管理办法",
                    meta={"source_org": "国资委", "category": "Enterprise Compliance",
                          "publish_date": "2023-06-28"})

    def test_duck_typing_compatibility_verified(self):
        """
        chunking.py only accepts strings/list[str]. No isinstance() against V1 ParsedDocument.
        → DUCK-TYPING COMPATIBILITY = VERIFIED
        """
        import inspect
        chunker_src = inspect.getsource(chunk_pdf_pages_to_parents) + \
                      inspect.getsource(split_children_from_parent)
        # No isinstance(..., ParsedDocument) reference of any kind.
        assert "isinstance" not in chunker_src
        assert "ParsedDocument" not in chunker_src  # nominal type absent

    def test_metadata_excluded_from_evidence_text(self):
        doc = self._make_v2_doc()
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)
        evidence = legacy.text
        # Heading + Paragraph must be present
        assert "中央企业法律纠纷案件管理办法" in evidence
        assert "公司法" in evidence
        assert "备案" in evidence
        # Metadata must be absent by default
        assert "文章来源" not in evidence
        assert "发布时间" not in evidence
        # Unknown: debug marker text retained (not silently dropped)
        assert "CMS div stripped" in evidence
        assert "<!-- UNKNOWN BLOCK" in evidence

    def test_heading_paragraph_table_rendered_in_evidence(self):
        doc = self._make_v2_doc()
        evidence = LegacyAdapter.to_legacy_text(doc)
        # HEADING render
        assert "# 中央企业法律纠纷案件管理办法" in evidence
        assert "## 第一章 总则" in evidence
        # TABLE render (caption + header + row)
        assert "**表 1 案件分级标准**" in evidence
        assert "| 类别 | 标准 | 报送时限 |" in evidence
        assert "| 重大案件 | 标的额 ≥ 5000 万元 | 30 日 |" in evidence
        # LIST render
        assert "- （一）明确处理标准" in evidence
        assert "  - 1. 先法律评估后决策" in evidence

    def test_smoke_actual_chunker_produces_parent_chunks(self):
        """
        ParsedDocument V2 → LegacyAdapter → legacy.segments →
          chunk_pdf_pages_to_parents(list(segments)) → ParentChunk list
        → NO DB / Milvus. Pure in-memory structural flow.
        """
        doc = self._make_v2_doc()
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)

        # Feed segments into ACTUAL chunker
        page_texts = list(legacy.segments)  # tuple[str] → list[str]
        parents = chunk_pdf_pages_to_parents(page_texts, max_chars=600)

        # We got at least 1 parent
        assert isinstance(parents, list) and len(parents) >= 1
        for p in parents:
            assert isinstance(p, ParentChunk)
            assert isinstance(p.title, str) and p.title
            assert isinstance(p.text, str)
            assert isinstance(p.page_no, int) and p.page_no >= 1

    def test_smoke_actual_chunker_produces_child_chunks(self):
        """
        parents → split_children_from_parent() → ChildChunk list
        Verifies ParentChunk text is a string current chunker can consume.
        """
        doc = self._make_v2_doc()
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)
        parents = chunk_pdf_pages_to_parents(list(legacy.segments), max_chars=600)

        all_children: list[ChildChunk] = []
        for idx, parent in enumerate(parents):
            children = split_children_from_parent(parent.text, parent_index=idx,
                                                  child_chars=256, overlap=64)
            for c in children:
                assert isinstance(c, ChildChunk)
                assert c.parent_index == idx
                assert c.chunk_index >= 0
                assert isinstance(c.text, str) and c.text.strip()
            all_children.extend(children)

        # Must have produced children (inputs are long enough)
        assert len(all_children) >= 3, "Expected ≥ 3 child chunks across parents"

        # Parent index references must be valid
        max_pindex = max(c.parent_index for c in all_children) if all_children else -1
        assert max_pindex < len(parents), "child.parent_index out of parents range"

    def test_smoke_evidence_survives_in_child_text(self):
        """Crucial: does real chunker output contain substantive evidence snippets?"""
        doc = self._make_v2_doc()
        legacy = LegacyAdapter.to_legacy_parsed_document(doc)
        parents = chunk_pdf_pages_to_parents(list(legacy.segments), max_chars=800)
        all_child_texts: list[str] = []
        for idx, p in enumerate(parents):
            for c in split_children_from_parent(p.text, parent_index=idx, child_chars=256):
                all_child_texts.append(c.text)
        merged = "\n".join(all_child_texts)
        # Substantive content (Heading / Para / Table cell) reachable via children
        assert "法律纠纷案件" in merged
        assert "公司法" in merged
        assert "重大案件" in merged
        assert "5000 万元" in merged  # table content preserved via legacy md table
        # Metadata still excluded in children
        assert "文章来源" not in merged
        assert "发布时间" not in merged


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
