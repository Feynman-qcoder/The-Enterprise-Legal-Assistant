"""
Test Chunk Metadata Contract V1 — 16+ frozen validation invariants.

ISOLATED.  Uses stdlib unittest only.
Reads: chunk_metadata_contract_v1.py
Does NOT import: Parser / Cleaner / Normalizer / QGate / Milvus / MySQL.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from chunk_metadata_contract_v1 import (  # noqa: E402
    ChunkMetadata,
    ChunkStructuralContext,
    ChunkProvenance,
    ChunkTableContext,
    ChunkQualityFlags,
    ChunkParentChildInfo,
    ChunkContractViolation,
    compute_chunk_id_contract,
    DOCUMENT_METADATA_PROJECTION_MAP,
    ChunkLevel,
    FilterCandidateClass,
    EmbeddingPolicy,
    FieldSourceOwner,
    MetadataLayer,
    SourceFormat,
    CHUNK_METADATA_CONTRACT_V1_INVARIANTS,
)


# ============================================================================
# Helpers — build a known-valid "happy path" ChunkMetadata scaffold
# ============================================================================

SAMPLE_SOURCE_SHA256 = "a" * 64   # 64 hex chars, placeholder
SAMPLE_DOCUMENT_ID = "LEGAL_DATA_001"
SAMPLE_LOGICAL_DOCUMENT_ID = "LAW_001"
SAMPLE_SOURCE_FILE = r"D:\xiaoyi\data_source\derived\md\LEGAL_DATA_001_个人信息保护法.md"
SAMPLE_SOURCE_FORMAT = "md"
SAMPLE_PARSER_NAME = "txt_md_wrapper_v2"
SAMPLE_PARSER_VERSION = "2.1.0-frozen"


def _make_happy_provenance(
    block_orders: list[int] | None = None,
    pages: list[int] | None = None,
    parser_name: str = SAMPLE_PARSER_NAME,
    parser_version: str = SAMPLE_PARSER_VERSION,
) -> ChunkProvenance:
    bos = [2, 3, 4] if block_orders is None else block_orders
    return ChunkProvenance(
        source_block_orders=bos,
        source_block_ids=[f"{SAMPLE_DOCUMENT_ID}_b{b:04d}" for b in bos],
        page_start=pages[0] if pages else None,
        page_end=pages[-1] if pages else None,
        page_numbers=pages or [],
        parser_name=parser_name,
        parser_version=parser_version,
    )


def _make_happy_chunk(
    *,
    chunk_index: int = 0,
    block_orders: list[int] | None = None,
    pages: list[int] | None = None,
    overrides: dict[str, Any] | None = None,
) -> ChunkMetadata:
    """Build a fully-valid sample.  Use overrides to test specific invariants."""
    bos = [2, 3, 4] if block_orders is None else block_orders
    prov = _make_happy_provenance(bos, pages)
    start = prov.original_block_order_start
    end = prov.original_block_order_end
    cid = compute_chunk_id_contract(
        source_sha256=SAMPLE_SOURCE_SHA256,
        document_id=SAMPLE_DOCUMENT_ID,
        chunk_index=chunk_index,
        original_block_order_start=start,
        original_block_order_end=end,
    )
    base = dict(
        chunk_id=cid,
        document_id=SAMPLE_DOCUMENT_ID,
        logical_document_id=SAMPLE_LOGICAL_DOCUMENT_ID,
        source_file=SAMPLE_SOURCE_FILE,
        source_sha256=SAMPLE_SOURCE_SHA256,
        source_format=SAMPLE_SOURCE_FORMAT,
        chunk_index=chunk_index,
        # ---- LAYER B: projected doc metadata (sample, not all available) ----
        title="中华人民共和国个人信息保护法",
        source_org="全国人民代表大会常务委员会",
        publish_date="2021-08-20",
        creation_date=None,
        effective_date="2021-11-01",
        expiry_date=None,
        document_number=None,
        document_index_number=None,
        document_type="法律",
        authority_level="法律",
        legal_status="现行有效",
        jurisdiction="全国",
        region=None,
        theme_category=None,
        source_url="https://www.cac.gov.cn/2021-08/20/c_1631050028355286.htm",
        structural=ChunkStructuralContext(
            chapter="第一章 总则",
            section=None,
            article="第一条",
            heading_path=[
                "中华人民共和国个人信息保护法",
                "第一章 总则",
            ],
            heading_level=1,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
        provenance=prov,
        table_context=ChunkTableContext(contains_table=False),
        parent_child=ChunkParentChildInfo(
            parent_chunk_id=None, chunk_level=ChunkLevel.ATOMIC.value
        ),
        quality=ChunkQualityFlags(
            quality_verdict="PASS",
            quality_warnings=[],
            has_review_content=False,
            review_rule_ids=[],
        ),
        extras={},
    )
    if overrides:
        base.update(overrides)
    return ChunkMetadata(**base)


# ============================================================================
# 16+ Validation Invariants
# ============================================================================

class TestChunkIdentityDeterminism(unittest.TestCase):
    """§3 — chunk_id contract determinism."""

    def test_chunk_id_deterministic_same_inputs(self):
        """Same inputs → identical chunk_id."""
        kwargs = dict(
            source_sha256=SAMPLE_SOURCE_SHA256,
            document_id=SAMPLE_DOCUMENT_ID,
            chunk_index=3,
            original_block_order_start=5,
            original_block_order_end=8,
        )
        a = compute_chunk_id_contract(**kwargs)
        b = compute_chunk_id_contract(**kwargs)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in a))

    def test_chunk_id_different_inputs_produce_different_id(self):
        base = dict(
            source_sha256=SAMPLE_SOURCE_SHA256,
            document_id=SAMPLE_DOCUMENT_ID,
            chunk_index=0,
            original_block_order_start=0,
            original_block_order_end=0,
        )
        a = compute_chunk_id_contract(**base)
        for mutation in (
            dict(chunk_index=1),
            dict(original_block_order_start=1, original_block_order_end=1),
            dict(source_sha256="b" * 64),
            dict(document_id="OTHER_DOC"),
        ):
            b = compute_chunk_id_contract(**{**base, **mutation})
            self.assertNotEqual(a, b, f"mutation {mutation} produced same id")

    def test_chunk_id_invalid_inputs_raise(self):
        with self.assertRaises(ChunkContractViolation):
            compute_chunk_id_contract(
                source_sha256="",
                document_id=SAMPLE_DOCUMENT_ID,
                chunk_index=0,
                original_block_order_start=0,
                original_block_order_end=0,
            )
        with self.assertRaises(ChunkContractViolation):
            compute_chunk_id_contract(
                source_sha256=SAMPLE_SOURCE_SHA256,
                document_id="",
                chunk_index=0,
                original_block_order_start=0,
                original_block_order_end=0,
            )
        with self.assertRaises(ChunkContractViolation):
            compute_chunk_id_contract(
                source_sha256=SAMPLE_SOURCE_SHA256,
                document_id=SAMPLE_DOCUMENT_ID,
                chunk_index=-1,
                original_block_order_start=0,
                original_block_order_end=0,
            )


class TestInvariant1ChunkId(unittest.TestCase):
    """INV1: chunk_id non-empty, length 32 hex lowercase."""

    def test_happy_path_ok(self):
        c = _make_happy_chunk()
        self.assertEqual(c.validate(), [])

    def test_empty_chunk_id_fail(self):
        c = _make_happy_chunk(overrides=dict(chunk_id=""))
        vs = c.validate()
        self.assertTrue(any("INV1" in v for v in vs), vs)

    def test_wrong_length_fail(self):
        c = _make_happy_chunk(overrides=dict(chunk_id="a" * 31))
        vs = c.validate()
        self.assertTrue(any("INV1" in v for v in vs), vs)

    def test_uppercase_fail(self):
        c = _make_happy_chunk(overrides=dict(chunk_id="A" * 32))
        vs = c.validate()
        self.assertTrue(any("INV1" in v for v in vs), vs)


class TestInvariant2DocumentIdentity(unittest.TestCase):
    """INV2: document identity non-empty."""

    def test_empty_document_id_fail(self):
        c = _make_happy_chunk(overrides=dict(document_id=""))
        vs = c.validate()
        self.assertTrue(any("INV2" in v for v in vs), vs)


class TestInvariant3SourceIdentity(unittest.TestCase):
    """INV3: source identity; source_sha256 length 64 hex; chunk_index >=0."""

    def test_empty_source_file_fail(self):
        c = _make_happy_chunk(overrides=dict(source_file=""))
        vs = c.validate()
        self.assertTrue(any("INV3" in v for v in vs), vs)

    def test_bad_sha256_length_fail(self):
        c = _make_happy_chunk(overrides=dict(source_sha256="abc"))
        vs = c.validate()
        self.assertTrue(any("INV3" in v for v in vs), vs)

    def test_negative_chunk_index_fail(self):
        c = _make_happy_chunk(overrides=dict(chunk_index=-1))
        vs = c.validate()
        self.assertTrue(any("INV3" in v for v in vs), vs)


class TestInvariant4_5_6_7BlockOrders(unittest.TestCase):
    """INV4 non-empty; INV5 monotonic; INV6 no dup; INV7 start/end consistent."""

    def test_empty_block_orders_fail(self):
        with self.assertRaises(ChunkContractViolation):
            _make_happy_provenance([])

    def test_non_monotonic_fail(self):
        with self.assertRaises(ChunkContractViolation):
            _make_happy_provenance([1, 3, 2])

    def test_duplicates_fail(self):
        with self.assertRaises(ChunkContractViolation):
            _make_happy_provenance([1, 2, 2, 3])

    def test_start_end_inconsistent_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkProvenance(
                source_block_orders=[2, 3, 4],
                original_block_order_start=1,   # mismatched (should be 2)
                original_block_order_end=4,
            )
        with self.assertRaises(ChunkContractViolation):
            ChunkProvenance(
                source_block_orders=[2, 3, 4],
                original_block_order_start=2,
                original_block_order_end=5,  # mismatched (should be 4)
            )

    def test_id_count_mismatch_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkProvenance(
                source_block_orders=[1, 2, 3],
                source_block_ids=["only_one"],
            )


class TestInvariant8_9Pages(unittest.TestCase):
    """INV8 page_start <= page_end; INV9 page_numbers within range."""

    def test_reversed_page_range_fail(self):
        with self.assertRaises(ChunkContractViolation):
            _make_happy_provenance([1, 2], pages=[10, 9, 8])  # page_end computed as 8 < 10 start
        # Explicit bad construction
        with self.assertRaises(ChunkContractViolation):
            ChunkProvenance(
                source_block_orders=[1],
                page_start=5,
                page_end=3,
            )

    def test_page_number_out_of_range_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkProvenance(
                source_block_orders=[1, 2],
                page_start=3,
                page_end=5,
                page_numbers=[3, 4, 6],   # 6 not in [3..5]
            )


class TestInvariant10StructuralNulls(unittest.TestCase):
    """chapter/section/article: may be null, must be non-empty if set."""

    def test_nulls_ok(self):
        s = ChunkStructuralContext(
            chapter=None, section=None, article=None,
            heading_path=[], heading_level=None,
            block_type="PARAGRAPH", content_type="LEGAL_TEXT",
        )
        # No exception raised

    def test_empty_string_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkStructuralContext(
                chapter="   ",   # whitespace-only is treated as empty in validate
                block_type="PARAGRAPH", content_type="LEGAL_TEXT",
            )
        # Also: block_type vocabulary
        with self.assertRaises(ChunkContractViolation):
            ChunkStructuralContext(block_type="FOO", content_type="LEGAL_TEXT")
        # content_type vocabulary
        with self.assertRaises(ChunkContractViolation):
            ChunkStructuralContext(block_type="PARAGRAPH", content_type="BAR")
        # heading_level range
        with self.assertRaises(ChunkContractViolation):
            ChunkStructuralContext(heading_level=0, block_type=None)
        with self.assertRaises(ChunkContractViolation):
            ChunkStructuralContext(heading_level=10, block_type=None)


class TestInvariant11SentinelsForbidden(unittest.TestCase):
    """No internal sentinels in any public field value."""

    def test_sentinel_in_title_fail(self):
        c = _make_happy_chunk(overrides=dict(title="prefix__METADATA_BLOCK_VALUE__suffix"))
        vs = c.validate()
        self.assertTrue(any("INV11" in v for v in vs), vs)

    def test_sentinel_in_structural_chapter_fail(self):
        structural = ChunkStructuralContext(
            chapter="第一编 __CHUNKER_INTERNAL__ contract",
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        )
        c = _make_happy_chunk(overrides=dict(structural=structural))
        vs = c.validate()
        self.assertTrue(any("INV11" in v for v in vs), vs)


class TestInvariant12StrictSchema(unittest.TestCase):
    """Unknown extra fields rejected (extras must be empty)."""

    def test_extras_not_empty_fail(self):
        c = _make_happy_chunk(overrides=dict(extras={"future_field_x": 1}))
        vs = c.validate()
        self.assertTrue(any("INV12" in v for v in vs), vs)


class TestInvariant13TableConsistency(unittest.TestCase):
    """TABLE metadata consistent; table block orders subset of provenance."""

    def test_table_flag_without_refs_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkTableContext(contains_table=True)

    def test_table_flag_false_but_refs_set_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkTableContext(contains_table=False, table_ids=["t1"])

    def test_table_orders_not_in_provenance_fail(self):
        prov = _make_happy_provenance([10, 11, 12])
        table = ChunkTableContext(
            contains_table=True,
            source_table_block_orders=[11, 20],   # 20 not in provenance
        )
        c = _make_happy_chunk(overrides=dict(provenance=prov, table_context=table,
                                             chunk_index=1))
        vs = c.validate()
        self.assertTrue(any("INV13" in v for v in vs), vs)

    def test_table_orders_within_provenance_ok(self):
        prov = _make_happy_provenance([10, 11, 12])
        table = ChunkTableContext(
            contains_table=True,
            source_table_block_orders=[11],
            table_ids=["doc_b0011_table"],
            table_header_preserved=True,
            table_structure_preserved=True,
            table_part_index=0,
            table_parts_total=2,
        )
        c = _make_happy_chunk(
            chunk_index=1,
            block_orders=[10, 11, 12],
            overrides=dict(provenance=prov, table_context=table),
        )
        # no INV13 violation
        vs = c.validate()
        self.assertFalse(any("INV13" in v for v in vs), vs)


class TestInvariant14ParentChildOptional(unittest.TestCase):
    """parent_chunk_id optional; vocabulary validated."""

    def test_chunk_level_wrong_value_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkParentChildInfo(chunk_level="SUPER")

    def test_parent_child_happy_variants_ok(self):
        # ATOMIC (flat)
        ChunkParentChildInfo(chunk_level=ChunkLevel.ATOMIC.value)
        # PARENT
        ChunkParentChildInfo(chunk_level="PARENT", parent_chunk_id=None)
        # CHILD with parent reference
        ChunkParentChildInfo(
            chunk_level=ChunkLevel.CHILD.value,
            parent_chunk_id="a" * 32,
        )
        # null (unknown strategy)
        ChunkParentChildInfo(chunk_level=None, parent_chunk_id=None)

    def test_bad_parent_chunk_id_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkParentChildInfo(parent_chunk_id="")
        with self.assertRaises(ChunkContractViolation):
            ChunkParentChildInfo(parent_document_id="")


class TestInvariant15DocMetadataNoOverwriteNoFabrication(unittest.TestCase):
    """No LWW.  No pseudo-null strings."""

    def test_pseudo_null_fabrication_fail(self):
        c = _make_happy_chunk(overrides=dict(source_org="UNKNOWN"))
        vs = c.validate()
        self.assertTrue(any("INV15" in v for v in vs), vs)

    def test_apply_projection_null_inherited_ok(self):
        c = _make_happy_chunk(overrides=dict(
            # Reset projection fields to None so projection can fill them
            title=None, source_org=None, publish_date=None,
            effective_date=None, document_type=None,
            authority_level=None, legal_status=None, jurisdiction=None,
            source_url=None,
        ))
        doc_meta = {
            "title": "个人信息保护法",
            "source_org": "全国人大常委会",
            "publish_date": "2021-08-20",
            "document_type": "法律",
            # fields NOT in the projection map should be ignored
            "custom_field": "should_be_ignored",
        }
        c2 = c.apply_document_metadata_projection(doc_meta)
        self.assertEqual(c2.title, "个人信息保护法")
        self.assertEqual(c2.source_org, "全国人大常委会")
        # custom field NOT propagated
        self.assertFalse(hasattr(c2, "custom_field"))
        # validate ok
        self.assertEqual(c2.validate(), [])

    def test_apply_projection_overwrite_forbidden(self):
        c = _make_happy_chunk()  # title already non-null
        with self.assertRaises(ChunkContractViolation):
            c.apply_document_metadata_projection({"title": "A different title"})

    def test_apply_projection_sentinel_in_source_rejected(self):
        c = _make_happy_chunk(overrides=dict(title=None))
        with self.assertRaises(ChunkContractViolation):
            c.apply_document_metadata_projection({
                "title": "prefix__METADATA_BLOCK_VALUE__",
            })


class TestInvariant16RoundTripStable(unittest.TestCase):
    """to_dict → from_dict → to_dict stable."""

    def test_happy_chunk_roundtrip_ok(self):
        c = _make_happy_chunk()
        d1 = c.to_dict()
        s1 = json.dumps(d1, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        c2 = ChunkMetadata.from_dict(d1)
        d2 = c2.to_dict()
        s2 = json.dumps(d2, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(s1, s2, "round-trip json serialization differs")

    def test_from_dict_unknown_key_raises(self):
        c = _make_happy_chunk()
        d = c.to_dict()
        d["_spurious_future_field"] = 1
        with self.assertRaises(ChunkContractViolation):
            ChunkMetadata.from_dict(d)


class TestInvariant17ChunkIdDeterminismDeep(unittest.TestCase):
    """Chunk's actual chunk_id matches contract rule when computable."""

    def test_mismatch_reported(self):
        c = _make_happy_chunk()
        # Force wrong chunk id
        object.__setattr__(c, "chunk_id", "0" * 32)
        vs = c.validate()
        self.assertTrue(any("INV17" in v for v in vs), vs)

    def test_happy_path_automatically_correct(self):
        c = _make_happy_chunk(chunk_index=5, block_orders=[10, 11, 12, 13, 14])
        vs = c.validate()
        self.assertFalse(any("INV17" in v for v in vs), vs)


class TestQualityFlagsVocabulary(unittest.TestCase):
    """§11 Quality flags vocabulary + never-embedded check."""

    def test_allowed_verdicts(self):
        for v in ("PASS", "WARNING", "POLICY_REVIEW", "REVIEW", None):
            ChunkQualityFlags(quality_verdict=v)

    def test_bad_verdict_fail(self):
        with self.assertRaises(ChunkContractViolation):
            ChunkQualityFlags(quality_verdict="REJECT")


class TestDocumentMetadataProjectionTableComplete(unittest.TestCase):
    """§4 Mapping table: all 18 INHERIT Metadata Contract V1 fields present."""

    def test_all_inherit_fields_covered(self):
        EXPECTED_INHERIT_FIELDS = {
            "document_id", "logical_document_id", "title", "source_org",
            "publish_date", "creation_date", "effective_date", "expiry_date",
            "document_number", "document_index_number", "document_type",
            "authority_level", "legal_status", "jurisdiction", "region",
            "theme_category", "source_url", "source_file",
        }
        actual = set(DOCUMENT_METADATA_PROJECTION_MAP.keys())
        self.assertEqual(actual, EXPECTED_INHERIT_FIELDS,
                         "Projection map must cover exactly the 18 frozen INHERIT fields")

    def test_filter_and_citation_classification_consistent(self):
        for field, proj in DOCUMENT_METADATA_PROJECTION_MAP.items():
            self.assertIsInstance(proj.filterable, FilterCandidateClass)
            self.assertIsInstance(proj.citation_relevant, bool)
            self.assertIsInstance(proj.embedding_policy, EmbeddingPolicy)
            # Ensure no field has NEVER_EMBED policy in projection (it might be
            # appropriate for internal quality only, not projected metadata)
            self.assertNotEqual(proj.embedding_policy, EmbeddingPolicy.NEVER_EMBED,
                                f"Document metadata field {field} has NEVER_EMBED — "
                                "use RETRIEVAL_FILTER_ONLY or CITATION_ONLY instead")


class TestEnumSurfaceStable(unittest.TestCase):
    """Enums stable — no unexpected value removal."""

    def test_source_format_matches_manifest_values(self):
        # From manifest samples: md / pdf / docx + html / txt (canonical 53 has 20/7/26)
        for fmt in ("md", "pdf", "docx", "html", "txt"):
            self.assertIsNotNone(SourceFormat(fmt), f"SourceFormat missing {fmt}")

    def test_metadata_layers_exactly_five(self):
        self.assertEqual(
            {e.value for e in MetadataLayer},
            {"A_IDENTITY", "B_DOCUMENT_METADATA", "C_STRUCTURAL",
             "D_PROVENANCE", "E_RETRIEVAL_RUNTIME"},
        )

    def test_chunk_level_three_values(self):
        self.assertEqual(
            {e.value for e in ChunkLevel}, {"PARENT", "CHILD", "ATOMIC"},
        )

    def test_field_source_owners(self):
        self.assertEqual(
            {e.value for e in FieldSourceOwner},
            {"PARSER", "METADATA_NORMALIZER", "CHUNKER", "QGATE",
             "DERIVED_DETERMINISTIC"},
        )


class TestInvariantsListFrozen(unittest.TestCase):
    """Contract invariant list: >= 16 entries, INV1..INV17 present."""

    def test_invariants_count_and_prefixes(self):
        count = len(CHUNK_METADATA_CONTRACT_V1_INVARIANTS)
        self.assertGreaterEqual(count, 16, f"Need >= 16 invariants, got {count}")
        # Expected INV1..INV17 (inclusive) prefixes
        expected = {f"INV{i}_" for i in range(1, 18)}
        actual_prefixes = set()
        for inv in CHUNK_METADATA_CONTRACT_V1_INVARIANTS:
            for e in expected:
                if inv.startswith(e):
                    actual_prefixes.add(e)
                    break
        self.assertGreaterEqual(len(actual_prefixes), 16, actual_prefixes)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
