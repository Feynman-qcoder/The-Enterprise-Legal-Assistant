"""
Chunk Metadata Contract V1 — Real Data Probe (READ-ONLY, NO ACTUAL CHUNKING).

Goal:
  Select 5 representative ParsedDocuments from canonical 53 corpus:
    1. MD (LEGAL_DATA_001 — 个人信息保护法, legal hierarchy via MD headings)
    2. PDF (LEGAL_LAW_001 — 民法典, 编章节条层级 example)
    3. DOCX (ENT_CONTRACT_003 — 合同节水管理项目服务合同)
    4. TABLE-containing doc (ENT_CONTRACT_003 DOCX contains party/signature TABLEs)
    5. Legal hierarchy example (same as PDF for explicit chapter/section/article demo)

Constraints:
  - READ-ONLY: never modify Parser/Cleaner/Normalizer/QGate/Manifest.
  - NO actual chunking algorithm.  Construct SAMPLE ChunkMetadata objects.
  - NO embedding / DB writes / production changes.
  - If source file unreachable, still build a deterministic contract-valid
    SAMPLE using metadata from ingest_manifest_v1.jsonl + known block ordering.

Output:
  chunk_metadata_real_probe.csv — one row per constructed sample probe, with
    scenario description, key contract fields, and validation result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chunk_metadata_contract_v1 import (  # noqa: E402
    ChunkMetadata,
    ChunkStructuralContext,
    ChunkProvenance,
    ChunkTableContext,
    ChunkParentChildInfo,
    ChunkQualityFlags,
    ChunkLevel,
    compute_chunk_id_contract,
    ChunkContractViolation,
)

CORPUS = Path(r"D:\xiaoyi\data_source")
MANIFEST = Path(r"D:\xiaoyi\data_source\_meta\ingest_manifest_v1.jsonl")
OUTPUT_CSV = THIS_DIR / "chunk_metadata_real_probe.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def find_source_file(manifest_entry: dict[str, Any]) -> Path | None:
    """Scan corpus to resolve canonical_file."""
    cf = manifest_entry.get("canonical_file", "")
    basename = Path(cf).name
    for root, dirs, files in os.walk(CORPUS):
        # prune noisy index folders
        dirs[:] = [d for d in dirs if d not in {"_meta", "_cache"}]
        for fn in files:
            if fn == basename:
                return Path(root) / fn
    return None


def safe_sha256(path: Path | None, fallback_manifest: dict[str, Any]) -> str:
    """Get source_sha256: prefer manifest normalized_sha256 if present,
    else read the file and SHA it, else deterministic placeholder
    (0-padded 64-hex manifest keyed by document id)."""
    nh = fallback_manifest.get("normalized_sha256")
    if isinstance(nh, str) and len(nh) == 64:
        return nh.lower()
    if path is not None and path.exists():
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            pass
    # Deterministic fallback: sha256(document_id + canonical_file) padded to 64
    seed = (
        fallback_manifest.get("document_id", "")
        + "|" + fallback_manifest.get("canonical_file", "")
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_sample_chunk(
    *,
    scenario: str,
    manifest: dict[str, Any],
    source_path: Path | None,
    block_orders: list[int],
    page_numbers: list[int] | None,
    structural: ChunkStructuralContext,
    table_context: ChunkTableContext | None = None,
    chunk_index: int = 0,
    parser_name: str = "",
    parser_version: str = "",
    parent_child: ChunkParentChildInfo | None = None,
    quality: ChunkQualityFlags | None = None,
    doc_metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a contract sample and return probe row dict."""

    source_format = (manifest.get("format") or "md").lower()
    source_sha256 = safe_sha256(source_path, manifest)
    document_id = manifest.get("logical_document_id") or manifest.get("canonical_file", "")
    logical_document_id = manifest.get("logical_document_id")
    source_file = str(source_path) if source_path is not None else (
        f"<corpus>/{manifest.get('canonical_file', '')}"
    )

    # Default parser metadata by format
    if not parser_name:
        parser_name = {
            "md": "txt_md_wrapper_v2",
            "txt": "txt_md_wrapper_v2",
            "pdf": "pdf_parser_v2_poc",
            "docx": "docx_parser_v2",
            "html": "html_parser_v2",
        }.get(source_format, "generic_parser_v2")
    if not parser_version:
        parser_version = {
            "md": "2.1.0-frozen",
            "pdf": "2.1.0-frozen",
            "docx": "2.4.2-frozen-patched",
            "html": "2.1.0-frozen",
        }.get(source_format, "1.0.0")

    # Provenance
    provenance = ChunkProvenance(
        source_block_orders=block_orders,
        source_block_ids=[f"{document_id}_b{b:04d}" for b in block_orders],
        page_start=page_numbers[0] if page_numbers else None,
        page_end=page_numbers[-1] if page_numbers else None,
        page_numbers=list(page_numbers) if page_numbers else [],
        parser_name=parser_name,
        parser_version=parser_version,
    )

    cid = compute_chunk_id_contract(
        source_sha256=source_sha256,
        document_id=document_id,
        chunk_index=chunk_index,
        original_block_order_start=provenance.original_block_order_start,
        original_block_order_end=provenance.original_block_order_end,
    )

    # Layer B document metadata projection
    doc_meta = dict(manifest)
    doc_meta["document_id"] = document_id

    cm = ChunkMetadata(
        chunk_id=cid,
        document_id=document_id,
        logical_document_id=logical_document_id,
        source_file=source_file,
        source_sha256=source_sha256,
        source_format=source_format,
        chunk_index=chunk_index,
        # --- Layer B fields from manifest where available ---
        title=manifest.get("title"),
        source_org=manifest.get("source_org"),
        publish_date=manifest.get("publish_date"),
        creation_date=manifest.get("creation_date"),
        effective_date=manifest.get("effective_date"),
        expiry_date=manifest.get("expiry_date"),
        document_number=manifest.get("document_number"),
        document_index_number=manifest.get("document_index_number"),
        document_type=manifest.get("document_type"),
        authority_level=manifest.get("authority_level"),
        legal_status=manifest.get("status"),
        jurisdiction=manifest.get("jurisdiction"),
        region=manifest.get("region"),
        theme_category=manifest.get("theme_category"),
        source_url=manifest.get("source_page_url"),
        structural=structural,
        provenance=provenance,
        table_context=table_context or ChunkTableContext(contains_table=False),
        parent_child=parent_child or ChunkParentChildInfo(chunk_level=ChunkLevel.ATOMIC.value),
        quality=quality or ChunkQualityFlags(quality_verdict="PASS"),
        extras={},
    )
    if doc_metadata_overrides:
        # Apply overrides via rebuilding (no LWW)
        cm = ChunkMetadata(**{**asdict(cm), **doc_metadata_overrides})

    violations = cm.validate()
    passed = (len(violations) == 0)
    probe_json_summary = json.dumps(cm.to_dict(), ensure_ascii=False, sort_keys=True)
    return {
        "probe_scenario": scenario,
        "document_id": document_id,
        "logical_document_id": logical_document_id or "",
        "canonical_file": manifest.get("canonical_file", ""),
        "source_format": source_format,
        "chunk_index": chunk_index,
        "chunk_id": cm.chunk_id,
        "source_block_orders": block_orders,
        "structural_chapter": structural.chapter or "",
        "structural_section": structural.section or "",
        "structural_article": structural.article or "",
        "structural_block_type": structural.block_type or "",
        "structural_content_type": structural.content_type or "",
        "contains_table": cm.table_context.contains_table,
        "table_structure_preserved": cm.table_context.table_structure_preserved,
        "page_start": provenance.page_start or "",
        "page_end": provenance.page_end or "",
        "parent_chunk_id": cm.parent_child.parent_chunk_id or "",
        "chunk_level": cm.parent_child.chunk_level or "",
        "quality_verdict": cm.quality.quality_verdict or "",
        "validation_pass": passed,
        "violations_count": len(violations),
        "violations": " || ".join(violations),
        "contract_version": cm.contract_version,
        "chunk_metadata_json": probe_json_summary,
    }


# ---------------------------------------------------------------------------
# Main: 5 scenarios
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = load_manifest()
    by_logical = {row.get("logical_document_id", ""): row for row in manifest}

    probes: list[dict[str, Any]] = []

    # --- Scenario 1: MD legal hierarchy (LEGAL_DATA_001) ---
    m = by_logical.get("LAW_001")
    if m is None:
        m = manifest[0]
    sp = find_source_file(m)
    probes.append(build_sample_chunk(
        scenario="S1_MD_LEGAL_HIERARCHY_ARTICLE_1",
        manifest=m,
        source_path=sp,
        block_orders=[0, 1, 2, 3],   # title + 第一章 + 第一条 + 条文1段
        page_numbers=None,
        structural=ChunkStructuralContext(
            chapter="第一章 总则",
            section=None,
            article="第一条",
            heading_path=["中华人民共和国个人信息保护法", "第一章 总则"],
            heading_level=1,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
    ))

    # --- Scenario 2: PDF Civil Code chapter / section / article combo ---
    m = by_logical.get("LAW_007")
    if m is None:
        # fallback find any pdf
        m = next((x for x in manifest if x.get("format") == "pdf"), manifest[0])
    sp = find_source_file(m)
    probes.append(build_sample_chunk(
        scenario="S2_PDF_CIVIL_CODE_CHAPTER_SECTION_ARTICLE",
        manifest=m,
        source_path=sp,
        block_orders=[50, 51, 52, 53, 54],
        page_numbers=[3, 4],
        structural=ChunkStructuralContext(
            chapter="第三编 合同",
            section="第二章 合同的订立",
            article="第四百七十一条",
            heading_path=[
                "中华人民共和国民法典",
                "第三编 合同",
                "第一分编 通则",
                "第二章 合同的订立",
            ],
            heading_level=2,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
    ))

    # --- Scenario 3: DOCX contract (ENT_CONTRACT_001) ---
    m = by_logical.get("CONTRACT_001")
    if m is None:
        m = next((x for x in manifest if x.get("format") == "docx"), manifest[-1])
    sp = find_source_file(m)
    probes.append(build_sample_chunk(
        scenario="S3_DOCX_CONTRACT_PREAMBLE",
        manifest=m,
        source_path=sp,
        block_orders=[0, 1, 2],
        page_numbers=[1],
        structural=ChunkStructuralContext(
            chapter=None,
            section=None,
            article=None,
            heading_path=[
                m.get("title", "") or "合同节水管理项目服务合同",
                "合同当事人",
            ],
            heading_level=1,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
    ))

    # --- Scenario 4: DOCX with TABLE (party/signature info table) ---
    # Same docx as scenario 3 but pick block-range with a TABLE
    m4 = by_logical.get("CONTRACT_001")
    if m4 is None:
        m4 = next((x for x in manifest if x.get("format") == "docx"), manifest[-1])
    sp4 = find_source_file(m4)
    TABLE_BLOCK_ORDER = 15  # Synthesized: sample chunk contains block#15 which is TABLE
    probes.append(build_sample_chunk(
        scenario="S4_DOCX_TABLE_PARTY_SIGNATURE_INFO",
        manifest=m4,
        source_path=sp4,
        block_orders=[14, TABLE_BLOCK_ORDER, 16],  # paragraph + TABLE + paragraph
        page_numbers=[2, 3],
        structural=ChunkStructuralContext(
            chapter=None,
            section="七、其他约定",
            article=None,
            heading_path=[
                m4.get("title", "") or "合同",
                "附件：合同签署页",
            ],
            heading_level=2,
            block_type="TABLE",
            content_type="MIXED_TEXT_TABLE",
        ),
        table_context=ChunkTableContext(
            contains_table=True,
            table_ids=[f"{(m4.get('logical_document_id') or 'DOCXCTR')}_b{TABLE_BLOCK_ORDER:04d}_table"],
            source_table_block_orders=[TABLE_BLOCK_ORDER],
            table_header_preserved=True,
            table_structure_preserved=True,
            table_part_index=0,
            table_parts_total=1,
        ),
        chunk_index=1,
    ))

    # --- Scenario 5: Parent/Child strategy demonstration on PDF Civil Code ---
    m5 = by_logical.get("LAW_007")
    if m5 is None:
        m5 = next((x for x in manifest if x.get("format") == "pdf"), manifest[0])
    sp5 = find_source_file(m5)
    # PARENT
    parent_probe = build_sample_chunk(
        scenario="S5_PDF_PARENT_CHILD_DEMO_PARENT",
        manifest=m5,
        source_path=sp5,
        block_orders=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        page_numbers=[5, 6, 7],
        structural=ChunkStructuralContext(
            chapter="第三编 合同",
            section="第二章 合同的订立",
            article=None,
            heading_path=[
                "中华人民共和国民法典",
                "第三编 合同",
                "第二章 合同的订立",
            ],
            heading_level=2,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
        chunk_index=0,
        parent_child=ChunkParentChildInfo(
            parent_chunk_id=None, chunk_level=ChunkLevel.PARENT.value,
        ),
    )
    probes.append(parent_probe)
    # CHILD (reuse parent chunk_id as parent_chunk_id reference)
    parent_cid = parent_probe["chunk_id"]
    probes.append(build_sample_chunk(
        scenario="S5_PDF_PARENT_CHILD_DEMO_CHILD",
        manifest=m5,
        source_path=sp5,
        block_orders=[102, 103],
        page_numbers=[6],
        structural=ChunkStructuralContext(
            chapter="第三编 合同",
            section="第二章 合同的订立",
            article="第四百七十二条",
            heading_path=[
                "中华人民共和国民法典",
                "第三编 合同",
                "第二章 合同的订立",
            ],
            heading_level=2,
            block_type="PARAGRAPH",
            content_type="LEGAL_TEXT",
        ),
        chunk_index=1,
        parent_child=ChunkParentChildInfo(
            parent_chunk_id=parent_cid,
            chunk_level=ChunkLevel.CHILD.value,
        ),
    ))

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = list(probes[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in probes:
            w.writerow(p)

    # Console summary
    total = len(probes)
    passed = sum(1 for p in probes if p["validation_pass"])
    print(f"chunk_metadata_real_probe.csv written: {OUTPUT_CSV}")
    print(f"TOTAL probes: {total}, PASS: {passed}, FAIL: {total - passed}")
    for p in probes:
        status = "PASS" if p["validation_pass"] else (
            f"FAIL [{p['violations']}]"
        )
        print(f"  - {p['probe_scenario']}: {status}")


if __name__ == "__main__":
    main()
