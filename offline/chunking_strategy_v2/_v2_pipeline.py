"""
Chunking Strategy V2 — FR8 Metrics engine, FR9/FR10/FR11 document-level report,
and pipeline main entry that produces ALL deliverables in one pass.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields as _dc_fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from _v2_utils import (  # noqa: E402
    FR7_FROZEN_THRESHOLDS,
    OUT_DIR,
    LoadedDoc,
    dispatch_parse_and_clean,
    load_manifest_entries,
    len_metric,
    normalize_ws,
    ChunkMetadata,
    ChunkContractViolation,
)
from _v2_strategies import (  # noqa: E402
    V2Chunk,
    chunk_strategy_a,
    chunk_strategy_b,
    parent_child_projection,
)

# Frozen thresholds (from FR7; DO NOT ALTER AFTER THIS POINT)
OVERSIZED = FR7_FROZEN_THRESHOLDS["OVERSIZED_THRESHOLD_CHARS"]
VERY_SMALL = FR7_FROZEN_THRESHOLDS["VERY_SMALL_THRESHOLD_CHARS"]

# Representative document logical IDs (Task 8)
REP_LEGAL_PDF_ID_SUBSTR = "中华人民共和国民法典"
REP_LAW_MD_ID_SUBSTR = "个人信息保护法"
REP_CONTRACT_DOCX_ID_SUBSTR = "合同节水管理项目服务合同"
REP_TABLE_HEAVY_SUBSTR = REP_CONTRACT_DOCX_ID_SUBSTR
REP_LONG_LEGAL_SUBSTR = "中华人民共和国民法典"


# ---------------------------------------------------------------------------
# Dataclass → dict helper (CMCV1 objects contain nested dataclasses)
# ---------------------------------------------------------------------------
def deep_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        out: dict[str, Any] = {}
        for f in _dc_fields(obj):
            out[f.name] = deep_to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple) and not isinstance(obj, (str, bytes)):
        return [deep_to_dict(x) for x in obj]
    if isinstance(obj, list):
        return [deep_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_to_dict(v) for k, v in obj.items()}
    # Handle Enums (ChunkLevel, BlockType): convert to name string
    if hasattr(obj, "name") and hasattr(obj, "value") and obj.__class__.__name__ not in {"str", "int"}:
        return str(obj.name) if not isinstance(obj, (str, int)) else obj
    return obj


# ---------------------------------------------------------------------------
# Chunk → JSON wrapper (chunk_metadata + text)
# ---------------------------------------------------------------------------
def chunk_to_record(chunk: V2Chunk) -> dict[str, Any]:
    return {
        "strategy": chunk.strategy,
        "text": chunk.text,
        "text_len_chars": len(chunk.text or ""),
        "chunk_metadata": deep_to_dict(chunk.metadata),
    }


# ---------------------------------------------------------------------------
# Metrics: per-document and aggregate
# ---------------------------------------------------------------------------
def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _duplicated_text_ratio(chunks: list[V2Chunk]) -> float:
    """Estimate duplicated ratio as a fraction of total chars.

    Use 8-byte rolling window (deterministic) over chunk text. Shingles that
    appear in 2+ different chunks contribute their char count once.
    This produces a size-ordered fair estimate that also correctly counts
    Strategy A overlap.
    """
    W = 8
    total_chars = 0
    counter: Counter[str] = Counter()
    for c in chunks:
        t = c.text or ""
        total_chars += len(t)
        if len(t) < W:
            continue
        for i in range(len(t) - W + 1):
            s = t[i:i + W]
            counter[s] += 1
    # Count chars in chunks covered by duplicate shingles
    dup_chars = 0
    for c in chunks:
        t = c.text or ""
        if len(t) < W:
            continue
        for i in range(len(t) - W + 1):
            if counter[t[i:i + W]] >= 2:
                dup_chars += 1
    if total_chars == 0:
        return 0.0
    return min(1.0, dup_chars / total_chars)


@dataclass
class DocMetrics:
    logical_document_id: str
    document_id: str
    source_format: str
    title: str | None
    document_type: str | None

    strategy: str  # "A" or "B"
    total_chunks: int
    avg_len_chars: float
    median_len_chars: float
    p95_len_chars: float
    min_len_chars: int
    max_len_chars: int
    oversized_chunks: int
    very_small_chunks: int
    source_block_coverage_pct: float
    duplicated_text_ratio: float
    cross_article_chunks: int
    cross_section_chunks: int
    heading_orphan_count: int
    article_orphan_count: int
    table_chunks: int
    table_relationship_loss: int
    provenance_failures: int
    metadata_contract_failures: int
    parse_skip_reason: str | None = None  # if doc was skipped


def compute_doc_metrics(
    strategy: str,
    loaded: LoadedDoc | None,
    chunks: list[V2Chunk],
    table_audit_rows: list[dict[str, Any]],
    parse_skip_reason: str | None = None,
) -> DocMetrics:
    if loaded is None:
        # Produce a placeholder row so manifest-accounting keeps 53 entries
        return DocMetrics(
            logical_document_id="UNKNOWN",
            document_id="UNKNOWN",
            source_format="unknown",
            title=None,
            document_type=None,
            strategy=strategy,
            total_chunks=0,
            avg_len_chars=0.0,
            median_len_chars=0.0,
            p95_len_chars=0.0,
            min_len_chars=0,
            max_len_chars=0,
            oversized_chunks=0,
            very_small_chunks=0,
            source_block_coverage_pct=0.0,
            duplicated_text_ratio=0.0,
            cross_article_chunks=0,
            cross_section_chunks=0,
            heading_orphan_count=0,
            article_orphan_count=0,
            table_chunks=0,
            table_relationship_loss=0,
            provenance_failures=0,
            metadata_contract_failures=0,
            parse_skip_reason=parse_skip_reason,
        )

    lengths = sorted(len(c.text or "") for c in chunks)
    total_chunks = len(chunks)
    title = loaded.title
    if isinstance(title, (bytes, bytearray)):
        title = str(title, "utf-8", errors="replace")
    title = str(title) if title is not None else None

    avg = float(statistics.mean(lengths)) if lengths else 0.0
    median = float(statistics.median(lengths)) if lengths else 0.0
    p95 = _percentile(lengths, 0.95) if lengths else 0.0
    min_len = lengths[0] if lengths else 0
    max_len = lengths[-1] if lengths else 0
    oversized = sum(1 for l in lengths if l > OVERSIZED)
    verysmall = sum(1 for l in lengths if l < VERY_SMALL)

    total_blocks = len(loaded.blocks)
    covered: set[int] = set()
    for c in chunks:
        covered.update(c._block_orders)
    coverage = (len(covered) / total_blocks * 100.0) if total_blocks > 0 else 0.0

    dup_ratio = _duplicated_text_ratio(chunks) if chunks else 0.0

    cross_article = sum(1 for c in chunks if len(c._articles) >= 2)
    cross_section = sum(1 for c in chunks if len(c._sections) >= 2)

    heading_orphans = 0
    article_orphans = 0
    for c in chunks:
        st = c.metadata.structural
        txt = c.text or ""
        if st.content_type == "HEADING_ONLY":
            heading_orphans += 1
        # Article orphan: LEGAL_TEXT, single article known, text just article label (<=30 chars)
        if (st.content_type in {"LEGAL_TEXT", "UNKNOWN", "MIXED_TEXT_TABLE"}
                and len(c._articles) == 1 and len(txt) <= 30):
            # Must NOT just be a heading chunk
            if st.block_type != "HEADING":
                article_orphans += 1

    table_chunks = sum(1 for c in chunks if c.metadata.table_context.contains_table)
    # TABLE relationship loss: baseline A can have relationship_loss_check != OK;
    # B by construction should be all OK.
    rel_loss_rows = [r for r in table_audit_rows if r.get("relationship_loss_check") != "OK"]
    rel_loss = len(rel_loss_rows)

    # Provenance failures + Contract failures: validate using CMCV1 `.validate()`
    provenance_failures = 0
    metadata_contract_failures = 0
    for c in chunks:
        violations: list[str] = []
        try:
            violations = c.metadata.validate() or []
        except ChunkContractViolation as e:
            metadata_contract_failures += 1
            provenance_failures += 1
            violations = [str(e)]
        except Exception:
            metadata_contract_failures += 1
            continue
        if violations:
            # Violations list non-empty → at least 1 invariant fail
            metadata_contract_failures += 1
            # provenance-related? search strings
            if any("PROVENANCE" in v.upper() or "BLOCK_ORDERS" in v.upper() for v in violations):
                provenance_failures += 1
        else:
            # provenance invariant manual sanity: source_block_orders monotonic strictly increasing?
            bos = c.metadata.provenance.source_block_orders
            monotonic_ok = True
            for i in range(1, len(bos)):
                if bos[i] <= bos[i - 1]:
                    monotonic_ok = False
                    break
            if not monotonic_ok:
                provenance_failures += 1

    return DocMetrics(
        logical_document_id=loaded.logical_document_id,
        document_id=loaded.document_id,
        source_format=loaded.source_format,
        title=title,
        document_type=loaded.document_type,
        strategy=strategy,
        total_chunks=total_chunks,
        avg_len_chars=round(avg, 2),
        median_len_chars=round(median, 2),
        p95_len_chars=round(p95, 2),
        min_len_chars=min_len,
        max_len_chars=max_len,
        oversized_chunks=oversized,
        very_small_chunks=verysmall,
        source_block_coverage_pct=round(coverage, 3),
        duplicated_text_ratio=round(dup_ratio, 5),
        cross_article_chunks=cross_article,
        cross_section_chunks=cross_section,
        heading_orphan_count=heading_orphans,
        article_orphan_count=article_orphans,
        table_chunks=table_chunks,
        table_relationship_loss=rel_loss,
        provenance_failures=provenance_failures,
        metadata_contract_failures=metadata_contract_failures,
    )


def aggregate_metrics(dms: list[DocMetrics]) -> dict[str, Any]:
    if not dms:
        return {"TOTAL_CHUNKS": 0}
    agg: dict[str, Any] = {"STRATEGY": dms[0].strategy}
    # Sum integer counters
    for key in (
        "total_chunks", "oversized_chunks", "very_small_chunks",
        "cross_article_chunks", "cross_section_chunks",
        "heading_orphan_count", "article_orphan_count",
        "table_chunks", "table_relationship_loss",
        "provenance_failures", "metadata_contract_failures",
    ):
        agg[key.upper()] = int(sum(getattr(d, key) or 0 for d in dms))
    # Length distribution across ALL chunks
    # We approximate AVG/MEDIAN/P95/MIN/MAX by averaging doc-level weighted by total_chunks
    total_c = agg["TOTAL_CHUNKS"]
    if total_c > 0:
        agg["AVG_LEN_CHARS"] = round(
            sum((getattr(d, "avg_len_chars") or 0.0) * (d.total_chunks or 0) for d in dms) / total_c,
            2,
        )
        # median / p95 approx as weighted average across doc medians (ballpark; fine for offline aggregate)
        agg["MEDIAN_LEN_CHARS_APPROX"] = round(
            sum((getattr(d, "median_len_chars") or 0.0) * (d.total_chunks or 0) for d in dms) / total_c,
            2,
        )
        agg["P95_LEN_CHARS_APPROX"] = round(
            sum((getattr(d, "p95_len_chars") or 0.0) * (d.total_chunks or 0) for d in dms) / total_c,
            2,
        )
        agg["MIN_LEN_CHARS"] = min((d.min_len_chars for d in dms if d.total_chunks > 0), default=0)
        agg["MAX_LEN_CHARS"] = max((d.max_len_chars for d in dms if d.total_chunks > 0), default=0)
        agg["AVG_COVERAGE_PCT"] = round(
            sum(getattr(d, "source_block_coverage_pct") or 0.0 for d in dms) / len([d for d in dms if d.parse_skip_reason is None]) if any(d.parse_skip_reason is None for d in dms) else 0.0,
            3,
        )
        agg["AVG_DUPLICATED_TEXT_RATIO"] = round(
            sum(getattr(d, "duplicated_text_ratio") or 0.0 for d in dms) / len([d for d in dms if d.parse_skip_reason is None]) if any(d.parse_skip_reason is None for d in dms) else 0.0,
            5,
        )
    return agg


# ---------------------------------------------------------------------------
# Human review package (FR9/FR10: 20-30 representative pairs)
# ---------------------------------------------------------------------------
@dataclass
class ReviewCase:
    case_id: str
    document: str
    context: str
    strategy_a_text: str
    strategy_b_text: str
    risk_reason: str


def generate_human_review_pairs(
    loaded_map: dict[str, LoadedDoc],
    a_chunks_map: dict[str, list[V2Chunk]],
    b_chunks_map: dict[str, list[V2Chunk]],
    table_audit_map: dict[str, tuple[list, list]],  # logical_id -> (a_audit, b_audit)
) -> list[ReviewCase]:
    cases: list[ReviewCase] = []

    # Helper: return doc chunks filtered by a predicate
    rep_keys_order = [
        "legal_pdf", "law_md", "contract_docx", "table_heavy_docx", "long_legal"
    ]
    # Resolve each rep key to first-match doc id
    reps: dict[str, list[str]] = {k: [] for k in rep_keys_order}
    for lid, loaded in loaded_map.items():
        title = str(loaded.title or "")
        if REP_LEGAL_PDF_ID_SUBSTR in loaded.logical_document_id or REP_LEGAL_PDF_ID_SUBSTR in title:
            reps["legal_pdf"].append(lid)
            reps["long_legal"].append(lid)
        if REP_LAW_MD_ID_SUBSTR in loaded.logical_document_id or REP_LAW_MD_ID_SUBSTR in title:
            reps["law_md"].append(lid)
        if REP_CONTRACT_DOCX_ID_SUBSTR in loaded.logical_document_id or REP_CONTRACT_DOCX_ID_SUBSTR in title:
            reps["contract_docx"].append(lid)
            reps["table_heavy_docx"].append(lid)

    def doc_for(k: str) -> tuple[str | None, LoadedDoc | None]:
        for lid in reps.get(k, []):
            if lid in loaded_map:
                return lid, loaded_map[lid]
        return None, None

    def pick_chunks_with(lid: str, pred, strategy: str):
        cs = a_chunks_map if strategy == "A" else b_chunks_map
        return [c for c in cs.get(lid, []) if pred(c)]

    # Target: 20-30 cases
    # 1) Boundary-sensitive: cross-article chunks in A — compare with corresponding chunks in B
    for rep_k in rep_keys_order:
        lid, loaded = doc_for(rep_k)
        if not lid or loaded is None:
            continue
        # Find 3 most-cross-article A chunks (by _articles len)
        a_cross = sorted(
            [c for c in a_chunks_map.get(lid, []) if len(c._articles) >= 2],
            key=lambda c: (-len(c._articles), -len(c.text or "")),
        )[:3]
        for ac in a_cross:
            # Find B chunks whose block orders intersect with A's block orders
            intersect_b = [
                bc for bc in b_chunks_map.get(lid, [])
                if set(bc._block_orders) & set(ac._block_orders)
            ]
            if not intersect_b:
                continue
            context = (
                f"cross-article ({len(ac._articles)} articles); "
                f"A articles={ac._articles}; "
                f"articles seen={ac._articles[:3]}..."
            )
            cases.append(ReviewCase(
                case_id=f"boundary_{rep_k}_{len(cases)+1:03d}",
                document=f"{loaded.logical_document_id} (title={loaded.title})",
                context=context,
                strategy_a_text=(ac.text or ""),
                strategy_b_text=("\n\n".join(c.text or "" for c in intersect_b[:2])),
                risk_reason=(
                    "A 跨越多条法条边界；"
                    "B 应保持 article 边界未合并。比较是否有 A 将"
                    "独立法条的 义务/否定/日期 条款合并导致混淆的风险。"
                ),
            ))

    # 2) TABLE: for every TABLE-heavy doc, compare table chunks between A/B
    for rep_k in ("table_heavy_docx", "contract_docx", "long_legal"):
        lid, loaded = doc_for(rep_k)
        if not lid or loaded is None:
            continue
        # B tables first (B has clean TABLE chunks)
        b_tables = [c for c in b_chunks_map.get(lid, []) if c.metadata.table_context.contains_table]
        for bt in b_tables[:4]:  # cap per doc
            # Find A chunks intersecting same block
            same_blocks = [
                ac for ac in a_chunks_map.get(lid, [])
                if ac.metadata.table_context.contains_table
                and set(ac.metadata.table_context.source_table_block_orders)
                & set(bt.metadata.table_context.source_table_block_orders)
            ]
            if not same_blocks:
                continue
            context = (
                f"TABLE block_orders={bt.metadata.table_context.source_table_block_orders}; "
                f"B part_index={bt.metadata.table_context.table_part_index}/"
                f"{bt.metadata.table_context.table_parts_total}; "
                f"B header_preserved={bt.metadata.table_context.table_header_preserved}"
            )
            cases.append(ReviewCase(
                case_id=f"table_{rep_k}_{len(cases)+1:03d}",
                document=f"{loaded.logical_document_id} (title={loaded.title})",
                context=context,
                strategy_a_text=(same_blocks[0].text or "")[:1200],
                strategy_b_text=(bt.text or ""),
                risk_reason=(
                    "TABLE 表头-行值关系是否可独立理解？比较 A 可能丢失表头；"
                    "B 的所有分片都应带表头 materialization。"
                ),
            ))

    # 3) Oversized: B chunks with text > B_HARD (should be 0; pick any > B_TARGET for review)
    for lid, loaded in loaded_map.items():
        for bc in b_chunks_map.get(lid, []):
            if len(bc.text or "") > OVERSIZED:
                # find matching A intersecting blocks
                a_related = [ac for ac in a_chunks_map.get(lid, []) if set(ac._block_orders) & set(bc._block_orders)]
                if not a_related:
                    continue
                cases.append(ReviewCase(
                    case_id=f"oversized_{len(cases)+1:03d}",
                    document=f"{loaded.logical_document_id} (title={loaded.title})",
                    context=(
                        f"B OVERSIZED len={len(bc.text or '')} chars "
                        f"(metrics-thr-OVERSIZED={OVERSIZED}, B_HARD_MAX="
                        f"{FR7_FROZEN_THRESHOLDS['B_HARD_MAX']}); structural=("
                        f"chapter={bc.metadata.structural.chapter}; "
                        f"section={bc.metadata.structural.section}; "
                        f"article={bc.metadata.structural.article})"
                    ),
                    strategy_a_text=(a_related[0].text or "")[:1200],
                    strategy_b_text=(bc.text or "")[:1200],
                    risk_reason="B 超过硬上限。检查文本内容是否为无法分段的超长句（需降级策略）。",
                ))
                if len([c for c in cases if c.case_id.startswith("oversized_")]) >= 3:
                    break
        if len([c for c in cases if c.case_id.startswith("oversized_")]) >= 3:
            break

    # 4) Very small: B very small chunks vs A
    for lid, loaded in loaded_map.items():
        small_b = [
            bc for bc in b_chunks_map.get(lid, [])
            if len(bc.text or "") < VERY_SMALL
        ]
        if not small_b:
            continue
        for bc in small_b[:2]:
            a_related = [ac for ac in a_chunks_map.get(lid, []) if set(ac._block_orders) & set(bc._block_orders)]
            if not a_related:
                continue
            cases.append(ReviewCase(
                case_id=f"verysmall_{len(cases)+1:03d}",
                document=f"{loaded.logical_document_id} (title={loaded.title})",
                context=(
                    f"VERY SMALL B len={len(bc.text or '')} chars (< VERY_SMALL="
                    f"{VERY_SMALL}); B block_orders={bc._block_orders}"
                ),
                strategy_a_text=(a_related[0].text or "")[:1200],
                strategy_b_text=(bc.text or ""),
                risk_reason=(
                    "B 极小 chunk 检查是否为孤立 HEADING 或法条标签。"
                    "小 adjacent 单元是否应该合并？"
                ),
            ))
        if len([c for c in cases if c.case_id.startswith("verysmall_")]) >= 4:
            break

    # 5) Disagreement cases: A vs B cover identical blocks but produce different chunks
    # (cross-section or heading_orphan_count in A but not B)
    for rep_k in ("law_md", "contract_docx", "legal_pdf"):
        lid, loaded = doc_for(rep_k)
        if not lid or loaded is None:
            continue
        a_head = [c for c in a_chunks_map.get(lid, []) if c.metadata.structural.content_type == "HEADING_ONLY"][:2]
        for ac in a_head:
            related_b = [bc for bc in b_chunks_map.get(lid, []) if set(bc._block_orders) & set(ac._block_orders)]
            if not related_b:
                continue
            cases.append(ReviewCase(
                case_id=f"disagree_{rep_k}_{len(cases)+1:03d}",
                document=f"{loaded.logical_document_id} (title={loaded.title})",
                context=(
                    f"A HEADING_ONLY len={len(ac.text or '')}; B same block_orders "
                    f"→ len={[len(c.text or '') for c in related_b]}"
                ),
                strategy_a_text=(ac.text or ""),
                strategy_b_text=("\n\n".join((c.text or "") for c in related_b[:2]))[:1200],
                risk_reason=(
                    "章节标题是独立 chunk 是否会影响检索？"
                    "对比 A 的独立 heading chunk 与 B 的标题+正文合并 chunk。"
                ),
            ))

    # Cap / floor to 20-30
    while len(cases) < 22:
        # pad with extra cross-section cases
        for lid, loaded in list(loaded_map.items())[:10]:
            a_cross_sec = [c for c in a_chunks_map.get(lid, []) if len(c._sections) >= 2]
            for ac in a_cross_sec[:2]:
                related_b = [bc for bc in b_chunks_map.get(lid, []) if set(bc._block_orders) & set(ac._block_orders)]
                if not related_b:
                    continue
                cases.append(ReviewCase(
                    case_id=f"pad_crosssec_{len(cases)+1:03d}",
                    document=f"{loaded.logical_document_id} (title={loaded.title})",
                    context=(
                        f"A CROSS-SECTION (sections={len(ac._sections)}) len="
                        f"{len(ac.text or '')}"
                    ),
                    strategy_a_text=(ac.text or "")[:1200],
                    strategy_b_text=("\n\n".join((c.text or "") for c in related_b[:2]))[:1200],
                    risk_reason="A 是否跨章节合并导致法律上下文混淆？B 是否正确保持边界？",
                ))
                if len(cases) >= 30:
                    break
            if len(cases) >= 30:
                break
        if len(cases) >= 30:
            break
    if len(cases) > 30:
        # Prefer boundary, table, disagreement; drop pad extras last
        priority_order = {"boundary": 0, "table": 1, "oversized": 2, "verysmall": 3, "disagree": 4, "pad_crosssec": 5}
        cases_sorted = sorted(cases, key=lambda c: (priority_order.get(c.case_id.split("_")[0], 99), c.case_id))
        cases = cases_sorted[:30]
    # Reset case_ids to stable numbering
    for i, c in enumerate(cases, 1):
        parts = c.case_id.split("_", 1)
        c.case_id = f"R{i:03d}_{parts[0]}" if len(parts) > 1 else f"R{i:03d}_{c.case_id}"
    return cases


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    manifest_entries: list[dict[str, Any]]
    loaded_map: dict[str, LoadedDoc]           # logical_id → LoadedDoc (only parsed ok)
    skipped: list[tuple[str, str]]             # logical_id_or_filename, reason
    a_chunks_map: dict[str, list[V2Chunk]]     # logical_id → chunks
    b_chunks_map: dict[str, list[V2Chunk]]
    a_doc_metrics: list[DocMetrics]
    b_doc_metrics: list[DocMetrics]
    a_agg: dict[str, Any]
    b_agg: dict[str, Any]
    table_audit_map: dict[str, tuple[list, list]]
    pc_probe_status: str
    review_cases: list[ReviewCase]
    output_json: dict[str, Any]                # chunking_strategy_v2.json content
    run_summary: dict[str, Any]


def run_pipeline() -> PipelineResult:
    t0 = time.time()
    entries = load_manifest_entries()
    assert len(entries) == 53, f"Manifest expected 53 entries, got {len(entries)}"

    loaded_map: dict[str, LoadedDoc] = {}
    skipped: list[tuple[str, str]] = []
    a_chunks_map: dict[str, list[V2Chunk]] = {}
    b_chunks_map: dict[str, list[V2Chunk]] = {}
    table_audit_map: dict[str, tuple[list, list]] = {}
    a_doc_metrics: list[DocMetrics] = []
    b_doc_metrics: list[DocMetrics] = []

    # Iterate entries — deterministic manifest order
    for entry in entries:
        lid = entry.get("logical_document_id") or entry.get("document_id") or f"ID_{entry.get('canonical_file','?')}"
        result = dispatch_parse_and_clean(entry)
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "SKIP":
            reason = result[1]
            skipped.append((lid, reason))
            a_doc_metrics.append(compute_doc_metrics("A", None, [], [], parse_skip_reason=reason))
            b_doc_metrics.append(compute_doc_metrics("B", None, [], [], parse_skip_reason=reason))
            continue
        loaded: LoadedDoc = result  # type: ignore[assignment]
        loaded_map[lid] = loaded

        # Run strategies
        try:
            a_chunks, a_ta = chunk_strategy_a(loaded)
        except Exception as e:
            a_chunks, a_ta = [], []
            skipped.append((lid, f"STRATEGY_A_ERROR_{type(e).__name__}:{str(e)[:60]}"))
        try:
            b_chunks, b_ta = chunk_strategy_b(loaded)
        except Exception as e:
            b_chunks, b_ta = [], []
            skipped.append((lid, f"STRATEGY_B_ERROR_{type(e).__name__}:{str(e)[:60]}"))

        # Determine determinism (FR12): re-run both once and compare.
        # Only run on representative subset (first 5) to save time, OR run all — we do all docs.
        a_chunks2, a_ta2 = ([], [])
        try:
            a_chunks2, a_ta2 = chunk_strategy_a(loaded)
        except Exception:
            pass
        if len(a_chunks) != len(a_chunks2):
            skipped.append((lid, f"DETERMINISM_A_FAILED chunks {len(a_chunks)} != {len(a_chunks2)}"))
        else:
            for i, (c1, c2) in enumerate(zip(a_chunks, a_chunks2)):
                if c1.text != c2.text or c1._block_orders != c2._block_orders:
                    skipped.append((lid, f"DETERMINISM_A_FAILED diff at chunk {i}"))
                    break
        b_chunks2, b_ta2 = ([], [])
        try:
            b_chunks2, b_ta2 = chunk_strategy_b(loaded)
        except Exception:
            pass
        if len(b_chunks) != len(b_chunks2):
            skipped.append((lid, f"DETERMINISM_B_FAILED chunks {len(b_chunks)} != {len(b_chunks2)}"))
        else:
            for i, (c1, c2) in enumerate(zip(b_chunks, b_chunks2)):
                if c1.text != c2.text or c1._block_orders != c2._block_orders:
                    skipped.append((lid, f"DETERMINISM_B_FAILED diff at chunk {i}"))
                    break

        a_chunks_map[lid] = a_chunks
        b_chunks_map[lid] = b_chunks
        table_audit_map[lid] = (a_ta, b_ta)
        a_doc_metrics.append(compute_doc_metrics("A", loaded, a_chunks, a_ta))
        b_doc_metrics.append(compute_doc_metrics("B", loaded, b_chunks, b_ta))

    # FR6 Parent/Child probe (per-doc; overall PASS if ALL pass / FAIL if ANY FAIL)
    pc_probe_status_per_doc: list[str] = []
    for lid, loaded in loaded_map.items():
        _, pc_pass = parent_child_projection(loaded, b_chunks_map.get(lid, []))
        pc_probe_status_per_doc.append(pc_pass)
    if not pc_probe_status_per_doc:
        pc_probe_status = "FAIL"
    elif all(s == "PASS" for s in pc_probe_status_per_doc):
        pc_probe_status = "PASS"
    else:
        fails = sum(1 for s in pc_probe_status_per_doc if s != "PASS")
        pc_probe_status = f"FAIL({fails}/{len(pc_probe_status_per_doc)})"

    # Aggregate
    a_agg = aggregate_metrics(a_doc_metrics)
    b_agg = aggregate_metrics(b_doc_metrics)

    # FR10 human review pairs
    review_cases = generate_human_review_pairs(loaded_map, a_chunks_map, b_chunks_map, table_audit_map)

    # ---- Gate decisions (FR11) ----
    failures: list[str] = []
    # 1. Contract failures must be 0 for both
    if a_agg.get("METADATA_CONTRACT_FAILURES", 0) != 0:
        failures.append("A METADATA_CONTRACT_FAILURES>0")
    if b_agg.get("METADATA_CONTRACT_FAILURES", 0) != 0:
        failures.append("B METADATA_CONTRACT_FAILURES>0")
    # 2. Source block coverage must be 100% for both (avg, but per-doc required for docs actually loaded)
    for dm in a_doc_metrics:
        if dm.parse_skip_reason is None and abs(dm.source_block_coverage_pct - 100.0) > 1e-6:
            failures.append(f"A coverage<100% at {dm.logical_document_id}: {dm.source_block_coverage_pct}")
            break
    for dm in b_doc_metrics:
        if dm.parse_skip_reason is None and abs(dm.source_block_coverage_pct - 100.0) > 1e-6:
            failures.append(f"B coverage<100% at {dm.logical_document_id}: {dm.source_block_coverage_pct}")
            break
    # 3. Determinism passes? skipped list must NOT contain DETERMINISM
    det_fail = [s for s in skipped if "DETERMINISM_" in s[1]]
    if det_fail:
        failures.append("DETERMINISM_FAILURES: " + "; ".join(f"{lid}={r}" for lid, r in det_fail[:5]))
    # 4. Table relationship loss: B must be 0
    if b_agg.get("TABLE_RELATIONSHIP_LOSS", 0) != 0:
        failures.append(f"B TABLE_RELATIONSHIP_LOSS={b_agg['TABLE_RELATIONSHIP_LOSS']}")
    # 5. 53 docs accounted in document metrics
    if len(a_doc_metrics) != 53 or len(b_doc_metrics) != 53:
        failures.append(f"doc metrics rows !=53: A={len(a_doc_metrics)}, B={len(b_doc_metrics)}")

    gate_status = "PASS" if not failures else ("PASS_WITH_REVIEW" if len(failures) <= 3 else "FAIL")

    # OFFLINE_PREFERRED_STRATEGY
    # Per spec §11: Prefer B if B passes structural/safety checks
    # (cross-article merging not worse, cross-section not worse, table relationships
    # preserved, CMCV1 clean, full block coverage, provenance clean).
    # Heading orphans are expected to be HIGHER for a structure-aware strategy because
    # B intentionally breaks chunks at heading boundaries, so we explicitly do NOT
    # penalize B on heading_orphan_count or article_orphan_count.
    b_wins_structure = (
        b_agg.get("CROSS_ARTICLE_CHUNKS", 10**9) <= a_agg.get("CROSS_ARTICLE_CHUNKS", 10**9)
        and b_agg.get("CROSS_SECTION_CHUNKS", 10**9) <= a_agg.get("CROSS_SECTION_CHUNKS", 10**9)
    )
    b_table_ok = b_agg.get("TABLE_RELATIONSHIP_LOSS", 1) == 0
    b_cmcv1_clean = b_agg.get("METADATA_CONTRACT_FAILURES", -1) == 0
    a_cmcv1_clean = a_agg.get("METADATA_CONTRACT_FAILURES", -1) == 0
    b_prov_clean = b_agg.get("PROVENANCE_FAILURES", -1) == 0
    # FR7 metrics: block coverage (per-strategy mean across loaded docs). A/B both 100
    # is enforced by test suite and already computed inside document metric aggregation.
    b_structural_safe = (
        b_wins_structure
        and b_table_ok
        and b_cmcv1_clean
        and a_cmcv1_clean
        and b_prov_clean
    )
    if gate_status == "PASS" and b_structural_safe:
        preferred = "B"
    elif gate_status == "FAIL":
        preferred = "A"
    else:
        preferred = "REVIEW"

    # Per spec §11 READY_FOR_RETRIEVAL_AB_EVAL = YES iff B passes structural/safety
    # AND Parent/Child probe is PASS (optional projection demonstrated supported).
    ready_for_retrieval_ab_eval = (
        b_structural_safe
        and pc_probe_status == "PASS"
    )

    run_summary = {
        "STARTED_AT": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "FINISHED_AT": datetime.now(timezone.utc).isoformat(),
        "ELAPSED_SEC": round(time.time() - t0, 3),
        "MANIFEST_ENTRIES": len(entries),
        "DOCS_LOADED_OK": len(loaded_map),
        "DOCS_SKIPPED": len(skipped),
        "SKIPPED_LIST": [
            {"logical_document_id": lid, "reason": reason}
            for (lid, reason) in skipped
        ],
        "FR7_FROZEN_THRESHOLDS": FR7_FROZEN_THRESHOLDS,
        "BGE_TOKENIZER_AVAILABLE": None,  # filled by caller if needed
        "PARENT_CHILD_PROBE": pc_probe_status,
        "FAILURES": failures,
        "CHUNKING_STRATEGY_V2_OFFLINE": gate_status,
        "OFFLINE_PREFERRED_STRATEGY": preferred,
        "READY_FOR_RETRIEVAL_AB_EVAL": "YES" if ready_for_retrieval_ab_eval else "NO",
    }

    # Explicit structured gates + decisions for downstream consumers (T11 + tools)
    gate_info = {
        "STATUS": gate_status,
        "TRIPLET": {
            "CHUNKING_STRATEGY_V2_OFFLINE": gate_status,
            "OFFLINE_PREFERRED_STRATEGY": preferred,
            "READY_FOR_RETRIEVAL_AB_EVAL": "YES" if ready_for_retrieval_ab_eval else "NO",
        },
        "CRITERIA": {
            "GATE_ALL_TESTS": gate_status == "PASS",
            "B_STRUCTURAL_SAFE": b_structural_safe,
            "B_WINS_CROSS_ARTICLE": b_agg.get("CROSS_ARTICLE_CHUNKS", -1) <= a_agg.get("CROSS_ARTICLE_CHUNKS", -2),
            "B_WINS_CROSS_SECTION": b_agg.get("CROSS_SECTION_CHUNKS", -1) <= a_agg.get("CROSS_SECTION_CHUNKS", -2),
            "B_TABLE_RELATIONSHIP_LOSS_ZERO": b_table_ok,
            "CMCV1_CLEAN": bool(b_cmcv1_clean and a_cmcv1_clean),
            "PROVENANCE_CLEAN": b_prov_clean and (a_agg.get("PROVENANCE_FAILURES", -1) == 0),
            "PARENT_CHILD_PROBE_PASS": pc_probe_status == "PASS",
        },
    }

    output_json = {
        "run_summary": run_summary,
        "gate": gate_info,
        "decisions": gate_info["TRIPLET"],
        "strategy_a_aggregate": a_agg,
        "strategy_b_aggregate": b_agg,
    }

    return PipelineResult(
        manifest_entries=entries,
        loaded_map=loaded_map,
        skipped=skipped,
        a_chunks_map=a_chunks_map,
        b_chunks_map=b_chunks_map,
        a_doc_metrics=a_doc_metrics,
        b_doc_metrics=b_doc_metrics,
        a_agg=a_agg,
        b_agg=b_agg,
        table_audit_map=table_audit_map,
        pc_probe_status=pc_probe_status,
        review_cases=review_cases,
        output_json=output_json,
        run_summary=run_summary,
    )


# ---------------------------------------------------------------------------
# Deliverables writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_all_deliverables(res: PipelineResult) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    # 1. strategy_a_chunks.json
    a_all_records: list[dict[str, Any]] = []
    for lid in sorted(res.a_chunks_map.keys()):
        for c in res.a_chunks_map[lid]:
            a_all_records.append(chunk_to_record(c))
    p = OUT_DIR / "strategy_a_chunks.json"
    _write_json(p, {"count": len(a_all_records), "chunks": a_all_records})
    files["strategy_a_chunks"] = p

    # 2. strategy_b_chunks.json
    b_all_records: list[dict[str, Any]] = []
    for lid in sorted(res.b_chunks_map.keys()):
        for c in res.b_chunks_map[lid]:
            b_all_records.append(chunk_to_record(c))
    p = OUT_DIR / "strategy_b_chunks.json"
    _write_json(p, {"count": len(b_all_records), "chunks": b_all_records})
    files["strategy_b_chunks"] = p

    # 3. chunking_ab_metrics.csv
    p = OUT_DIR / "chunking_ab_metrics.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        # Collect all aggregate keys from both
        keys = sorted(set(res.a_agg.keys()) | set(res.b_agg.keys()))
        keys = [k for k in keys if k != "STRATEGY"]
        w = csv.writer(f)
        w.writerow(["STRATEGY"] + keys)
        w.writerow(["A"] + [res.a_agg.get(k, "") for k in keys])
        w.writerow(["B"] + [res.b_agg.get(k, "") for k in keys])
    files["ab_metrics"] = p

    # 4. chunking_ab_document_metrics.csv
    p = OUT_DIR / "chunking_ab_document_metrics.csv"
    fieldnames = [f.name for f in _dc_fields(DocMetrics)]
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for dm in res.a_doc_metrics:
            w.writerow(asdict(dm))
        for dm in res.b_doc_metrics:
            w.writerow(asdict(dm))
    files["doc_metrics"] = p

    # 5. table_chunk_audit.csv
    p = OUT_DIR / "table_chunk_audit.csv"
    all_rows: list[dict[str, Any]] = []
    for lid, (a_a, b_a) in res.table_audit_map.items():
        for r in a_a:
            r = dict(r)
            r.setdefault("strategy", "A")
            all_rows.append(r)
        for r in b_a:
            r = dict(r)
            r.setdefault("strategy", "B")
            all_rows.append(r)
    if all_rows:
        fields_audit = list(all_rows[0].keys())
    else:
        fields_audit = [
            "strategy", "chunk_id", "document_id", "logical_document_id",
            "block_order", "table_id", "table_part_index", "table_parts_total",
            "header_preserved", "structure_preserved", "text_len_chars",
            "rows_total", "rows_in_chunk", "headers_materialized",
            "relationship_loss_check",
        ]
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields_audit, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    files["table_audit"] = p

    # 6. chunking_ab_human_review.csv (FR10)
    p = OUT_DIR / "chunking_ab_human_review.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "case_id", "document", "context", "Strategy_A", "Strategy_B",
            "risk_reason", "用户选择", "用户备注",
        ])
        for c in res.review_cases:
            w.writerow([
                c.case_id, c.document, c.context,
                c.strategy_a_text, c.strategy_b_text, c.risk_reason,
                "", "",  # user fields blank
            ])
    files["human_review_csv"] = p

    # 7. chunking_ab_review_zh.md (FR10)
    p = OUT_DIR / "chunking_ab_review_zh.md"
    lines = [
        "# Chunking Strategy V2 — 人工评审包（A/B 20–30 例）",
        "",
        "> 本文件仅用于人工评审。**禁止**由模型自动判定胜者。",
        "> 对于每个 case，请在末尾 `用户选择` 列填入 `A` / `B` / `两者均可` / `两者都差`，",
        "> 并在 `用户备注` 中说明理由（如：A 合并两条导致否定含义混淆 / B 表头保留正确）。",
        "",
        f"- 生成时间：{res.run_summary['FINISHED_AT']}",
        f"- docs 总数：{res.run_summary['MANIFEST_ENTRIES']}",
        f"- docs loaded OK：{res.run_summary['DOCS_LOADED_OK']}",
        f"- OFFLINE 决策：CHUNKING_STRATEGY_V2_OFFLINE = **{res.run_summary['CHUNKING_STRATEGY_V2_OFFLINE']}**",
        f"- OFFLINE_PREFERRED_STRATEGY = **{res.run_summary['OFFLINE_PREFERRED_STRATEGY']}**",
        "",
        "## Cases",
        "",
        "| 编号 | 文档 | 上下文 | 风险理由 | 用户选择 | 用户备注 |",
        "|---|---|---|---|---|---|",
    ]
    for c in res.review_cases:
        lines.append(
            f"| {c.case_id} | {c.document[:60]} | {c.context[:140]} | {c.risk_reason[:60]} |  |  |"
        )
    lines.extend([
        "",
        "## 详细对比",
        "",
    ])
    for c in res.review_cases:
        lines.append(f"### {c.case_id}")
        lines.append("")
        lines.append(f"- **文档**: {c.document}")
        lines.append(f"- **上下文**: {c.context}")
        lines.append(f"- **风险点**: {c.risk_reason}")
        lines.append(f"- **用户选择**: ______（A / B / 两者均可 / 两者都差）")
        lines.append(f"- **用户备注**:")
        lines.append("  > ")
        lines.append("")
        lines.append("**Strategy A**:")
        lines.append("```text")
        lines.append(c.strategy_a_text or "(空)")
        lines.append("```")
        lines.append("")
        lines.append("**Strategy B**:")
        lines.append("```text")
        lines.append(c.strategy_b_text or "(空)")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    files["human_review_md"] = p

    # 8. chunking_strategy_v2.json (FR12)
    p = OUT_DIR / "chunking_strategy_v2.json"
    final_json = dict(res.output_json)
    final_json["deliverables"] = {k: str(v.relative_to(OUT_DIR)) for k, v in files.items()}
    _write_json(p, final_json)
    files["v2_json"] = p

    return files


def write_chunking_strategy_md(
    res: PipelineResult, files_written: dict[str, Path]
) -> Path:
    p = OUT_DIR / "chunking_strategy_v2.md"
    a = res.a_agg
    b = res.b_agg
    s = res.run_summary
    def row(metric: str, label: str, compare: bool = True) -> str:
        va = a.get(metric, "")
        vb = b.get(metric, "")
        cmp_cell = ""
        if compare:
            try:
                fva = float(va) if va != "" else 0.0
                fvb = float(vb) if vb != "" else 0.0
                if fva < fvb:
                    cmp_cell = "A 更优"
                elif fva > fvb:
                    cmp_cell = "B 更优"
                else:
                    cmp_cell = "持平"
            except Exception:
                cmp_cell = "—"
        return f"| {label} | {va} | {vb} | {cmp_cell} |"

    md = f"""# Chunking Strategy V2 — Offline A/B 评估报告

> 类型：CHUNKING IMPLEMENTATION + OFFLINE COMPARISON ONLY。
> 不包含 Embedding、检索、Milvus/MySQL/Redis 写入或上游改写。

## 1. 运行摘要

| 项目 | 值 |
|---|---|
| 开始时间 (UTC) | {s['STARTED_AT']} |
| 结束时间 (UTC) | {s['FINISHED_AT']} |
| 总耗时 (秒) | {s['ELAPSED_SEC']} |
| Manifest 条目数 | {s['MANIFEST_ENTRIES']} |
| 成功解析 | {s['DOCS_LOADED_OK']} / {s['MANIFEST_ENTRIES']} |
| 跳过条目 | {s['DOCS_SKIPPED']} |
| BGE-M3 tokenizer 可用 | {'是' if s.get('BGE_TOKENIZER_AVAILABLE') else '否 (仅使用 chars 主度量)'} |
| Parent/Child Probe | **{s['PARENT_CHILD_PROBE']}** |
| **CHUNKING_STRATEGY_V2_OFFLINE** | **{s['CHUNKING_STRATEGY_V2_OFFLINE']}** |
| **OFFLINE_PREFERRED_STRATEGY** | **{s['OFFLINE_PREFERRED_STRATEGY']}** |
| **READY_FOR_RETRIEVAL_AB_EVAL** | **{s['READY_FOR_RETRIEVAL_AB_EVAL']}** |

### 1.1 Gate 故障（如有）

{f'- 无' if not s['FAILURES'] else chr(10).join('- ' + x for x in s['FAILURES'])}

### 1.2 跳过的文档列表

"""
    if s["SKIPPED_LIST"]:
        md += "| logical_document_id | reason |\n|---|---|\n"
        for it in s["SKIPPED_LIST"]:
            md += f"| {it['logical_document_id']} | {it['reason']} |\n"
    else:
        md += "- （无）\n"
    md += f"""
## 2. 策略定义

### Strategy A — BASELINE

- **算法**：滑动窗口（生产 `chunking.py`）
- **chunk size**：{FR7_FROZEN_THRESHOLDS['A_WINDOW']} chars（normalize_ws 后）
- **overlap**：{FR7_FROZEN_THRESHOLDS['A_OVERLAP']} chars；step = {FR7_FROZEN_THRESHOLDS['A_STEP']} chars
- **长度度量**：Python len(normalize_ws(text)) 字符数
- **输入表示**：按 ParsedDocumentV2 blocks 顺序 flatten → 单一文本；TABLE 以 `to_markdown()` 参与切片

### Strategy B — STRUCTURE-AWARE HYBRID

- **优先边界**：chapter > section > article > HEADING (level≤2 或下降) > PARAGRAPH / LIST_ITEM > TABLE 独立处理
- **禁止跨强边界合并**：不同 `article` 永不合并；新 section/chapter heading 强制分组断开；同 article 下小 unit 可合并（FR2.2）
- **Oversized 递归降级顺序**：paragraph/双换行 → 句号句末 → 逗号顿号 → 硬长度切分
- **TABLE 独立处理**：小表单 chunk（TABLE_SMALL_ROWS={FR7_FROZEN_THRESHOLDS['TABLE_SMALL_ROWS']} 或 TABLE_SMALL_CHARS={FR7_FROZEN_THRESHOLDS['TABLE_SMALL_CHARS']}）；大表按 TABLE_GROUP_ROWS={FR7_FROZEN_THRESHOLDS['TABLE_SPLIT_GROUP_ROWS']} 行分组 + 所有分片重复 materialized header（`table_header_preserved=True`）
- **长度参数**：target_max = {FR7_FROZEN_THRESHOLDS['B_TARGET_MAX_CHARS']} chars；hard_max = {FR7_FROZEN_THRESHOLDS['B_HARD_MAX']} chars

## 3. FR7 冻结阈值

| 阈值 | 值 | 含义 |
|---|---|---|
"""
    for k, v in FR7_FROZEN_THRESHOLDS.items():
        md += f"| {k} | {v} | 参考 Task 7 指标阈值 / 策略参数\n"
    md += """
## 4. 指标总览 (chunking_ab_metrics.csv)

| 指标 | Strategy A | Strategy B | 离线优方 (仅参考) |
|---|---|---|---|
"""
    for metric, label in [
        ("TOTAL_CHUNKS", "TOTAL_CHUNKS"),
        ("AVG_LEN_CHARS", "AVG chunk length (chars)"),
        ("MEDIAN_LEN_CHARS_APPROX", "MEDIAN length (加权近似)"),
        ("P95_LEN_CHARS_APPROX", "P95 length (加权近似)"),
        ("MIN_LEN_CHARS", "MIN length"),
        ("MAX_LEN_CHARS", "MAX length"),
        ("OVERSIZED_CHUNKS", f"OVERSIZED_CHUNKS (> {FR7_FROZEN_THRESHOLDS['OVERSIZED_THRESHOLD_CHARS']})"),
        ("VERY_SMALL_CHUNKS", f"VERY_SMALL_CHUNKS (< {FR7_FROZEN_THRESHOLDS['VERY_SMALL_THRESHOLD_CHARS']})"),
        ("AVG_COVERAGE_PCT", "AVG SOURCE_BLOCK_COVERAGE (%)"),
        ("AVG_DUPLICATED_TEXT_RATIO", "AVG DUPLICATED_TEXT_RATIO"),
        ("CROSS_ARTICLE_CHUNKS", "CROSS_ARTICLE_CHUNKS (越低越好)"),
        ("CROSS_SECTION_CHUNKS", "CROSS_SECTION_CHUNKS (越低越好)"),
        ("HEADING_ORPHAN_COUNT", "HEADING_ORPHAN_COUNT"),
        ("ARTICLE_ORPHAN_COUNT", "ARTICLE_ORPHAN_COUNT"),
        ("TABLE_CHUNKS", "TABLE_CHUNKS"),
        ("TABLE_RELATIONSHIP_LOSS", "TABLE_RELATIONSHIP_LOSS (必须 B=0)"),
        ("PROVENANCE_FAILURES", "PROVENANCE_FAILURES (应为 0)"),
        ("METADATA_CONTRACT_FAILURES", "METADATA_CONTRACT_FAILURES (应为 0)"),
    ]:
        md += row(metric, label) + "\n"

    md += f"""
## 5. 每文档指标

完整 CSV：`chunking_ab_document_metrics.csv`（53 doc × A/B = 106 行）。

### 5.1 代表文档 A/B 对照（9 指标）

"""
    # Per-rep-doc quick table
    rep_keys_display = [
        "LEGAL_LAW_中华人民共和国民法典",
        "LEGAL_DATA_个人信息保护法",
        "ENT_CONTRACT_合同节水管理项目服务合同",
    ]

    def find_doc_metrics(substr: str, strategy: str) -> DocMetrics | None:
        dms = res.a_doc_metrics if strategy == "A" else res.b_doc_metrics
        for dm in dms:
            if substr in (dm.logical_document_id or "") or substr in (dm.title or ""):
                return dm
        return None

    md += "| 代表文档 | 策略 | total_chunks | avg_len | oversized | very_small | cov% | cross_article | cross_section | table_chunks | contract_fail |\n"
    md += "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for rep in rep_keys_display:
        for s_ in ("A", "B"):
            dm = find_doc_metrics(rep, s_)
            if dm is None:
                continue
            md += (
                f"| {rep} | {s_} | {dm.total_chunks} | {dm.avg_len_chars} | "
                f"{dm.oversized_chunks} | {dm.very_small_chunks} | "
                f"{dm.source_block_coverage_pct} | {dm.cross_article_chunks} | "
                f"{dm.cross_section_chunks} | {dm.table_chunks} | "
                f"{dm.metadata_contract_failures} |\n"
            )

    md += f"""
## 6. TABLE 审计

参见 `table_chunk_audit.csv`。每文档每 TABLE chunk 一行，列：strategy, chunk_id, table_id, part, header_preserved, structure_preserved, relationship_loss_check。

## 7. 人工评审包

请评审 20–30 对代表 case：

- **Markdown 详细版**：`chunking_ab_review_zh.md`
- **CSV 版**：`chunking_ab_human_review.csv`

用户字段：**用户选择**（A / B / 两者均可 / 两者都差）与 **用户备注** 保持空白，由真实评审者填写。禁止 AI 直接填胜者。

case 构成（由 FR10 规则自动挑选）：

- boundary 敏感：A 的跨 article chunk vs B 的对应 chunks（约 10 例）
- table：A/B 同 TABLE block（约 6–8 例）
- oversized：B 超出硬上限的告警样本（0–3 例）
- very-small：B 产生的极小 chunk，核对是否应当合并（2–4 例）
- disagreement：A 为独立 HEADING_ONLY vs B 标题+正文合并（3–5 例）
- cross-section 兜底补齐到 20–30 例

## 8. Parent/Child Probe

本阶段 Parent/Child 仅作为 OPTIONAL projection 探针验证 CMCV1 Layer E 支持。结果：**{res.pc_probe_status}**。不作为 OFFLINE_PREFERRED_STRATEGY 的直接决策依据。

## 9. STOP 检查点（§15）

本任务停止在 **OFFLINE CHUNKING A/B** 阶段。以下动作尚未发生也不应被触发：

- Embedding / 写入 Milvus / MySQL / Redis
- 检索 A/B 比较（BM25 / dense / rerank）
- Re-ingestion / 重新预处理 53 文档
- 修改 Parser / Cleaner / MetadataNormalizer / QGate / Chunk Metadata Contract V1

## 10. 交付物清单

| 文件名 | 描述 |
|---|---|
| `chunking_strategy_v2.md` | 本报告 |
| `chunking_strategy_v2.json` | 机器可消费运行摘要 + 聚合指标 + 交付物索引 |
| `strategy_a_chunks.json` | Strategy A 全部 chunk（包装 `chunk_metadata` + `text`） |
| `strategy_b_chunks.json` | Strategy B 全部 chunk（同上结构） |
| `chunking_ab_metrics.csv` | 聚合指标对比（2 行） |
| `chunking_ab_document_metrics.csv` | 106 行（53 doc × A/B）每文档指标 |
| `chunking_ab_review_zh.md` | 人工评审 Markdown 包 |
| `chunking_ab_human_review.csv` | 人工评审 CSV 包（用户字段空） |
| `table_chunk_audit.csv` | TABLE chunk 审计 |
| `test_chunking_strategy_v2.py` | 自动化测试套件（Task 13） |
"""
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    res = run_pipeline()
    # Fill tokenizer info
    from _v2_utils import tokenizer_available
    res.run_summary["BGE_TOKENIZER_AVAILABLE"] = bool(tokenizer_available())
    res.output_json["run_summary"] = res.run_summary

    files_written = write_all_deliverables(res)
    md_path = write_chunking_strategy_md(res, files_written)
    files_written["v2_md"] = md_path

    # Console summary
    print("=" * 72)
    print("CHUNKING STRATEGY V2 OFFLINE — PIPELINE SUMMARY")
    print("=" * 72)
    for k, v in res.run_summary.items():
        if k == "SKIPPED_LIST":
            continue
        if k == "FAILURES":
            print(f"  {k}: {v!r}")
            continue
        print(f"  {k}: {v}")
    print("-" * 72)
    print(f"  Strategy A total chunks: {res.a_agg.get('TOTAL_CHUNKS')}")
    print(f"  Strategy B total chunks: {res.b_agg.get('TOTAL_CHUNKS')}")
    print(f"  Parent/Child probe: {res.pc_probe_status}")
    print(f"  Review cases generated: {len(res.review_cases)}")
    print("-" * 72)
    print("  Deliverables:")
    for k, v in files_written.items():
        print(f"    {k:24s} -> {v}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
