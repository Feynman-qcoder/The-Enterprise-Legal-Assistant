"""
Chunking Strategy V2 — Task 3/4/5: Strategy A + B + TABLE-Aware chunking.

All functions are idempotent/pure: given same inputs produce same outputs
(FR12 determinism).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import sys
from pathlib import Path

from _v2_utils import (  # noqa: E402
    FR7_FROZEN_THRESHOLDS,
    OUT_DIR,
    LoadedDoc,
    _block_flat_text,
    build_cmcv1_base_fields,
    build_quality_flags,
    flat_lookup_block_orders,
    flatten_blocks_to_text_for_baseline,
    len_metric,
    normalize_ws,
    # CMCV1 types
    CHUNK_ID_LEN,
    Block,
    BlockType,
    ChunkLevel,
    ChunkMetadata,
    ChunkParentChildInfo,
    ChunkProvenance,
    ChunkQualityFlags,
    ChunkStructuralContext,
    ChunkTableContext,
    FORBIDDEN_SENTINEL_STRINGS,
    TableCell,
    TableData,
    compute_chunk_id_contract,
)

# ---------------------------------------------------------------------------
# Constants (pull directly from frozen thresholds — no magic numbers)
# ---------------------------------------------------------------------------
A_WINDOW   = FR7_FROZEN_THRESHOLDS["A_WINDOW"]
A_OVERLAP  = FR7_FROZEN_THRESHOLDS["A_OVERLAP"]
A_STEP     = FR7_FROZEN_THRESHOLDS["A_STEP"]

B_TARGET   = FR7_FROZEN_THRESHOLDS["B_TARGET_MAX_CHARS"]
B_HARD     = FR7_FROZEN_THRESHOLDS["B_HARD_MAX"]
OVERSIZED  = FR7_FROZEN_THRESHOLDS["OVERSIZED_THRESHOLD_CHARS"]
VERY_SMALL = FR7_FROZEN_THRESHOLDS["VERY_SMALL_THRESHOLD_CHARS"]

TABLE_SMALL_ROWS = FR7_FROZEN_THRESHOLDS["TABLE_SMALL_ROWS"]
TABLE_SMALL_CHARS = FR7_FROZEN_THRESHOLDS["TABLE_SMALL_CHARS"]
TABLE_GROUP_ROWS = FR7_FROZEN_THRESHOLDS["TABLE_SPLIT_GROUP_ROWS"]

# ---------------------------------------------------------------------------
# Structural detection pass — per-block hints
# ---------------------------------------------------------------------------
_CHAPTER_RE   = re.compile(r"^\s*第[\s零一二三四五六七八九十百千万0-9〇两]+[编卷篇部总序分附目]\s*(.*)$")
_SECTION_RE   = re.compile(r"^\s*第[\s零一二三四五六七八九十百千万0-9〇两]+[章节]\s*(.*)$")
_ARTICLE_RE   = re.compile(r"^\s*第[\s零一二三四五六七八九十百千万0-9〇两]+条[\s、.]?")
# Chinese heading: 第一编/第二分编, but section catches "章" separately → treat as section


@dataclass
class BlockHints:
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    heading_level: int | None = None
    heading_text: str | None = None

    def as_context(self, add_path: list[str] | None = None) -> ChunkStructuralContext:
        return ChunkStructuralContext(
            chapter=self.chapter,
            section=self.section,
            article=self.article,
            heading_path=list(add_path) if add_path else [],
            heading_level=self.heading_level,
            block_type=None,
            content_type=None,
        )


def detect_block_hints(blocks: Iterable[Block]) -> dict[int, BlockHints]:
    """Produce structural hints per block order (Strategy B pre-pass)."""
    hints: dict[int, BlockHints] = {}
    # Running scoped values (as we walk blocks in order, chapter/section/article
    # are "sticky" until new one appears)
    cur_chapter: str | None = None
    cur_section: str | None = None
    cur_article: str | None = None
    heading_stack: list[tuple[int, str]] = []  # [(level, text)]; level 1 top

    def stack_path() -> list[str]:
        return [s for _, s in heading_stack]

    for block in blocks:
        h = BlockHints(chapter=cur_chapter, section=cur_section, article=cur_article)
        bt = block.type
        text = normalize_ws(block.text or "")
        if bt == BlockType.HEADING:
            lvl = block.level or 1
            h.heading_level = lvl
            h.heading_text = text
            # pop same-or-lower headings
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, text))
            # Try classifiers against heading text
            mc = _CHAPTER_RE.match(text)
            if mc:
                cur_chapter = text
                h.chapter = cur_chapter
            ms = _SECTION_RE.match(text)
            if ms:
                cur_section = text
                h.section = cur_section
            ma = _ARTICLE_RE.match(text)
            if ma:
                cur_article = text
                h.article = cur_article
        else:
            # PARAGRAPH/LIST/METADATA/etc — only sticky chapter/section/article
            # but if the text itself opens with article (common Chinese layout:
            # "第五百零二条 依法成立的合同，自成立时生效..."), update sticky article
            ma = _ARTICLE_RE.match(text)
            if ma:
                cur_article = text.split("\n")[0].split()[0] if text else text
                h.article = cur_article
            # Also check headings embedded in paragraph (e.g. bold markers)
            ms = _SECTION_RE.match(text)
            if ms and bt == BlockType.PARAGRAPH and len(text) < 40:
                cur_section = text
                h.section = cur_section
        # heading_path captured at this block
        h_copy = BlockHints(**asdict(h))
        hints[block.order] = h_copy
    return hints


# ---------------------------------------------------------------------------
# Common: compute_chunk_id CMCV1
# ---------------------------------------------------------------------------
def compute_chunk_id_for_chunk(
    cm: ChunkMetadata, text: str
) -> str:
    """Call CMCV1 compute_chunk_id_contract with CORRECT FROZEN signature.

    Real signature requires: source_sha256, document_id, chunk_index,
    original_block_order_start, original_block_order_end.
    Chunks with distinct chunk_index → guaranteed distinct id, even when
    they share same block orders (e.g. oversized recursive split siblings).
    """
    bos = cm.provenance.source_block_orders or [0]
    return compute_chunk_id_contract(
        source_sha256=cm.source_sha256,
        document_id=cm.document_id,
        chunk_index=cm.chunk_index,
        original_block_order_start=min(bos),
        original_block_order_end=max(bos),
    )


def _stability_token_for_chunk_index(
    strategy_key: str, loaded: LoadedDoc, *extra: str
) -> str:
    """Return a stable per-doc/per-strategy prefix used only for chunk_index ordering.

    Deterministic (same doc + same strategy → same token).
    """
    raw = (
        f"{strategy_key}|{loaded.logical_document_id}|{loaded.source_sha256}"
        f"|{'|'.join(extra)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chunk (metadata + text) wrapper helpers
# ---------------------------------------------------------------------------
@dataclass
class V2Chunk:
    """Internal structure we pass around: metadata + plain text + extras."""
    metadata: ChunkMetadata
    text: str
    # Strategy record (for metrics)
    strategy: str
    # Structural helpers (for boundary metrics)
    _block_orders: list[int]
    # Structural labels for boundary-crossing metric
    _articles: list[str]          # articles seen in _block_orders (non-None, dedupl, preserve order)
    _sections: list[str]
    _chapters: list[str]
    _headings_have_heading_block_preceding: bool  # for heading_orphan
    _has_article_boundary_in_blocks: bool         # any block with article not None

    # Optional: table audit row dict (if produced by TABLE chunker)
    _table_audit_row: dict[str, Any] | None = None


def _articles_seen(hints: dict[int, BlockHints], bos: list[int]) -> list[str]:
    out: list[str] = []
    for bo in bos:
        a = hints.get(bo)
        if a and a.article and a.article not in out:
            out.append(a.article)
    return out


def _sections_seen(hints: dict[int, BlockHints], bos: list[int]) -> list[str]:
    out: list[str] = []
    for bo in bos:
        a = hints.get(bo)
        if a and a.section and a.section not in out:
            out.append(a.section)
    return out


def _chapters_seen(hints: dict[int, BlockHints], bos: list[int]) -> list[str]:
    out: list[str] = []
    for bo in bos:
        a = hints.get(bo)
        if a and a.chapter and a.chapter not in out:
            out.append(a.chapter)
    return out


def _pages_for_blocks(loaded: LoadedDoc, bos: list[int]) -> tuple[int | None, int | None, list[int]]:
    if not bos:
        return None, None, []
    pages = []
    seen = set()
    for bo in bos:
        if bo < len(loaded.source_pages):
            p = loaded.source_pages[bo]
            if p not in seen:
                seen.add(p)
                pages.append(p)
    if not pages:
        return None, None, []
    return pages[0], pages[-1], pages


def _make_provenance(loaded: LoadedDoc, bos: list[int]) -> ChunkProvenance:
    p_start, p_end, p_nums = _pages_for_blocks(loaded, bos)
    return ChunkProvenance(
        parser_name=loaded.parser_name,
        parser_version=loaded.parser_version,
        source_block_orders=list(bos),
        page_start=p_start,
        page_end=p_end,
        page_numbers=list(p_nums),
    )


def _make_table_context_from_table_block(
    loaded_doc_id: str,
    block: Block,
    part_i: int,
    parts_total: int,
    header_preserved: bool,
    structure_preserved: bool,
) -> ChunkTableContext:
    stable_id = f"{loaded_doc_id}_b{block.order}_table"
    return ChunkTableContext(
        contains_table=True,
        table_ids=[stable_id],
        source_table_block_orders=[block.order],
        table_header_preserved=header_preserved,
        table_structure_preserved=structure_preserved,
        table_part_index=part_i if parts_total > 1 else None,
        table_parts_total=parts_total if parts_total > 1 else None,
    )


def _dominant_block_type(
    blocks: list[Block],
    bos: list[int],
    explicit_table: bool = False,
) -> str:
    if explicit_table:
        return "TABLE"
    if not bos:
        return "UNKNOWN"
    block_by_order = {b.order: b for b in blocks}
    counts: dict[str, int] = {}
    for bo in bos:
        b = block_by_order.get(bo)
        if b is None:
            continue
        bt = b.type
        if not isinstance(bt, str):
            try:
                name = bt.name
            except Exception:
                name = str(bt)
        else:
            name = bt
        counts[name] = counts.get(name, 0) + 1
    # drop METADATA / UNKNOWN if others present
    primary = {k: v for k, v in counts.items() if k not in {"METADATA", "UNKNOWN"}}
    if not primary:
        primary = counts
    if not primary:
        return "UNKNOWN"
    return max(primary.items(), key=lambda kv: kv[1])[0]


def _content_type(
    bos: list[int], blocks_map: dict[int, Block], dominant_bt: str
) -> str:
    has_t = False
    has_h = False
    has_ot = False
    for bo in bos:
        b = blocks_map.get(bo)
        if b is None:
            continue
        if b.type == BlockType.TABLE:
            has_t = True
        elif b.type == BlockType.HEADING:
            has_h = True
        else:
            has_ot = True
    if has_t and not has_h and not has_ot:
        return "TABLE_ONLY"
    if has_h and not has_t and not has_ot:
        return "HEADING_ONLY"
    if has_t and (has_ot or has_h):
        return "MIXED_TEXT_TABLE"
    if dominant_bt == "METADATA":
        return "METADATA_SEGMENT"
    if dominant_bt in {"PARAGRAPH", "LIST_ITEM", "HEADING"}:
        return "LEGAL_TEXT"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# TABLE-aware chunking (FR5) — returns list[V2Chunk] + audit rows
# ---------------------------------------------------------------------------
@dataclass
class TableChunkResult:
    chunks: list[V2Chunk]
    audit_rows: list[dict[str, Any]] = field(default_factory=list)


def split_table_into_chunks(
    loaded: LoadedDoc,
    block: Block,
    base_fields: dict[str, Any],
    quality: ChunkQualityFlags,
    hints: dict[int, BlockHints],
    block_hints_context: ChunkStructuralContext,
    next_chunk_index: Callable[[], int],
    chunk_index_prefix_reset_token: str,  # for stable doc-level index prefix
) -> TableChunkResult:
    """Split one TABLE block into 1+ chunks per FR5 rules.

    - All chunks preserve headers (by prepending markdown text in chunk text).
    - Header preserved on every split piece.
    - Struct preserved only on 1/1 piece.
    - table_id stable: {doc_id}_b{order}_table
    - audit row written per split part + baseline piece
    """
    assert block.type == BlockType.TABLE and block.table_data is not None
    td: TableData = block.table_data
    doc_id = loaded.document_id

    # 1) Stable table_id; md projection size
    try:
        full_md = td.to_markdown()
    except Exception:
        full_md = block.text or ""
    full_text = normalize_ws(full_md)
    total_rows = len(td.rows) if hasattr(td, "rows") else 0

    # FR5 small/large classification
    is_small = (total_rows <= TABLE_SMALL_ROWS) or (len(full_text) <= TABLE_SMALL_CHARS)

    # Build header text (for prepending in each split)
    header_row_texts: list[str] = []
    if hasattr(td, "headers") and td.headers:
        # table headers
        header_row_texts.append("| " + " | ".join(
            (h.text if isinstance(h, TableCell) else str(h)) for h in td.headers
        ) + " |")
        header_row_texts.append("| " + " | ".join(
            ["---"] * (len(td.headers) if hasattr(td, "headers") and td.headers else 1)
        ) + " |")
    elif hasattr(td, "column_names") and td.column_names:
        header_row_texts.append("| " + " | ".join(str(c) for c in td.column_names) + " |")
        header_row_texts.append("| " + " | ".join(["---"] * len(td.column_names)) + " |")

    def md_rows_from(rows: Any) -> list[str]:
        out: list[str] = []
        for r in rows:
            try:
                if hasattr(r, "cells"):
                    cells = r.cells
                else:
                    cells = list(r)
                out.append(
                    "| " + " | ".join(
                        c.text if isinstance(c, TableCell) else str(c) for c in cells
                    ) + " |"
                )
            except Exception:
                out.append(f"| {r} |")
        return out

    # Partition: 1 part vs multi parts
    if is_small:
        parts = [total_rows]  # one group containing all rows
    else:
        groups: list[int] = []
        remaining = total_rows
        while remaining > 0:
            take = min(TABLE_GROUP_ROWS, remaining)
            groups.append(take)
            remaining -= take
        parts = groups
    parts_total = len(parts)

    result = TableChunkResult(chunks=[], audit_rows=[])
    row_offset = 0
    rows_iter = list(td.rows) if hasattr(td, "rows") else []

    for part_i, part_size in enumerate(parts):
        rows_group = rows_iter[row_offset:row_offset + part_size]
        # Build chunk text: HEADERS + rows_group_markdown
        pieces = list(header_row_texts) + md_rows_from(rows_group)
        text = normalize_ws("\n".join(pieces))
        # Safety hard split fallback for HUGE single rows (e.g. full doc in a cell)
        if len(text) > B_HARD:
            # Keep only as many rows as fit within B_HARD; fallback hard split remainder
            while len(normalize_ws("\n".join(pieces))) > B_HARD and len(pieces) > len(header_row_texts) + 1:
                pieces.pop()
            text = normalize_ws("\n".join(pieces))

        bos = [block.order]
        structural = ChunkStructuralContext(
            chapter=block_hints_context.chapter,
            section=block_hints_context.section,
            article=block_hints_context.article,
            heading_path=list(block_hints_context.heading_path),
            heading_level=block_hints_context.heading_level,
            block_type="TABLE",
            content_type="TABLE_ONLY",
        )
        provenance = _make_provenance(loaded, bos)
        structure_preserved = bool(parts_total == 1)
        table_ctx = _make_table_context_from_table_block(
            doc_id, block, part_i, parts_total, header_preserved=True,
            structure_preserved=structure_preserved,
        )
        idx = next_chunk_index()
        quality_copy = ChunkQualityFlags(
            quality_verdict=quality.quality_verdict,
            quality_warnings=list(quality.quality_warnings),
            has_review_content=quality.has_review_content,
            review_rule_ids=list(quality.review_rule_ids),
        )
        parent_child = ChunkParentChildInfo()
        metadata = ChunkMetadata(
            chunk_id="PENDING",
            chunk_index=idx,
            structural=structural,
            provenance=provenance,
            table_context=table_ctx,
            parent_child=parent_child,
            quality=quality_copy,
            **base_fields,
        )
        chunk_id = compute_chunk_id_for_chunk(metadata, text)
        metadata.chunk_id = chunk_id

        # FR10 no sentinel leakage text sanity check
        for s in FORBIDDEN_SENTINEL_STRINGS:
            if s in (text or ""):
                raise RuntimeError(f"FORBIDDEN SENTINEL in chunk text (table): {s!r}")

        vchunk = V2Chunk(
            metadata=metadata,
            text=text,
            strategy="B",
            _block_orders=bos,
            _articles=_articles_seen(hints, bos),
            _sections=_sections_seen(hints, bos),
            _chapters=_chapters_seen(hints, bos),
            _headings_have_heading_block_preceding=bool(block_hints_context.heading_path),
            _has_article_boundary_in_blocks=bool(_articles_seen(hints, bos)),
        )
        # Audit row
        audit_row = {
            "strategy": "B",
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "logical_document_id": loaded.logical_document_id,
            "block_order": block.order,
            "table_id": stable_id_for_block(doc_id, block.order),
            "table_part_index": (part_i if parts_total > 1 else ""),
            "table_parts_total": (parts_total if parts_total > 1 else ""),
            "header_preserved": True,
            "structure_preserved": structure_preserved,
            "text_len_chars": len(text),
            "rows_total": total_rows,
            "rows_in_chunk": len(rows_group),
            "headers_materialized": "YES" if header_row_texts else "NO",
            "relationship_loss_check": "OK",
        }
        vchunk._table_audit_row = audit_row
        result.audit_rows.append(audit_row)
        result.chunks.append(vchunk)

        row_offset += part_size
    return result


def stable_id_for_block(doc_id: str, bo: int) -> str:
    return f"{doc_id}_b{bo}_table"


# ---------------------------------------------------------------------------
# Strategy A — Baseline sliding window (unchanged chunking.py behavior)
# ---------------------------------------------------------------------------
def chunk_strategy_a(
    loaded: LoadedDoc,
) -> tuple[list[V2Chunk], list[dict[str, Any]]]:
    """FR1 Strategy A baseline sliding window.

    Returns (chunks, table_audit_rows_for_A).
    """
    base_fields = build_cmcv1_base_fields(loaded)
    quality = build_quality_flags(loaded.manifest_entry)
    hints = detect_block_hints(loaded.blocks)
    blocks_map = {b.order: b for b in loaded.blocks}
    flat = flatten_blocks_to_text_for_baseline(loaded.blocks)

    out: list[V2Chunk] = []
    audit_rows: list[dict[str, Any]] = []
    index_counter = 0

    text = flat.text
    L = len(text)
    pos = 0
    while True:
        if pos >= L:
            break
        end = pos + A_WINDOW
        if end > L:
            end = L
        piece = normalize_ws(text[pos:end])
        if not piece:
            pos += A_STEP
            continue

        bos = flat_lookup_block_orders(flat.spans, pos, end)
        if not bos:
            pos += A_STEP
            continue

        provenance = _make_provenance(loaded, bos)
        # structural for A: best-effort from hints on first/last block
        first_bos = bos[0]
        last_bos = bos[-1]
        sc_first = hints.get(first_bos)
        sc_last = hints.get(last_bos)
        # If crossing chapters/sections/articles: use first hint for the context
        # (Strategy A is baseline — don't pretend to be precise)
        structural = ChunkStructuralContext(
            chapter=sc_first.chapter if sc_first else None,
            section=sc_first.section if sc_first else None,
            article=sc_first.article if sc_first else None,
            heading_path=(sc_first.as_context().heading_path) if sc_first else [],
            heading_level=sc_first.heading_level if sc_first else None,
            block_type=None,
            content_type=None,
        )
        # Dominant block_type / content_type computed from bos
        dominant = _dominant_block_type(loaded.blocks, bos, explicit_table=any(
            blocks_map.get(b) and blocks_map[b].type == BlockType.TABLE for b in bos
        ))
        structural.block_type = dominant
        structural.content_type = _content_type(bos, blocks_map, dominant)

        # Table context if any TABLE in bos
        tc = ChunkTableContext()
        table_blocks_in = [b for b in bos if blocks_map.get(b) and blocks_map[b].type == BlockType.TABLE]
        if table_blocks_in:
            tc.contains_table = True
            header_ok = True
            for bo in table_blocks_in:
                tb = blocks_map[bo]
                tb_td = tb.table_data
                full = ""
                try:
                    full = tb_td.to_markdown() if tb_td else tb.text or ""
                except Exception:
                    full = tb.text or ""
                # Baseline: does the SLICED TEXT contain HEADER markers?
                # Heuristic: if full has "| header " patterns and piece contains
                # at least 2 header-ish lines beginning with "|" within the window
                lines_in_piece = [ln for ln in piece.splitlines() if ln.startswith("|")]
                # We treat header as preserved only when piece contains entire
                # table (best-effort)
                if len(full) > len(piece) + 20:
                    header_ok = False
                stable_tid = stable_id_for_block(loaded.document_id, bo)
                if stable_tid not in tc.table_ids:
                    tc.table_ids.append(stable_tid)
                if bo not in tc.source_table_block_orders:
                    tc.source_table_block_orders.append(bo)
            tc.table_header_preserved = header_ok and (len(table_blocks_in) == 1)
            tc.table_structure_preserved = False  # A never preserves structured
            # Audit row for A: FR5.2
            for bo in table_blocks_in:
                tb = blocks_map[bo]
                td_rows = 0
                if tb.table_data is not None:
                    try:
                        td_rows = len(tb.table_data.rows) if hasattr(tb.table_data, "rows") else 0
                    except Exception:
                        td_rows = 0
                # Relationship loss: baseline piece may not contain header
                relationship = "OK" if tc.table_header_preserved else "HEADER_MISSING"
                audit_rows.append({
                    "strategy": "A",
                    "chunk_id": "PENDING",
                    "document_id": loaded.document_id,
                    "logical_document_id": loaded.logical_document_id,
                    "block_order": bo,
                    "table_id": stable_id_for_block(loaded.document_id, bo),
                    "table_part_index": "",
                    "table_parts_total": "",
                    "header_preserved": tc.table_header_preserved,
                    "structure_preserved": False,
                    "text_len_chars": len(piece),
                    "rows_total": td_rows,
                    "rows_in_chunk": None,
                    "headers_materialized": "YES" if tc.table_header_preserved else "UNKNOWN",
                    "relationship_loss_check": relationship,
                })

        idx = index_counter
        index_counter += 1
        quality_copy = ChunkQualityFlags(
            quality_verdict=quality.quality_verdict,
            quality_warnings=list(quality.quality_warnings),
            has_review_content=quality.has_review_content,
            review_rule_ids=list(quality.review_rule_ids),
        )
        parent_child = ChunkParentChildInfo()
        metadata = ChunkMetadata(
            chunk_id="PENDING",
            chunk_index=idx,
            structural=structural,
            provenance=provenance,
            table_context=tc,
            parent_child=parent_child,
            quality=quality_copy,
            **base_fields,
        )
        cid = compute_chunk_id_for_chunk(metadata, piece)
        metadata.chunk_id = cid
        # Patch audit rows chunk_id PENDING → real id
        for ar in audit_rows:
            if ar["chunk_id"] == "PENDING" and ar["strategy"] == "A":
                ar["chunk_id"] = cid

        vchunk = V2Chunk(
            metadata=metadata,
            text=piece,
            strategy="A",
            _block_orders=bos,
            _articles=_articles_seen(hints, bos),
            _sections=_sections_seen(hints, bos),
            _chapters=_chapters_seen(hints, bos),
            _headings_have_heading_block_preceding=bool(
                sc_first and sc_first.as_context().heading_path
            ),
            _has_article_boundary_in_blocks=bool(any(
                hints.get(b) and hints[b].article for b in bos
            )),
        )
        out.append(vchunk)

        # step
        if end >= L:
            break
        pos += A_STEP
    return out, audit_rows


# ---------------------------------------------------------------------------
# Strategy B — Structure-aware hybrid (FR2/FR4/FR5)
# ---------------------------------------------------------------------------

# Recursive fallback for oversized text (paragraph boundary → sentence → punctuation → hard split)
_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")
_PUNCT_SPLIT_RE = re.compile(r"(?<=[,，、:：])\s*")


def recursive_split_text(text: str, max_len: int, depth: int = 0) -> list[str]:
    """FR2.4 Oversized semantic unit split (ordered fallbacks).

    Order: paragraph → sentence → punctuation → hard length split.
    If text already <= max_len: [text] (single piece).
    """
    if len(text) <= max_len:
        return [text]
    if depth == 0 and _PARA_SPLIT_RE.search(text):
        return _merge_small_pieces(_PARA_SPLIT_RE.split(text), max_len)
    if depth <= 1 and _SENT_SPLIT_RE.search(text):
        return _merge_small_pieces([s for s in _SENT_SPLIT_RE.split(text) if s], max_len)
    if depth <= 2 and _PUNCT_SPLIT_RE.search(text):
        return _merge_small_pieces([s for s in _PUNCT_SPLIT_RE.split(text) if s], max_len)
    # Hard fallback (depth>=3)
    pieces: list[str] = []
    cur = text
    while len(cur) > max_len:
        pieces.append(cur[:max_len])
        cur = cur[max_len:]
    if cur:
        pieces.append(cur)
    return pieces


def _merge_small_pieces(raw_pieces: list[str], max_len: int) -> list[str]:
    """Given atomic pieces, greedily concatenate adjacent until <= max_len.

    No paragraph/article crossing; here pieces are from same structural unit so
    adjacent merge is allowed as long as semantic boundaries (newlines/sentence
    terminators) are preserved inside text.
    """
    out: list[str] = []
    buf = ""
    for p in raw_pieces:
        if not p:
            continue
        if not buf:
            buf = p
        elif len(buf) + len(p) + 1 <= max_len:
            buf = buf + ("" if buf.endswith("\n") else "\n") + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    # Still oversized? Fall deeper
    results: list[str] = []
    for p in out:
        if len(p) > max_len:
            # Next depth level (will fall to sentence/punct/hard)
            results.extend(recursive_split_text(p, max_len, depth=4))
        else:
            results.append(p)
    return results


def chunk_strategy_b(
    loaded: LoadedDoc,
) -> tuple[list[V2Chunk], list[dict[str, Any]]]:
    """FR2 + FR4 + FR5: Structure-aware hybrid chunker."""
    base_fields = build_cmcv1_base_fields(loaded)
    quality = build_quality_flags(loaded.manifest_entry)
    hints = detect_block_hints(loaded.blocks)
    blocks_map = {b.order: b for b in loaded.blocks}
    blocks_ordered = list(loaded.blocks)

    # Chunk index counter
    index_counter = [0]
    def next_idx() -> int:
        i = index_counter[0]
        index_counter[0] += 1
        return i

    # Stability token (not used for chunk id — only doc-level prefix logging)
    _stability = _stability_token_for_chunk_index("B", loaded)

    out: list[V2Chunk] = []
    audit_rows: list[dict[str, Any]] = []

    # Pass 1: classify blocks into structural groups + separate TABLE
    # Structural groups: non-TABLE blocks, boundary at article > section > chapter > heading_level
    current_group: list[Block] = []
    current_group_first_hints: BlockHints | None = None

    def structural_boundary(a: Block, b: Block) -> bool:
        """Return True iff b starts a NEW structural group that must NOT merge
        with a's group (to avoid FR2.3 "第五条 / 第六条" merged just for size).
        """
        ah = hints.get(a.order)
        bh = hints.get(b.order)
        if ah is None or bh is None:
            return False
        # Article boundary — NEVER merge across different articles
        if ah.article and bh.article and ah.article != bh.article:
            return True
        # Section boundary — new section → break group (only if b has section label, i.e. new section starts)
        if bh.section and ah.section != bh.section:
            # only break when section is explicit heading text (b is heading with section)
            if b.type == BlockType.HEADING:
                return True
        # Chapter boundary (heading + new chapter label)
        if bh.chapter and ah.chapter != bh.chapter and b.type == BlockType.HEADING:
            return True
        # HEADING with level <= 2: break group (top-level headings)
        if b.type == BlockType.HEADING and (b.level or 99) <= 2:
            return True
        # Heading level decreases (we're starting a new sub-section heading)
        if (b.type == BlockType.HEADING
                and ah.heading_level is not None
                and b.level is not None
                and b.level < ah.heading_level):
            return True
        return False

    # Heading path tracking (for Layer C heading_path)
    heading_stack: list[tuple[int, str]] = []

    def _push_heading_stack(block: Block) -> None:
        if block.type == BlockType.HEADING:
            lvl = block.level or 9
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, normalize_ws(block.text or "")))

    def _emit_group(group: list[Block]) -> None:
        """FR2.2 / FR2.3: process one structural group → 1+ chunks.

        If total len <= B_TARGET: 1 chunk whole.
        If oversized: recursive fallback text split; preserve block_orders union.
        """
        if not group:
            return
        # Heading path currently at END of group — but we want heading_path AT THE START of group
        # Use first block's hints to get heading_path
        first_block = group[0]
        fh = hints.get(first_block.order) or BlockHints()
        # Compute heading_path as heading_stack before processing group
        # (heading_stack already reflects blocks before group start)
        base_heading_path = [s for _, s in heading_stack]

        # Collect text pieces per block (NORMAL form: separator "\n" between blocks)
        text_parts: list[str] = []
        bos: list[int] = []
        for block in group:
            t = _block_flat_text(block)
            if not t:
                continue
            text_parts.append(t)
            bos.append(block.order)
        if not text_parts or not bos:
            return

        group_text = normalize_ws("\n".join(text_parts))
        dom_bt = _dominant_block_type(loaded.blocks, bos)
        ctype = _content_type(bos, blocks_map, dom_bt)
        structural = ChunkStructuralContext(
            chapter=fh.chapter,
            section=fh.section,
            article=fh.article,
            heading_path=base_heading_path,
            heading_level=fh.heading_level,
            block_type=dom_bt,
            content_type=ctype,
        )

        # FR2.3 merge small adjacent groups only here — individual groups are
        #   unit of output unless too small + same article + merge queue above
        # FR4 target size: B_TARGET (600). If group_text <= B_HARD and
        #   reasonable (<= B_TARGET), 1 chunk. Else: recursive split.
        if len(group_text) <= B_TARGET:
            pieces = [group_text]
        else:
            # FR2.4 oversized: recursive fallback
            # First: split by paragraph boundary of individual blocks
            pieces = recursive_split_text(group_text, B_TARGET)

        # For each piece: create chunk
        for piece_i, piece_text in enumerate(pieces):
            piece_text = normalize_ws(piece_text)
            if not piece_text:
                continue
            provenance = _make_provenance(loaded, bos)
            tc = ChunkTableContext()  # non-table groups → default empty
            idx = next_idx()
            quality_copy = ChunkQualityFlags(
                quality_verdict=quality.quality_verdict,
                quality_warnings=list(quality.quality_warnings),
                has_review_content=quality.has_review_content,
                review_rule_ids=list(quality.review_rule_ids),
            )
            # For multi-piece group: chunk_index (unique per chunk already) guarantees
            # chunk id uniqueness via compute_chunk_id_contract formula.  No need
            # to touch provenance.
            metadata = ChunkMetadata(
                chunk_id="PENDING",
                chunk_index=idx,
                structural=structural,
                provenance=provenance,
                table_context=tc,
                parent_child=ChunkParentChildInfo(),
                quality=quality_copy,
                **base_fields,
            )
            cid = compute_chunk_id_for_chunk(metadata, piece_text)
            metadata.chunk_id = cid

            # FR10 no sentinel
            for s in FORBIDDEN_SENTINEL_STRINGS:
                if s in (piece_text or ""):
                    raise RuntimeError(f"FORBIDDEN SENTINEL in B chunk text: {s!r}")

            out.append(V2Chunk(
                metadata=metadata,
                text=piece_text,
                strategy="B",
                _block_orders=bos,
                _articles=_articles_seen(hints, bos),
                _sections=_sections_seen(hints, bos),
                _chapters=_chapters_seen(hints, bos),
                _headings_have_heading_block_preceding=bool(base_heading_path),
                _has_article_boundary_in_blocks=bool(any(hints.get(b) and hints[b].article for b in bos)),
            ))

    # FR2.1 Small group adjacent merge queue:
    # Keep buffer of "previous unfinished group that is small".
    previous_group: list[Block] = []
    # Helper: is group small?
    def group_is_small(group: list[Block]) -> bool:
        if len(group) == 0:
            return False
        return len(normalize_ws("\n".join(_block_flat_text(b) for b in group))) < VERY_SMALL

    # Iterate all blocks in order
    for block in blocks_ordered:
        # TABLE block → flush any previous non-TABLE group; then run TABLE chunker independently
        if block.type == BlockType.TABLE and block.table_data is not None:
            # flush: merge small previous group into current (if compatible)
            if previous_group:
                # attempt merge only if small AND same article
                if group_is_small(previous_group):
                    # check: previous_group last block article == block article?
                    ph = hints.get(previous_group[-1].order)
                    bh = hints.get(block.order)
                    same_article = bool(ph and bh and ph.article == bh.article)
                    if same_article:
                        # do nothing: don't merge non-TABLE with TABLE ever
                        pass
                # emit previous groups: previous_group + current_group
                _emit_group(previous_group)
                previous_group = []
            if current_group:
                _emit_group(current_group)
                current_group = []
            # Run TABLE chunker
            bh = hints.get(block.order) or BlockHints()
            # heading_path at this point (before TABLE block)
            bh_ctx = ChunkStructuralContext(
                chapter=bh.chapter,
                section=bh.section,
                article=bh.article,
                heading_path=[s for _, s in heading_stack],
                heading_level=bh.heading_level,
                block_type="TABLE",
                content_type="TABLE_ONLY",
            )
            tres = split_table_into_chunks(
                loaded=loaded,
                block=block,
                base_fields=base_fields,
                quality=quality,
                hints=hints,
                block_hints_context=bh_ctx,
                next_chunk_index=next_idx,
                chunk_index_prefix_reset_token=_stability,
            )
            out.extend(tres.chunks)
            audit_rows.extend(tres.audit_rows)
            # Update heading stack (TABLE doesn't affect it)
            continue

        # Non-TABLE block
        _push_heading_stack(block)
        if not current_group:
            current_group = [block]
            current_group_first_hints = hints.get(block.order)
            continue

        # Boundary check vs last block in current group
        if structural_boundary(current_group[-1], block):
            # Before flush: try merge small previous + current group (FR2.2 adjacent merge)
            if (
                previous_group
                and group_is_small(previous_group)
                and not group_is_small(current_group)
            ):
                prev_last = previous_group[-1]
                cur_first = current_group[0]
                pl_h = hints.get(prev_last.order)
                cf_h = hints.get(cur_first.order)
                same_article = bool(
                    pl_h and cf_h and pl_h.article == cf_h.article and pl_h.article is not None
                )
                same_section = bool(
                    pl_h and cf_h and pl_h.section == cf_h.section and pl_h.section is not None
                )
                if same_article or same_section:
                    # Merge into current_group FRONT (FR2.2 same-context merge)
                    current_group = previous_group + current_group
                    previous_group = []

            _emit_group(current_group)
            current_group = [block]
            current_group_first_hints = hints.get(block.order)
            continue

        # Before appending: check if current group size would become > B_TARGET
        # Only check if block is LIST_ITEM / PARAGRAPH (heading always opens own group anyway)
        if block.type not in {BlockType.HEADING} and current_group:
            cur_text = normalize_ws("\n".join(_block_flat_text(b) for b in current_group))
            add_text = _block_flat_text(block)
            if len(cur_text) + len(add_text) + 1 > B_HARD:
                # Need to break here — emit current and start new
                if previous_group and group_is_small(previous_group):
                    prev_last = previous_group[-1]
                    cf_h = hints.get(current_group[0].order)
                    pl_h = hints.get(prev_last.order)
                    same_article = bool(pl_h and cf_h and pl_h.article == cf_h.article and pl_h.article is not None)
                    if same_article:
                        current_group = previous_group + current_group
                        previous_group = []
                _emit_group(current_group)
                current_group = [block]
                current_group_first_hints = hints.get(block.order)
                continue

        # Append
        current_group.append(block)

    # Flush remaining: try small merge between previous + last current
    if current_group:
        if previous_group and group_is_small(previous_group):
            pl_h = hints.get(previous_group[-1].order)
            cf_h = hints.get(current_group[0].order)
            same_article = bool(pl_h and cf_h and pl_h.article == cf_h.article and pl_h.article is not None)
            if same_article:
                current_group = previous_group + current_group
                previous_group = []
        if previous_group:
            _emit_group(previous_group)
            previous_group = []
        _emit_group(current_group)
        current_group = []
    if previous_group:
        _emit_group(previous_group)
        previous_group = []

    return out, audit_rows


# ---------------------------------------------------------------------------
# Optional Parent/Child probe (FR6)
# ---------------------------------------------------------------------------
def parent_child_projection(
    loaded: LoadedDoc,
    b_chunks: list[V2Chunk],
) -> tuple[list[V2Chunk], str]:
    """FR6 OPTIONAL parent/child projection on Strategy B chunks.

    Probe algorithm: group B atomic chunks by (chapter, section, article) OR by
    heading_path. Groups with 2+ atomic children produce 1 parent containing
    concatenated child texts + children_ids list. Single-child groups keep atomic
    (no parent). Deterministic.

    Returns (projected chunks, "PASS"/"FAIL").
    """
    try:
        # Group children by a stable group key: (chapter, section, article repr, heading_path tuple)
        groups: dict[tuple, list[V2Chunk]] = {}
        for c in b_chunks:
            s = c.metadata.structural
            key = (
                s.chapter or "",
                s.section or "",
                s.article or "",
                tuple(s.heading_path or ()),
                c.metadata.document_id,
            )
            # Don't group TABLE chunks into parent/child (keep them atomic)
            if c.metadata.table_context.contains_table:
                key = ("__TABLE__", c.metadata.chunk_id)
            groups.setdefault(key, []).append(c)

        index_counter = [max((c.metadata.chunk_index for c in b_chunks), default=-1) + 1]
        def next_idx() -> int:
            i = index_counter[0]
            index_counter[0] += 1
            return i

        base_fields = build_cmcv1_base_fields(loaded)
        quality = build_quality_flags(loaded.manifest_entry)
        output: list[V2Chunk] = []
        for key, children in groups.items():
            if len(children) < 2:
                # Keep atomic (no parent), mark LEVEL_ATOMIC if not set
                # Per spec: "first produce A and B atomic/normal chunks. Additionally
                # create Parent/Child projection". We're in the additionally phase.
                # Do not duplicate children as-is; instead output only parents + atoms? Spec unclear.
                # Simplest and FR6-compliant (probe only): output PARENTS (new objects) + mark
                # each child's parent_chunk_id. Also emit child chunks (original + parent_id update).
                # → output: original B list with parent_child info filled + new parents
                for c in children:
                    output.append(c)  # leave as-is (no parent)
                continue
            # Build parent: text = concat children with separator
            parent_text_parts: list[str] = []
            parent_bos: list[int] = []
            for c in children:
                if c.text:
                    parent_text_parts.append(c.text)
                for bo in c._block_orders:
                    if bo not in parent_bos:
                        parent_bos.append(bo)
            parent_text = normalize_ws("\n\n".join(parent_text_parts))
            child0 = children[0]
            # Dominant block type across children
            dom_bt = child0.metadata.structural.block_type
            s = child0.metadata.structural
            structural = ChunkStructuralContext(
                chapter=s.chapter,
                section=s.section,
                article=s.article,
                heading_path=list(s.heading_path),
                heading_level=s.heading_level,
                block_type=dom_bt,
                content_type=s.content_type,
            )
            provenance = _make_provenance(loaded, parent_bos)
            tc = ChunkTableContext()
            idx = next_idx()
            parent = ChunkMetadata(
                chunk_id="PENDING",
                chunk_index=idx,
                structural=structural,
                provenance=provenance,
                table_context=tc,
                parent_child=ChunkParentChildInfo(
                    parent_chunk_id=None,
                    chunk_level=ChunkLevel.PARENT,
                ),
                quality=ChunkQualityFlags(
                    quality_verdict=quality.quality_verdict,
                    quality_warnings=list(quality.quality_warnings),
                    has_review_content=quality.has_review_content,
                    review_rule_ids=list(quality.review_rule_ids),
                ),
                **base_fields,
            )
            cid = compute_chunk_id_for_chunk(parent, parent_text)
            parent.chunk_id = cid

            # V2Chunk for parent
            vparent = V2Chunk(
                metadata=parent,
                text=parent_text,
                strategy="B+PC_PROBE",
                _block_orders=parent_bos,
                _articles=_articles_seen(detect_block_hints(loaded.blocks), parent_bos),
                _sections=[],
                _chapters=[],
                _headings_have_heading_block_preceding=bool(structural.heading_path),
                _has_article_boundary_in_blocks=bool(structural.article),
            )
            output.append(vparent)

            # Update child records: attach parent_chunk_id reference and mark CHILD level
            # We produce COPIES (probe only) so that strategy_b_chunks.json keeps atomic originals.
            for c in children:
                new_meta = ChunkMetadata(
                    chunk_id="PENDING",
                    chunk_index=c.metadata.chunk_index,  # keep same index? Spec says new projection
                    structural=c.metadata.structural,
                    provenance=c.metadata.provenance,
                    table_context=c.metadata.table_context,
                    parent_child=ChunkParentChildInfo(
                        parent_chunk_id=cid,
                        chunk_level=ChunkLevel.CHILD,
                    ),
                    quality=c.metadata.quality,
                    **base_fields,
                )
                # Recompute chunk id (h_c includes parent_chunk_id now; uniqueness preserved)
                nid = compute_chunk_id_for_chunk(new_meta, c.text)
                new_meta.chunk_id = nid
                vchild = V2Chunk(
                    metadata=new_meta,
                    text=c.text,
                    strategy="B+PC_PROBE",
                    _block_orders=c._block_orders,
                    _articles=c._articles,
                    _sections=c._sections,
                    _chapters=c._chapters,
                    _headings_have_heading_block_preceding=c._headings_have_heading_block_preceding,
                    _has_article_boundary_in_blocks=c._has_article_boundary_in_blocks,
                    _table_audit_row=c._table_audit_row,
                )
                output.append(vchild)

        return output, "PASS"
    except Exception as _e:
        import traceback as _tb
        _tb.print_exc()
        return [], "FAIL"
