# =============================================================================
# Xiaoyi Enterprise Legal RAG — PDF Parser V2 Production Implementation V1
# =============================================================================
# Deploy target (once sandbox write restriction released / after human review):
#   copy: F:\DataBase\trae_work\RAG\production_deploy\modules\ingestion\pdf_parser_v2.py
#   → to: D:\xiaoyi\Legal_System\modules\ingestion\pdf_parser_v2.py
#
# FROZEN Production Route (from Closing Verification V1, immutable RULES 1-7):
#
#   PDF → Geometry-aware Extraction (PyMuPDF evidence)
#       → Layout Normalizer (round 4, typography features)
#       → Xiaoyi Page-local Reading Order Reconstruction
#           (single / multi / full-width-heading + columns)
#       → Table Detection Router
#           RULE1 lines_strict (Stage1, ONLY DEFAULT)
#           RULE2  text strategy = DISABLED Production V1
#           RULE3  lines fallback IFF stage1 TRUE count == 0 AND HIGH_SIGNAL
#       → RULE4 Form / Semantic Table Gate (6-class deterministic)
#       → RULE5 Table bbox ownership (center-in-bbox) — no double emission
#       → RULE6 No adjacent-text merge guess (colspan/rowspan default 1,1)
#       → Semantic Block Builder (6 Frozen BlockTypes ONLY)
#       → ParsedDocument V2 Contract (RULE7 — DO NOT MODIFY CONTRACT MODULE)
#       → LegacyAdapter → Current Chunker (unchanged)
#
# Non-goals (THIS Production V1 scope):
#   - NO OCR (scanned/image-only → UNKNOWN + warning)
#   - NO aggressive header/footer remover → Cleaner V2 (NOT started yet)
#   - NO 20-doc Frozen Shadow → reserved for NEXT independent task
#   - NO PyMuPDF license conclusion → NEEDS_REVIEW (stays open)
# =============================================================================

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Frozen Contract (RULE7: MUST NOT MODIFY the contract module).
# Task 20: deploy root resolved relative to this file (repo root = parents[2]).
_D_PROJECT = Path(__file__).resolve().parents[2]
if str(_D_PROJECT) not in sys.path:
    sys.path.insert(0, str(_D_PROJECT))

from modules.ingestion.parsed_document_v2 import (  # noqa: E402
    Block,
    BlockType,
    ContractViolation,
    LegacyAdapter,
    ParsedDocument,
    Provenance,
    TableCell,
    TableData,
)

# --- PyMuPDF engine (geometry evidence only; Xiaoyi = semantic reconstruction)
try:
    import pymupdf  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PDF Parser V2 Production requires PyMuPDF (expected conda env xiaoyi_rag, 1.28.2)."
    ) from exc


# =============================================================================
# §34 Parser Identity (production stable; no poc/spike/closing terms)
# =============================================================================
PARSER_NAME = "pdf_parser_v2"
PARSER_VERSION = "2.0.0"


# =============================================================================
# §33 Stable warning keys
# =============================================================================
W_SCANNED = "SCANNED_OR_IMAGE_ONLY_PAGE"
W_TABLE_AMBIG = "TABLE_AMBIGUITY"
W_MERGED_CELL = "MERGED_CELL_NOT_RECOVERED"
W_READING_ORDER = "READING_ORDER_AMBIGUITY"
W_EMPTY_PAGE = "EMPTY_PAGE"
W_ANOMALY = "PDF_EXTRACTION_ANOMALY"


# =============================================================================
# §12 Production Config (thresholds centralized; small & deterministic)
# =============================================================================

@dataclass
class PdfParserV2Config:
    """Production V1 thresholds. Frozen Router Rules enforced by code.

    RULE2 Safety: text_strategy_enable MUST stay False in Production V1.
      Even if a caller flips it to True, parse_pdf_v2() will REFUSE to call
      find_tables(strategy="text") because RULE2 mandates DISABLED for V1.
      The field exists ONLY for forward-compat experimentation in isolated
      dev runs; clearly marked EXPERIMENTAL / DISABLED.
    """

    # --- geometry ---
    bbox_round_precision: int = 4

    # --- reading order ---
    line_y_tolerance: float = 2.0
    column_gap_ratio: float = 0.08       # fraction of page width treated as gap
    paragraph_gap_ratio: float = 0.025
    heading_font_size_delta: float = 1.8
    min_line_words: int = 1

    # --- table router / gate (frozen rules hard-enforced in code) ---
    table_strategy_default: str = "lines_strict"   # RULE1 (do not change)
    enable_lines_fallback: bool = True              # RULE3 enable flag
    min_table_rows: int = 3
    min_table_cols: int = 2
    form_colon_ratio_threshold: float = 0.08        # > TH → form-like signal
    signature_signal_threshold: int = 1             # >= TH → signature layout signal
    table_struct_keywords: Tuple[str, ...] = (
        "表", "表格", "一览表", "明细表", "附表", "清单", "汇总表",
    )

    # --- EXPERIMENTAL / DISABLED (RULE2 Production V1 DISABLED) ---
    text_strategy_enable: bool = False  # MUST STAY False.
    text_strategy_experimental_note: str = (
        "[EXPERIMENTAL / DISABLED RULE2] text strategy DISABLED for Production V1. "
        "Re-enable requires: separate task, 20-doc labeled GT + dedicated text-FP "
        "classifier training + independent Gate (NOT covered here)."
    )

    # --- warning cap ---
    max_warnings_per_doc: int = 50


# =============================================================================
# Internal layout primitives (NEVER exposed to Frozen public Contract)
# =============================================================================

@dataclass
class _Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    flags: int
    font_name: str
    block_no: int
    line_no: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class _Line:
    words: List[_Word]
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    is_bold: bool
    text: str
    col: int = 0       # 0 = full-width heading; 1..N = column index

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class _PageGeometry:
    page_index_0: int
    page_number_1: int
    width: float
    height: float
    blocks: List[Tuple]
    words: List[_Word]
    lines: List[_Line]
    spans: List[Dict[str, Any]]
    plain_text: str
    rect_drawings_count: int = 0
    line_drawings_count: int = 0

    @property
    def empty(self) -> bool:
        return not self.plain_text.strip() and len(self.words) == 0


@dataclass
class _TableCandidate:
    strategy: str
    bbox: Tuple[float, float, float, float]
    rows: int
    cols: int
    cells_grid: List[List[str]]
    classification: str
    classification_confidence: float = 0.0
    signal_details: Dict[str, Any] = field(default_factory=dict)
    pymu_obj: Any = None

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


# =============================================================================
# §13 Geometry-aware extraction (PyMuPDF → Xiaoyi evidence primitives)
# =============================================================================

def _round(v: float, n: int = 4) -> float:
    try:
        return round(float(v), n)
    except Exception:
        return 0.0


def extract_page_geometry(
    page: "pymupdf.Page",
    page_index_0: int,
    cfg: PdfParserV2Config,
) -> _PageGeometry:
    """Evidence extraction. Deterministic; all coordinates round cfg.precision.

    - blocks: 7-tuples from page.get_text("blocks")  (coords round 4)
    - words: 8-tuple → typed _Word (font info recovered from dict spans)
    - spans: normalized list of dicts (text / bbox / size / flags / font / block_no / line_no)
    - drawing counts (rect / line) — used by HIGH_SIGNAL and decor-rect FP Q4.
    """
    prec = cfg.bbox_round_precision
    width = float(page.rect.width)
    height = float(page.rect.height)

    raw_blocks = list(page.get_text("blocks") or [])
    norm_blocks: List[Tuple] = []
    for b in raw_blocks:
        if len(b) >= 4:
            nb = [_round(b[0], prec), _round(b[1], prec), _round(b[2], prec), _round(b[3], prec)]
            for rest in list(b[4:]):
                nb.append(rest)
            norm_blocks.append(tuple(nb))
        else:
            norm_blocks.append(tuple(b))

    raw_words = list(page.get_text("words") or [])
    words: List[_Word] = []

    dict_data = page.get_text("dict") or {}
    span_fonts: Dict[Tuple[int, int], Tuple[float, int, str]] = {}
    spans_for_out: List[Dict[str, Any]] = []
    for block in dict_data.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        b_no = int(block.get("number", 0))
        for ln_idx, line in enumerate(block.get("lines", []) or []):
            for span in line.get("spans", []) or []:
                key = (b_no, ln_idx)
                fs = float(span.get("size", 10.0))
                flags = int(span.get("flags", 0) or 0)
                font = str(span.get("font", ""))
                prev = span_fonts.get(key)
                if prev is None or fs > prev[0]:
                    span_fonts[key] = (fs, flags, font)
                sbbox = list(span.get("bbox", (0, 0, 0, 0)))
                while len(sbbox) < 4:
                    sbbox.append(0.0)
                span_out = {
                    "text": str(span.get("text", "")),
                    "bbox": (_round(sbbox[0], prec), _round(sbbox[1], prec),
                             _round(sbbox[2], prec), _round(sbbox[3], prec)),
                    "size": fs,
                    "flags": flags,
                    "font": font,
                    "block_no": b_no,
                    "line_no": ln_idx,
                }
                spans_for_out.append(span_out)

    for w in raw_words:
        if len(w) < 8:
            continue
        x0, y0, x1, y1, text, b_no, l_no, _w_no = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
        b_no_i = int(b_no); l_no_i = int(l_no)
        fs, flags, font = span_fonts.get((b_no_i, l_no_i), (10.0, 0, ""))
        words.append(_Word(
            text=str(text),
            x0=_round(x0, prec), y0=_round(y0, prec),
            x1=_round(x1, prec), y1=_round(y1, prec),
            font_size=fs, flags=flags, font_name=font,
            block_no=b_no_i, line_no=l_no_i,
        ))

    paths = list(page.get_drawings() or [])
    rect_count = 0
    line_count = 0
    for item in paths:
        if not isinstance(item, dict):
            continue
        items_list = item.get("items", []) or []
        for it in items_list:
            if isinstance(it, (list, tuple)) and it:
                tk = it[0] if isinstance(it[0], str) else ""
                if tk == "re":
                    rect_count += 1
                elif tk == "l":
                    line_count += 1
        kind = item.get("type", "")
        if kind == "re":
            rect_count += 1
        elif kind == "l":
            line_count += 1

    plain_text = (page.get_text("text") or "").strip()

    return _PageGeometry(
        page_index_0=page_index_0,
        page_number_1=page_index_0 + 1,
        width=width, height=height,
        blocks=norm_blocks, words=words,
        lines=[],
        spans=spans_for_out,
        plain_text=plain_text,
        rect_drawings_count=rect_count,
        line_drawings_count=line_count,
    )


# =============================================================================
# §14 Layout Normalizer + Reading Order Reconstruction (page-local ONLY §15)
# =============================================================================

_LEGAL_CH_RE = re.compile(
    r"^\s*第[一二三四五六七八九十百千万零〇两0-9]+[章节编篇部卷节总]+\s*[：:.]?\s*"
)
_LEGAL_ART_RE = re.compile(r"^\s*第[一二三四五六七八九十百千万零〇两0-9]+条\b")


def normalize_layout_elements(geom: _PageGeometry, cfg: PdfParserV2Config) -> List[_Line]:
    """Group words into lines using (block_no,line_no) when available; y-bucket fallback.

    Deterministic; output lines sorted by geometry position.
    """
    prec = cfg.bbox_round_precision
    tol = cfg.line_y_tolerance

    clusters: Dict[Tuple[int, int], List[_Word]] = {}
    residual: List[_Word] = []
    # Heuristic: if block_no == 0 for all words and words count high, PDF lacks proper
    # block/line info → treat ALL as residual and use y-bucket.
    all_zero_blob = bool(geom.words) and all(w.block_no == 0 and w.line_no == 0 for w in geom.words) and len(geom.words) > 50
    for w in geom.words:
        key = (w.block_no, w.line_no)
        if all_zero_blob:
            residual.append(w)
            continue
        clusters.setdefault(key, []).append(w)

    all_line_groups: List[List[_Word]] = list(clusters.values())
    if residual:
        rlist = sorted(residual, key=lambda w: (w.cy, w.cx))
        bucket: List[_Word] = []
        bucket_cy: Optional[float] = None
        for w in rlist:
            if bucket_cy is None:
                bucket = [w]; bucket_cy = w.cy; continue
            if abs(w.cy - bucket_cy) <= tol:
                bucket.append(w)
            else:
                all_line_groups.append(bucket)
                bucket = [w]; bucket_cy = w.cy
        if bucket:
            all_line_groups.append(bucket)

    lines_out: List[_Line] = []
    for grp in all_line_groups:
        if not grp:
            continue
        grp_sorted = sorted(grp, key=lambda w: (w.cx,))
        words_ok = [w for w in grp_sorted if w.text.strip()]
        if not words_ok and cfg.min_line_words >= 1:
            continue
        x0 = min(w.x0 for w in grp_sorted)
        y0 = min(w.y0 for w in grp_sorted)
        x1 = max(w.x1 for w in grp_sorted)
        y1 = max(w.y1 for w in grp_sorted)
        fsize = max(w.font_size for w in grp_sorted)
        bold = any(((w.flags & 16) != 0) for w in grp_sorted)
        text = " ".join(w.text for w in grp_sorted)
        lines_out.append(_Line(
            words=grp_sorted,
            x0=_round(x0, prec), y0=_round(y0, prec),
            x1=_round(x1, prec), y1=_round(y1, prec),
            font_size=_round(fsize, 1), is_bold=bool(bold), text=text,
        ))
    return lines_out


def reconstruct_reading_order(
    geom: _PageGeometry,
    cfg: PdfParserV2Config,
) -> List[_Line]:
    """Single / Multi / Full-width-heading + columns reconstruction.

    Deterministic. Page-local; NEVER cross-page merge here.
    """
    lines = geom.lines
    if not lines:
        return []
    W = geom.width
    gap_th = W * cfg.column_gap_ratio

    # Step 1: full-width heading mask (width > 0.9 * page_w)
    fw_mask = [l.width > W * 0.9 for l in lines]
    non_fw = [l for l, f in zip(lines, fw_mask) if not f]

    # Step 2: find column x-breaks among line-centers of non-fw lines
    col_breaks: List[float] = []
    if non_fw:
        xs = sorted({round(l.cx, 1) for l in non_fw})
        for a, b in zip(xs, xs[1:]):
            if b - a >= gap_th:
                col_breaks.append((a + b) / 2.0)

    if col_breaks:
        n_cols = len(col_breaks) + 1
        for i, l in enumerate(lines):
            if fw_mask[i]:
                l.col = 0
                continue
            cidx = 1
            for cb in col_breaks:
                if l.cx < cb:
                    break
                cidx += 1
            l.col = cidx
        tol = cfg.line_y_tolerance * 2.0
        lines_sorted = sorted(
            lines,
            key=lambda l: (int((l.cy // tol) * tol * 100), l.col, l.cy, l.x0),
        )
        return lines_sorted

    return sorted(lines, key=lambda l: (l.cy, l.x0))


# =============================================================================
# §16/17/18  Table Detection Router + High Signal + Gate (RULE 1/2/3/4)
# =============================================================================

_TABLE_CLASS_TRUE = "TRUE_SEMANTIC_TABLE"
_TABLE_CLASS_FORM = "FORM_LAYOUT"
_TABLE_CLASS_LABEL = "LABEL_VALUE_LAYOUT"
_TABLE_CLASS_SIG = "SIGNATURE_LAYOUT"
_TABLE_CLASS_OTHER = "OTHER_FALSE_POSITIVE"
_TABLE_CLASS_UNCERTAIN = "UNCERTAIN"

_SIGNATURE_SIGNALS: Tuple[str, ...] = (
    "签字", "签名", "盖章", "公章", "法定代表人", "委托代理人",
    "开户银行", "邮政编码", "（签字）", "(签字)", "（盖章）",
)


def _stable_json(obj: Any, *, prec: int = 4) -> str:
    def _norm(o: Any) -> Any:
        if isinstance(o, float):
            return round(o, prec)
        if isinstance(o, (list, tuple)):
            return [_norm(x) for x in o]
        if isinstance(o, dict):
            return {str(k): _norm(v) for k, v in sorted(o.items())}
        return o
    return json.dumps(_norm(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pymu_table_cells_grid(t: Any) -> List[List[str]]:
    """Extract plain cells from a pymupdf Table."""
    try:
        cells: List[List[str]] = []
        for row in (t.extract() or []):
            rr: List[str] = []
            for c in (row or []):
                rr.append("" if c is None else str(c).strip())
            cells.append(rr)
        if cells:
            max_w = max(len(r) for r in cells)
            cells = [r + [""] * (max_w - len(r)) for r in cells]
        return cells
    except Exception:
        return []


def _pymu_table_bbox(t: Any, prec: int = 4) -> Tuple[float, float, float, float]:
    try:
        b = t.bbox
        if b is None:
            return (0.0, 0.0, 0.0, 0.0)
        return (_round(b[0], prec), _round(b[1], prec), _round(b[2], prec), _round(b[3], prec))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def _high_signal_table_evidence(
    geom: _PageGeometry,
    cfg: PdfParserV2Config,
) -> Tuple[bool, Dict[str, Any]]:
    """RULE3 HIGH_SIGNAL:

    (a) structural "表" keyword OR (decor rects + multi-col)
    AND
    (b) multi-column alignment evidence
    AND
    (c) low colon density + low signature signals
    """
    plain = geom.plain_text
    kw_hits = sum(1 for kw in cfg.table_struct_keywords if kw in plain)
    if geom.lines:
        xs = [round(l.x0, 0) for l in geom.lines if l.text.strip()]
        freq: Dict[float, int] = {}
        for x in xs:
            freq[x] = freq.get(x, 0) + 1
        sorted_x = sorted(freq.items(), key=lambda kv: -kv[1])
        total = sum(freq.values()) or 1
        acc = 0
        n_x = 0
        for _, cnt in sorted_x:
            acc += cnt; n_x += 1
            if acc / total >= 0.6:
                break
        multi_col = n_x >= 2 and len(geom.lines) >= 6
    else:
        multi_col = False
    line_count_lines = max(len(plain.splitlines()), 1)
    colon_density_lines = (plain.count("：") + plain.count(":")) / line_count_lines
    sig_hits = sum(1 for s in _SIGNATURE_SIGNALS if s in plain)
    low_noise = (
        colon_density_lines < cfg.form_colon_ratio_threshold
        and sig_hits < cfg.signature_signal_threshold
    )
    decor_rect_enough = geom.rect_drawings_count >= 20
    ok = (
        (kw_hits >= 1 or (decor_rect_enough and multi_col))
        and multi_col
        and low_noise
    )
    details = {
        "table_keyword_hits": kw_hits,
        "multi_col": multi_col,
        "colon_density_lines": round(colon_density_lines, 4),
        "signature_hits": sig_hits,
        "rect_drawings": geom.rect_drawings_count,
        "decor_rect_enough": decor_rect_enough,
        "low_noise": low_noise,
    }
    return bool(ok), details


def classify_table_candidate(
    cand: _TableCandidate,
    geom: _PageGeometry,
    cfg: PdfParserV2Config,
) -> Tuple[str, Dict[str, Any]]:
    """RULE4 6-class deterministic Form / Semantic Gate.

    Only TRUE_SEMANTIC_TABLE may emit BlockType.TABLE.
    """
    grid = cand.cells_grid
    if (not grid) or len(grid) < cfg.min_table_rows or (grid and len(grid[0]) < cfg.min_table_cols):
        return _TABLE_CLASS_OTHER, {"reason": "shape_below_min"}

    header_row = grid[0]
    first_col = [r[0].strip() for r in grid if r]
    cells_non_empty = 0
    colon_cells = 0
    for row in grid:
        for c in row:
            cs = c.strip()
            if not cs:
                continue
            cells_non_empty += 1
            if "：" in cs or ":" in cs:
                colon_cells += 1
    colon_ratio = colon_cells / max(cells_non_empty, 1)

    joined_all = "\n".join(" ".join(r) for r in grid)
    sig_hits_cells = sum(1 for s in _SIGNATURE_SIGNALS if s in joined_all)

    # Numeric seq first column (1..N)
    # Critical Hardening Root A: fullwidth digit support (０-９ / U+FF10..FF19)
    # for num_first & numeric body recognition — SHADOW-019 uses fullwidth
    # １/２/３ in catalog layouts, so must not bypass the rejection gate.
    _FW = str.maketrans({"０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
                         "５": "5", "６": "6", "７": "7", "８": "8", "９": "9"})

    def _norm_digits(s: str) -> str:
        return s.translate(_FW)

    num_first = 0
    for f in first_col:
        fn = _norm_digits(f)
        fm = re.match(r"^\s*(\d+)\s*$", fn)
        if fm:
            try:
                n = int(fm.group(1))
                if 1 <= n <= 5000:
                    num_first += 1
            except ValueError:
                pass
    num_seq_ok = num_first >= 2 or (len(first_col) >= 3 and num_first >= 1)

    header_no_colon = all(("：" not in h and ":" not in h) for h in header_row if h)
    header_short = all(len(h) <= 24 for h in header_row if h)
    noun_like_header = header_no_colon and header_short and any(
        any("\u4e00" <= ch <= "\u9fff" or ch.isalnum() for ch in h)
        for h in header_row
        if h
    )

    header_keys = ("名称", "规格", "型号", "单位", "数量", "单价", "金额",
                   "价格", "备注", "序号", "编号", "地址", "日期", "产地",
                   "年份", "时间", "电话", "联系人", "厂家", "品牌",
                   "能力", "功率")
    hk_hits = sum(1 for h in header_row for k in header_keys if k in h)

    details = {
        "rows": len(grid),
        "cols": len(grid[0]) if grid else 0,
        "colon_ratio_cells": round(colon_ratio, 4),
        "sig_hits_in_cells": sig_hits_cells,
        "numeric_first_col_count": num_first,
        "numeric_seq_ok": num_seq_ok,
        "header_noun_like": noun_like_header,
        "header_keyword_hits": hk_hits,
    }

    # Critical Hardening Root A: list/catalog layout structural signals (derived
    # from SHADOW-004 / SHADOW-019 Root Cause diagnosis; NOT filename special case).
    #
    # KEY INSIGHT (from FIX-001/FIX-008 vs SHADOW-004/019 signal analysis):
    #   TRUE semantic bordered tables — PyMuPDF find_tables(lines_strict) often
    #   returns low populated_density (0.05..0.35) due to line-art artifacts
    #   producing extra empty-cell slots + many cols (>= 8).
    #   Catalog/list/TOC layouts — HIGH populated_density (>= 0.70) because
    #   every visual line has something, typically with <= 4 columns and a
    #   numeric first column (序号) but WITHOUT real numeric-data columns.
    cells_total = sum(len(r) for r in grid)
    populated_density = cells_non_empty / max(cells_total, 1)
    row_count = len(grid)
    col_count = len(grid[0]) if grid else 0
    body_rows = grid[1:] if row_count > 1 else grid

    def _cell_numeric(v: str) -> bool:
        if not v:
            return False
        v = _norm_digits(v.strip())
        v = v.replace(",", "").replace("，", "")
        v = v.replace("%", "").replace("％", "")
        # Strip currency prefixes / units — generous numeric recogniser so we
        # count true numeric-data columns correctly (e.g. ¥12.50 / 100元).
        v = re.sub(r"^[¥￥$€£]+\s*", "", v)
        v = re.sub(r"\s*(元|RMB|CNY|万)$", "", v)
        if re.match(r"^-?\d+(?:\.\d+)?$", v.strip()):
            return True
        return False

    numeric_body_rows = 0
    long_cell_rows = 0
    very_long_cells = 0
    for r in body_rows:
        numeric_cols_in_row = sum(1 for c in r if _cell_numeric(c))
        if numeric_cols_in_row >= 2:
            numeric_body_rows += 1
        row_lens = [len(str(c)) for c in r]
        if row_lens and max(row_lens) >= 35:
            long_cell_rows += 1
        very_long_cells += sum(1 for l in row_lens if l >= 80)

    numeric_data_rows_ratio = numeric_body_rows / max(len(body_rows), 1)
    long_cell_row_ratio = long_cell_rows / max(len(body_rows), 1)
    second_col_long = 0
    if col_count >= 2:
        for r in body_rows:
            if len(r) >= 2 and len(str(r[1])) >= 30:
                second_col_long += 1
    second_col_long_ratio = second_col_long / max(len(body_rows), 1)

    # ========================================================================
    # § Narrow Recall Hardening V1: STRUCTURAL SCHEMA EVIDENCE CHECK
    # ------------------------------------------------------------------------
    # Surface pattern (rows≥10, cols≤4, num_seq_ok, low numeric ratio) is
    # NECESSARY but NOT SUFFICIENT for list/catalog.
    #
    # TRUE semantic narrow tables (even short-text ones) — typically expose
    # at least TWO non-sequence body columns that are:
    #   (a) populated in ≥ 60-70% of body rows, AND
    #   (b) correspond to distinct header cells (schema), AND
    #   (c) each contributes ≥ 4% of body text share.
    #
    # FALSE list/catalog layouts (even with long titles) — typically degenerate
    # into a sequence column plus exactly ONE rich descriptor column; any
    # other candidate columns are sparse / near-empty line-art artifacts.
    # ========================================================================
    col_occupancies: List[float] = []
    body_text_len_by_col: List[int] = [0] * col_count
    for r in body_rows:
        for ci in range(col_count):
            cell_val = "" if ci >= len(r) else str(r[ci])
            if cell_val.strip():
                body_text_len_by_col[ci] += len(cell_val)
    for ci in range(col_count):
        n_empty_this_col = 0
        for r in body_rows:
            if ci >= len(r) or not str(r[ci]).strip():
                n_empty_this_col += 1
        col_occupancies.append(1.0 - (n_empty_this_col / max(len(body_rows), 1)))
    non_seq_col_occ = col_occupancies[1:] if col_count > 1 else []
    non_seq_body_len = body_text_len_by_col[1:]
    total_non_seq_body_len = max(sum(non_seq_body_len), 1)
    # Rich non-seq cols: ≥ 70% body row occupancy
    rich_non_seq_cols_70 = sum(1 for f in non_seq_col_occ if f >= 0.70)
    # Multi-role non-seq cols: ≥ 60% occupancy AND ≥ 4% of body text share
    multi_role_non_seq_cols = 0
    for ci in range(1, col_count):
        occ = col_occupancies[ci] if ci < len(col_occupancies) else 0
        text_share = (
            (body_text_len_by_col[ci] / total_non_seq_body_len)
            if total_non_seq_body_len else 0
        )
        if occ >= 0.60 and text_share >= 0.04:
            multi_role_non_seq_cols += 1
    # Single-descriptor dominance: degenerate to one descriptor column
    if non_seq_body_len and sum(non_seq_body_len) > 0:
        max_non_seq_share = max(non_seq_body_len) / sum(non_seq_body_len)
    else:
        max_non_seq_share = 0.0
    any_secondary_occupied = any(
        f >= 0.10 for f in (non_seq_col_occ[1:] if len(non_seq_col_occ) > 1 else [])
    )
    single_descriptor_dominant = (
        (col_count <= 2)
        or (max_non_seq_share >= 0.92 and not any_secondary_occupied)
    )
    # Header → body correspondence: non-empty header cells map to cols with
    # ≥ 20% body occupancy.
    header_nonempty_idx = [
        i for i in range(col_count)
        if i < len(header_row) and header_row[i].strip()
    ]
    if header_nonempty_idx:
        hdr_body_correspond = sum(
            1 for i in header_nonempty_idx
            if i < len(col_occupancies) and col_occupancies[i] >= 0.20
        ) / len(header_nonempty_idx)
    else:
        hdr_body_correspond = 0.0

    # STRUCTURAL TRUE override: unambiguous 2-D role schema → do NOT reject
    # as list/catalog, regardless of rows / cols / numeric_ratio / cell len.
    structural_true_schema = (
        (rich_non_seq_cols_70 >= 2 and not single_descriptor_dominant)
        or multi_role_non_seq_cols >= 2
    ) and (hdr_body_correspond >= 0.66)

    list_catalog_reject = False
    # Catalog/LIST rejection #0 — narrow low-numeric surface pattern.
    #   STRUCTURAL SAVE: structural_true_schema → skip, it's a TRUE narrow
    #   semantic table (检查项目/管理要求/备注 or 风险类型/等级/责任部门 etc).
    if (
        not structural_true_schema
        and row_count >= 10
        and col_count <= 4
        and num_seq_ok
        and numeric_data_rows_ratio < 0.15
    ):
        list_catalog_reject = True
    # Catalog/LIST rejection #1 — dense small-grid list-with-序号 shape.
    if (
        not structural_true_schema
        and row_count >= 3
        and col_count <= 4
        and num_seq_ok
        and numeric_data_rows_ratio < 0.2
        and (long_cell_row_ratio >= 0.35 or second_col_long_ratio >= 0.5)
        and populated_density >= 0.80
    ):
        list_catalog_reject = True
    # Catalog/LIST rejection #2 — mid-density long small-grid catalog shape.
    if (
        not structural_true_schema
        and not list_catalog_reject
        and row_count >= 12
        and col_count <= 4
        and populated_density >= 0.65
        and numeric_data_rows_ratio < 0.2
        and (long_cell_row_ratio >= 0.25 or second_col_long_ratio >= 0.3)
    ):
        list_catalog_reject = True

    details.update({
        "populated_density": round(populated_density, 4),
        "numeric_body_rows_with_2plus_cols": numeric_body_rows,
        "numeric_data_rows_ratio": round(numeric_data_rows_ratio, 4),
        "long_cell_row_ratio": round(long_cell_row_ratio, 4),
        "second_col_long_ratio": round(second_col_long_ratio, 4),
        "very_long_cells_ge80": very_long_cells,
        # — Narrow Recall Hardening V1 structural signals —
        "col_occupancies_non_seq": [round(f, 3) for f in non_seq_col_occ],
        "rich_non_seq_cols_ge70_pct": rich_non_seq_cols_70,
        "multi_role_non_seq_cols_ge60pct_4pct_share": multi_role_non_seq_cols,
        "hdr_body_correspond": round(hdr_body_correspond, 4),
        "single_descriptor_dominant": single_descriptor_dominant,
        "max_non_seq_text_share": round(max_non_seq_share, 4),
        "structural_true_schema": structural_true_schema,
        "list_catalog_reject": list_catalog_reject,
    })

    # --- Decision tree (deterministic, no random) ---
    if sig_hits_cells >= 2:
        return _TABLE_CLASS_SIG, {**details, "reason": "sig_cells>=2"}
    if colon_ratio >= max(0.25, cfg.form_colon_ratio_threshold * 3):
        if sig_hits_cells >= 1:
            return _TABLE_CLASS_FORM, {**details, "reason": "high_colon+sig"}
        if not noun_like_header:
            return _TABLE_CLASS_LABEL, {**details, "reason": "colon_ratio_high"}

    # Critical Hardening Root A: explicit list/catalog layout short-circuit
    # BEFORE TRUE gates. Preserves TRUE numeric tables — only rejects when
    # numeric_data_rows_ratio still <0.2 (no true numeric-data content).
    if list_catalog_reject and numeric_data_rows_ratio < 0.2:
        return _TABLE_CLASS_OTHER, {**details, "reason": "list_or_catalog_layout_like"}

    if noun_like_header and hk_hits >= 2 and (num_seq_ok or len(grid) >= cfg.min_table_rows + 1):
        return _TABLE_CLASS_TRUE, {**details, "reason": "noun_hdr+keys+seq", "conf": 0.9}
    if num_seq_ok and hk_hits >= 1 and colon_ratio < cfg.form_colon_ratio_threshold:
        return _TABLE_CLASS_TRUE, {**details, "reason": "seq+hdr+low_noise", "conf": 0.85}

    if colon_ratio > cfg.form_colon_ratio_threshold * 1.5:
        return _TABLE_CLASS_FORM, {**details, "reason": "borderline_form"}
    if hk_hits >= 1 and colon_ratio < cfg.form_colon_ratio_threshold:
        return _TABLE_CLASS_UNCERTAIN, {**details, "reason": "weak_true_signal", "conf": 0.55}
    return _TABLE_CLASS_OTHER, {**details, "reason": "catchall_other"}


def detect_table_candidates(
    page: "pymupdf.Page",
    geom: _PageGeometry,
    cfg: PdfParserV2Config,
) -> Tuple[List[_TableCandidate], Dict[str, Any]]:
    """RULE 1/2/3 Router.

    RULE1 Stage1: always lines_strict (default).
    RULE3 Stage2 lines fallback: only if Stage1 TRUE count == 0 AND HIGH_SIGNAL passed.
    RULE2 text strategy: Production V1 DISABLED. NEVER call find_tables("text").
          Even if cfg.text_strategy_enable accidentally True → SAFETY GUARD block.
    """
    prec = cfg.bbox_round_precision
    # --- Stage 1 RULE1 ---
    s1_err = None
    s1_raw: List[Any] = []
    try:
        fto = page.find_tables(strategy="lines_strict")
        s1_raw = list(getattr(fto, "tables", []) or [])
    except Exception as exc:
        s1_err = f"{type(exc).__name__}: {exc}"

    stage1: List[_TableCandidate] = []
    for t in s1_raw:
        grid = _pymu_table_cells_grid(t)
        if not grid:
            continue
        bb = _pymu_table_bbox(t, prec)
        cand = _TableCandidate(
            strategy="lines_strict", bbox=bb,
            rows=len(grid), cols=len(grid[0]) or 0,
            cells_grid=grid, classification=_TABLE_CLASS_UNCERTAIN,
            pymu_obj=t,
        )
        cls, details = classify_table_candidate(cand, geom, cfg)
        cand.classification = cls
        cand.signal_details = details
        cand.classification_confidence = float(details.get("conf", 0.0))
        stage1.append(cand)

    s1_true = sum(1 for c in stage1 if c.classification == _TABLE_CLASS_TRUE)
    info: Dict[str, Any] = {
        "stage1_strategy": "lines_strict",
        "stage1_detected": len(stage1),
        "stage1_true": s1_true,
        "stage1_error": s1_err,
    }

    # --- Stage 2 (RULE3) conditional lines fallback ---
    stage2: List[_TableCandidate] = []
    info["stage2_run"] = False
    if cfg.enable_lines_fallback and s1_true == 0:
        hs_ok, hs_details = _high_signal_table_evidence(geom, cfg)
        info["high_signal"] = hs_details
        info["high_signal_passed"] = bool(hs_ok)
        if hs_ok:
            info["stage2_run"] = True
            info["stage2_strategy"] = "lines"
            s2_err = None
            try:
                fto2 = page.find_tables(strategy="lines")
                s2_raw = list(getattr(fto2, "tables", []) or [])
            except Exception as exc:
                s2_err = f"{type(exc).__name__}: {exc}"
                s2_raw = []
            info["stage2_error"] = s2_err
            for t in s2_raw:
                grid = _pymu_table_cells_grid(t)
                if not grid:
                    continue
                bb = _pymu_table_bbox(t, prec)
                cand = _TableCandidate(
                    strategy="lines", bbox=bb,
                    rows=len(grid), cols=len(grid[0]) or 0,
                    cells_grid=grid, classification=_TABLE_CLASS_UNCERTAIN,
                    pymu_obj=t,
                )
                cls, details = classify_table_candidate(cand, geom, cfg)
                cand.classification = cls
                cand.signal_details = details
                cand.classification_confidence = float(details.get("conf", 0.0))
                stage2.append(cand)
            info["stage2_detected"] = len(stage2)
            info["stage2_true"] = sum(1 for c in stage2 if c.classification == _TABLE_CLASS_TRUE)

    # --- RULE2 enforcement ---
    info["text_strategy_disabled_RULE2"] = True
    info["text_strategy_enable_cfg_value"] = bool(cfg.text_strategy_enable)
    if cfg.text_strategy_enable:
        # Safety guard: even if bool accidentally flipped → DO NOT call find_tables("text").
        info["text_strategy_force_blocked"] = True

    all_cand: List[_TableCandidate] = list(stage1)
    if info.get("stage2_run"):
        s1_true_bboxes = [c.bbox for c in stage1 if c.classification == _TABLE_CLASS_TRUE]
        for c in stage2:
            if any(_bbox_iou(c.bbox, b1) >= 0.7 for b1 in s1_true_bboxes):
                continue
            all_cand.append(c)
    all_cand.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return all_cand, info


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0); ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0); inter = iw * ih
    if inter <= 0:
        return 0.0
    aA = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    bA = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return inter / (aA + bA - inter)


# =============================================================================
# §21 / RULE6 TableData Direct Mapping (NO markdown roundtrip; NO text-merge guess)
# =============================================================================

def table_to_tabledata(cand: _TableCandidate, cfg: PdfParserV2Config) -> Tuple[TableData, List[str]]:
    warnings: List[str] = []
    grid = cand.cells_grid
    if not grid or not grid[0]:
        raise ContractViolation("TRUE_SEMANTIC_TABLE candidate has empty grid")
    cols = len(grid[0])
    # RULE6: no geometry evidence of merge available at this API level → all (1,1) + warning.
    warnings.append(W_MERGED_CELL)
    headers_raw = list(grid[0])
    while len(headers_raw) < cols:
        headers_raw.append("")
    headers = [TableCell(text=str(h), colspan=1, rowspan=1, is_header=True) for h in headers_raw]
    out_rows: List[List[TableCell]] = []
    for ri, row in enumerate(grid):
        if ri == 0:
            continue
        rr = list(row)
        while len(rr) < cols:
            rr.append("")
        rr = rr[:cols]
        out_rows.append([TableCell(text=str(c), colspan=1, rowspan=1, is_header=False) for c in rr])
    caption: Optional[str] = None
    return TableData(headers=headers, rows=out_rows, caption=caption), warnings


# =============================================================================
# §24 / RULE5 Table bbox ownership
# =============================================================================

def _point_in_rect(cx: float, cy: float, bbox, pad: float = 1.5) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 - pad) <= cx <= (x1 + pad) and (y0 - pad) <= cy <= (y1 + pad)


def enforce_table_ownership(
    ordered_lines: List[_Line],
    confirmed_tables: List[_TableCandidate],
) -> List[_Line]:
    """Center-in-bbox ownership. Any line whose center is inside a confirmed table
    bbox is dropped from paragraph flow (one semantic owner RULE5).
    """
    if not confirmed_tables:
        return ordered_lines
    kept: List[_Line] = []
    for l in ordered_lines:
        if any(_point_in_rect(l.cx, l.cy, t.bbox, pad=1.5) for t in confirmed_tables):
            continue
        kept.append(l)
    return kept


# =============================================================================
# §27 Heading / §28 Paragraphs / §29 LIST_ITEM
# =============================================================================

_LIST_ITEM_PREFIXES = (
    re.compile(r"^\s*[一二三四五六七八九十百千万零〇两]+[、.．]\s"),
    re.compile(r"^\s*[（(][一二三四五六七八九十百千万零〇两]+[)）]\s*"),
    re.compile(r"^\s*[0-9]+[.、．]\s"),
    re.compile(r"^\s*[（(][0-9]+[)）]\s*"),
    re.compile(r"^\s*[\u2022\u25E6\u25AA\u2023\u2043\uF0B7\u00B7]\s"),
)


def _is_list_item(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # §29 法律条文 NOT LIST_ITEM
    if _LEGAL_ART_RE.match(t):
        return False
    return any(p.match(t) for p in _LIST_ITEM_PREFIXES)


def _is_heading(line: _Line, ordered_lines: List[_Line], cfg: PdfParserV2Config):
    text = line.text.strip()
    if not text:
        return False, None
    if _is_list_item(text):
        return False, None
    score = 0.0
    legal_ch = bool(_LEGAL_CH_RE.match(text))
    if legal_ch:
        score += 2.0
    sizes = [ln.font_size for ln in ordered_lines if ln.font_size > 0]
    if sizes:
        s2 = sorted(sizes)
        med = s2[len(s2) // 2]
        if line.font_size >= med + cfg.heading_font_size_delta:
            score += 1.6
        elif line.font_size >= med + 1.0:
            score += 0.8
    if line.is_bold and len(text) <= 40:
        score += 0.6
    level = None
    if score >= 2.0 or legal_ch:
        level = 1
        s3 = sorted(sizes, reverse=True) if sizes else []
        if s3 and line.font_size < s3[0] - 0.5:
            level = 2
        if s3 and line.font_size < s3[0] - 3.0:
            level = 3
        if level is None:
            level = 2
        return True, max(1, min(9, int(level)))
    return False, None


@dataclass
class _BlockBuild:
    type: BlockType
    text: str
    level: Optional[int] = None
    table_data: Optional[TableData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# §26 Semantic Block Builder (6 Frozen BlockTypes ONLY)
# =============================================================================

def _para_groups_and_lines(ordered_lines: List[_Line], cfg: PdfParserV2Config, page_h: float):
    """Group ordered_lines into paragraph groups, returning list of (y0, [lines]).

    Same logic as paragraph flush used elsewhere; shared for interleaving key consistency.
    """
    gap_th = page_h * cfg.paragraph_gap_ratio
    groups: List[Tuple[float, List[_Line]]] = []
    buf: List[_Line] = []
    def flush():
        nonlocal buf
        if not buf:
            return
        groups.append((buf[0].cy, list(buf)))
        buf = []
    for line in ordered_lines:
        if not line.text.strip():
            continue
        if not buf:
            buf = [line]; continue
        last = buf[-1]
        gap = abs(line.y0 - last.y1) if line.y0 >= last.y1 else abs(line.y1 - last.y0)
        dcol = line.col != last.col and line.col != 0 and last.col != 0
        h1 = _is_heading(line, ordered_lines, cfg)[0]
        h2 = _is_heading(last, ordered_lines, cfg)[0]
        lb = _is_list_item(last.text) or _is_list_item(line.text)
        fd = abs(line.font_size - last.font_size) >= max(cfg.heading_font_size_delta, 2.0)
        if dcol or gap >= gap_th or h1 or h2 or lb or fd:
            flush()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return groups


def _page_is_scanned_or_image_only(page_geom: _PageGeometry) -> bool:
    r"""Critical Hardening Root B: fail-closed generic scanned/image-only detector.

    Pure evidence: text sparse (span chars OR words low) AND (heavy image OR
    heavy vector-rendered drawings). No filename / page special case; no OCR.

    Key coverage:
      * raster-image pages (image_blocks / image_area_ratio)
      * pure vector-rendered scanned pages: bazillions of tiny line draws
        per pixel line (SHADOW-017 p4 → line_draws=52976) OR 200+ rect tiles
    """
    span_chars_total = 0
    img_blocks = 0
    for s in page_geom.spans:
        try:
            span_chars_total += len(str(s.get("text", "")) or "")
        except Exception:
            pass
    for b in page_geom.blocks:
        try:
            if len(b) >= 7 and int(b[6]) == 1:
                img_blocks += 1
        except Exception:
            continue
    plain_empty = not page_geom.plain_text.strip()
    text_sparse = span_chars_total < 40 or (plain_empty and len(page_geom.words) < 8)
    if not text_sparse:
        return False
    page_area = max(1.0, float(page_geom.width) * float(page_geom.height))
    image_area_sum = 0.0
    for b in page_geom.blocks:
        try:
            if len(b) >= 7 and int(b[6]) == 1:
                x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                image_area_sum += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        except Exception:
            continue
    image_area_ratio = image_area_sum / page_area
    rect_draws = int(page_geom.rect_drawings_count) or 0
    line_draws = int(page_geom.line_drawings_count) or 0
    image_objects = img_blocks

    # 1. Image-area coverage: typical embedded image → scanned page.
    if image_area_ratio >= 0.35 and text_sparse:
        return True
    # 2. Multiple small image blocks + no text.
    if image_objects >= 2 and text_sparse and span_chars_total < 15:
        return True
    # 3. Rect-rendered / bitmap-like page (many small rect tiles).
    if plain_empty and len(page_geom.words) == 0 and rect_draws >= 200:
        return True
    # 4. Vector-rendered scan (e.g. SHADOW-017 p4: rects=67, line_draws=52976).
    #    Bazillions of tiny 'l' segments per pixel raster line → huge line_draws.
    if (
        plain_empty
        and len(page_geom.words) == 0
        and line_draws >= 2000
    ):
        return True
    # 5. Medium rect + line drawing evidence + at least 1 image object anchor.
    if (
        plain_empty
        and len(page_geom.words) == 0
        and rect_draws + line_draws >= 40
        and image_objects >= 1
    ):
        return True
    return False


def build_semantic_blocks(
    geom: _PageGeometry,
    ordered_lines: List[_Line],
    confirmed_tables: List[_TableCandidate],
    source_file: str,
    cfg: PdfParserV2Config,
) -> Tuple[List[Block], List[str]]:
    warnings: List[str] = []
    page_number_1 = geom.page_number_1

    # Critical Hardening Root B: fail-closed SCANNED detection BEFORE
    # W_EMPTY_PAGE early return — so pure vector-rendered scanned pages
    # aren't silently dropped.  SCANNED also takes precedence over EMPTY
    # (we want an observable warning, not a silent EMPTY_PAGE).
    scanned_detected = _page_is_scanned_or_image_only(geom)
    if scanned_detected:
        warnings.append(f"{W_SCANNED}@p{page_number_1}")
        out_blocks_scan: List[Block] = [Block(
            block_id="__placeholder__",
            type=BlockType.UNKNOWN,
            text=(
                f"[Page {page_number_1} appears scanned or image-only. "
                f"Production V1 OCR disabled; no text fabricated.]"
            ),
            order=0,
            page=page_number_1,
            metadata={
                "image_only_page": True,
                "rect_drawings": int(geom.rect_drawings_count or 0),
                "line_drawings": int(geom.line_drawings_count or 0),
            },
        )]
        return out_blocks_scan, warnings

    if geom.empty:
        warnings.append(W_EMPTY_PAGE)
        return [], warnings

    # A) confirmed TRUE tables
    table_builds: List[Tuple[float, float, _BlockBuild]] = []  # (y0, x0, build)
    for t in confirmed_tables:
        if t.classification != _TABLE_CLASS_TRUE:
            continue
        td, td_w = table_to_tabledata(t, cfg)
        warnings.extend(td_w)
        md = {
            "table_strategy": t.strategy,
            "table_bbox": list(t.bbox),
            "table_shape": (t.rows, t.cols),
            "table_gate_details": t.signal_details,
        }
        table_builds.append((t.bbox[1], t.bbox[0], _BlockBuild(
            type=BlockType.TABLE, text=td.to_markdown(), table_data=td, metadata=md,
        )))

    # B) non-table line groups → heading / para / list
    groups = _para_groups_and_lines(ordered_lines, cfg, max(geom.height, 1.0))
    para_builds: List[Tuple[float, float, _BlockBuild]] = []
    for (y_g, lines_g) in groups:
        full = "\n".join(l.text for l in lines_g).strip()
        if not full:
            continue
        if len(lines_g) == 1:
            h_ok, h_lv = _is_heading(lines_g[0], ordered_lines, cfg)
            if h_ok and h_lv is not None:
                para_builds.append((y_g, lines_g[0].x0, _BlockBuild(type=BlockType.HEADING, text=full, level=h_lv)))
                continue
            if _is_list_item(lines_g[0].text):
                para_builds.append((y_g, lines_g[0].x0, _BlockBuild(type=BlockType.LIST_ITEM, text=full)))
                continue
        para_builds.append((y_g, lines_g[0].x0, _BlockBuild(type=BlockType.PARAGRAPH, text=full)))

    # C) Interleave tables + paragraphs by geometry (y0 then x0 then type ties (para first = 0, table = 1) )
    keyed: List[Tuple[float, float, int, _BlockBuild]] = []
    for (y, x, bld) in para_builds:
        keyed.append((y, x, 0, bld))
    for (y, x, bld) in table_builds:
        keyed.append((y, x, 1, bld))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    interleaved = [t[3] for t in keyed]

    out_blocks: List[Block] = []
    for bld in interleaved:
        out_blocks.append(Block(
            block_id="__placeholder__",
            type=bld.type,
            text=bld.text,
            order=0,
            level=bld.level,
            table_data=bld.table_data,
            page=page_number_1,
            metadata=dict(bld.metadata),
        ))

    # D) Scanned / image-only page placeholder
    if len(geom.words) < 8 and geom.rect_drawings_count + geom.line_drawings_count > 30 and not geom.plain_text.strip():
        warnings.append(W_SCANNED)
        out_blocks.append(Block(
            block_id="__placeholder__",
            type=BlockType.UNKNOWN,
            text=f"[Placeholder: page {page_number_1} appears scanned/image-only (V1 no OCR)]",
            order=0,
            page=page_number_1,
            metadata={"image_only_page": True, "rects": geom.rect_drawings_count},
        ))

    if not out_blocks:
        out_blocks.append(Block(
            block_id="__placeholder__",
            type=BlockType.UNKNOWN,
            text="",
            order=0,
            page=page_number_1,
            metadata={"empty_page": True},
        ))
        warnings.append(W_EMPTY_PAGE)

    return out_blocks, warnings


# =============================================================================
# Document Entry — parse_pdf_v2() → ParsedDocument V2 Frozen Contract
# =============================================================================

def parse_pdf_v2(
    path,
    *,
    cfg: Optional[PdfParserV2Config] = None,
    document_id: Optional[str] = None,
) -> ParsedDocument:
    """Production V2 entry. Returns ParsedDocument V2 (Frozen Contract).

    §43 Error Handling: malformed/corrupt PDF will raise ValueError (typed) — never
    silently returns empty doc.
    """
    cfg = cfg or PdfParserV2Config()
    pth = Path(path)
    if not pth.is_file():
        raise FileNotFoundError(f"PDF not found: {pth}")
    source_file = str(pth)
    if not document_id:
        stem_safe = re.sub(r"[^A-Za-z0-9_\-]", "_", pth.stem)[:40]
        try:
            suffix = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:8]
        except Exception:
            suffix = uuid.uuid4().hex[:8]
        document_id = f"pdfv2_{stem_safe}_{suffix}"

    warnings: List[str] = []
    native_meta: Dict[str, Any] = {}
    page_count = 0
    all_page_blocks: List[Block] = []

    doc_handle: Optional["pymupdf.Document"] = None
    try:
        doc_handle = pymupdf.open(source_file)
    except Exception as exc:
        raise ValueError(f"PDF_OPEN_FAILED: {type(exc).__name__}: {exc}") from exc
    try:
        page_count = int(doc_handle.page_count)
        native_meta = dict(doc_handle.metadata or {})
        for pidx in range(page_count):
            try:
                page = doc_handle[pidx]
            except Exception as exc:
                warnings.append(f"{W_ANOMALY}@p{pidx+1}: PAGE_OPEN {type(exc).__name__}")
                continue
            try:
                geom = extract_page_geometry(page, pidx, cfg)
            except Exception as exc:
                warnings.append(f"{W_ANOMALY}@p{pidx+1}: GEOM {type(exc).__name__}")
                del page
                continue
            geom.lines = normalize_layout_elements(geom, cfg)
            ordered_lines = reconstruct_reading_order(geom, cfg)
            try:
                candidates, _tinfo = detect_table_candidates(page, geom, cfg)
            except Exception as exc:
                warnings.append(f"{W_ANOMALY}@p{pidx+1}: TABLE {type(exc).__name__}")
                candidates = []
            confirmed_true: List[_TableCandidate] = [
                c for c in candidates if c.classification == _TABLE_CLASS_TRUE
            ]
            uncert_n = sum(1 for c in candidates if c.classification == _TABLE_CLASS_UNCERTAIN)
            if uncert_n > 0:
                warnings.append(f"{W_TABLE_AMBIG}@p{pidx+1}: uncertain={uncert_n}")
            owned_lines = enforce_table_ownership(ordered_lines, confirmed_true)
            blocks_p, warns_p = build_semantic_blocks(geom, owned_lines, confirmed_true, source_file, cfg)
            all_page_blocks.extend(blocks_p)
            warnings.extend(warns_p)
            del page
    finally:
        if doc_handle is not None:
            try:
                doc_handle.close()
            except Exception:
                pass

    if len(warnings) > cfg.max_warnings_per_doc:
        truncated_note = f"... warnings truncated (original count: {len(warnings)})"
        warnings = warnings[: cfg.max_warnings_per_doc] + [truncated_note]

    # ---- METADATA lightweight (§32: NO heavy regex)
    title = native_meta.get("title") or pth.stem
    meta: Dict[str, Any] = {
        "source_file": source_file,
        "page_count": int(page_count),
        "pdf_native_metadata": dict(native_meta),
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
    }

    # ---- Global renumber (Frozen Contract: index == order == provenance.block_order)
    ParsedDocument.prepare_blocks(document_id, all_page_blocks)
    for b in all_page_blocks:
        b.attach_provenance(source_file)

    # ---- METADATA block (document header) prepend → renumber + attach provenance again
    meta_lines = [
        f"source_file: {pth.name}",
        f"title: {title}",
        f"page_count: {page_count}",
        f"parser: {PARSER_NAME}/{PARSER_VERSION}",
    ]
    if native_meta.get("author"):
        meta_lines.append(f"pdf_author: {native_meta['author']}")
    meta_block = Block(
        block_id="__placeholder__",
        type=BlockType.METADATA,
        text="\n".join(meta_lines),
        order=0, page=1,
        metadata={"role": "document_metadata_header"},
    )
    all_page_blocks.insert(0, meta_block)
    ParsedDocument.prepare_blocks(document_id, all_page_blocks)
    for b in all_page_blocks:
        b.attach_provenance(source_file)

    doc = ParsedDocument(
        document_id=document_id,
        source_file=source_file,
        title=str(title),
        metadata=meta,
        blocks=all_page_blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        warnings=list(warnings),
    )
    doc.validate()
    return doc


# =============================================================================
# §53 Convenience: V2 → V1-compatible (LegacyAdapter) for quick chunker smoke
# =============================================================================

def parse_pdf_v2_to_legacy(path, *, cfg: Optional[PdfParserV2Config] = None):
    """Parse V2 → LegacyParsedDocument (segments per-page). For quick dispatcher /
    chunker integration smoke. Per §53 this callable is a parallel explicit v2
    entrypoint; the canonical parse_document() dispatcher stays on V1 pypdf route
    until explicit feature-flag swap (requires human approval, not done here).
    """
    doc = parse_pdf_v2(path, cfg=cfg)
    return LegacyAdapter.to_legacy_parsed_document(doc)


__all__ = [
    "PdfParserV2Config",
    "W_SCANNED", "W_TABLE_AMBIG", "W_MERGED_CELL", "W_READING_ORDER", "W_EMPTY_PAGE", "W_ANOMALY",
    "extract_page_geometry",
    "normalize_layout_elements",
    "reconstruct_reading_order",
    "detect_table_candidates",
    "classify_table_candidate",
    "table_to_tabledata",
    "enforce_table_ownership",
    "build_semantic_blocks",
    "parse_pdf_v2",
    "parse_pdf_v2_to_legacy",
    "PARSER_NAME", "PARSER_VERSION",
]
