"""
PDF Parser V2 — POC (Proof of Concept)
=======================================
Frozen Contract target: ParsedDocument V2
Dependency: pypdf 6.x (PyMuPDF/fitz NOT available, BLOCKED_BY_DEPENDENCY noted)

Pipeline:
  Raw PDF (pypdf)
  → visitor_text fragments (x, y, font_size, font_name) per page
  → y-invert to top=0 convention + normalize font size units
  → line grouping (y-tolerance bucket)
  → cross-page header/footer detection (repetition + position)
  → table region detection (grid alignment heuristic)
  → table bbox ownership mask (drop duplicate text inside tables)
  → column detection (x-gap heuristic)
  → reading-order reconstruction (y-desc → column → x-asc)
  → paragraph grouping (nearby lines, same column, compatible font)
  → heading / paragraph / list / table / metadata classification
  → Provenance attach + renumber_blocks
  → ParsedDocument V2

Scope: POC only. No production integration. No new dependencies installed.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# =====================================================================
# Ensure project import path — ONLY for contract imports (safe)
# =====================================================================
_PROJECT_ROOT = Path(r"D:\xiaoyi\Legal_System")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.ingestion.parsed_document_v2 import (  # noqa: E402  (FROZEN)
    Block,
    BlockType,
    ContractViolation,
    LegacyAdapter,
    ParsedDocument,
    Provenance,
    TableCell,
    TableData,
)

from pypdf import PdfReader  # noqa: E402  (already in project deps)


PARSER_NAME = "pdf_parser_v2_poc"
PARSER_VERSION = "0.1.0-pypdf"
BLOCKED_BY_DEPENDENCY_NOTE = (
    "POC built on pypdf 6.x visitor_text. PyMuPDF (fitz) is NOT installed in the environment. "
    "Per task rules, no dependency was added. Recommend adding PyMuPDF>=1.24 for production "
    "PDF layout extraction (richer blocks API, native table detection, font encoding robustness)."
)


# =====================================================================
# Geometry primitives
# =====================================================================

_Y_TOLERANCE_PT = 3.0   # lines with y within this band share a line
_X_TOLERANCE_PT = 2.0   # x tolerance for column alignment bucketing
_LINE_GAP_FACTOR = 1.4  # if gap > this × median_line_h → new paragraph


@dataclass
class Fragment:
    """Atomic glyph/fragment from pypdf visitor_text, with TOP=0 y convention."""
    text: str
    x: float         # left x (PDF units, unchanged)
    y_top: float     # y with TOP=0 convention
    y_raw: float     # original PDF y (BOTTOM=0, kept for debug)
    fs_raw: float    # raw font size from pypdf (may be CTM-scaled)
    font: str
    page: int        # 1-based

    @property
    def y_bottom(self) -> float:
        """Approximate visual bottom (y_top + fs-scaled height)."""
        return self.y_top + max(3.0, self.fs_raw * 0.012)


@dataclass
class Line:
    """Group of fragments on the same visual horizontal band."""
    fragments: list[Fragment]
    page: int
    y_top: float
    y_bottom: float
    x_min: float
    x_max: float
    text: str = ""
    fs_median: float = 0.0
    font_mode: str = ""

    def recompute(self) -> None:
        if not self.fragments:
            return
        # Sort fragments left→right within line
        self.fragments.sort(key=lambda f: (f.x, f.y_top))
        # Join with single space only if previous doesn't end with space and next doesn't start with one
        parts: list[str] = []
        prev_x_end: Optional[float] = None
        prev_fs: Optional[float] = None
        for f in self.fragments:
            w_guess = max(2.0, f.fs_raw * 0.006)
            if prev_x_end is not None and f.x > prev_x_end + w_guess * 1.5:
                parts.append(" ")
            parts.append(f.text)
            prev_x_end = f.x + len(f.text) * w_guess
            prev_fs = f.fs_raw
        self.text = "".join(parts).strip()
        self.x_min = min(f.x for f in self.fragments)
        self.x_max = max(f.x + max(2.0, f.fs_raw * 0.006) * len(f.text) for f in self.fragments)
        fs_vals = sorted(f.fs_raw for f in self.fragments)
        self.fs_median = fs_vals[len(fs_vals) // 2] if fs_vals else 0.0
        from collections import Counter
        fonts = Counter(f.font for f in self.fragments)
        self.font_mode = fonts.most_common(1)[0][0] if fonts else ""


@dataclass
class TableRegion:
    page: int
    x_min: float
    x_max: float
    y_top: float
    y_bottom: float
    header_indices: list[int] = field(default_factory=list)
    n_cols: int = 0
    n_rows: int = 0
    caption: str = ""
    # Debug
    reason: str = ""


@dataclass
class ColumnSpec:
    x_start: float
    x_end: float


# =====================================================================
# Step 1: Raw extraction → fragments
# =====================================================================

def _extract_page_fragments(page: Any, page_num: int, page_w: float, page_h: float) -> list[Fragment]:
    """Use pypdf visitor_text to collect raw fragments.

    CRITICAL: we do NOT pre-invert y here, because pypdf's tm[5] direction
    can be INCONSISTENT across pages within the same PDF (due to mixed
    content-stream transforms / XObjects).  We calibrate y-axis direction
    per-page downstream using plain extract_text() token order as ground-truth
    signal.
    """
    frags: list[Fragment] = []

    def visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        if not text or not text.strip():
            return
        raw_x = float(tm[4])
        raw_y = float(tm[5])
        # Filter spurious fragments: significant text (≥2 chars) at origin (0,0)
        # e.g. GF-2024-0113 Page 1  "制定" lands at (0,0) incorrectly.
        if len(text.strip()) >= 2 and abs(raw_x) < 1e-6 and abs(raw_y) < 1e-6:
            return
        fs_raw = float(font_size or 0.0)
        font = ""
        if isinstance(font_dict, dict):
            bf = font_dict.get("/BaseFont", "")
            font = bf if isinstance(bf, str) else str(bf)
        # Keep raw coordinates; visual ordering will be calibrated later
        frags.append(Fragment(
            text=text,
            x=raw_x,
            y_top=raw_y,            # temporarily raw_y; will be normalized later
            y_raw=raw_y,
            fs_raw=fs_raw,
            font=font,
            page=page_num,
        ))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        pass

    # Fallback: if visitor gave 0 fragments (e.g., certain font encodings or
    # scanned-first-pages like 民法典 Page 1), fall back to plain extract_text
    # and split into lines with approximate geometry.
    if not frags:
        plain = page.extract_text() or ""
        if plain.strip():
            lines_plain = plain.splitlines()
            n_lines = max(1, len(lines_plain))
            step = page_h / (n_lines + 1)
            approx_fs = (page_h / n_lines) * 0.6 if n_lines > 0 else 10.0
            for i, ln in enumerate(lines_plain):
                if not ln.strip():
                    continue
                # Fallback fragments: raw_y is intentionally such that ASC order
                # matches plain top→bottom (i=0 → raw_y LARGE = top, consistent
                # with "descending raw = top→bottom" convention we'll detect).
                raw_y = page_h - step * (i + 0.5)
                frags.append(Fragment(
                    text=ln,
                    x=20.0,
                    y_top=raw_y,
                    y_raw=raw_y,
                    fs_raw=approx_fs * 100.0,  # fake scaling for downstream
                    font="/FallbackPlain",
                    page=page_num,
                ))

    return frags


# =====================================================================
# Step 1b: Per-page y-axis direction calibration
# =====================================================================

# pypdf tm[5] direction varies by page (some pages raw_y DESC = top→bottom,
# others raw_y ASC = top→bottom).  Use plain extract_text token order as
# ground-truth signal to pick direction per page.

def _calibrate_y_direction(
    page: Any,
    frags: list[Fragment],
    page_h: float,
) -> tuple[str, dict[str, Any]]:
    """Return ('asc' | 'desc', debug_info).

    'asc'  → smaller y_raw = visually TOP  (sort by y_raw ascending for top→bottom)
    'desc' → larger  y_raw = visually TOP  (sort by y_raw descending for top→bottom)
    """
    info: dict[str, Any] = {}
    if not frags:
        info["chosen"] = "desc"
        info["plain_tokens_count"] = 0
        info["reason"] = "no_fragments_fallback"
        return "desc", info

    # --- Plain text "golden" token order (first significant tokens) ---
    try:
        plain_text = page.extract_text() or ""
    except Exception:
        plain_text = ""

    def _sig_tokens(s: str, limit: int = 60) -> list[str]:
        # Collect substrings that are "meaningful": any run of ≥1 CJK chars
        # OR ≥2 alnum chars.  This tolerates whitespace/punctuation variations
        # between plain extraction and geometry reconstruction.
        out: list[str] = []
        for m in re.finditer(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_\-]{2,}", s):
            out.append(m.group(0))
            if len(out) >= limit:
                break
        return out

    plain_tokens = _sig_tokens(plain_text, limit=80)
    info["plain_tokens_count"] = len(plain_tokens)
    if len(plain_tokens) < 6:
        # Not enough signal; fall back to majority convention: DESC for legal PDF
        info["chosen"] = "desc"
        info["reason"] = "too_few_plain_tokens_fallback_desc"
        return "desc", info

    # --- Generate candidate band order for both directions ---
    def _candidate_tokens(direction: str, limit: int = 80) -> list[str]:
        # Band fragments by y_tolerance
        key_func = (lambda f: f.y_raw) if direction == "asc" else (lambda f: -f.y_raw)
        sorted_f = sorted(frags, key=lambda f: (round(key_func(f) if direction == "asc" else -key_func(f), 0), f.x))
        # Actually simpler: sort by y_raw in the direction
        if direction == "asc":
            sf = sorted(frags, key=lambda f: (round(f.y_raw, 0), f.x))
        else:
            sf = sorted(frags, key=lambda f: (-round(f.y_raw, 0), f.x))
        # Band by y
        bands_text: list[str] = []
        band_y = -1.0
        buf: list[str] = []
        for f in sf:
            if band_y < 0 or abs(f.y_raw - band_y) <= _Y_TOLERANCE_PT:
                buf.append(f.text)
                if band_y < 0:
                    band_y = f.y_raw
                else:
                    band_y = (band_y + f.y_raw) / 2
            else:
                bands_text.append("".join(buf))
                buf = [f.text]
                band_y = f.y_raw
        if buf:
            bands_text.append("".join(buf))
        # Collect sig tokens from first ~15 bands (enough to cover header, title, TOC start)
        out: list[str] = []
        for bt in bands_text[:18]:
            for m in re.finditer(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_\-]{2,}", bt):
                out.append(m.group(0))
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out

    asc_tokens = _candidate_tokens("asc")
    desc_tokens = _candidate_tokens("desc")
    info["asc_tokens_sample"] = asc_tokens[:12]
    info["desc_tokens_sample"] = desc_tokens[:12]

    # Score: for each of first N plain tokens (N=min(len,30)), check if it
    # appears within first K candidate tokens (K=min(len,40)).  This is a
    # "early token matching accuracy" score.
    def _score(plain: list[str], cand: list[str]) -> float:
        if not cand:
            return 0.0
        n = min(len(plain), 36)
        cand_set_limited: set[str] = set(cand[: min(len(cand), 50)])
        hits = 0
        # Penalize if plain tokens appear in reversed rank in cand vs plain
        last_rank = -1
        rank_penalty = 0
        for i in range(n):
            tok = plain[i]
            if tok in cand_set_limited:
                hits += 1
                # Rank within candidate list
                try:
                    r = cand.index(tok)
                except ValueError:
                    r = last_rank
                if r < last_rank:
                    rank_penalty += 1
                last_rank = max(last_rank, r)
        score = hits / max(1, n) - rank_penalty * 0.03
        return max(0.0, score)

    asc_score = _score(plain_tokens, asc_tokens)
    desc_score = _score(plain_tokens, desc_tokens)
    info["asc_score"] = round(asc_score, 3)
    info["desc_score"] = round(desc_score, 3)

    # Decision: require ≥15% relative difference, else default desc
    # (desc is legal PDF majority and page 1 convention for FIX-001)
    diff = asc_score - desc_score
    if diff > 0.08 and asc_score > 0.1:
        chosen = "asc"
    elif diff < -0.08 and desc_score > 0.1:
        chosen = "desc"
    else:
        # Tie-break: whichever has the plain token "第" or "条" or "一" earlier
        # in a realistic position, but default to desc (PDF-standard-ish)
        chosen = "desc"
        # But if asc score is strictly higher (even by tiny), use asc
        if asc_score > desc_score:
            chosen = "asc"
    info["chosen"] = chosen
    return chosen, info


def _apply_y_calibration(frags: list[Fragment], direction: str, page_h: float) -> None:
    """In-place update each fragment.y_top so that SMALL y_top = visually TOP.

    After this, sorting by y_top ASC is guaranteed to be the correct visual
    top→bottom order (independent of raw tm[5] direction on the page).
    """
    if direction == "asc":
        # Smaller y_raw = visually top.  Already matches desired convention.
        for f in frags:
            f.y_top = f.y_raw
    else:
        # Larger y_raw = visually top.  Invert to small=top convention.
        # Use page_h as a safe scale; but clamp negatives to 0.
        for f in frags:
            f.y_top = max(0.0, page_h - f.y_raw)


# =====================================================================
# Step 2: Fragments → Lines (y bucket)
# =====================================================================

def _fragments_to_lines(frags: list[Fragment], page: int) -> list[Line]:
    """Group fragments by approximate y band, sort within line by x."""
    if not frags:
        return []
    # Sort by y_top asc
    sorted_f = sorted(frags, key=lambda f: (round(f.y_top, 0), f.x))
    bands: list[list[Fragment]] = []
    band_y: list[float] = []
    for f in sorted_f:
        placed = False
        for i, by in enumerate(band_y):
            if abs(f.y_top - by) <= _Y_TOLERANCE_PT:
                bands[i].append(f)
                # update band y to running mean
                band_y[i] = (by * len(bands[i]) + f.y_top) / (len(bands[i]) + 1)
                placed = True
                break
        if not placed:
            bands.append([f])
            band_y.append(f.y_top)
    lines: list[Line] = []
    for i, (band_f, by) in enumerate(zip(bands, band_y)):
        xs = [f.x for f in band_f]
        y_ts = [f.y_top for f in band_f]
        y_bs = [f.y_bottom for f in band_f]
        line = Line(
            fragments=list(band_f),
            page=page,
            y_top=min(y_ts),
            y_bottom=max(y_bs),
            x_min=min(xs),
            x_max=max(xs) + 1.0,
        )
        line.recompute()
        if line.text:
            lines.append(line)
    # Sort top→bottom
    lines.sort(key=lambda ln: (ln.y_top, ln.x_min))
    return lines


# =====================================================================
# Step 3: Header / Footer detection (cross-page repetition + geometry)
# =====================================================================

def _detect_header_footer(
    all_page_lines: list[list[Line]],
    page_h_list: list[float],
) -> tuple[dict[int, set[int]], dict[int, set[int]], list[str]]:
    """Return (header_lines_by_page_idx, footer_lines_by_page_idx, warnings).

    A line is candidate header/footer if:
      - its normalized text repeats on ≥2 pages at similar vertical band
      - OR it's pure page-number pattern in top 5% or bottom 5% band
    """
    warnings: list[str] = []
    header_map: dict[int, set[int]] = {i: set() for i in range(len(all_page_lines))}
    footer_map: dict[int, set[int]] = {i: set() for i in range(len(all_page_lines))}

    if len(all_page_lines) < 2:
        return header_map, footer_map, warnings

    # Build signature → list[(page_idx, line_idx)]
    sig_positions: dict[str, list[tuple[int, int, float, float]]] = {}
    for pi, lines in enumerate(all_page_lines):
        page_h = page_h_list[pi] if pi < len(page_h_list) else 842.0
        for li, ln in enumerate(lines):
            norm = re.sub(r"\s+", " ", ln.text).strip()
            # Normalize digits to N for page-number pattern detection
            norm_d = re.sub(r"\d+", "N", norm)
            norm_key = norm_d if re.fullmatch(r"[N·\-\s·.]{1,8}", norm_d) else norm[:80]
            y_ratio = ln.y_top / page_h if page_h > 0 else 0.0
            sig_positions.setdefault(norm_key, []).append((pi, li, y_ratio, ln.y_top))

    pure_page_num = re.compile(r"^[\s\-·.]*\d{1,3}[\s\-·.]*$")
    for sig, occs in sig_positions.items():
        if len(occs) < 2:
            continue
        # Check if all occurrences are in top 8% OR all in bottom 8% (relative to page height)
        top_ratios = [o[2] for o in occs]
        all_top = all(r <= 0.08 for r in top_ratios)
        all_bottom = all(r >= 0.90 for r in top_ratios)
        all_mix = all_top or all_bottom
        if not all_mix and len(occs) >= max(3, len(all_page_lines) // 2):
            # Repetition across many pages regardless of band
            all_mix = True
        if not all_mix:
            continue
        for pi, li, _r, _y in occs:
            if _r <= 0.08:
                header_map[pi].add(li)
            elif _r >= 0.90:
                footer_map[pi].add(li)
            elif len(occs) >= max(3, len(all_page_lines) // 2):
                # Strong cross-page repetition → treat as band based on its own ratio
                if _r <= 0.15:
                    header_map[pi].add(li)
                elif _r >= 0.85:
                    footer_map[pi].add(li)

    # Additionally: pure page numbers in top or bottom band even if not repeating
    for pi, lines in enumerate(all_page_lines):
        page_h = page_h_list[pi] if pi < len(page_h_list) else 842.0
        for li, ln in enumerate(lines):
            if not pure_page_num.match(ln.text):
                continue
            ratio = ln.y_top / page_h if page_h > 0 else 0.0
            if ratio <= 0.05:
                header_map[pi].add(li)
            elif ratio >= 0.93:
                footer_map[pi].add(li)

    return header_map, footer_map, warnings


# =====================================================================
# Step 4: Column detection
# =====================================================================

def _detect_columns(lines: list[Line], page_w: float) -> list[ColumnSpec]:
    """Detect up to 2 columns by looking for large x-gap between line starts.

    Returns column x-boundaries; single-column returns [ColumnSpec(0, page_w)].
    """
    if not lines:
        return [ColumnSpec(0, max(page_w, 1.0))]

    # Collect line x_min (only lines longer than ~10 chars to avoid indent noise)
    x_mins: list[float] = []
    for ln in lines:
        if len(ln.text) >= 8:
            x_mins.append(ln.x_min)
    if not x_mins:
        return [ColumnSpec(0, max(page_w, 1.0))]

    # Try 2-column hypothesis: split at mid_x, count lines that are unambiguously left or right
    mid = page_w / 2
    left_lines = [ln for ln in lines if len(ln.text) >= 8 and ln.x_max < mid - 20]
    right_lines = [ln for ln in lines if len(ln.text) >= 8 and ln.x_min > mid + 20]
    total_eligible = len(left_lines) + len(right_lines)
    if total_eligible < 6:
        return [ColumnSpec(0, max(page_w, 1.0))]

    # If ≥60% of eligible long lines are split evenly, assume 2-column layout
    ratio = min(len(left_lines), len(right_lines)) / max(1, max(len(left_lines), len(right_lines)))
    if ratio >= 0.4 and (len(left_lines) + len(right_lines)) >= 8:
        # Find actual column boundary: the gap between max left x_min and min right x_min
        col_boundary = mid
        if left_lines and right_lines:
            max_left_x = max(ln.x_max for ln in left_lines)
            min_right_x = min(ln.x_min for ln in right_lines)
            if min_right_x > max_left_x:
                col_boundary = (max_left_x + min_right_x) / 2
        return [ColumnSpec(0, col_boundary), ColumnSpec(col_boundary, page_w)]

    return [ColumnSpec(0, max(page_w, 1.0))]


def _assign_column(line: Line, cols: list[ColumnSpec]) -> int:
    if len(cols) <= 1:
        return 0
    # Use x_min of line; tie-break by center
    cx = (line.x_min + line.x_max) / 2
    for i, c in enumerate(cols):
        if c.x_start <= cx <= c.x_end:
            return i
    # fallback: nearest
    return min(range(len(cols)), key=lambda i: abs((cols[i].x_start + cols[i].x_end) / 2 - cx))


# =====================================================================
# Step 5: Table detection & extraction
# =====================================================================

# Heuristic for table: a dense cluster of lines where multiple lines share
# the same column x-boundaries (≥3 distinct x-anchors) forming a grid.
# Also accept: lines starting with table header keywords followed by tabular alignment.

_TABLE_HEADER_HINT_RE = re.compile(r"(序号|编号|项目|名称|标的|规格|单位|数量|单价|金额|价格|产地|品种|商标)\s*(序号|编号|项目|名称|标的|规格|单位|数量|单价|金额|价格|产地|品种|商标)?")


def _quantize(v: float, step: float = 6.0) -> float:
    return round(v / step) * step


def _detect_table_regions(
    lines: list[Line],
    page: int,
    exclude_indices: set[int],
) -> list[TableRegion]:
    """Detect grid-like regions using x-anchor clustering."""
    regions: list[TableRegion] = []
    n = len(lines)
    if n < 2:
        return regions

    # For each line, compute its x-anchor set (quantized x positions of internal fragments)
    line_anchors: list[set[float]] = []
    for li, ln in enumerate(lines):
        anchors: set[float] = set()
        for f in ln.fragments:
            if f.text.strip():
                anchors.add(_quantize(f.x, 6.0))
        # Also add boundary anchors for each whitespace-separated token
        tokens = re.split(r"\s{2,}", ln.text)
        if len(tokens) >= 3:
            # spread them along x_min..x_max proportionally
            for ti in range(len(tokens) + 1):
                ratio = ti / max(1, len(tokens))
                anchors.add(_quantize(ln.x_min + ratio * (ln.x_max - ln.x_min), 6.0))
        line_anchors.append(anchors)

    # Find contiguous line blocks where ≥3 anchors are shared by ≥80% of lines in block
    i = 0
    while i < n:
        if i in exclude_indices:
            i += 1
            continue
        # Try extending window
        best_window: Optional[tuple[int, int, int, list[float]]] = None  # (start, end, n_cols, shared_anchors)
        for j in range(i + 1, min(i + 60, n)):
            if j in exclude_indices:
                break
            window_lines = lines[i : j + 1]
            if len(window_lines) < 2:
                continue
            # Shared anchors across majority
            counter: dict[float, int] = {}
            for k in range(i, j + 1):
                for a in line_anchors[k]:
                    counter[a] = counter.get(a, 0) + 1
            win_len = len(window_lines)
            threshold = max(2, int(win_len * 0.6))
            shared = sorted([a for a, c in counter.items() if c >= threshold])
            if len(shared) >= 3:
                # Candidate table; record longest with most columns
                if best_window is None or len(shared) > best_window[2] or (j - i) > (best_window[1] - best_window[0]):
                    best_window = (i, j, len(shared), shared)
        if best_window is not None:
            s, e, n_cols, anchors = best_window
            n_rows = e - s + 1
            # ================================================================
            # STRICT VALIDATION — reduce false positives
            # FIX-006 民法典 had 260 false-positive tables without these gates.
            # ================================================================
            # Header hint: require ≥1 keyword match in first 1-2 lines AND those
            # lines must have ≥3 distinct fragment x-anchors (= looks like a
            # multi-column header row).  This prevents common legal prose words
            # "名称/单位/数量/..." that appear ALONE in a paragraph from firing.
            hint_lines_passed = 0
            for k in range(s, min(e + 1, s + 2)):
                kw_match = bool(_TABLE_HEADER_HINT_RE.search(lines[k].text or ""))
                enough_anchors = len(line_anchors[k]) >= 3
                if kw_match and enough_anchors:
                    hint_lines_passed += 1
            header_hint_present = (hint_lines_passed >= 1)
            # (1) Row + Column size gate
            if header_hint_present:
                rows_ok = n_rows >= 2
                cols_ok = n_cols >= 3
            else:
                rows_ok = n_rows >= 14
                cols_ok = n_cols >= 6
            if not (rows_ok and cols_ok):
                i += 1
                continue
            # (2) Grid density: ≥50% of lines must have ≥n_cols-1 distinct anchors
            dense_count = 0
            for k in range(s, e + 1):
                if len(line_anchors[k]) >= max(3, n_cols - 1):
                    dense_count += 1
            if dense_count < max(2, int(n_rows * 0.5)):
                i += 1
                continue
            # (3) Content quality gate (destroys 民法典 punctuation-only false grids):
            #     Compute "meaningful chars per expected cell".  Low density = not a table.
            #     Strip spaces + Chinese/English punctuation from each line; count residue.
            _punct = set(" ，。！？、；：""''【】《》（）()()[]\\-—…·,.!?;:\"'`~@#$%^&*+=<>\\/|")
            total_ch = 0
            meaningful_cells_min_2ch = 0
            for k in range(s, e + 1):
                raw = lines[k].text
                meaningful = "".join(c for c in raw if c not in _punct)
                mlen = len(meaningful)
                total_ch += mlen
                # Count "cells with ≥2 meaningful chars" via fragment split
                cells_text = [f.text.strip() for f in lines[k].fragments if f.text.strip()]
                if not cells_text:
                    cells_text = [raw]
                for ct in cells_text:
                    cm = "".join(c for c in ct if c not in _punct)
                    if len(cm) >= 2:
                        meaningful_cells_min_2ch += 1
            expected_cells = n_rows * max(n_cols, 2)
            density = total_ch / max(1, expected_cells)
            # For hint tables: density ≥ 1.0, cells_≥2ch ≥ n_cols
            # For no-hint (huge grid) tables: density ≥ 2.0, cells_≥2ch ≥ n_rows
            if header_hint_present:
                content_ok = density >= 1.0 and meaningful_cells_min_2ch >= max(n_cols, 3)
            else:
                content_ok = density >= 2.0 and meaningful_cells_min_2ch >= max(n_rows, 5)
            if not content_ok:
                i += 1
                continue

            header_idx = [s]  # first line is likely header
            all_x = []
            all_y_top = []
            all_y_bot = []
            for k in range(s, e + 1):
                ln = lines[k]
                all_x.append(ln.x_min)
                all_x.append(ln.x_max)
                all_y_top.append(ln.y_top)
                all_y_bot.append(ln.y_bottom)
            caption = lines[s].text if lines[s].text and _TABLE_HEADER_HINT_RE.search(lines[s].text) else ""
            regions.append(TableRegion(
                page=page,
                x_min=min(all_x) - 2.0,
                x_max=max(all_x) + 2.0,
                y_top=min(all_y_top) - 2.0,
                y_bottom=max(all_y_bot) + 2.0,
                header_indices=header_idx,
                n_cols=n_cols,
                n_rows=n_rows,
                caption=caption,
                reason=f"x-anchors={n_cols}, rows={n_rows}, hint={header_hint_present}",
            ))
            i = e + 1
        else:
            i += 1

    return regions


def _extract_table_from_region(
    lines: list[Line],
    region: TableRegion,
    s_line_idx: int,
    e_line_idx_incl: int,
) -> Optional[TableData]:
    """Convert a table region's lines → TableData (headers + rows)."""
    region_lines = lines[s_line_idx : e_line_idx_incl + 1]
    if not region_lines:
        return None

    # Build column boundaries by clustering fragment x-anchors across the region
    anchor_counter: dict[float, int] = {}
    for ln in region_lines:
        seen_in_line: set[float] = set()
        for f in ln.fragments:
            a = _quantize(f.x, 5.0)
            if a not in seen_in_line:
                anchor_counter[a] = anchor_counter.get(a, 0) + 1
                seen_in_line.add(a)
        # Also add line x_min as anchor
        a = _quantize(ln.x_min, 5.0)
        if a not in seen_in_line:
            anchor_counter[a] = anchor_counter.get(a, 0) + 1

    # Pick anchors that appear in ≥40% of lines (or at least 2 lines)
    threshold = max(2, int(len(region_lines) * 0.4))
    col_edges = sorted(a for a, c in anchor_counter.items() if c >= threshold)
    if len(col_edges) < 2:
        # Fallback: split by whitespace in header row
        header_line = region_lines[0]
        tokens = [t for t in re.split(r"\s{2,}", header_line.text) if t.strip()]
        if len(tokens) < 2:
            tokens = header_line.text.split()
        if len(tokens) >= 2:
            # Create synthetic edges
            width = region.x_max - region.x_min
            ncols = len(tokens)
            col_edges = [region.x_min + (width * i / ncols) for i in range(ncols + 1)]
        else:
            return None

    n_cols = len(col_edges)

    def col_of_x(x: float) -> int:
        # Assign to nearest boundary
        for ci in range(n_cols - 1):
            mid = (col_edges[ci] + col_edges[ci + 1]) / 2
            if x < mid:
                return ci
        return n_cols - 2

    # Parse each line into row cells
    rows_cells: list[list[str]] = []
    for ln in region_lines:
        row: list[str] = [""] * max(1, n_cols - 1)
        # Sort fragments by x
        fs = sorted(ln.fragments, key=lambda frag: frag.x)
        for f in fs:
            if not f.text.strip():
                continue
            ci = col_of_x(f.x)
            if 0 <= ci < len(row):
                sep = "" if not row[ci] or row[ci].endswith(" ") else " "
                row[ci] += sep + f.text.strip()
        # Strip cells
        row = [c.strip() for c in row]
        rows_cells.append(row)

    if not rows_cells:
        return None

    # Normalize width: pad/truncate to header width
    header_cells = rows_cells[0]
    width = len(header_cells)
    for i in range(len(rows_cells)):
        r = rows_cells[i]
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            # Merge overflow into last cell
            r = r[:width - 1] + [" ".join(r[width - 1 :])]
        rows_cells[i] = r

    header_td = [TableCell(text=c, is_header=True) for c in rows_cells[0]]
    body_td = [[TableCell(text=c) for c in row] for row in rows_cells[1:]]

    # Handle all-empty rows (skip)
    body_td = [row for row in body_td if any(c.text for c in row)]

    try:
        td = TableData(headers=header_td, rows=body_td, caption=region.caption or None)
        return td
    except ContractViolation:
        return None


# =====================================================================
# Step 6: Reading order sort (top→bottom, column→column, left→right)
# =====================================================================

def _sort_reading_order(
    lines: list[Line],
    cols: list[ColumnSpec],
    drop_set: set[int],
) -> list[tuple[int, Line]]:
    """Return list of (original_line_idx, Line) in reading order."""
    items: list[tuple[int, int, float, Line]] = []
    for li, ln in enumerate(lines):
        if li in drop_set:
            continue
        col = _assign_column(ln, cols)
        # Primary: y_top bucketed
        y_bucket = round(ln.y_top / 2.0)  # bucket for tolerance
        items.append((col, y_bucket, ln.x_min, ln))
    # Sort by (column, y desc, x asc) — but we want top FIRST → y ascending?
    # Wait: y_top=0 is page TOP. So lower y_top = visually earlier.
    # So sorting by y_bucket ascending gives top→bottom.
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    result: list[tuple[int, Line]] = []
    seen: set[int] = set()
    for col, yb, xm, ln in items:
        # find original index
        for orig_idx, line in enumerate(lines):
            if line is ln and orig_idx not in seen:
                seen.add(orig_idx)
                result.append((orig_idx, line))
                break
    return result


# =====================================================================
# Step 7: Paragraph grouping + Block classification
# =====================================================================

_HEADING_NUMBER_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零〇○两\d]+[编章节条款目节条项款])[\s:：.]")
_CLAUSE_NUM_RE = re.compile(r"^\s*\d+(\.\d+)*[\s、.．:：]")
_LIST_NUM_RE = re.compile(r"^\s*([（(]\s*[一二三四五六七八九十百零〇\d]+\s*[)）]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]|\d+[.、．])")


def _classify_line_and_level(
    line: Line,
    page_body_fs_median: float,
    prev_line_text: str,
) -> tuple[BlockType, Optional[int], dict[str, Any]]:
    """Return (block_type_hint_for_line, heading_level_if_any, metadata)."""
    txt = line.text.strip()
    if not txt:
        return BlockType.UNKNOWN, None, {}
    meta: dict[str, Any] = {}

    # Font size relative
    rel_fs = line.fs_median / page_body_fs_median if page_body_fs_median > 0 else 1.0

    # Heading detection (multi-signal)
    heading_match = _HEADING_NUMBER_RE.match(txt)
    level: Optional[int] = None
    is_heading = False
    if heading_match:
        is_heading = True
        w = heading_match.group(1)
        if "编" in w:
            level = 1
        elif "章" in w:
            level = 2
        elif "节" in w:
            level = 3
        elif "条" in w:
            level = 5
        elif "款" in w:
            level = 6
        elif "项" in w:
            level = 6
        elif "目" in w:
            level = 4
        else:
            level = 4
    if _CLAUSE_NUM_RE.match(txt) and rel_fs >= 0.95:
        is_heading = True
        if level is None:
            num_part = txt.split()[0] if txt.split() else ""
            dots = num_part.count(".")
            level = min(7, 3 + dots)

    # Short + big font → heading
    if rel_fs >= 1.3 and len(txt) <= 40:
        is_heading = True
        if level is None:
            level = 2 if rel_fs >= 1.6 else 3
    if rel_fs >= 1.15 and len(txt) <= 25 and not is_heading:
        is_heading = True
        level = level or 4

    # LIST_ITEM detection
    if not is_heading and _LIST_NUM_RE.match(txt):
        # Count indent
        indent_level = 0
        stripped = txt.lstrip()
        indent_chars = len(txt) - len(stripped)
        if indent_chars >= 2:
            indent_level = 1
        if indent_chars >= 6:
            indent_level = 2
        meta["list_level"] = indent_level
        return BlockType.LIST_ITEM, None, meta

    if is_heading:
        if level is None:
            level = 5
        return BlockType.HEADING, level, meta

    return BlockType.PARAGRAPH, None, meta


@dataclass
class UnmergedBlock:
    page: int
    lines_in: list[Line]
    type: BlockType
    level: Optional[int]
    table_data: Optional[TableData]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    bbox_x_min: float = 0.0
    bbox_x_max: float = 0.0
    bbox_y_top: float = 0.0
    bbox_y_bottom: float = 0.0
    debug_src: str = ""


def _group_and_classify(
    ordered_lines: list[tuple[int, Line]],
    page_fs_median: float,
    page: int,
    table_extractions: dict[int, TableData],
) -> list[UnmergedBlock]:
    """Take ordered (line_idx, line) → produce semantic blocks with paragraph merging."""
    blocks: list[UnmergedBlock] = []
    if not ordered_lines:
        return blocks

    current_paragraph: Optional[UnmergedBlock] = None

    prev_txt = ""
    for orig_idx, ln in ordered_lines:
        # If this line's origin was absorbed into TABLE, emit the table block once
        if orig_idx in table_extractions:
            # Flush current paragraph if any
            if current_paragraph is not None and current_paragraph.text.strip():
                blocks.append(current_paragraph)
                current_paragraph = None
            td = table_extractions[orig_idx]
            tb = UnmergedBlock(
                page=page,
                lines_in=[ln],
                type=BlockType.TABLE,
                level=None,
                table_data=td,
                text=td.to_markdown(),
                metadata={"source": "table_region", "orig_line_start": orig_idx},
                bbox_x_min=ln.x_min,
                bbox_x_max=ln.x_max,
                bbox_y_top=ln.y_top,
                bbox_y_bottom=ln.y_bottom,
                debug_src=f"table_at_line_{orig_idx}",
            )
            blocks.append(tb)
            prev_txt = "[TABLE]"
            continue

        btype, lvl, meta = _classify_line_and_level(ln, page_fs_median, prev_txt)
        txt = ln.text.strip()

        if btype in (BlockType.HEADING, BlockType.LIST_ITEM):
            # HEADING always starts a new block (don't merge into prior paragraph).
            # LIST_ITEM: also starts a NEW semantic block; HOWEVER a list item may
            # span multiple physical PDF line wraps, so we allow subsequent
            # non-list/non-heading PARAGRAPH lines to merge into it via the
            # paragraph-wrap code below (set current_paragraph = ub below).
            if current_paragraph is not None and current_paragraph.text.strip():
                blocks.append(current_paragraph)
                current_paragraph = None
            ub = UnmergedBlock(
                page=page,
                lines_in=[ln],
                type=btype,
                level=lvl,
                table_data=None,
                text=txt,
                metadata=dict(meta),
                bbox_x_min=ln.x_min,
                bbox_x_max=ln.x_max,
                bbox_y_top=ln.y_top,
                bbox_y_bottom=ln.y_bottom,
                debug_src=f"{btype.value}_line_{orig_idx}",
            )
            if btype == BlockType.HEADING:
                # Headings never merge with following text.
                blocks.append(ub)
                current_paragraph = None
            else:
                # LIST_ITEM.  Subsequent PDF line wraps (that are classified as
                # PARAGRAPH and not themselves LIST_ITEM / HEADING) can merge
                # into current_paragraph via wrap rules below.
                current_paragraph = ub
            prev_txt = txt
            continue

        # PARAGRAPH — possibly merge with previous paragraph
        if current_paragraph is None:
            current_paragraph = UnmergedBlock(
                page=page,
                lines_in=[ln],
                type=BlockType.PARAGRAPH,
                level=None,
                table_data=None,
                text=txt,
                metadata=dict(meta),
                bbox_x_min=ln.x_min,
                bbox_x_max=ln.x_max,
                bbox_y_top=ln.y_top,
                bbox_y_bottom=ln.y_bottom,
                debug_src=f"para_start_{orig_idx}",
            )
        else:
            # Check vertical gap — too large → new paragraph
            prev_bot = current_paragraph.bbox_y_bottom
            cur_top = ln.y_top
            line_h = max(3.0, page_fs_median * 0.012) if page_fs_median > 0 else 5.0
            gap = max(0.0, cur_top - prev_bot)
            # Also check indent change / punctuation / font mismatch
            indent_diff = abs(ln.x_min - current_paragraph.bbox_x_min)
            ends_with_stop = current_paragraph.text.rstrip().endswith(("。", "！", "？", ".", "!", "?", "；", ";", "：", ":"))
            font_compat = (abs(ln.fs_median - page_fs_median) <= page_fs_median * 0.2) or page_fs_median == 0
            # ---- Relaxed wrap: short tail line is almost always a wrap carry-over ----
            cur_len = len(txt)
            short_tail_wrap = (
                not ends_with_stop
                and cur_len <= 20
                and gap <= line_h * (_LINE_GAP_FACTOR + 3.0)
                and font_compat
                and indent_diff <= 70.0
            )
            if short_tail_wrap:
                # Force merge, no flush.  Do not insert separator (CJK line wraps)
                sep = ""
                current_paragraph.text += sep + txt
                current_paragraph.lines_in.append(ln)
                current_paragraph.bbox_x_min = min(current_paragraph.bbox_x_min, ln.x_min)
                current_paragraph.bbox_x_max = max(current_paragraph.bbox_x_max, ln.x_max)
                current_paragraph.bbox_y_bottom = max(current_paragraph.bbox_y_bottom, ln.y_bottom)
            elif (
                not ends_with_stop
                and indent_diff <= 30.0
                and gap <= line_h * (_LINE_GAP_FACTOR + 0.5)
                and font_compat
            ):
                # Medium wrap: previous line does NOT end with sentence-final punct,
                # indent change is small, gap is small, font matches.
                # This is a normal CJK line-wrapped paragraph continuation.
                sep = ""  # no space for CJK wrap joins
                current_paragraph.text += sep + txt
                current_paragraph.lines_in.append(ln)
                current_paragraph.bbox_x_min = min(current_paragraph.bbox_x_min, ln.x_min)
                current_paragraph.bbox_x_max = max(current_paragraph.bbox_x_max, ln.x_max)
                current_paragraph.bbox_y_bottom = max(current_paragraph.bbox_y_bottom, ln.y_bottom)
            elif (
                gap > line_h * _LINE_GAP_FACTOR
                or indent_diff > 12.0
                or ends_with_stop
                or not font_compat
            ):
                # Flush previous paragraph
                blocks.append(current_paragraph)
                current_paragraph = UnmergedBlock(
                    page=page,
                    lines_in=[ln],
                    type=BlockType.PARAGRAPH,
                    level=None,
                    table_data=None,
                    text=txt,
                    metadata=dict(meta),
                    bbox_x_min=ln.x_min,
                    bbox_x_max=ln.x_max,
                    bbox_y_top=ln.y_top,
                    bbox_y_bottom=ln.y_bottom,
                    debug_src=f"para_start_{orig_idx}",
                )
            else:
                # Merge: append line text
                sep = "" if current_paragraph.text.endswith(("-", "—")) else " "
                current_paragraph.text += sep + txt
                current_paragraph.lines_in.append(ln)
                current_paragraph.bbox_x_min = min(current_paragraph.bbox_x_min, ln.x_min)
                current_paragraph.bbox_x_max = max(current_paragraph.bbox_x_max, ln.x_max)
                current_paragraph.bbox_y_bottom = max(current_paragraph.bbox_y_bottom, ln.y_bottom)

        prev_txt = txt

    if current_paragraph is not None and current_paragraph.text.strip():
        blocks.append(current_paragraph)
    return blocks


# =====================================================================
# Debug info carrier
# =====================================================================

@dataclass
class PDebug:
    pages_count: int = 0
    fragments_per_page: list[int] = field(default_factory=list)
    lines_per_page: list[int] = field(default_factory=list)
    raw_extract_samples: list[str] = field(default_factory=list)
    sort_baseline_samples: list[str] = field(default_factory=list)
    reconstructed_samples: list[str] = field(default_factory=list)
    header_footer_per_page: list[tuple[int, int]] = field(default_factory=list)
    columns_per_page: list[int] = field(default_factory=list)
    tables_per_page: list[int] = field(default_factory=list)
    table_details: list[dict[str, Any]] = field(default_factory=list)
    y_calibration_per_page: list[dict[str, Any]] = field(default_factory=list)
    dropped_lines: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =====================================================================
# Main POC parser
# =====================================================================

def _body_fs_median(lines: list[Line]) -> float:
    """Compute approximate body font size median (quantile-based to reject headings)."""
    if not lines:
        return 0.0
    vals = sorted(ln.fs_median for ln in lines if ln.fs_median > 0)
    if not vals:
        return 0.0
    # Take 40th percentile (headings are upper tail)
    idx = int(len(vals) * 0.4)
    return vals[min(idx, len(vals) - 1)]


def parse_pdf_v2_poc(
    pdf_path: str | Path,
    *,
    document_id: Optional[str] = None,
    debug: bool = True,
) -> tuple[ParsedDocument, PDebug]:
    """Parse PDF → ParsedDocument V2 (Frozen Contract) + debug info.

    Deterministic: same input bytes → same output structure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)

    d = PDebug(pages_count=n_pages)

    # ---- Page-level iteration ----
    all_page_lines: list[list[Line]] = []
    all_page_w: list[float] = []
    all_page_h: list[float] = []

    raw_samples: list[str] = []
    sort_samples: list[str] = []

    for pi in range(n_pages):
        page = reader.pages[pi]
        page_w = float(page.mediabox.width)
        page_h = float(page.mediabox.height)
        all_page_w.append(page_w)
        all_page_h.append(page_h)

        # Extract fragments (RAW coordinates, y direction unknown per page)
        frags = _extract_page_fragments(page, pi + 1, page_w, page_h)
        d.fragments_per_page.append(len(frags))

        # ---- CRITICAL: per-page y-axis direction calibration ----
        # pypdf tm[5] direction is INCONSISTENT across pages.  Use the plain
        # extract_text token order as ground-truth signal to decide whether
        # raw_y ASC or DESC means visual top→bottom on this specific page.
        direction, calib_info = _calibrate_y_direction(page, frags, page_h)
        calib_info["page"] = pi + 1
        d.y_calibration_per_page.append(calib_info)
        # Now normalize y_top in-place so small y_top = visually top
        _apply_y_calibration(frags, direction, page_h)

        # Group to lines (now using normalized y_top)
        lines = _fragments_to_lines(frags, pi + 1)
        all_page_lines.append(lines)
        d.lines_per_page.append(len(lines))

        # Debug samples: raw extract vs sort baseline (first 2 pages only)
        if pi < 2:
            plain = (page.extract_text() or "")[:600]
            raw_samples.append(f"== Page {pi+1} plain (600c) ==\n{plain}")
            geom_sorted = sorted(
                lines, key=lambda ln: (round(ln.y_top / 2.0), ln.x_min)
            )
            sort_txt = "\n".join(ln.text for ln in geom_sorted[:30])
            sort_samples.append(f"== Page {pi+1} geom-sort (30 lines) ==\n{sort_txt}")

    d.raw_extract_samples = raw_samples
    d.sort_baseline_samples = sort_samples

    # ---- Cross-page header/footer detection ----
    header_map, footer_map, hf_warns = _detect_header_footer(all_page_lines, all_page_h)
    d.warnings.extend(hf_warns)
    for pi in range(n_pages):
        n_h = len(header_map.get(pi, set()))
        n_f = len(footer_map.get(pi, set()))
        d.header_footer_per_page.append((n_h, n_f))
        if n_h:
            for li in header_map[pi]:
                if li < len(all_page_lines[pi]):
                    d.dropped_lines.append({
                        "page": pi + 1, "kind": "HEADER", "text": all_page_lines[pi][li].text[:60],
                    })
        if n_f:
            for li in footer_map[pi]:
                if li < len(all_page_lines[pi]):
                    d.dropped_lines.append({
                        "page": pi + 1, "kind": "FOOTER", "text": all_page_lines[pi][li].text[:60],
                    })

    # ---- Build final blocks across all pages ----
    doc_warnings: list[str] = []
    doc_warnings.append(BLOCKED_BY_DEPENDENCY_NOTE)
    doc_warnings.extend(hf_warns)

    doc_metadata: dict[str, Any] = {
        "filename": pdf_path.name,
        "extension": ".pdf",
        "mime_type": "application/pdf",
        "page_count": n_pages,
        "source": pdf_path.name,
        "parser": f"{PARSER_NAME}/{PARSER_VERSION}",
    }
    # Try native PDF metadata (titles, etc.)
    try:
        info = reader.metadata or {}
        for k in ("/Title", "/Author", "/Subject", "/Creator", "/CreationDate"):
            if k in info and info[k]:
                v = str(info[k])
                if v.strip():
                    doc_metadata[f"pdf_{k.lstrip('/').lower()}"] = v[:200]
    except Exception:
        pass

    unmerged_all: list[UnmergedBlock] = []
    recon_samples: list[str] = []

    for pi in range(n_pages):
        lines = all_page_lines[pi]
        page_w = all_page_w[pi]
        page_h = all_page_h[pi]

        # Build drop set (header + footer)
        drop_set: set[int] = set()
        drop_set.update(header_map.get(pi, set()))
        drop_set.update(footer_map.get(pi, set()))

        # Column detection
        cols = _detect_columns(lines, page_w)
        d.columns_per_page.append(len(cols))

        # Body font size
        kept_lines = [ln for i, ln in enumerate(lines) if i not in drop_set]
        fs_med = _body_fs_median(kept_lines) if kept_lines else 0.0

        # Table detection (on non-dropped lines, indices refer to positions in `lines`)
        # Exclude already-dropped lines from table detection
        table_regions = _detect_table_regions(lines, pi + 1, drop_set)
        d.tables_per_page.append(len(table_regions))

        # Map lines to their table extraction (key = start line index within page's `lines`)
        table_extractions: dict[int, TableData] = {}
        # Also track all line indices that are OWNED by a table (should NOT emit as text)
        table_owned_line_indices: set[int] = set()

        for ri, region in enumerate(table_regions):
            # Find which lines are inside the region bbox
            region_line_idxs: list[int] = []
            for li, ln in enumerate(lines):
                if li in drop_set:
                    continue
                if (
                    ln.y_top >= region.y_top - 0.5
                    and ln.y_bottom <= region.y_bottom + 0.5
                    and ln.x_min >= region.x_min - 5.0
                    and ln.x_max <= region.x_max + 5.0
                ):
                    region_line_idxs.append(li)
            if len(region_line_idxs) < 2:
                continue
            s = region_line_idxs[0]
            e = region_line_idxs[-1]
            td = _extract_table_from_region(lines, region, s, e)
            if td is not None and td.headers and len(td.headers) >= 2:
                table_extractions[s] = td
                for li2 in range(s, e + 1):
                    table_owned_line_indices.add(li2)
                d.table_details.append({
                    "page": pi + 1,
                    "region": ri,
                    "start_line": s,
                    "end_line": e,
                    "n_cols": len(td.headers),
                    "n_rows": len(td.rows),
                    "caption": td.caption or "",
                    "reason": region.reason,
                    "header_sample": [c.text[:20] for c in td.headers[:5]],
                })
            elif td is None:
                # Table region not convertible → still mark owned if dense grid
                if len(region_line_idxs) >= 3:
                    doc_warnings.append(
                        f"Page {pi+1}: Table region detected (rows={len(region_line_idxs)}, reason={region.reason}) "
                        f"but column extraction failed → lines emitted as PARAGRAPH."
                    )

        # Reading order sort (drop headers/footers + table-owned lines except TABLE start key kept in table_extractions)
        order_drop = set(drop_set)
        order_drop.update(table_owned_line_indices)
        # But keep the TABLE start line (it will trigger the table block emission)
        for start_li in table_extractions:
            order_drop.discard(start_li)

        ordered = _sort_reading_order(lines, cols, order_drop)
        # Add page to reconstruction sample (first 2 pages)
        if pi < 2:
            sample_lines = [ln.text for _, ln in ordered[:25]]
            recon_samples.append(f"== Page {pi+1} reconstructed (25 lines) ==\n" + "\n".join(sample_lines))

        # Group + classify
        page_blocks = _group_and_classify(
            ordered, fs_med, pi + 1, table_extractions,
        )
        unmerged_all.extend(page_blocks)

    d.reconstructed_samples = recon_samples

    # ---- Assemble V2 Document ----
    if document_id is None:
        # Deterministic ID from file path hash + size + mtime
        st = pdf_path.stat()
        h = hashlib.sha1(f"{pdf_path}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:16]
        document_id = f"pdfpoc_{h}"

    # Title heuristic: first HEADING or first significant non-empty block text
    title = pdf_path.stem
    for ub in unmerged_all:
        if ub.type is BlockType.HEADING and len(ub.text) >= 4:
            title = ub.text[:120]
            break
        if ub.type is BlockType.PARAGRAPH and 4 <= len(ub.text) <= 80 and ub.page <= 2:
            title = ub.text[:80]
            break
    doc_metadata["title"] = title

    # Build Block list (Frozen Contract strict)
    blocks: list[Block] = []
    source_file_str = str(pdf_path)

    for i, ub in enumerate(unmerged_all):
        bid = Block.make_block_id(document_id, i)
        if ub.type is BlockType.TABLE:
            assert ub.table_data is not None
            blk = Block(
                block_id=bid,
                type=BlockType.TABLE,
                text=ub.table_data.to_markdown(),
                order=i,
                level=None,
                table_data=ub.table_data,
                page=ub.page,
                metadata=dict(ub.metadata),
            )
        elif ub.type is BlockType.HEADING:
            blk = Block(
                block_id=bid,
                type=BlockType.HEADING,
                text=ub.text,
                order=i,
                level=ub.level or 5,
                table_data=None,
                page=ub.page,
                metadata=dict(ub.metadata),
            )
        else:
            blk = Block(
                block_id=bid,
                type=ub.type,
                text=ub.text,
                order=i,
                level=None,
                table_data=None,
                page=ub.page,
                metadata=dict(ub.metadata),
            )
        blk.attach_provenance(source_file_str)
        blocks.append(blk)

    # Strict pre-construction renumber (per Frozen Contract)
    ParsedDocument.prepare_blocks(document_id, blocks)

    try:
        doc = ParsedDocument(
            document_id=document_id,
            source_file=source_file_str,
            title=title,
            metadata=doc_metadata,
            blocks=blocks,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            warnings=doc_warnings,
        )
        doc.validate()
    except ContractViolation as e:
        doc_warnings.append(f"CONTRACT_VIOLATION_RECOVERED: {e}")
        # Recover: force renumber and try again
        ParsedDocument.prepare_blocks(document_id, blocks)
        doc = ParsedDocument(
            document_id=document_id,
            source_file=source_file_str,
            title=title,
            metadata=doc_metadata,
            blocks=blocks,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            warnings=doc_warnings + [f"POST-RENUMBER_RECOVERY: {e}"],
        )

    return doc, d


# =====================================================================
# Preview generators (for POC deliverables)
# =====================================================================

def generate_structured_preview(doc: ParsedDocument, d: PDebug) -> str:
    lines: list[str] = []
    lines.append("Document")
    lines.append(f"parser: {doc.parser_name}/{doc.parser_version}")
    lines.append(f"source: {doc.source_file}")
    lines.append(f"title: {doc.title}")
    lines.append(f"block_count: {len(doc.blocks)}")
    lines.append(f"page_count: {d.pages_count}")
    if doc.warnings:
        lines.append("warnings:")
        for w in doc.warnings[:5]:
            lines.append(f"  - {w[:200]}")
        if len(doc.warnings) > 5:
            lines.append(f"  ... +{len(doc.warnings)-5} more")
    lines.append("")
    for i, b in enumerate(doc.blocks):
        lines.append(f"Block {i:03d}")
        lines.append(f"type: {b.type.value}")
        lines.append(f"page: {b.page}")
        lines.append(f"order: {b.order}")
        if b.type is BlockType.HEADING:
            lines.append(f"level: {b.level}")
        md = b.metadata or {}
        if md:
            short_md = ", ".join(f"{k}={v!r}"[:60] for k, v in list(md.items())[:3])
            lines.append(f"meta: {{{short_md}}}")
        if b.provenance_stored:
            lines.append(f"prov: page={b.provenance_stored.page} block_order={b.provenance_stored.block_order}")
        if b.type is BlockType.TABLE and b.table_data is not None:
            lines.append("TABLE:")
            md_table = b.table_data.to_markdown()
            for tline in md_table.splitlines():
                lines.append(f"    {tline}")
        else:
            t = b.text.replace("\n", "\n    ")
            lines.append(f"text: {t[:1000]}")
        lines.append("")
    return "\n".join(lines)


def generate_legacy_preview(doc: ParsedDocument) -> str:
    return LegacyAdapter.to_legacy_text(doc, include_metadata=False)


def generate_debug_preview(doc: ParsedDocument, d: PDebug) -> str:
    out: list[str] = []
    out.append("# PDF Parser V2 POC Debug Report")
    out.append("")
    out.append(f"Source: {doc.source_file}")
    out.append(f"Pages: {d.pages_count}")
    out.append(f"Blocks: {len(doc.blocks)}")
    out.append("")

    out.append("## Pages summary")
    out.append("| Page | Fragments | Lines | Columns | Header dropped | Footer dropped | Tables |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pi in range(d.pages_count):
        fp = d.fragments_per_page[pi] if pi < len(d.fragments_per_page) else 0
        lp = d.lines_per_page[pi] if pi < len(d.lines_per_page) else 0
        cp = d.columns_per_page[pi] if pi < len(d.columns_per_page) else 1
        hd = d.header_footer_per_page[pi][0] if pi < len(d.header_footer_per_page) else 0
        fd = d.header_footer_per_page[pi][1] if pi < len(d.header_footer_per_page) else 0
        tp = d.tables_per_page[pi] if pi < len(d.tables_per_page) else 0
        out.append(f"| {pi+1} | {fp} | {lp} | {cp} | {hd} | {fd} | {tp} |")
    out.append("")

    out.append("## Raw extract samples (plain pypdf order)")
    for s in d.raw_extract_samples:
        out.append("```")
        out.append(s)
        out.append("```")
        out.append("")
    out.append("## Sort baseline (geom y-sort only)")
    for s in d.sort_baseline_samples:
        out.append("```")
        out.append(s)
        out.append("```")
        out.append("")
    out.append("## Reconstructed reading order")
    for s in d.reconstructed_samples:
        out.append("```")
        out.append(s)
        out.append("```")
        out.append("")

    out.append("## Table extractions")
    if d.table_details:
        for t in d.table_details:
            out.append(f"- Page {t['page']} region {t.get('region','?')}: "
                       f"{t['n_cols']} cols × {t['n_rows']+1} rows, caption={t.get('caption','')[:40]!r}, "
                       f"reason={t.get('reason','')}, headers={t.get('header_sample',[])}")
    else:
        out.append("- None")
    out.append("")

    out.append("## Dropped header/footer lines")
    if d.dropped_lines:
        for dl in d.dropped_lines:
            out.append(f"- Page {dl['page']} [{dl['kind']}]: {dl['text']!r}")
    else:
        out.append("- None")
    out.append("")

    out.append("## Column decisions per page")
    for pi, nc in enumerate(d.columns_per_page):
        out.append(f"- Page {pi+1}: {nc} column(s)")
    out.append("")

    out.append("## Warnings")
    for w in doc.warnings:
        out.append(f"- {w}")
    out.append("")

    return "\n".join(out)


# =====================================================================
# Convenience: main for POC execution
# =====================================================================

FIXTURES: list[tuple[str, str, Path]] = [
    (
        "FIX-001",
        "ENT_CONTRACT_011_农副产品买卖合同",
        Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_011_农副产品买卖合同（市场监管总局2025版）.pdf"),
    ),
    (
        "FIX-006",
        "LEGAL_LAW_001_中华人民共和国民法典",
        Path(r"D:\xiaoyi\data_source\01_public_legal\raw\pdf\LEGAL_LAW_001_中华人民共和国民法典.pdf"),
    ),
    (
        "FIX-008",
        "ENT_CONTRACT_037_建设工程施工合同",
        Path(r"D:\xiaoyi\data_source\02_enterprise_public\raw\pdf\ENT_CONTRACT_037_建设工程施工合同（住房城乡建设部、国家工商总局2017版）.pdf"),
    ),
    (
        "ADD-002",
        "LEGAL_LAW_002_中华人民共和国公司法",
        Path(r"D:\xiaoyi\data_source\01_public_legal\raw\pdf\LEGAL_LAW_002_中华人民共和国公司法.pdf"),
    ),
]


def run_all_fixtures(preview_dir: str | Path) -> dict[str, tuple[ParsedDocument, PDebug, Path]]:
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[ParsedDocument, PDebug, Path]] = {}
    for fid, fname, fpath in FIXTURES:
        print(f"[pdf_poc] Parsing {fid}: {fpath.name} ...")
        try:
            doc, dbg = parse_pdf_v2_poc(fpath)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            raise
        base = f"{fid}_{fname}"
        structured_path = preview_dir / f"{base}.structured.md"
        legacy_path = preview_dir / f"{base}.legacy.md"
        debug_path = preview_dir / f"{base}.debug.md"
        structured_path.write_text(generate_structured_preview(doc, dbg), encoding="utf-8")
        legacy_path.write_text(generate_legacy_preview(doc), encoding="utf-8")
        debug_path.write_text(generate_debug_preview(doc, dbg), encoding="utf-8")
        print(f"  blocks={len(doc.blocks)}, pages={dbg.pages_count}, tables={sum(dbg.tables_per_page)}")
        results[fid] = (doc, dbg, preview_dir)
    return results


if __name__ == "__main__":
    out_dir = Path(r"D:\xiaoyi\data_source\_meta\pdf_parser_v2_poc")
    run_all_fixtures(out_dir)
    print("[pdf_poc] Done.")
