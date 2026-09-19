"""
Xiaoyi Enterprise Legal RAG — DOCX Parser V2 (Production Candidate Core Hardening V1)
=====================================================================================

Scope:
  Frozen ParsedDocument V2 Contract output from DOCX native structure (OOXML)
  using:
    python-docx (high-level Document / Paragraph / Table)
    + limited OOXML fallback (lxml / zip) for:
        - Numbering XML (word/numbering.xml) resolution
        - Unified inline traversal (w:r / w:hyperlink / w:ins / w:del / smartTag / sdt)
        - Revision CURRENT policy diagnostics
        - Core / App / Custom properties

Production vs POC (refactor from POC 4100 line monolith to primitives):
  - NumberingResolver (§6 P0-1)
  - extract_inline_content (§8 P0-2)
  - Revision CURRENT policy (§9 P0-3)
  - Classifier (§14 Heading Policy, §7 Numbering Principles)
  - Table converter (§11-13 Table Hardening)
  - Document-order traversal (§15)

Design:
  No fixture-specific hacks.
  No schema modifications. Frozen Contract only (see parsed_document_v2.py).
  Provenance.page = None (never fake page numbers, §13).

Author: Xiaoyi Legal Copilot Team (Trae generated, POC→Production refactor)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

# Frozen Contract - 100% DO NOT MODIFY
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Task 20: repo-root anchored
from modules.ingestion.parsed_document_v2 import (  # noqa: E402
    Block,
    BlockType,
    LegacyAdapter,
    ParsedDocument,
    Provenance,
    TableCell,
    TableData,
)

from docx import Document  # noqa: E402
from docx.oxml.ns import qn as _qn, nsmap  # noqa: E402
from docx.oxml.table import CT_Tbl, CT_Tc  # noqa: E402
from docx.oxml.text.paragraph import CT_P  # noqa: E402
from docx.table import Table, _Cell, _Row  # noqa: E402
from docx.text.paragraph import Paragraph  # noqa: E402

import xml.etree.ElementTree as ET  # noqa: E402

# =============================================================================
# CONSTANTS
# =============================================================================

PARSER_NAME = "docx_parser_v2"
PARSER_VERSION = "1.0.0-core-hardening-v1"

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{NS_W}}}"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
R = f"{{{NS_REL}}}"

# Native Word Heading → Frozen level. Covers English & Chinese localized style names.
NATIVE_HEADING_MAP: Dict[str, int] = {}
for _lv in range(1, 10):
    NATIVE_HEADING_MAP[f"Heading {_lv}"] = _lv
    NATIVE_HEADING_MAP[f"标题 {_lv}"] = _lv
    NATIVE_HEADING_MAP[f"Heading{_lv}"] = _lv
    NATIVE_HEADING_MAP[f"标题{_lv}"] = _lv
    # Common suffix variants from OOXML styleId normalization
    NATIVE_HEADING_MAP[f"Heading{_lv}_1"] = _lv
    NATIVE_HEADING_MAP[f"Heading_{_lv}"] = _lv
NATIVE_HEADING_MAP["Title"] = 1  # document title maps to level 1 (Frozen Contract HEADING requires 1..9)
NATIVE_HEADING_MAP["标题"] = 1
NATIVE_HEADING_MAP["Subtitle"] = 2
NATIVE_HEADING_MAP["副标题"] = 2
# Heading (no level number, generic default to level-1)
NATIVE_HEADING_MAP["Heading"] = 1

# Legal chapter regex (NOT article — 第一章/第一节/第一篇 -> HEADING when style absent)
_LEGAL_CHAPTER_RE = re.compile(
    r"^\s*第[一二三四五六七八九十百千万零〇〇两0-9]+[章节编篇部卷节条总]+\s*[：:.]?\s*"
)
_LEGAL_ARTICLE_ONLY_RE = re.compile(r"^\s*第[一二三四五六七八九十百千万零〇〇两0-9]+条\s")
# List prefix fallback (§7 fallback; never OVERRIDE native numPr)
_LIST_PREFIX_CN1_RE = re.compile(r"^\s*[一二三四五六七八九十百千万零〇〇两]+[、.．]\s")
_LIST_PREFIX_CN2_RE = re.compile(r"^\s*[（(][一二三四五六七八九十百千万零〇〇两]+[)）]\s*")
_LIST_PREFIX_AR1_RE = re.compile(r"^\s*[0-9]+[.、．]\s")
_LIST_PREFIX_AR2_RE = re.compile(r"^\s*[（(][0-9]+[)）]\s*")
_LIST_PREFIX_BULLET_RE = re.compile(r"^\s*[\u2022\u25E6\u25AA\u2023\u2043\uF0B7\u00B7]\s")


@dataclass
class InlineResult:
    """§8 Unified inline extraction output (single paragraph / single table cell)."""
    text: str
    # Per-block revision diagnostics (count under CURRENT policy)
    inserted_revision_count: int = 0
    deleted_revision_count: int = 0  # not in text (CURRENT policy), but detected (not silent loss)
    inserted_revision_chars: int = 0
    deleted_revision_chars: int = 0
    has_revision_marks: bool = False  # ins or del present
    # Hyperlinks: [(visible_text, rel_id_or_url)]
    hyperlinks: List[Tuple[str, str]] = field(default_factory=list)
    hyperlink_count: int = 0


@dataclass
class NumberingInfo:
    """§6 Numbering resolver output per paragraph (when applicable)."""
    is_list: bool
    list_level: Optional[int] = None  # 0-based (ilvl)
    num_id: Optional[str] = None
    abstract_num_id: Optional[str] = None
    number_format: Optional[str] = None  # bullet / decimal / lowerLetter / chineseCounting ...
    level_text_template: Optional[str] = None  # lvlText value, e.g. "%1.%2."
    rendered_prefix: Optional[str] = None  # best-effort visible prefix (e.g. "1.2.3." or "•")


# =============================================================================
# INLINE: Unified extraction (P0-2 + Revision CURRENT P0-3)
# =============================================================================

def _iter_direct_children(element: ET.Element) -> Iterable[ET.Element]:
    """Yield direct children, preserving document order."""
    for child in list(element):
        yield child


def _collect_texts(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{W}t"))


def extract_inline_content(
    block_xml: ET.Element,
    revision_mode: str = "CURRENT",
    collect_text_only: bool = False,
) -> InlineResult:
    """§8 Unified inline extraction. Traverses children of a <w:p> or <w:tc> in DOCUMENT ORDER.

    Handles:
      w:r  (regular run)
      w:hyperlink → visible text (text box content of hyperlink's runs)
      w:ins → CURRENT policy: KEEP visible text (§9: inserted = current evidence)
      w:del → CURRENT policy: SKIP visible text (NOT in current evidence). BUT increment counter (no silent loss).
      w:smartTag → descend into children for w:r / w:ins / ...
      w:sdt (structured doc tag) → w:sdtContent then w:p / w:r children

    Args:
      block_xml: <w:p> or <w:tc> (or ancestor) XML element.
      revision_mode: Only "CURRENT" implemented in this Production V1.
      collect_text_only: If true, ignore hyperlinks/revision details slightly faster.

    Returns:
      InlineResult with text + diagnostics. Text order NEVER altered.
    """
    assert revision_mode == "CURRENT", (
        f"Only CURRENT mode implemented in Production V1. Got {revision_mode!r}"
    )

    res = InlineResult(text="")
    # Stack-based iterative DFS over child axis (not // descendants!) to preserve mixed order.
    # Each frame = (node, is_inside_w_del)
    stack: List[Tuple[ET.Element, bool]] = []
    # Push direct children in REVERSE so we pop left-first (preserves doc order)
    for child in reversed(list(block_xml)):
        stack.append((child, False))

    while stack:
        node, inside_del = stack.pop()
        tag_local = node.tag.split("}")[-1] if "}" in node.tag else node.tag

        # --- Terminal inline runs ---
        if tag_local == "r":
            t_text = _collect_texts(node)
            if t_text:
                # Determine status if wrapped by del (ancestor) or ins
                # inside_del already propagated by w:del handler below
                if inside_del:
                    # §9 CURRENT: SKIP from evidence, count only
                    res.deleted_revision_chars += len(t_text)
                    res.has_revision_marks = True
                else:
                    res.text += t_text
            continue

        # --- Hyperlink (non-terminal: contains w:r children; relationship stored) ---
        if tag_local == "hyperlink":
            rel_id = node.get(f"{R}id", "")
            anchor_text_before = res.text
            # Descend: push children (w:r / w:ins ... inside hyperlink)
            # We need to capture visible text specifically for hyperlink metadata
            # So we recurse here instead of stack to isolate anchor length
            sub = extract_inline_content(node, revision_mode="CURRENT", collect_text_only=True)
            if sub.text:
                if inside_del:
                    res.deleted_revision_chars += len(sub.text)
                else:
                    res.text += sub.text
            if sub.text and not collect_text_only:
                res.hyperlinks.append((sub.text, rel_id))
                res.hyperlink_count += 1
            # propagate counts
            res.inserted_revision_count += sub.inserted_revision_count
            res.deleted_revision_count += sub.deleted_revision_count
            res.inserted_revision_chars += sub.inserted_revision_chars
            res.deleted_revision_chars += sub.deleted_revision_chars
            if sub.has_revision_marks:
                res.has_revision_marks = True
            continue

        # --- w:ins (Inserted revision) → §9 CURRENT: treat as visible evidence ---
        if tag_local == "ins":
            # Count this ins node (per-revision, not per-run)
            res.inserted_revision_count += 1
            res.has_revision_marks = True
            chars_before = len(res.text)
            # Push w:r / w:p descendants of ins, inside_del=False still (ins is VISIBLE)
            for child in reversed(list(node)):
                stack.append((child, inside_del))
            # We can't easily count chars here; count via a simpler way:
            # Collect text once then diff (for diagnostics).
            ins_t = _collect_texts(node)
            res.inserted_revision_chars += len(ins_t)
            continue

        # --- w:del (Deleted revision) → §9 CURRENT: NOT visible; only count ---
        if tag_local == "del":
            res.deleted_revision_count += 1
            res.has_revision_marks = True
            # Push descendants with inside_del=True to block text from being appended.
            for child in reversed(list(node)):
                stack.append((child, True))
            # Also count chars for diagnostics
            del_t = _collect_texts(node)
            res.deleted_revision_chars += len(del_t)
            continue

        # --- w:smartTag — non-revision container, transparent descend ---
        if tag_local in ("smartTag", "customXml", "sdtContent", "sdt", "pict", "rPr", "pPr", "tblPr", "trPr", "tcPr", "hyperlink", "bookmarkStart", "bookmarkEnd", "commentRangeStart", "commentRangeEnd"):
            if tag_local in ("rPr", "pPr", "tblPr", "trPr", "tcPr", "bookmarkStart", "bookmarkEnd", "commentRangeStart", "commentRangeEnd"):
                continue  # Properties-only / structural markers, no inline text
            # Descend transparently, inheriting inside_del flag
            for child in reversed(list(node)):
                stack.append((child, inside_del))
            continue

        # --- Unknown tags: descend anyway to not miss text containers ---
        # (e.g. w:fldSimple, w:fldChar, w:instrText are mostly ignored; we only seek <w:t> inside "good" containers)
        # Skip pure property/metadata leaf if it's definitely not a text container
        if tag_local in ("noBreakHyphen", "softHyphen", "tab", "br", "cr", "sym", "drawing", "object", "pict", "lastRenderedPageBreak"):
            continue
        # Default: safe descend (inside_del propagated)
        for child in reversed(list(node)):
            stack.append((child, inside_del))

    # Hyperlink count already added via hyperlink handler; make sure aggregate hyperlink_count correct
    if res.hyperlink_count == 0 and res.hyperlinks:
        res.hyperlink_count = len(res.hyperlinks)
    return res


# =============================================================================
# NUMBERING: NumberingResolver (P0-1 §6-§7)
# =============================================================================

class NumberingResolver:
    """Parses word/numbering.xml and resolves per-paragraph numbering metadata.

    Native Evidence Priority (§7):
      1. w:numPr exists → LIST_ITEM (MEMBERSHIP)
      2. numId + ilvl → look up in abstractNum → w:lvl → numFmt / lvlText / start
      3. Override per num → w:lvlOverride (startOverride, new lvl)
      4. Best-effort rendered_prefix (simple case only)
    """

    def __init__(self, doc_zip: zipfile.ZipFile):
        self.has_numbering_part: bool = False
        self.abstract_nums: Dict[str, Dict[int, Dict[str, Any]]] = {}  # abstractNumId -> {ilvl -> {fmt, txt, start}}
        self.nums: Dict[str, Dict[str, Any]] = {}  # numId -> {abstractNumId, overrides:{ilvl -> override_dict}}
        self.warning_messages: List[str] = []

        if "word/numbering.xml" not in doc_zip.namelist():
            return
        self.has_numbering_part = True
        try:
            root = ET.fromstring(doc_zip.read("word/numbering.xml"))
        except ET.ParseError as e:
            self.warning_messages.append(f"numbering.xml parse error: {e}")
            return

        # Parse abstractNum
        for a in root.findall(f"{W}abstractNum"):
            a_id = a.get(f"{W}abstractNumId")
            if a_id is None:
                continue
            lvls: Dict[int, Dict[str, Any]] = {}
            for lvl in a.findall(f"{W}lvl"):
                ilvl_s = lvl.get(f"{W}ilvl")
                if ilvl_s is None:
                    continue
                ilvl = int(ilvl_s)
                fmt_node = lvl.find(f"{W}numFmt")
                txt_node = lvl.find(f"{W}lvlText")
                start_node = lvl.find(f"{W}start")
                lvls[ilvl] = {
                    "numFmt": fmt_node.get(f"{W}val") if fmt_node is not None else None,
                    "lvlText": txt_node.get(f"{W}val") if txt_node is not None else None,
                    "start": int(start_node.get(f"{W}val")) if start_node is not None and start_node.get(f"{W}val", "").isdigit() else 1,
                }
            self.abstract_nums[a_id] = lvls

        # Parse num (instance with abstract ref + overrides)
        for n in root.findall(f"{W}num"):
            n_id = n.get(f"{W}numId")
            if n_id is None:
                continue
            abs_ref = n.find(f"{W}abstractNumId")
            abs_id = abs_ref.get(f"{W}val") if abs_ref is not None else None
            overrides: Dict[int, Dict[str, Any]] = {}
            for o in n.findall(f"{W}lvlOverride"):
                ilvl_s = o.get(f"{W}ilvl")
                if ilvl_s is None:
                    continue
                ilvl = int(ilvl_s)
                o_dict: Dict[str, Any] = {}
                st = o.find(f"{W}startOverride")
                if st is not None:
                    o_dict["startOverride"] = st.get(f"{W}val")
                lvl_mod = o.find(f"{W}lvl")
                if lvl_mod is not None:
                    fm = lvl_mod.find(f"{W}numFmt")
                    tx = lvl_mod.find(f"{W}lvlText")
                    if fm is not None:
                        o_dict["numFmt"] = fm.get(f"{W}val")
                    if tx is not None:
                        o_dict["lvlText"] = tx.get(f"{W}val")
                overrides[ilvl] = o_dict
            self.nums[n_id] = {"abstractNumId": abs_id, "overrides": overrides}

    # --- helpers ---
    @staticmethod
    def _chinese_counting(n: int) -> str:
        # Sufficient for list numbers typically 1~999
        digits = "零一二三四五六七八九"
        if n <= 0:
            return str(n)
        if n < 10:
            return digits[n]
        if n < 20:
            return "十" + (digits[n % 10] if n % 10 != 0 else "")
        if n < 100:
            tens, rem = divmod(n, 10)
            return digits[tens] + "十" + (digits[rem] if rem else "")
        if n < 1000:
            hundreds, rest = divmod(n, 100)
            s = digits[hundreds] + "百"
            if rest == 0:
                return s
            if rest < 10:
                s += "零"
            return s + NumberingResolver._chinese_counting(rest)
        return str(n)

    @staticmethod
    def _format_num(n: int, num_fmt: Optional[str]) -> str:
        if not num_fmt:
            return str(n)
        f = num_fmt.lower()
        if f in ("decimal", "decimalzeropadded", "decimalenclosedcircle", "decimalenclosedfullstop", "chinesedigital"):
            return str(n)
        if f in ("chineseccounting", "chineselettersimplified", "chineselettertraditional"):
            return NumberingResolver._chinese_counting(n)
        if f in ("lowerletter", "loweralpha"):
            # a-z aa-az ...
            return "".join(chr(ord("a") + ((x - 1) % 26)) for x in [((n - 1) // 26) + 1 if n > 26 else n]) if n <= 26 else chr(ord("a") + (n - 1) % 26)
        if f in ("upperletter", "upperalpha"):
            return chr(ord("A") + (n - 1) % 26) if n <= 26 else chr(ord("A") + (n - 1) % 26)
        if f in ("lowerroman",):
            return NumberingResolver._roman(n, lower=True)
        if f in ("upperroman",):
            return NumberingResolver._roman(n, lower=False)
        if f in ("bullet", "none"):
            return ""
        return str(n)

    @staticmethod
    def _roman(n: int, lower: bool = False) -> str:
        vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        s = ""
        for v, sym in vals:
            while n >= v:
                s += sym
                n -= v
        return s.lower() if lower else s

    # --- Public interface ---
    def resolve(self, paragraph_xml: ET.Element) -> NumberingInfo:
        """Get native numbering info for a <w:p> element."""
        pPr = paragraph_xml.find(f"{W}pPr")
        if pPr is None:
            return NumberingInfo(False)
        numPr = pPr.find(f"{W}numPr")
        if numPr is None:
            return NumberingInfo(False)
        numId_el = numPr.find(f"{W}numId")
        ilvl_el = numPr.find(f"{W}ilvl")
        if numId_el is None:
            return NumberingInfo(False)  # malformed
        num_id = numId_el.get(f"{W}val")
        ilvl = int(ilvl_el.get(f"{W}val")) if ilvl_el is not None and ilvl_el.get(f"{W}val", "").isdigit() else 0
        info = NumberingInfo(True, list_level=ilvl, num_id=num_id)

        if not self.has_numbering_part or num_id not in self.nums:
            return info  # unknown num_id
        num_def = self.nums[num_id]
        abs_id = num_def.get("abstractNumId")
        info.abstract_num_id = abs_id
        if abs_id is None or abs_id not in self.abstract_nums:
            return info
        abs_lvls = self.abstract_nums[abs_id]
        lvl_def = abs_lvls.get(ilvl, {})
        # Apply override if any
        ovr = num_def.get("overrides", {}).get(ilvl, {})
        fmt = ovr.get("numFmt") if "numFmt" in ovr else lvl_def.get("numFmt")
        txt = ovr.get("lvlText") if "lvlText" in ovr else lvl_def.get("lvlText")
        start_raw = ovr.get("startOverride") if "startOverride" in ovr else lvl_def.get("start")
        start = int(start_raw) if isinstance(start_raw, str) and start_raw.isdigit() else (start_raw if isinstance(start_raw, int) else 1)
        info.number_format = fmt
        info.level_text_template = txt

        # Best-effort rendered_prefix: lvlText %1.%2.%3. → substitute 1 for each level <= ilvl
        if txt is not None:
            # Simple deterministic substitution (no full level counters state for Production V1)
            # We substitute "%N" with start value of corresponding abstract level (best-effort)
            def sub(m):
                try:
                    i = int(m.group(1)) - 1
                    # Only support up to ilvl; higher levels substitute with their own abstract lvl start if known
                    if i == ilvl:
                        return NumberingResolver._format_num(start, fmt)
                    # else fall back to start if i<len(abs_lvls) else 1
                    other_lvl_def = abs_lvls.get(i, {})
                    other_start = other_lvl_def.get("start", 1)
                    other_fmt = other_lvl_def.get("numFmt") or fmt
                    return NumberingResolver._format_num(other_start, other_fmt)
                except Exception:
                    return m.group(0)
            rendered = re.sub(r"%(\d+)", sub, txt)
            info.rendered_prefix = rendered
        elif fmt == "bullet":
            # Bullet without lvlText is common
            info.rendered_prefix = "\u2022"
        return info


# =============================================================================
# PARAGRAPH CLASSIFIER (Heading Policy §14 + Numbering §7)
# =============================================================================

def _level_from_outlineLvl(p_xml: ET.Element) -> Optional[int]:
    pPr = p_xml.find(f"{W}pPr")
    if pPr is None:
        return None
    ol = pPr.find(f"{W}outlineLvl")
    if ol is None:
        return None
    v = ol.get(f"{W}val")
    if v is None or not v.isdigit():
        return None
    # OOXML outlineLvl 0-8 → Heading level 1-9
    return int(v) + 1


def _classify_paragraph(
    p: Paragraph,
    p_xml: ET.Element,
    trimmed_text: str,
    num_info: NumberingInfo,
) -> Tuple[BlockType, Optional[int], Dict[str, Any]]:
    """Return (block_type, heading_level_or_None, extra_metadata)."""
    extra: Dict[str, Any] = {}
    if p.style is not None:
        extra["style_name"] = p.style.name or ""
        extra["style_id"] = p.style.style_id or ""

    # --- 1. Native Heading Style (strongest) §14 ---
    native_level = None
    if p.style is not None:
        name = p.style.name or ""
        if name in NATIVE_HEADING_MAP:
            native_level = NATIVE_HEADING_MAP[name]
        else:
            # Fuzzy: starts with Heading/标题 ends with digit
            m1 = re.match(r"^Heading\s*([1-9])$", name)
            m2 = re.match(r"^标题\s*([1-9])$", name)
            m3 = re.match(r"^Heading\s*([1-9])_?\d*$", name)
            for m in (m1, m2, m3):
                if m:
                    native_level = int(m.group(1))
                    break

    if native_level is None:
        native_level = _level_from_outlineLvl(p_xml)

    if native_level is not None:
        return (BlockType.HEADING, native_level, extra)

    # --- 2. Native Numbering (§7: MEMBERSHIP > visual regex) ---
    if num_info.is_list:
        return (BlockType.LIST_ITEM, None, {
            **extra,
            "list_level": num_info.list_level,
            "num_id": num_info.num_id,
            "num_abstract_num_id": num_info.abstract_num_id,
            "number_format": num_info.number_format,
            "level_text_template": num_info.level_text_template,
            "rendered_numbering_prefix": num_info.rendered_prefix,
            "numbering_source": "native_numbering_xml",
        })

    # --- 3. Legal Chapter heuristic (第一章/第一节/第一篇 -> HEADING; NEVER 第一条 as HEADING per §9) ---
    if trimmed_text and _LEGAL_CHAPTER_RE.match(trimmed_text):
        # Double check: It's NOT an article ("第一条") which is a LIST_ITEM per §9
        if not _LEGAL_ARTICLE_ONLY_RE.match(trimmed_text):
            # Assign a level based on structural keyword (Frozen Contract requires 1..9 int for HEADING)
            sem_lv = 2  # default (第一节 default is sub to 第一章 lv1)
            if re.search(r"第[一二三四五六七八九十百千万零〇〇两0-9]+[编篇部卷]", trimmed_text):
                sem_lv = 1  # Top level
            elif re.search(r"第[一二三四五六七八九十百千万零〇〇两0-9]+章", trimmed_text):
                sem_lv = 1
            elif re.search(r"第[一二三四五六七八九十百千万零〇〇两0-9]+节", trimmed_text):
                sem_lv = 2
            elif re.search(r"第[一二三四五六七八九十百千万零〇〇两0-9]+[总目次分]", trimmed_text):
                sem_lv = 1
            # Clamp to Frozen Contract 1..9
            sem_lv = max(1, min(9, sem_lv))
            return (BlockType.HEADING, sem_lv, {**extra, "heading_source": "semantic_legal_chapter_regex"})

    # --- 3b. Semantic Heading H1 fallback: very short standalone <30 chars title-like at top doc with colon rare (defense)
    # Disabled: avoid cosmetic (per §9 don't fake headings)


    # --- 4. Semantic LIST_ITEM fallback (never override native numbering) ---
    if trimmed_text and (
        _LIST_PREFIX_CN1_RE.match(trimmed_text)
        or _LIST_PREFIX_CN2_RE.match(trimmed_text)
        or _LIST_PREFIX_AR1_RE.match(trimmed_text)
        or _LIST_PREFIX_AR2_RE.match(trimmed_text)
        or _LIST_PREFIX_BULLET_RE.match(trimmed_text)
        or _LEGAL_ARTICLE_ONLY_RE.match(trimmed_text)
    ):
        return (BlockType.LIST_ITEM, None, {**extra, "numbering_source": "semantic_prefix_regex"})

    # --- 5. Default PARAGRAPH ---
    return (BlockType.PARAGRAPH, None, extra)


# =============================================================================
# TABLE PRODUCTION HARDENER (§11-13)
# =============================================================================

def _convert_docx_table(table: Table) -> Tuple[TableData, Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "rows": 0,
        "cols": 0,
        "merged_cells_count": 0,
        "vmerge_continue_count": 0,
        "nested_tables_cells": 0,
        "empty_cells_count": 0,
    }
    rows_data: List[List[TableCell]] = []
    headers: List[TableCell] = []

    real_rows = list(table.rows)
    diag["rows"] = len(real_rows)
    if not real_rows:
        return TableData(headers=[], rows=[]), diag

    n_cols_guess = len(real_rows[0].cells)
    diag["cols"] = n_cols_guess

    all_rows: List[List[TableCell]] = []
    for ri, row in enumerate(real_rows):
        row_cells: List[TableCell] = []
        # Access underlying CT_Row for merge info via tc XML
        ct_row = row._tr
        ct_tcs: List = list(ct_row.findall(f"{W}tc"))
        for ci, (cell, ct_tc) in enumerate(zip(list(row.cells), ct_tcs)):
            # Extract cell's paragraphs with inline (including NESTED tables inside cell — §13 no silent loss)
            # Recursive descent: all <w:p> at any depth within <w:tc> (covers nested tbl/tc/p)
            cell_paragraphs_xml = list(cell._tc.findall(f".//{W}p"))
            cell_texts: List[str] = []
            total_ins = total_del = 0
            for cpx in cell_paragraphs_xml:
                ir = extract_inline_content(cpx)
                if ir.text.strip() or len(cell_texts) > 0 and cpx is not cell_paragraphs_xml[-1]:
                    cell_texts.append(ir.text)
                total_ins += ir.inserted_revision_count
                total_del += ir.deleted_revision_count
            # Nested tables detection
            nested = cell._tc.findall(f".//{W}tbl")
            nested_count = len(nested)
            if nested_count > 0:
                diag["nested_tables_cells"] += nested_count
            # Merge info via XML attributes on <w:tcPr>
            tcPr = ct_tc.find(f"{W}tcPr")
            colspan = 1
            rowspan = 1  # Frozen Contract invariant: rowspan >= 1 (always)
            if tcPr is not None:
                gs = tcPr.find(f"{W}gridSpan")
                if gs is not None:
                    try:
                        span = int(gs.get(f"{W}val", "1"))
                        if span > 1:
                            colspan = span
                            diag["merged_cells_count"] += 1
                    except (ValueError, TypeError):
                        pass
                vm = tcPr.find(f"{W}vMerge")
                if vm is not None:
                    v = vm.get(f"{W}val")
                    if v == "continue" or v is None:
                        # Frozen TableCell doesn't support "continue" sentinel (rowspan must be >=1)
                        # Best-effort per §12: rowspan=1, diagnostic record
                        diag["vmerge_continue_count"] += 1
                        diag["merged_cells_count"] += 1
                    elif v == "restart":
                        rowspan = 2  # best-effort guess (no full vertical span reconstruction)
                        diag["merged_cells_count"] += 1
            cell_text = "\n".join(cell_texts).strip()
            if not cell_text:
                diag["empty_cells_count"] += 1
            row_cells.append(TableCell(
                text=cell_text,
                colspan=colspan,
                rowspan=rowspan,
                is_header=(ri == 0),
            ))
        all_rows.append(row_cells)

    # Normalize all rows to SAME logical cell count (§12 best-effort structural mapping).
    # Frozen Contract invariant: len(each row) == len(headers). Use max-length as
    # baseline to avoid truncating visible evidence (never silent loss per §12).
    if all_rows:
        target_len = max(len(r) for r in all_rows)
        for r_idx, row in enumerate(all_rows):
            if len(row) < target_len:
                pad = target_len - len(row)
                diag.setdefault("padded_empty_cell_total", 0)
                diag["padded_empty_cell_total"] += pad
                all_rows[r_idx] = row + [
                    TableCell(text="", colspan=1, rowspan=1, is_header=(r_idx == 0))
                    for _ in range(pad)
                ]
            elif len(row) > target_len:
                # Should be impossible (target_len = max); defense-in-depth
                diag.setdefault("truncated_cell_total", 0)
                diag["truncated_cell_total"] += (len(row) - target_len)
                all_rows[r_idx] = row[:target_len]
        headers = all_rows[0]
        rows_data = all_rows[1:]
    else:
        headers, rows_data = [], []

    # Warning if nested tables present
    return TableData(headers=headers, rows=rows_data), diag


# =============================================================================
# DIAGNOSTICS INVENTORIES (Revision / Hyperlink / Textbox / HF / Pagebreak)
# =============================================================================

def _collect_inventories(doc_xml: ET.Element, doc_zip: zipfile.ZipFile, document: Document) -> Dict[str, Any]:
    inv: Dict[str, Any] = {}
    # Revision
    ins_count = len(doc_xml.findall(f".//{W}ins"))
    del_count = len(doc_xml.findall(f".//{W}del"))
    r_ins = len(doc_xml.findall(f".//{W}rIns"))
    r_del = len(doc_xml.findall(f".//{W}rDel"))
    inv["revision_inventory"] = {
        "w_ins_count": ins_count,
        "w_del_count": del_count,
        "w_rIns_count": r_ins,
        "w_rDel_count": r_del,
    }
    # Hyperlink
    hyperlinks = doc_xml.findall(f".//{W}hyperlink")
    # stdlib ET has no .getparent(); count paragraphs that contain hyperlink children
    # by iterating paragraph candidates (not all parents). For inventory, counts
    # are observational only — so use distinct paragraphs with hyperlink descendants
    paras_with_hyper: Set[str] = set()
    for p in doc_xml.findall(f".//{W}p"):
        if len(p.findall(f".//{W}hyperlink")) > 0:
            paras_with_hyper.add(id(p))
    inv["hyperlink_inventory"] = {
        "hyperlink_element_count": len(hyperlinks),
        "paragraphs_with_hyperlinks": len(paras_with_hyper),
    }
    # Text box / drawing inline text content (not visible via python-docx paragraphs)
    txbx_contents = doc_xml.findall(f".//{W}txbxContent")
    txbx_text_total = 0
    for tc in txbx_contents:
        txbx_text_total += sum(len(t.text or "") for t in tc.findall(f".//{W}t"))
    inv["txbx_inventory"] = {
        "w_txbx_content_count": len(txbx_contents),
        "w_txbx_content_text_total": txbx_text_total,
    }
    # Page breaks
    last_pb = len(doc_xml.findall(f".//{W}lastRenderedPageBreak"))
    hard_br = 0
    for br in doc_xml.findall(f".//{W}br"):
        if br.get(f"{W}type") == "page":
            hard_br += 1
    inv["page_break_inventory"] = {
        "w_lastRenderedPageBreak_count": last_pb,
        "w_hard_page_break_count": hard_br,
    }
    # Header/footer via python-docx sections
    sections = list(document.sections)
    header_flags = [not s.header.is_linked_to_previous for s in sections]
    footer_flags = [not s.footer.is_linked_to_previous for s in sections]
    inv["header_footer_inventory"] = {
        "section_count": len(sections),
        "unique_section_headers": sum(1 for f in header_flags if f),
        "unique_section_footers": sum(1 for f in footer_flags if f),
    }
    # Custom properties existence
    inv["package_part_presence"] = {
        "has_word_numbering_xml": "word/numbering.xml" in doc_zip.namelist(),
        "has_word_footnotes_xml": "word/footnotes.xml" in doc_zip.namelist(),
        "has_word_endnotes_xml": "word/endnotes.xml" in doc_zip.namelist(),
        "has_word_comments_xml": "word/comments.xml" in doc_zip.namelist(),
        "has_docProps_core_xml": "docProps/core.xml" in doc_zip.namelist(),
        "has_docProps_app_xml": "docProps/app.xml" in doc_zip.namelist(),
        "has_docProps_custom_xml": "docProps/custom.xml" in doc_zip.namelist(),
    }
    return inv


# =============================================================================
# TITLE / DUPLICATE (§16)
# =============================================================================

def _extract_title_and_duplicate(
    document: Document,
    blocks: List[Block],
    source_file: str,
) -> Tuple[str, bool, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    # Candidates
    core_title = (getattr(document.core_properties, "title", "") or "").strip()
    notes["core_properties_title"] = core_title
    # First HEADING block if any
    first_heading = next((b.text.strip() for b in blocks if b.type is BlockType.HEADING and b.text.strip()), "")
    if first_heading:
        notes["first_heading_block_title"] = first_heading
    # First text-like PARAGRAPH block (document cover title usually)
    first_body_text = ""
    for b in blocks:
        if b.type in (BlockType.PARAGRAPH, BlockType.LIST_ITEM) and b.text.strip():
            first_body_text = b.text.strip()
            notes["first_body_paragraph_text"] = first_body_text
            break
    # Decide title with preference order
    candidates = [core_title, first_heading, first_body_text]
    title = ""
    for c in candidates:
        if c and len(c) < 200:
            title = c
            break
    if not title:
        # fallback to filename without extension
        p = Path(source_file)
        # remove ENT_CONTRACT_XXX prefix
        name = p.stem
        m = re.match(r"^[A-Z]+_\w+_(.+)$", name)
        if m:
            name = m.group(1)
        title = name.strip() or p.stem
        notes["title_fallback_reason"] = "no_core_no_heading_no_first_paragraph_nonempty"

    # Duplicate detection (§16: only mark, not delete)
    title_l = title.strip().lower()
    duplicate = False
    if title_l:
        # Does title_l appear elsewhere in body as a standalone short block?
        for b in blocks:
            if b is blocks[0]:
                continue
            bt = b.text.strip()
            if not bt:
                continue
            if bt.lower() == title_l and len(bt) < 200:
                duplicate = True
                notes["duplicate_first_occurrence_block_order"] = b.order
                break
    notes["title_candidates"] = candidates
    notes["title_chosen_from"] = (
        "core_properties" if title == core_title else
        "first_heading_block" if title == first_heading else
        "first_body_paragraph" if title == first_body_text else
        "filename_fallback"
    )
    return title, duplicate, notes


# =============================================================================
# PUBLIC: parse_docx_v2
# =============================================================================

def parse_docx_v2(path: str | Path) -> ParsedDocument:
    """Production DOCX Parser V2 entry point. Returns Frozen ParsedDocument.

    Hardening guarantees:
      - Document-order interleaving (§15)
      - Frozen Contract invariants (caller should .validate())
      - NumberingResolver native evidence > semantic fallback (§7)
      - Unified inline traversal w/ Revision CURRENT policy (§8-9)
      - Table 2D w/ merged/nested diagnostics (§11-13)
      - Heading policy: native > outlineLvl > legal chapter regex (§14)
      - Duplicate title: detect + mark only (§16, not Cleaner)
      - Provenance.page = None (NEVER fake pages, §13)
      - Deterministic (same file → same output)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DOCX not found: {p}")
    source_file = str(p.resolve())
    file_bytes = p.read_bytes()
    content_hash = hashlib.md5(file_bytes).hexdigest()
    file_size = len(file_bytes)

    document = Document(str(p))
    doc_zip = zipfile.ZipFile(p)

    # --- 1. Pre-parse inventories ---
    doc_xml_root = ET.fromstring(doc_zip.read("word/document.xml"))
    inventories = _collect_inventories(doc_xml_root, doc_zip, document)

    # --- 2. Numbering Resolver ---
    numbering = NumberingResolver(doc_zip)
    inventories["numbering_inventory"] = {
        "has_numbering_part": numbering.has_numbering_part,
        "abstract_num_count": len(numbering.abstract_nums),
        "num_count": len(numbering.nums),
    }
    if numbering.warning_messages:
        inventories["numbering_inventory"]["warnings"] = list(numbering.warning_messages)

    # --- 3. Document-order body traversal (§15) ---
    blocks: List[Block] = []
    order = 0
    total_inserted_rev_count = 0
    total_deleted_rev_count = 0
    total_inserted_rev_chars = 0
    total_deleted_rev_chars = 0
    paragraphs_with_hyperlinks = 0
    table_index = 0
    section_index = 0  # approximate (OOXML sectPr is interleaved, simplified here)

    def _make_provenance(order_index: int) -> Provenance:
        return Provenance(
            source_file=source_file,
            page=None,
            block_order=order_index,
        )

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            p_xml = item._p
            inline = extract_inline_content(p_xml)
            trimmed = inline.text.strip()
            # Accumulate global diagnostics
            total_inserted_rev_count += inline.inserted_revision_count
            total_deleted_rev_count += inline.deleted_revision_count
            total_inserted_rev_chars += inline.inserted_revision_chars
            total_deleted_rev_chars += inline.deleted_revision_chars
            if inline.hyperlink_count:
                paragraphs_with_hyperlinks += 1

            # Skip empty whitespace-only paragraphs (don't enter blocks list at all — Cleaner V2 responsibility but saves downstream work, §29 still OK as we're not semantic cleaning)
            if not trimmed and not inline.has_revision_marks and inline.hyperlink_count == 0:
                continue

            num_info = numbering.resolve(p_xml)
            btype, level, extra_meta = _classify_paragraph(item, p_xml, trimmed, num_info)
            # Merge revision / hyperlink diagnostics into block metadata (allowed open dict)
            block_meta: Dict[str, Any] = {**extra_meta}
            block_meta["section_index"] = section_index
            if inline.has_revision_marks:
                block_meta["revision_status"] = "CURRENT_with_marks"
                block_meta["inserted_revision_count"] = inline.inserted_revision_count
                block_meta["deleted_revision_count_detected"] = inline.deleted_revision_count
                block_meta["inserted_revision_chars"] = inline.inserted_revision_chars
                block_meta["deleted_revision_chars_detected"] = inline.deleted_revision_chars
            if inline.hyperlink_count:
                block_meta["hyperlink_count"] = inline.hyperlink_count
                # Keep only text (NOT urls) in block metadata to avoid schema drift
                block_meta["hyperlink_texts"] = [h[0] for h in inline.hyperlinks]
            if num_info.is_list and "num_id" not in block_meta:
                # redundant defense-in-depth
                block_meta["numbering_source"] = "native_numbering_xml"
                block_meta["list_level"] = num_info.list_level

            block_id = Block.make_block_id(content_hash, order)
            block = Block(
                block_id=block_id,
                type=btype,
                text=trimmed,
                order=order,
                level=level,
                table_data=None,
                page=None,
                metadata=block_meta,
            )
            block.attach_provenance(source_file)
            blocks.append(block)
            order += 1

        elif isinstance(item, Table):
            table_data, diag = _convert_docx_table(item)
            total_inserted_rev_count += 0  # table counts are per-cell in diag already
            table_meta: Dict[str, Any] = {
                "table_index": table_index,
                "table_rows": len(table_data.rows),
                "table_cols": len(table_data.headers),
                "has_merged_cells": (diag["merged_cells_count"] + diag["vmerge_continue_count"]) > 0,
                "has_nested_table": diag["nested_tables_cells"] > 0,
                "table_diagnostics": diag,
                "section_index": section_index,
            }
            block_id = Block.make_block_id(content_hash, order)
            block = Block(
                block_id=block_id,
                type=BlockType.TABLE,
                text="",  # Frozen Contract auto-fills: text = table_data.to_markdown() in __post_init__
                order=order,
                level=None,
                table_data=table_data,
                page=None,
                metadata=table_meta,
            )
            block.attach_provenance(source_file)
            blocks.append(block)
            order += 1
            table_index += 1

    # --- 4. Post-process (title + duplicate detection) ---
    title, duplicate, title_notes = _extract_title_and_duplicate(document, blocks, source_file)

    # --- 5. Document metadata ---
    cp = document.core_properties
    native_meta: Dict[str, Any] = {
        "source_file": source_file,
        "file_size_bytes": file_size,
        "docx_md5": content_hash,
        "duplicate_title_detected": duplicate,
        "title_notes": title_notes,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        # Native DOCX Core Properties (§21 A) - only non-empty (strictly no guessing)
        "native_core": {},
    }
    core_fields = [
        "title", "author", "subject", "keywords", "category", "comments", "content_status",
        "created", "modified", "last_modified_by", "last_printed", "revision",
        "identifier", "language", "version",
    ]
    for k in core_fields:
        v = getattr(cp, k, None)
        if isinstance(v, datetime):
            native_meta["native_core"][k] = v.isoformat()
        elif v is None:
            continue
        elif isinstance(v, str) and not v.strip():
            continue
        else:
            native_meta["native_core"][k] = v

    # Merge inventories into open metadata (Contract allows; observability / diagnostics)
    for k, v in inventories.items():
        native_meta[k] = v
    native_meta["total_inserted_revision_count"] = total_inserted_rev_count
    native_meta["total_deleted_revision_count"] = total_deleted_rev_count
    native_meta["total_inserted_revision_chars"] = total_inserted_rev_chars
    native_meta["total_deleted_revision_chars_detected"] = total_deleted_rev_chars
    native_meta["paragraphs_with_hyperlinks_count"] = paragraphs_with_hyperlinks
    native_meta["table_count"] = table_index

    # --- 6. Warnings (Contract: open list[str]) ---
    warnings: List[str] = []
    if duplicate:
        warnings.append("duplicate_title_detected: document metadata title appears again within body blocks")
    if numbering.warning_messages:
        warnings.extend(numbering.warning_messages)
    if total_deleted_rev_count > 0:
        warnings.append(
            f"revision_deleted_content_detected: CURRENT policy excludes {total_deleted_rev_count} "
            f"w:del revision elements from evidence text; see total_deleted_revision_chars_detected in metadata; "
            f"AUDIT/REVIEW mode would include them."
        )
    if diag_inv := inventories.get("txbx_inventory"):
        if diag_inv.get("w_txbx_content_count", 0) > 0:
            warnings.append(
                f"textbox_content_present: word document contains {diag_inv['w_txbx_content_count']} "
                f"w:txbxContent (text box / drawing inline text); current parser inventories only; "
                f"content not merged into evidence blocks. No evidence of critical loss in current fixtures."
            )
    if diag_inv := inventories.get("package_part_presence"):
        if diag_inv.get("has_word_footnotes_xml") or diag_inv.get("has_word_endnotes_xml") or diag_inv.get("has_word_comments_xml"):
            warnings.append(
                "supplementary_parts_present: footnotes/endnotes/comments parts exist; "
                "current parser does not extract them. NOT EXERCISED in this Production Core Hardening V1."
            )
    nested_total = sum(
        b.metadata.get("table_diagnostics", {}).get("nested_tables_cells", 0)
        for b in blocks if b.type is BlockType.TABLE and b.metadata
    )
    if nested_total:
        warnings.append(
            f"nested_table_cells: {nested_total} table cells contain nested tables; "
            "conservatively flatten into parent cell text + diagnostic, do NOT recursively produce TableData "
            "(Frozen Contract TableData lacks recursive sub-table field)."
        )

    parsed = ParsedDocument(
        document_id=content_hash,
        source_file=source_file,
        title=title,
        metadata=native_meta,
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        warnings=warnings,
    )
    parsed.validate()
    return parsed
