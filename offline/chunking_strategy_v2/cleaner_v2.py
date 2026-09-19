"""
Cleaner V2 — Block-Aware Core (ISOLATED).

=================================================================
SCOPE (Section 3 / 4 of task spec):
  Answer only: "KEEP / DROP / MOVE_METADATA / DEDUPLICATE / REVIEW?"

FORBIDDEN:
  - NO metadata canonical normalization
  - NO quality gate implementation
  - NO production ingestion
  - NO re-ingestion
  - NO modifications to parsed_document_v2.py, html_parser_v2.py,
    pdf_parser_v2.py, docx_parser_v2.py, LegacyAdapter, Chunker
  - NO import / call of MetadataNormalizer V1
  - NO LLM dependency
  - NO new dependencies

Actions (FROZEN enum, Section 2 / 6):
  KEEP, DROP, MOVE_METADATA, DEDUPLICATE, REVIEW
=================================================================
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import sys
from pathlib import Path

# =============================================================================
# Resolve import paths — support both f:\DataBase\trae_work\RAG layout
# AND D:\xiaoyi\Legal_System layout for ParsedDocument V2 Contract.
# PRIORITY: the Frozen Contract file in Legal_System/modules.
# =============================================================================

_LEGAL_SYS = Path(r"D:\xiaoyi\Legal_System")
_CONTRACT_VIA_LEGAL = _LEGAL_SYS / "modules" / "ingestion" / "parsed_document_v2.py"
_CONTRACT_VIA_WORK = Path(r"f:\DataBase\trae_work\RAG\parsed_document_v2.py")

if _CONTRACT_VIA_LEGAL.is_file():
    _sys_path_for_contract = str(_LEGAL_SYS)
elif _CONTRACT_VIA_WORK.is_file():
    _sys_path_for_contract = str(Path(r"f:\DataBase\trae_work\RAG"))
else:
    # Assume working directory / Legal_System importable as-is
    _sys_path_for_contract = None

if _sys_path_for_contract and _sys_path_for_contract not in sys.path:
    sys.path.insert(0, _sys_path_for_contract)

# --- ONLY import Frozen ParsedDocument V2 Contract ---
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

# =============================================================================
# Action Enum (FROZEN — 5 actions only, Section 2)
# =============================================================================

class CleanerAction(str, Enum):
    KEEP = "KEEP"
    DROP = "DROP"
    MOVE_METADATA = "MOVE_METADATA"
    DEDUPLICATE = "DEDUPLICATE"
    REVIEW = "REVIEW"


# =============================================================================
# Confidence scale (deterministic, not probabilistic)
# =============================================================================

class Confidence(str, Enum):
    HIGH = "HIGH"       # Near-zero risk of legal content loss
    MEDIUM = "MEDIUM"   # Strong evidence but human verification preferred
    LOW = "LOW"         # Weak signals — typically REVIEW instead


# =============================================================================
# Audit Trail: per-block Decision record
# =============================================================================

@dataclass
class BlockDecision:
    """Immutable record of one block's decision (Section 5 / 20 — Audit Trail)."""
    original_block_order: int
    block_type: str
    text_preview: str
    action: CleanerAction
    rule_id: str
    reason: str
    confidence: Confidence

    def to_row(self) -> dict[str, Any]:
        return {
            "original_block_order": self.original_block_order,
            "block_type": self.block_type,
            "action": self.action.value,
            "rule_id": self.rule_id,
            "confidence": self.confidence.value,
            "reason": self.reason,
            "text_preview": self.text_preview,
        }


# =============================================================================
# Metadata candidate (for Normalizer downstream, Section 10 / 18)
# =============================================================================

@dataclass
class MetadataCandidate:
    """
    Cleaner's output to downstream MetadataNormalizer.

    Cleaner ONLY records which block(s) look like metadata key/value pairs.
    It does NOT canonicalize field names, parse dates, resolve conflicts.
    """
    key_block_orders: list[int]
    value_block_orders: list[int]
    key_texts: list[str]
    value_texts: list[str]
    pattern: str  # e.g. "adjacent_key_value_pair", "single_kv_line"
    rule_id: str


# =============================================================================
# Duplicate group (audit for DEDUPLICATE action, Section 11 / 12)
# =============================================================================

@dataclass
class DuplicateGroup:
    block_orders: list[int]
    normalized_text: str
    reason: str
    rule_id: str
    kept_order: int  # which one is KEEP'd (the rest are DEDUPLICATE'd)


# =============================================================================
# CleaningResult — Structured audit output (OUTPUT CONTRACT, Section 5)
# =============================================================================

@dataclass
class CleaningResult:
    cleaned_document: ParsedDocument
    decisions: list[BlockDecision]
    dropped_blocks: list[BlockDecision] = field(default_factory=list)
    move_metadata_candidates: list[MetadataCandidate] = field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    review_blocks: list[BlockDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- aggregate stats ---
    @property
    def total_blocks(self) -> int:
        return len(self.decisions)

    @property
    def kept_count(self) -> int:
        return sum(1 for d in self.decisions if d.action is CleanerAction.KEEP)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped_blocks)

    @property
    def move_meta_count(self) -> int:
        return sum(1 for d in self.decisions if d.action is CleanerAction.MOVE_METADATA)

    @property
    def dedup_count(self) -> int:
        return sum(1 for d in self.decisions if d.action is CleanerAction.DEDUPLICATE)

    @property
    def review_count(self) -> int:
        return len(self.review_blocks)


# =============================================================================
# Known metadata key dictionary (Chinese + English, used only for MOVE_METADATA
# candidate detection — NOT for canonicalization)
# =============================================================================

KNOWN_METADATA_KEYS = frozenset([
    # Core legal doc metadata (common in HTML-derived gov pages)
    "标题", "发文机关", "发布日期", "发布时间", "成文日期", "发文字号",
    "索引号", "主题分类", "来源", "文章来源", "信息来源",
    "发布机构", "发文单位", "文    号", "标　　题", "索 引 号",
    "主 题 词", "主题词",
    # English variants (for bilingual docs)
    "Title", "Source", "Date", "Issued by", "Document Number",
])

# Regex: "Any Chinese chars + ：/:" (a label line ending in Chinese/English colon)
METADATA_KEY_LABEL_RE = re.compile(
    r"^\s*[\u4e00-\u9fffA-Za-z0-9（）()\s·・．.]{1,30}[：:]\s*$",
    re.UNICODE,
)

# =============================================================================
# UI / Boilerplate vocabulary (DROP signals, Section 9)
# Must be combined with: short length, isolated, known pattern context
# =============================================================================

# Exact short tokens that, when they appear ISOLATED (short block, no context),
# are nearly-certainly UI rather than legal substance.
STANDALONE_UI_TOKENS = frozenset([
    "关闭", "关闭窗口",
    "打印", "打印本页",
    "分享", "分享到", "一键分享",
    "返回顶部", "回到顶部",
    "纠错", "我要纠错",
    "收藏", "收藏本文",
    "扫一扫", "扫一扫在手机打开当前页", "扫一扫在手机上打开当前页",
    "二维码", "二维码生成专用",
    "上一篇", "下一篇",
    "责任编辑",
])

STANDALONE_UI_BRACKETED_RE = re.compile(
    r"^\s*[【\[\(]\s*(打印|关闭|纠错|分享|收藏|返回顶部)\s*[】\]\)]\s*$",
    re.UNICODE,
)

# High-confidence CMS markers (exact match or strong prefix match, Section 9)
CMS_MARKER_EXACT = frozenset([
    "JiaThis Button BEGIN",
    "JiaThis Button END",
    "Baidu Button BEGIN",
    "Baidu Button END",
    "Produced By CMS",
    "footer", "header",
])

CMS_MARKER_PREFIXES = (
    "start component",
    "end component",
    "start component HTML组件",
    "end component HTML组件",
)

CMS_MARKER_REGEXES = [
    re.compile(r"^\s*(?:start|end)\s+component(?:\s+.*)?$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*JiaThis\s+Button\s*(?:BEGIN|END)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Baidu\s+Button\s*(?:BEGIN|END)?\s*$", re.IGNORECASE),
    re.compile(r"京ICP备", re.UNICODE),
    re.compile(r"京公网安备", re.UNICODE),
    re.compile(r"版权所有", re.UNICODE),
    re.compile(r"网站群内容管理系统", re.UNICODE),
]


# =============================================================================
# V2.1 Patch constants: page-furniture, vertical banner, document identifier
# =============================================================================

# V2.1: Standalone pure Arabic or Chinese low-val page number tokens, ONLY to
# be used when format=PDF + provenance supports page region / context is short
# isolated. Explicitly PROTECT patterns like 第3条 / 3.1 / 第3章 / 3个月 etc.
# via negative lookahead + additional context checks in-code.
PDF_STANDALONE_PURE_DIGIT_RE = re.compile(r"^\s*\d{1,6}\s*$")

# V2.1: Vertical banner fragment detection (FIX-006 style):
# N lines of single rare/title Chinese characters stacked vertically.
# Heuristic: stripped text after removing newlines = N single CJK chars,
# len >= 4, no punctuation, no common legal-opening signals like 第,之,为,本.
PDF_VERT_BANNER_RE = re.compile(
    r"^[\u4e00-\u9fff](\s*\n\s*[\u4e00-\u9fff]){3,}$",
    re.UNICODE,
)
PDF_VERT_BANNER_BLACKLIST_PREFIX_CHARS = frozenset(
    ["第", "本", "依", "为", "对", "若", "如", "与", "及", "或", "根", "违", "合",
     "民", "法", "刑", "商", "行", "行", "契", "订", "签", "债", "物", "担",
     "公", "仲", "诉", "调", "解", "履"]
)

# V2.1: Document identifier conservative pattern (national doc numbers,
# contract text numbers, 文, etc.). Examples:
#   GF—2025—0151, SF-2023-0042, 京发改〔2024〕123号, 法释〔2020〕15号
DOC_IDENTIFIER_RE = re.compile(
    r"^\s*"
    r"("
    r"[A-Z]{2,5}[—\-]\d{3,5}[—\-]\d{3,6}"
    r"|"
    r"[\u4e00-\u9fffA-Za-z]{2,10}[〔\[\(]\d{3,5}[〕\]\)]\d{1,6}号"
    r"|"
    r"[A-Z]{2,5}\d{3,6}(?:[—\-][A-Z0-9]{1,8})?"
    r")\s*$",
    re.UNICODE,
)
DOC_IDENTIFIER_CONTEXT_KEY_RE = re.compile(
    r"(示范文本|合同编号|编号|文\s*号|发文字号|公文号|文号)",
    re.UNICODE,
)


# Pure separator (no semantic content whatsoever)
PURE_SEPARATOR_RE = re.compile(r"^[\s_＿\-—–・─=~．.•·\|]{3,}$", re.UNICODE)

# Standalone page number (NOT legal numbering like 第三条)
# NOTE: pure-digit page tokens are handled by separate R_DROP_PDF_PAGE_FURNITURE
# rule in _apply_drop_rules because they require PDF-format + context validation
# to avoid colliding with article numbers.
STANDALONE_PAGE_RE = [
    re.compile(r"^\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*第\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*[—\-–]\s*\d+\s*[—\-–]\s*$", re.UNICODE),
    re.compile(r"^\s*Page\s*\d+(?:\s*of\s*\d+)?\s*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^\s*第\s*[零〇一二三四五六七八九十百千]+\s*页\s*(?:/\s*共\s*[零〇一二三四五六七八九十百千]+\s*页)?\s*$", re.UNICODE),
]

# Navigation prefix: "上一篇: XXX" / "下一篇：YYY" / "责任编辑：XXX"
NAV_PREFIX_RE = re.compile(
    r"^\s*(上一篇|下一篇|责任编辑|分享到|分享至|一键分享)\s*[:：]",
    re.UNICODE,
)

# Pure URL standalone line (navigation chrome, not legal citation)
PURE_URL_STANDALONE_RE = re.compile(
    r"^\s*https?://[^\s]{4,}\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Legal paragraph protection signal: if a paragraph looks like substantive
# legal text even if it contains a UI word, we MUST keep it.
LEGAL_PARAGRAPH_SIGNAL_RE = re.compile(
    r"(第[一二三四五六七八九十百千零〇\d]+[章节条款项条])|"
    r"(根据《.+》)|"
    r"(当事人|合同|法律|法规|规定|办法|条例|应当|不得|必须|可以|有权|义务|责任)",
    re.UNICODE,
)

# =============================================================================
# Helpers
# =============================================================================

def _preview(text: str, limit: int = 80) -> str:
    s = text.strip().replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _norm_for_dup(text: str) -> str:
    """Normalization for duplicate detection: strip + ws collapse + case-lower."""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    return t


def _is_short_block(text: str, max_chars: int = 20) -> bool:
    return len(_normalize_ws(text)) <= max_chars


# =============================================================================
# BlockAwareCleaner — Core implementation
# =============================================================================

class BlockAwareCleaner:
    """
    Block-level cleaner operating on ParsedDocument V2.

    Principles:
      - KEEP by default (Section 8)
      - DROP only with HIGH confidence multi-signal evidence (Section 9)
      - MOVE_METADATA candidate detection only; NO canonicalization (Section 10)
      - DEDUPLICATE only local/adjacent with strong evidence (Section 11/12)
      - REVIEW ambiguous, UNKNOWN, mixed blocks (Section 13)
      - Format-independent first (Section 14)
    """

    CLEANER_VERSION = "v2.1.0-targeted"

    # ------------------------------------------------------------------ main
    def clean(self, doc: ParsedDocument) -> CleaningResult:
        """
        Clean a ParsedDocument V2.

        Guarantees:
          - Original doc is NEVER mutated.
          - Returns new, validated ParsedDocument.
          - Full audit trail (one BlockDecision per original block).
        """
        # --- IMMUTABLE INPUT GUARANTEE ---
        # We'll work with block text/type references; any new blocks are deep copies.
        original_blocks_sha = self._blocks_sha256(doc.blocks)

        n = len(doc.blocks)
        decisions: list[BlockDecision] = [None] * n  # type: ignore[list-item]
        warnings: list[str] = []

        # ----------------------------------------------------------
        # Pass 1: Initial default = KEEP for all blocks
        # ----------------------------------------------------------
        for i, blk in enumerate(doc.blocks):
            decisions[i] = BlockDecision(
                original_block_order=i,
                block_type=blk.type.value,
                text_preview=_preview(blk.text),
                action=CleanerAction.KEEP,
                rule_id="R_KEEP_DEFAULT",
                reason="Default KEEP per conservative policy (Section 8)",
                confidence=Confidence.HIGH,
            )

        # ----------------------------------------------------------
        # Pass 2: DROP detection (multi-signal, HIGH confidence only)
        # ----------------------------------------------------------
        self._apply_drop_rules(doc, decisions, warnings)

        # ----------------------------------------------------------
        # Pass 3: MOVE_METADATA candidate detection
        # ----------------------------------------------------------
        meta_candidates = self._apply_move_metadata_rules(doc, decisions, warnings)

        # ----------------------------------------------------------
        # HARD INVARIANT (Section 5): MOVE_METADATA MUST carry a preserved
        # payload. If a block is classified MOVE_METADATA but no staged
        # key/value pairs exist for it, fail-closed to REVIEW instead of
        # silently discarding information.
        # ----------------------------------------------------------
        # Build set of block indices actually staged by meta_candidates.
        _staged_move_indices: set[int] = set()
        for cand in meta_candidates:
            for _idx_list in (cand.key_block_orders, cand.value_block_orders):
                for _idx in _idx_list:
                    if isinstance(_idx, int) and 0 <= _idx < n:
                        _staged_move_indices.add(_idx)
        for _i in range(n):
            if decisions[_i].action is CleanerAction.MOVE_METADATA:
                if _i not in _staged_move_indices:
                    # Fallback: check if any candidate references this block
                    # anywhere OR carries non-empty text payload.
                    _any_ref = False
                    for cand in meta_candidates:
                        _payloads: list = []
                        _payloads.extend(getattr(cand, "key_texts", []) or [])
                        _payloads.extend(getattr(cand, "value_texts", []) or [])
                        _payloads.append(getattr(cand, "raw_block_text", None))
                        if _i in getattr(cand, "key_block_orders", []) or _i in getattr(cand, "value_block_orders", []):
                            _any_ref = True
                            break
                        if any(_pl for _pl in _payloads if isinstance(_pl, str) and _pl.strip()):
                            _any_ref = True
                            break
                    if not _any_ref:
                        _orig_rule = decisions[_i].rule_id
                        _orig_reason = decisions[_i].reason or ""
                        decisions[_i] = self._mk_dec(
                            _i, doc.blocks[_i], CleanerAction.REVIEW,
                            rule_id="R_REVIEW_METADATA_EMPTY_PAYLOAD",
                            reason=(
                                "Safety fence: block was MOVE_METADATA via "
                                f"{_orig_rule!r} but no metadata payload was "
                                "staged. Failing closed to REVIEW to avoid "
                                "information loss. Previous reason: "
                                f"{_orig_reason[:140]!r}"
                            ),
                            confidence=Confidence.HIGH,
                        )
                        warnings.append(
                            "MOVE_METADATA-empty-payload-fence-upgraded-to-REVIEW: "
                            f"block {_i} rule={_orig_rule}"
                        )
                        continue

        # ----------------------------------------------------------
        # Pass 4: DEDUPLICATE detection (local/adjacent only)
        # ----------------------------------------------------------
        dup_groups = self._apply_dedupe_rules(doc, decisions, warnings)

        # ----------------------------------------------------------
        # Pass 5: REVIEW detection (UNKNOWN + ambiguous + mixed)
        # ----------------------------------------------------------
        self._apply_review_rules(doc, decisions, warnings)

        # ----------------------------------------------------------
        # Pass 6: Legal Content Retention Safety Net (Section 7)
        #   Any DROP decision with legal-paragraph-like signal → REVIEW
        # ----------------------------------------------------------
        self._apply_legal_safety_net(doc, decisions, warnings)

        # ----------------------------------------------------------
        # Build: new ParsedDocument
        # ----------------------------------------------------------
        cleaned_doc = self._build_cleaned_document(
            doc, decisions, dup_groups, meta_candidates
        )

        # --- Validate: original NOT mutated ---
        assert self._blocks_sha256(doc.blocks) == original_blocks_sha, \
            "BUG: original ParsedDocument.blocks was mutated (Section 23)"

        # --- Validate new contract ---
        cleaned_doc.validate()

        # ----------------------------------------------------------
        # Aggregate audit sub-lists
        # ----------------------------------------------------------
        dropped_blocks = [d for d in decisions if d.action is CleanerAction.DROP]
        review_blocks = [d for d in decisions if d.action is CleanerAction.REVIEW]

        result = CleaningResult(
            cleaned_document=cleaned_doc,
            decisions=decisions,
            dropped_blocks=dropped_blocks,
            move_metadata_candidates=meta_candidates,
            duplicate_groups=dup_groups,
            review_blocks=review_blocks,
            warnings=warnings,
        )
        return result



    # ------------------------------------------------------------------
    # V2.1 helpers: safe format / attribute lookups for Frozen V2 contract.
    # The ParsedDocument V2 frozen contract does NOT define a "format" field.
    # We derive it from source_file suffix + parser_name as the ONLY allowed
    # general route.
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_format(doc) -> str:
        sf = (getattr(doc, "source_file", None) or "").lower()
        pn = (getattr(doc, "parser_name", None) or "").lower()
        if sf.endswith(".pdf") or "pdf" in pn.split("_"):
            return "pdf"
        if sf.endswith(".docx") or "docx" in pn.split("_"):
            return "docx"
        if sf.endswith((".html", ".htm")) or "html" in pn.split("_"):
            return "html"
        if sf.endswith(".doc"):
            return "docx"
        return "unknown"

    # =================================================================
    # Rule applications
    # =================================================================

    def _apply_drop_rules(
        self,
        doc: ParsedDocument,
        decisions: list[BlockDecision],
        warnings: list[str],
    ) -> None:
        """Section 9 DROP rules — HIGH confidence multi-signal only."""
        blocks = doc.blocks
        n = len(blocks)
        for i, blk in enumerate(blocks):
            if decisions[i].action is not CleanerAction.KEEP:
                continue

            # ================================================================
            # CRITICAL GUARDS (Section 13):
            #  - UNKNOWN blocks NEVER go through DROP (must be REVIEW first)
            #  - TABLE blocks NEVER go through text-level DROP rules
            #    (Parser already classified table; table ambiguity → REVIEW only)
            # ================================================================
            if blk.type is BlockType.UNKNOWN:
                continue
            if blk.type is BlockType.TABLE:
                continue

            text = blk.text
            stripped = text.strip()
            norm = _normalize_ws(text)

            # --- R_DROP_CMS_EXACT: exact CMS marker match ---
            if stripped in CMS_MARKER_EXACT:
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_CMS_EXACT",
                    reason=f"Exact match of known CMS marker '{stripped}'",
                    confidence=Confidence.HIGH,
                )
                continue

            # --- R_DROP_CMS_PREFIX: start/end component ---
            if any(stripped.startswith(p) for p in CMS_MARKER_PREFIXES):
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_CMS_PREFIX",
                    reason=f"CMS component marker prefix (start/end component)",
                    confidence=Confidence.HIGH,
                )
                continue

            # --- R_DROP_CMS_REGEX: high-confidence regex hits ---
            if any(rx.search(stripped) for rx in CMS_MARKER_REGEXES):
                # Must be a short-ish block; don't drop long paragraphs containing
                # these strings as substring (e.g. legal text about 京ICP备)
                if _is_short_block(text, 60):
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.DROP,
                        rule_id="R_DROP_CMS_REGEX",
                        reason="Short block matches CMS/ICP copyright regex (Section 9)",
                        confidence=Confidence.HIGH,
                    )
                    continue

            # --- R_DROP_PURE_SEPARATOR: no semantic content ---
            if stripped and PURE_SEPARATOR_RE.match(stripped):
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_PURE_SEPARATOR",
                    reason="Pure separator line (no semantic content)",
                    confidence=Confidence.HIGH,
                )
                continue

            # --- R_DROP_PAGE_NUMBER: standalone page number (NOT legal num) ---
            if any(rx.match(stripped) for rx in STANDALONE_PAGE_RE):
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_PAGE_NUMBER",
                    reason="Standalone page number line (Section 9)",
                    confidence=Confidence.HIGH,
                )
                continue

            # --- R_DROP_PDF_PAGE_FURNITURE: pure page-number in PDF with strong context ---
            # Must be extremely conservative to protect 第X条 / list numbering.
            if self._doc_format(doc) == "pdf" and blk.type in (BlockType.PARAGRAPH, BlockType.METADATA):
                if _is_short_block(text, 8) and PDF_STANDALONE_PURE_DIGIT_RE.match(stripped):
                    # Positive furniture signals:
                    #   1) at least one adjacent block (prev or next) is also a
                    #      pure-digit short block OR known page-furniture token
                    #   2) block has no legal signals (no 第X条 / 应当 / 合同 etc.)
                    #   3) block is NOT inside a paragraph that continues before/after
                    #      via "prev block ends with CN/EN punctuation" — we skip if
                    #      prev block ends with any of ,，。；;:：)）]】（( [（ because
                    #      pure digit after that may be clause list number, not page.
                    def _furniture_neighbor_digit(j: int) -> bool:
                        if j < 0 or j >= n:
                            return False
                        nb = blocks[j]
                        if nb.type not in (BlockType.PARAGRAPH, BlockType.METADATA):
                            return False
                        ns = nb.text.strip()
                        if not _is_short_block(nb.text, 12):
                            return False
                        if PDF_STANDALONE_PURE_DIGIT_RE.match(ns):
                            return True
                        if ns in ("—", "-", "–", "·", "•", "·"):
                            return True
                        return False
                    prev_text = blocks[i - 1].text if i > 0 else ""
                    prev_stripped_end = prev_text.rstrip()[-1:] if prev_text.rstrip() else ""
                    ends_with_continuation = prev_stripped_end in (
                        ",", "，", ".", "。", ";", "；", ":", "：", ")", "）", "]", "】", "、",
                    )
                    # Build doc-scale furniture band count FIRST so the
                    # punctuation-continuation veto can be overruled when the
                    # whole document exhibits a clear page-number furniture
                    # band (e.g. 10-page PDF has 10 pure-digit short blocks).
                    band_count_hint = 0
                    for jb in blocks:
                        if jb is blk:
                            continue
                        if jb.type in (BlockType.PARAGRAPH, BlockType.METADATA) and _is_short_block(jb.text, 8) and PDF_STANDALONE_PURE_DIGIT_RE.match(jb.text.strip()):
                            band_count_hint += 1
                    strong_band = band_count_hint >= 3
                    prev_has_legal = bool(LEGAL_PARAGRAPH_SIGNAL_RE.search(prev_text)) if prev_text else False
                    # Only honour the prev-punctuation continuation veto if:
                    #  - furniture band is NOT strong (few digits, ambiguous)
                    #  - OR the previous block actually contains legal prose
                    #    signals (proves we're inside a clause/list, not just
                    #    a paragraph footer right before a page-break number).
                    continuation_veto = bool(ends_with_continuation) and (
                        (not strong_band) or prev_has_legal
                    )
                    if (not LEGAL_PARAGRAPH_SIGNAL_RE.search(text)) and (not continuation_veto):
                        neigh_ok = _furniture_neighbor_digit(i - 1) or _furniture_neighbor_digit(i + 1)
                        # Allow doc-scale band (3+ other pure-digit short blocks)
                        # as sufficient structural evidence even when the
                        # immediate neighbours are not furniture-like.
                        if not neigh_ok:
                            neigh_ok = band_count_hint >= 3
                        if neigh_ok:
                            decisions[i] = self._mk_dec(
                                i, blk, CleanerAction.DROP,
                                rule_id="R_DROP_PDF_PAGE_FURNITURE",
                                reason=(
                                    "PDF context: isolated pure-digit short block surrounded "
                                    "by other page-furniture digits / separators (page number; "
                                    f"furniture_band={band_count_hint + 1})"
                                ),
                                confidence=Confidence.HIGH,
                            )
                            continue

            # --- R_DROP_PDF_VERT_BANNER: vertical publication header fragments ---
            if self._doc_format(doc) == "pdf" and blk.type is BlockType.PARAGRAPH:
                if _is_short_block(text, 40) and PDF_VERT_BANNER_RE.match(text):
                    chars_joined = re.sub(r"\s+", "", text)
                    if (
                        len(chars_joined) >= 4
                        and not LEGAL_PARAGRAPH_SIGNAL_RE.search(text)
                        and chars_joined[0] not in PDF_VERT_BANNER_BLACKLIST_PREFIX_CHARS
                    ):
                        decisions[i] = self._mk_dec(
                            i, blk, CleanerAction.DROP,
                            rule_id="R_DROP_PDF_VERT_BANNER",
                            reason=(
                                "PDF vertical-banner character fragment (repeated "
                                "per-page publication header / 公报 chars)"
                            ),
                            confidence=Confidence.MEDIUM,
                        )
                        continue

            # --- R_DROP_NAV_PREFIX: "上一篇:" / "下一篇：" / "责任编辑：" ---
            if NAV_PREFIX_RE.match(stripped):
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_NAV_PREFIX",
                    reason="Navigation residue prefix (上一篇/下一篇/责任编辑)",
                    confidence=Confidence.HIGH,
                )
                continue

            # --- R_DROP_UI_SHORT: short isolated UI token ---
            # Signals:
            #   a) stripped in STANDALONE_UI_TOKENS (exact set)
            #   b) OR STANDALONE_UI_BRACKETED_RE match (【打印】 etc.)
            #   c) AND block must be short (<= 15 chars ws-normalized)
            #   d) AND no legal paragraph signal present (extra safe)
            is_ui_exact = stripped in STANDALONE_UI_TOKENS
            is_ui_bracketed = bool(STANDALONE_UI_BRACKETED_RE.match(stripped))
            if (is_ui_exact or is_ui_bracketed) and _is_short_block(text, 15):
                if not LEGAL_PARAGRAPH_SIGNAL_RE.search(text):
                    reason = (
                        f"Short isolated UI token '{stripped}'"
                        + (" (bracketed form)" if is_ui_bracketed else "")
                    )
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.DROP,
                        rule_id="R_DROP_UI_SHORT",
                        reason=reason,
                        confidence=Confidence.HIGH,
                    )
                    continue

            # --- R_DROP_PURE_URL: standalone pure URL line with no context ---
            if PURE_URL_STANDALONE_RE.match(stripped) and _is_short_block(text, 200):
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.DROP,
                    rule_id="R_DROP_PURE_URL",
                    reason="Standalone URL line with no legal context (navigation chrome)",
                    confidence=Confidence.MEDIUM,
                )
                continue

    def _apply_move_metadata_rules(
        self,
        doc: ParsedDocument,
        decisions: list[BlockDecision],
        warnings: list[str],
    ) -> list[MetadataCandidate]:
        """
        Section 10 / 18 MOVE_METADATA detection.

        Produces candidates for downstream MetadataNormalizer V1.
        Does NOT canonicalize field names / values itself.

        Patterns detected:
          A) Adjacent pair: Block i = "X：" (ends with colon, known key label)
                           Block i+1 = "value"
          B) Single line:   "X：value" within one PARAGRAPH block
          C) Top-of-doc metadata-like region (order < 15, PARAGRAPH only)
        """
        candidates: list[MetadataCandidate] = []
        blocks = doc.blocks
        n = len(blocks)
        i = 0

        while i < n:
            blk = blocks[i]
            text = blk.text
            stripped = text.strip()
            dec = decisions[i]

            # Only consider PARAGRAPH blocks for metadata candidates
            # (Parser V2 emits some as PARAGRAPH; some as METADATA type — METADATA is already implicit)
            if blk.type in (BlockType.PARAGRAPH, BlockType.METADATA):
                pass
            else:
                i += 1
                continue

            # Skip blocks already DROP'd (they're not useful metadata candidates)
            if dec.action is CleanerAction.DROP:
                i += 1
                continue

            # --- Pattern A: adjacent label + value pair ---
            # IMPORTANT: only apply to PARAGRAPH blocks.
            # METADATA-type blocks must NOT trigger adjacent pairing; they are handled
            # exclusively by Pattern C (R_META_TYPE_MARKER). This prevents false pairing
            # like: _meta("文章来源：法规局") followed by _p("第一条 正文...").
            pattern_a_eligible = blk.type is BlockType.PARAGRAPH
            if pattern_a_eligible and i + 1 < n:
                label_match = METADATA_KEY_LABEL_RE.match(stripped)
                label_has_known_key = any(k in stripped for k in KNOWN_METADATA_KEYS)
                if label_match or label_has_known_key:
                    next_blk = blocks[i + 1]
                    next_stripped = next_blk.text.strip()
                    next_ok_type = next_blk.type in (BlockType.PARAGRAPH, BlockType.METADATA)
                    next_not_empty = bool(next_stripped)
                    next_not_drop = decisions[i + 1].action is not CleanerAction.DROP
                    next_not_heading = next_blk.type is not BlockType.HEADING
                    # V2.1 guard: reject clearly non-value next blocks:
                    #   pure digit page number (<=6 digits + short) / pure separator /
                    #   EM-DASH wrapped + digit page-furniture style banners.
                    next_norm_ws = _normalize_ws(next_stripped)
                    next_is_page_furniture = bool(
                        next_ok_type
                        and len(next_norm_ws) <= 8
                        and PDF_STANDALONE_PURE_DIGIT_RE.match(next_stripped)
                    )
                    next_is_all_sep = bool(PURE_SEPARATOR_RE.match(next_stripped))
                    next_is_vertical_banner = bool(
                        PDF_VERT_BANNER_RE.match(next_blk.text)
                    )
                    next_plausible_value = not (
                        next_is_page_furniture or next_is_all_sep or next_is_vertical_banner
                    )
                    # Also: next value must not look like substantive legal opening
                    # (i.e. 第X条 / 当事人...) if length >= 12
                    next_substantive = (
                        len(next_norm_ws) >= 12
                        and LEGAL_PARAGRAPH_SIGNAL_RE.search(next_blk.text)
                    )
                    next_plausible_value = next_plausible_value and not next_substantive
                    if (
                        next_ok_type
                        and next_not_empty
                        and next_not_drop
                        and next_not_heading
                        and next_plausible_value
                    ):
                        # Prevent false pairing: if "next block" looks like a heading
                        # or substantive legal text opening, it's NOT metadata value.
                        is_next_legal = LEGAL_PARAGRAPH_SIGNAL_RE.search(next_blk.text)
                        # Threshold 50 chars: any 50+ char block that also has legal signal
                        # is substantive legal body, not a metadata value.
                        is_next_too_long = len(_normalize_ws(next_blk.text)) > 50
                        if not (is_next_legal and is_next_too_long):
                            key_texts = [stripped]
                            val_texts = [next_stripped]
                            cand = MetadataCandidate(
                                key_block_orders=[i],
                                value_block_orders=[i + 1],
                                key_texts=key_texts,
                                value_texts=val_texts,
                                pattern="adjacent_key_value_pair",
                                rule_id="R_META_ADJACENT_PAIR",
                            )
                            candidates.append(cand)

                            # Mark both as MOVE_METADATA (only if still KEEP; don't override DROP/REVIEW)
                            if decisions[i].action is CleanerAction.KEEP:
                                decisions[i] = self._mk_dec(
                                    i, blk, CleanerAction.MOVE_METADATA,
                                    rule_id="R_META_ADJACENT_PAIR",
                                    reason=(
                                        "Metadata key candidate; adjacent label/value pair detected "
                                        "(Section 10, passed to MetadataNormalizer for canonicalization)"
                                    ),
                                    confidence=Confidence.MEDIUM,
                                )
                            if decisions[i + 1].action is CleanerAction.KEEP:
                                decisions[i + 1] = self._mk_dec(
                                    i + 1, next_blk, CleanerAction.MOVE_METADATA,
                                    rule_id="R_META_ADJACENT_PAIR",
                                    reason=(
                                        "Metadata value candidate; paired with preceding label block "
                                        "(Section 10, passed to MetadataNormalizer)"
                                    ),
                                    confidence=Confidence.MEDIUM,
                                )
                            i += 2  # Skip the value block we paired
                            continue

            # --- Pattern B: single-line "Key: Value" pattern ---
            # Only apply to PARAGRAPH blocks. METADATA-type blocks always go to Pattern C
            # (R_META_TYPE_MARKER) so rule_id correctly reflects Parser's explicit METADATA marking.
            pattern_b_eligible = blk.type is BlockType.PARAGRAPH
            single_kv_re = re.compile(
                r"^\s*[\u4e00-\u9fffA-Za-z0-9\s·・．.（）()]{1,20}[：:]\s*.{1,100}\s*$",
                re.UNICODE,
            )
            if pattern_b_eligible and single_kv_re.match(stripped):
                # Known key prefix makes this higher confidence
                if any(stripped.startswith(k) for k in KNOWN_METADATA_KEYS):
                    cand = MetadataCandidate(
                        key_block_orders=[i],
                        value_block_orders=[i],
                        key_texts=[stripped],
                        value_texts=[stripped],
                        pattern="single_kv_line",
                        rule_id="R_META_SINGLE_KV_LINE",
                    )
                    candidates.append(cand)
                    if decisions[i].action is CleanerAction.KEEP:
                        decisions[i] = self._mk_dec(
                            i, blk, CleanerAction.MOVE_METADATA,
                            rule_id="R_META_SINGLE_KV_LINE",
                            reason="Single-line metadata Key:Value pattern (Section 10)",
                            confidence=Confidence.MEDIUM,
                        )

            # --- Pattern C: already METADATA type block → real extraction + MOVE ---
            if blk.type is BlockType.METADATA and decisions[i].action is CleanerAction.KEEP:
                # V2.1: Attempt to extract structured k:v pairs from metadata text.
                # Parser metadata header is often:
                #   source_file: ...\n
                #   title: ...\n
                #   page_count: 10\n
                #   parser: pdf_parser_v2/x.y.z\n
                #   pdf_author: ...
                meta_keys, meta_vals = [], []
                for line in stripped.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # split on first " : " / "："
                    split_match = re.match(r"^([^：:]{1,40}?)\s*[:：]\s*(.*)$", line)
                    if split_match:
                        k, v = split_match.group(1).strip(), split_match.group(2).strip()
                        if k:
                            meta_keys.append(k)
                            meta_vals.append(v if v else "")
                if meta_keys and any(v != "" for v in meta_vals):
                    cand = MetadataCandidate(
                        key_block_orders=[i] * len(meta_keys),
                        value_block_orders=[i] * len(meta_vals),
                        key_texts=meta_keys,
                        value_texts=meta_vals,
                        pattern="parser_metadata_block_kv_extract",
                        rule_id="R_META_TYPE_MARKER",
                    )
                    candidates.append(cand)
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.MOVE_METADATA,
                        rule_id="R_META_TYPE_MARKER",
                        reason=(
                            "Parser METADATA marking; extracted "
                            f"{len(meta_keys)} k:v pair(s) for downstream MetadataNormalizer"
                        ),
                        confidence=Confidence.HIGH,
                    )
                else:
                    # No parseable key:value pairs in this METADATA-type block.
                    # Conservative: REVIEW rather than silent empty MOVE.
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.REVIEW,
                        rule_id="R_REVIEW_META_BODY_MIXED",
                        reason="METADATA type but no parseable k:v pairs for staging",
                        confidence=Confidence.LOW,
                    )

            # --- Pattern D: document-level identifier at document head (V2.1) ---
            # Conservative: block order < 10, identifier regex match,
            # contextual evidence: near a heading/title, adjacent to doc
            # origin/制定 unit or identifier context keywords.
            pattern_d_eligible = (
                blk.type is BlockType.PARAGRAPH
                and i < 10
                and decisions[i].action is CleanerAction.KEEP
                and self._doc_format(doc) in ("pdf", "docx", "html")
                and DOC_IDENTIFIER_RE.match(stripped)
            )
            if pattern_d_eligible:
                # Positive context:
                #   a) neighbor within ±3 is HEADING or contains 合同/示范/公告/令 等
                #   b) OR neighbor text contains DOC_IDENTIFIER_CONTEXT_KEY_RE
                ctx_ok = False
                window = range(max(0, i - 3), min(n, i + 4))
                for j in window:
                    if j == i:
                        continue
                    nb = blocks[j]
                    if nb.type is BlockType.HEADING:
                        ctx_ok = True
                        break
                    nt = nb.text
                    if DOC_IDENTIFIER_CONTEXT_KEY_RE.search(nt):
                        ctx_ok = True
                        break
                    if re.search(r"(示范文本|合同|协议|办法|规定|条例|公告|主席令|政府令|印发)", nt or ""):
                        ctx_ok = True
                        break
                if ctx_ok:
                    cand = MetadataCandidate(
                        key_block_orders=[i],
                        value_block_orders=[i],
                        key_texts=["document_identifier"],
                        value_texts=[stripped],
                        pattern="document_identifier_head",
                        rule_id="R_META_DOCUMENT_IDENTIFIER",
                    )
                    candidates.append(cand)
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.MOVE_METADATA,
                        rule_id="R_META_DOCUMENT_IDENTIFIER",
                        reason=(
                            "Document-head identifier pattern (e.g. 示范合同编号/文号), "
                            "passed to MetadataNormalizer for canonicalization"
                        ),
                        confidence=Confidence.MEDIUM,
                    )
                    # Do not increment i by 2; no "paired" next block consumed.

            i += 1

        # ================================================================
        # V2.1 INVARIANT (Section 5 of patch task):
        # If any block decision is MOVE_METADATA but no candidate references it
        # in non-empty value_texts → demote to REVIEW (KEEP-equivalent safe) to
        # prevent empty payload information loss.
        # ================================================================
        ref_by_val: set[int] = set()
        ref_by_key: set[int] = set()
        for cand in candidates:
            if cand.value_texts and any(v and v.strip() for v in cand.value_texts):
                ref_by_val.update(cand.value_block_orders)
                ref_by_key.update(cand.key_block_orders)
        for j, dec in enumerate(decisions):
            if dec.action is CleanerAction.MOVE_METADATA:
                if j not in ref_by_val and j not in ref_by_key:
                    decisions[j] = self._mk_dec(
                        j, doc.blocks[j], CleanerAction.REVIEW,
                        rule_id="R_REVIEW_MOVE_PAYLOAD_EMPTY_INVARIANT",
                        reason=(
                            "MOVE_METADATA invariant violation: no MetadataCandidate "
                            "referenced this block with non-empty value payload. "
                            "Fail-closed to REVIEW."
                        ),
                        confidence=Confidence.MEDIUM,
                    )
                    warnings.append(
                        f"order={j} MOVE→REVIEW demotion (empty payload invariant)."
                    )
        return candidates

    def _apply_dedupe_rules(
        self,
        doc: ParsedDocument,
        decisions: list[BlockDecision],
        warnings: list[str],
    ) -> list[DuplicateGroup]:
        """
        Section 11 / 12 DEDUPLICATE rules.

        STRICT:
          - Only LOCAL / ADJACENT duplicates (within 8 blocks)
          - Must have strong evidence: same normalized text
          - NEVER dedup long legal text just because wording repeats
          - Dedup applies to: duplicate titles, duplicate metadata region,
            exact repeated short boilerplate (boilerplate should be DROP though)
        """
        groups: list[DuplicateGroup] = []
        blocks = doc.blocks
        n = len(blocks)
        WINDOW = 8  # local adjacency window

        # Pass 1: heading/paragraph duplicate title (common: doc title + header repeats)
        for i in range(n):
            if decisions[i].action in (CleanerAction.DROP, CleanerAction.DEDUPLICATE):
                continue
            blk_i = blocks[i]
            norm_i = _norm_for_dup(blk_i.text)
            if not norm_i:
                continue

            # For short blocks (<= 80 chars): allow local dedup
            # For long blocks (> 80 chars): ONLY if type is HEADING (duplicate heading)
            short_text = len(norm_i) <= 80
            type_ok_for_dup = (
                short_text
                or blk_i.type is BlockType.HEADING
                or blk_i.type is BlockType.METADATA
            )
            if not type_ok_for_dup:
                continue

            # --- Legal Repeated Text Safety (Section 12) ---------------------------
            # CRITICAL: Distinguish between:
            #   A) "metadata region / title" duplicates (window 8 safe; common in CMS)
            #   B) "legal substantive body" text (contract repeated clauses, repeated
            #      article definitions, etc.). For B, window MUST be 1 (immediately
            #      adjacent only) AND NO heading boundary between i and j.
            #
            # Liberal window = ok if: block type is METADATA, OR block is flagged
            # MOVE_METADATA, OR it's a HEADING level 1 (doc title heading).
            # Everything else (PARAGRAPH/LIST_ITEM/regular HEADING/LIST_ITEM in legal
            # body) → distance 1 ONLY, plus heading boundary check below.
            _i_action = decisions[i].action
            liberal_window_ok = (
                blk_i.type is BlockType.METADATA
                or _i_action is CleanerAction.MOVE_METADATA
                or (blk_i.type is BlockType.HEADING and (blk_i.level or 1) == 1)
            )

            # Scan local window only (i+1 to min(i+WINDOW, n-1))
            for j in range(i + 1, min(i + WINDOW + 1, n)):
                if decisions[j].action in (CleanerAction.DROP, CleanerAction.DEDUPLICATE):
                    continue
                # Distance gate for legal substantive body (Section 12):
                if not liberal_window_ok:
                    # Strict mode: only IMMEDIATE neighbors can be duplicates.
                    # Any gap >= 2 could be intentional legal repeated text across
                    # sections (contract templates, identical definition paragraphs).
                    if j != i + 1:
                        continue
                    # Additionally: if ANY heading block (even level 1) OR
                    # substantive legal paragraph boundary was between i and j,
                    # also skip. Note: j == i+1 so this is trivially satisfied.

                blk_j = blocks[j]
                norm_j = _norm_for_dup(blk_j.text)
                if norm_j != norm_i:
                    continue

                # Liberal window extra check: heading-boundary crossing is NEVER ok
                # for dedup (cross-section boundary = intentional repeat possible).
                # Check between i+1 and j-1 for ANY HEADING block with legal structure
                # marker (第X条/章/节) OR any heading at level <=2.
                if liberal_window_ok and j > i + 1:
                    _crosses_boundary = False
                    strong_heading_re = re.compile(
                        r"第[一二三四五六七八九十百千零〇\d]+[章节条款项分编编篇部]", re.UNICODE
                    )
                    for _k in range(i + 1, j):
                        bk = blocks[_k]
                        if bk.type is BlockType.HEADING:
                            lv = bk.level or 99
                            if lv <= 2 or strong_heading_re.search(bk.text):
                                _crosses_boundary = True
                                break
                    if _crosses_boundary:
                        continue

                # Type consistency requirement: same block type, or both title-like
                if blk_i.type is not blk_j.type:
                    # Allow HEADING(1) + PARAGRAPH = duplicate title case only
                    both_title_like = (
                        {blk_i.type, blk_j.type} == {BlockType.HEADING, BlockType.PARAGRAPH}
                        and blk_i.type is BlockType.HEADING
                    )
                    if not both_title_like:
                        continue

                # Legal repetition safety (Section 12): long text (> 200 chars) NEVER dedup
                if len(norm_i) > 200:
                    continue

                # All checks passed → mark as DEDUPLICATE
                kept_order = i
                dedup_order = j
                # Heuristic: prefer to keep the one with HEADING type
                if blk_j.type is BlockType.HEADING and blk_i.type is not BlockType.HEADING:
                    kept_order = j
                    dedup_order = i

                # If we've already marked one, skip
                if decisions[dedup_order].action is CleanerAction.DEDUPLICATE:
                    continue

                decisions[dedup_order] = self._mk_dec(
                    dedup_order, blocks[dedup_order], CleanerAction.DEDUPLICATE,
                    rule_id="R_DEDUP_LOCAL_ADJACENT",
                    reason=(
                        f"Local duplicate (window={WINDOW}, kept_order={kept_order}): "
                        f"identical normalized text as order {kept_order}; "
                        f"type={blocks[dedup_order].type.value}; Section 11/12"
                    ),
                    confidence=Confidence.MEDIUM,
                )
                groups.append(DuplicateGroup(
                    block_orders=[kept_order, dedup_order],
                    normalized_text=norm_i,
                    reason=(
                        f"Pair {kept_order}↔{dedup_order}; keep={kept_order} "
                        f"({blocks[kept_order].type.value})"
                    ),
                    rule_id="R_DEDUP_LOCAL_ADJACENT",
                    kept_order=kept_order,
                ))

        return groups

    def _apply_review_rules(
        self,
        doc: ParsedDocument,
        decisions: list[BlockDecision],
        warnings: list[str],
    ) -> None:
        """Section 13 REVIEW rules — default for UNKNOWN; NEVER DROP UNKNOWN."""
        blocks = doc.blocks
        n = len(blocks)

        for i, blk in enumerate(blocks):
            dec = decisions[i]
            text = blk.text
            stripped = text.strip()
            norm = _normalize_ws(text)

            # --- R_REVIEW_UNKNOWN_DEFAULT: UNKNOWN → ALWAYS REVIEW (NEVER DROP) ---
            if blk.type is BlockType.UNKNOWN:
                # Only override if current is KEEP (don't downgrade DROP — but UNKNOWN
                # shouldn't have been DROP'd anyway per the default-REVIEW rule)
                if dec.action is CleanerAction.KEEP:
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.REVIEW,
                        rule_id="R_REVIEW_UNKNOWN_DEFAULT",
                        reason="BlockType.UNKNOWN requires human review (Section 13); never silent drop",
                        confidence=Confidence.HIGH,
                    )
                    continue

            # --- R_REVIEW_MIXED_BOILERPLATE_LEGAL: mixed content ---
            # Contains UI token but also substantive legal signal → can't safely DROP.
            # BUT: if STRONG legal evidence is present (第X条 / 明确法律条款 / 多条信号),
            # the block is almost certainly KEEP-worthy legal content — flagging it REVIEW
            # would just add noise without improving safety.
            has_ui = any(tok in stripped for tok in STANDALONE_UI_TOKENS)
            has_legal = bool(LEGAL_PARAGRAPH_SIGNAL_RE.search(text))
            if has_ui and has_legal and dec.action is CleanerAction.KEEP:
                # Check for STRONG legal: article number pattern (第X条/章/节) OR long substantive text
                strong_article_re = re.compile(
                    r"第[一二三四五六七八九十百千零〇\d]+[章节条款项条]", re.UNICODE
                )
                has_strong_legal = bool(strong_article_re.search(text)) or len(norm) >= 30
                if has_strong_legal:
                    # Strong legal signal → just KEEP (it's normal legal text that happens to mention a UI word)
                    pass  # leave KEEP in place
                else:
                    # Weak evidence of both → REVIEW (cannot safely tell)
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.REVIEW,
                        rule_id="R_REVIEW_MIXED_BOILERPLATE_LEGAL",
                        reason="Mixed UI token + legal paragraph signal (weak); cannot safely DROP",
                        confidence=Confidence.MEDIUM,
                    )
                    continue

            # --- R_REVIEW_META_BODY_MIXED: metadata label + substantive body ---
            label_match = METADATA_KEY_LABEL_RE.match(stripped)
            if label_match and len(norm) > 80 and dec.action is CleanerAction.KEEP:
                decisions[i] = self._mk_dec(
                    i, blk, CleanerAction.REVIEW,
                    rule_id="R_REVIEW_META_BODY_MIXED",
                    reason="Block looks like metadata label but is >80 chars (may mix metadata+body)",
                    confidence=Confidence.MEDIUM,
                )
                continue

            # --- R_REVIEW_AMBIGUOUS_SHORT ---
            # Strategy (Section 13 "异常超短但可能有法律意义的 block"):
            #  A) len 1-2 chars → almost always ambiguous → REVIEW
            #  B) len 3-5 chars AND pure bracket/list-item pattern (no substantive verb/noun)
            #     → REVIEW (e.g., "（三）", "(1)", "①")
            #  Otherwise KEEP even if short (e.g., "正文A" is 3 chars but clearly meaningful content)
            pure_listitem_re = re.compile(
                r"^[（\(\[【]\s*[一二三四五六七八九十百千零〇\d]{1,3}\s*[）\)\]】]$|"
                r"^[①②③④⑤⑥⑦⑧⑨⑩]$|"
                r"^[\-\*·・．.]\s*$",
                re.UNICODE,
            )
            tiny_len = len(norm) <= 2
            list_like_len = 3 <= len(norm) <= 5 and bool(pure_listitem_re.match(stripped))
            if norm and (tiny_len or list_like_len) and blk.type is BlockType.PARAGRAPH:
                if dec.action is CleanerAction.KEEP:
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.REVIEW,
                        rule_id="R_REVIEW_AMBIGUOUS_SHORT",
                        reason=(
                            f"Very short block ({len(norm)} chars; "
                            f"tiny={tiny_len}, list-item-like={list_like_len}) "
                            f"— ambiguous legal/noise"
                        ),
                        confidence=Confidence.LOW,
                    )
                    continue

            # --- R_REVIEW_TABLE_AMBIGUOUS: damaged / conflicting TABLE ---
            if blk.type is BlockType.TABLE and dec.action is CleanerAction.KEEP:
                if blk.table_data is not None:
                    td = blk.table_data
                    # Empty rows or empty headers → suspicious structure
                    if not td.rows or not td.headers:
                        decisions[i] = self._mk_dec(
                            i, blk, CleanerAction.REVIEW,
                            rule_id="R_REVIEW_TABLE_AMBIGUOUS",
                            reason="TABLE block missing rows/headers (damaged structure)",
                            confidence=Confidence.MEDIUM,
                        )
                        continue
                    # All cells empty → suspicious
                    all_empty_headers = all(not c.text.strip() for c in td.headers)
                    all_empty_rows = all(
                        all(not c.text.strip() for c in row) for row in td.rows
                    )
                    if all_empty_headers and all_empty_rows:
                        decisions[i] = self._mk_dec(
                            i, blk, CleanerAction.REVIEW,
                            rule_id="R_REVIEW_TABLE_AMBIGUOUS",
                            reason="TABLE block with all empty cells",
                            confidence=Confidence.MEDIUM,
                        )
                        continue

    def _apply_legal_safety_net(
        self,
        doc: ParsedDocument,
        decisions: list[BlockDecision],
        warnings: list[str],
    ) -> None:
        """
        Section 7 PRIMARY SAFETY PRINCIPLE.

        Legal Content Retention > Noise Removal.
        Any DROP decision that looks like it may contain legal content → REVIEW.
        """
        blocks = doc.blocks
        for i, blk in enumerate(blocks):
            dec = decisions[i]
            if dec.action is not CleanerAction.DROP:
                continue

            text = blk.text
            # If the block contains substantive legal signals, override DROP → REVIEW
            if LEGAL_PARAGRAPH_SIGNAL_RE.search(text):
                # But allow short UI tokens that happen to contain 1 character match
                # Only upgrade if text is > 15 chars (meaningful paragraph)
                if len(_normalize_ws(text)) > 15:
                    decisions[i] = self._mk_dec(
                        i, blk, CleanerAction.REVIEW,
                        rule_id="R_SAFETY_NET_LEGAL_DROP_OVERRIDE",
                        reason=(
                            f"Safety Net: DROP candidate '{dec.rule_id}' reverted to REVIEW — "
                            f"contains legal-paragraph signal and >15 chars (Section 7)"
                        ),
                        confidence=Confidence.HIGH,
                    )
                    warnings.append(
                        f"SAFETY_NET: order={i} reverted DROP→REVIEW "
                        f"(rule={dec.rule_id}); legal signal present"
                    )

    # =================================================================
    # Build cleaned document
    # =================================================================

    def _build_cleaned_document(
        self,
        original_doc: ParsedDocument,
        decisions: list[BlockDecision],
        dup_groups: list[DuplicateGroup],
        meta_candidates: list[MetadataCandidate],
    ) -> ParsedDocument:
        """
        Build new, validated ParsedDocument without mutating original.

        Inclusion rules (what goes into evidence blocks):
          - KEEP → always include
          - MOVE_METADATA → include (MetadataNormalizer decides downstream extraction;
                                we don't drop evidence silently)
          - DEDUPLICATE → exclude (redundant copy)
          - DROP → exclude (boilerplate/UI)
          - REVIEW → include (we don't drop, we flag for human; still part of evidence)

        Post-processing:
          - Deep copy blocks (never touch originals)
          - Explicit renumber_blocks() to fix order/provenance
          - validate()
        """
        orig_blocks = original_doc.blocks
        kept_block_indices: list[int] = []

        for i, dec in enumerate(decisions):
            action = dec.action
            if action is CleanerAction.KEEP:
                kept_block_indices.append(i)
            elif action is CleanerAction.MOVE_METADATA:
                # Include in evidence (Cleaner doesn't drop; Normalizer decides downstream)
                kept_block_indices.append(i)
            elif action is CleanerAction.REVIEW:
                # REVIEW blocks still included (no silent deletion per Section 20/23)
                kept_block_indices.append(i)
            elif action is CleanerAction.DEDUPLICATE:
                # Exclude redundant copy only (first occurrence is KEEP'd)
                continue
            elif action is CleanerAction.DROP:
                # Exclude boilerplate
                continue
            else:  # pragma: no cover
                # Unknown action → conservative include + REVIEW warning
                kept_block_indices.append(i)

        # --- Shallow copy of block list, then deep-copy each block ---
        new_blocks: list[Block] = []
        for orig_i in kept_block_indices:
            orig = orig_blocks[orig_i]
            if orig.table_data is not None:
                # Clone TableData
                old_td = orig.table_data
                new_headers = [
                    TableCell(
                        text=c.text,
                        colspan=c.colspan,
                        rowspan=c.rowspan,
                        is_header=c.is_header,
                    )
                    for c in old_td.headers
                ]
                new_rows = [
                    [
                        TableCell(
                            text=c.text,
                            colspan=c.colspan,
                            rowspan=c.rowspan,
                            is_header=c.is_header,
                        )
                        for c in row
                    ]
                    for row in old_td.rows
                ]
                new_td = TableData(
                    headers=new_headers,
                    rows=new_rows,
                    caption=old_td.caption,
                )
                # Build text fresh from new TableData (text must == to_markdown())
                new_text = new_td.to_markdown()
                new_prov = orig.provenance_stored
                new_meta = dict(orig.metadata)
                new_meta.setdefault("cleaner_v2", {})
                new_meta["cleaner_v2"]["original_block_order"] = orig_i
                new_meta["cleaner_v2"]["original_action"] = decisions[orig_i].action.value
                new_blocks.append(Block(
                    block_id=orig.block_id,
                    type=orig.type,
                    text=new_text,
                    order=orig.order,
                    level=orig.level,
                    table_data=new_td,
                    page=orig.page,
                    metadata=new_meta,
                    provenance_stored=new_prov,
                ))
            else:
                new_prov = orig.provenance_stored
                new_meta = dict(orig.metadata)
                new_meta.setdefault("cleaner_v2", {})
                new_meta["cleaner_v2"]["original_block_order"] = orig_i
                new_meta["cleaner_v2"]["original_action"] = decisions[orig_i].action.value
                new_blocks.append(Block(
                    block_id=orig.block_id,
                    type=orig.type,
                    text=orig.text,
                    order=orig.order,
                    level=orig.level,
                    table_data=None,
                    page=orig.page,
                    metadata=new_meta,
                    provenance_stored=new_prov,
                ))

        # --- EXPLICIT renumber (required by Contract) ---
        new_doc_id = original_doc.document_id
        ParsedDocument.prepare_blocks(new_doc_id, new_blocks)

        # --- Build new doc-level metadata (deep copy) ---
        new_metadata = copy.deepcopy(original_doc.metadata)
        new_metadata.setdefault("cleaning", {})
        new_metadata["cleaning"]["cleaner_v2_version"] = self.CLEANER_VERSION
        new_metadata["cleaning"]["cleaner_v2_original_block_count"] = len(orig_blocks)
        new_metadata["cleaning"]["cleaner_v2_cleaned_block_count"] = len(new_blocks)
        new_metadata["cleaning"]["cleaner_v2_decisions_summary"] = {
            "keep": sum(1 for d in decisions if d.action is CleanerAction.KEEP),
            "drop": sum(1 for d in decisions if d.action is CleanerAction.DROP),
            "move_metadata": sum(1 for d in decisions if d.action is CleanerAction.MOVE_METADATA),
            "deduplicate": sum(1 for d in decisions if d.action is CleanerAction.DEDUPLICATE),
            "review": sum(1 for d in decisions if d.action is CleanerAction.REVIEW),
        }
        # Audit reference: list of rule_ids applied (unique)
        rule_ids = sorted({d.rule_id for d in decisions})
        new_metadata["cleaning"]["cleaner_v2_rules_triggered"] = rule_ids

        # --- Warnings: append cleaner-specific warnings ---
        new_warnings = list(original_doc.warnings)
        new_warnings.append(
            f"cleaner_v2::version={self.CLEANER_VERSION}; "
            f"blocks {len(orig_blocks)}→{len(new_blocks)}; "
            f"drop={new_metadata['cleaning']['cleaner_v2_decisions_summary']['drop']} "
            f"review={new_metadata['cleaning']['cleaner_v2_decisions_summary']['review']}"
        )

        new_doc = ParsedDocument(
            document_id=new_doc_id,
            source_file=original_doc.source_file,
            title=original_doc.title,
            metadata=new_metadata,
            blocks=new_blocks,
            parser_name=original_doc.parser_name,
            parser_version=original_doc.parser_version,
            warnings=new_warnings,
        )
        new_doc.validate()
        return new_doc

    # =================================================================
    # Utils
    # =================================================================

    @staticmethod
    def _mk_dec(
        order: int,
        blk: Block,
        action: CleanerAction,
        *,
        rule_id: str,
        reason: str,
        confidence: Confidence,
    ) -> BlockDecision:
        return BlockDecision(
            original_block_order=order,
            block_type=blk.type.value,
            text_preview=_preview(blk.text),
            action=action,
            rule_id=rule_id,
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _blocks_sha256(blocks: list[Block]) -> str:
        h = hashlib.sha256()
        for b in blocks:
            h.update(f"{b.order}|{b.type.value}|{b.text}".encode("utf-8"))
        return h.hexdigest()


# =============================================================================
# Convenience public function
# =============================================================================

def clean_parsed_document_v2(doc: ParsedDocument) -> CleaningResult:
    """One-shot entrypoint: create a BlockAwareCleaner and clean doc."""
    cleaner = BlockAwareCleaner()
    return cleaner.clean(doc)


__all__ = [
    "CleanerAction",
    "Confidence",
    "BlockDecision",
    "MetadataCandidate",
    "DuplicateGroup",
    "CleaningResult",
    "BlockAwareCleaner",
    "clean_parsed_document_v2",
]
