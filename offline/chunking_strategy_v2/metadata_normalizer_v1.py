"""
MetadataNormalizerV1 — ISOLATED Cross-Format Implementation
============================================================

Scope (§0):
  Pure, stdlib-only metadata normalization module.
  - Reads Frozen ParsedDocument V2 only
  - NEVER mutates original ParsedDocument
  - NEVER deletes / reorders blocks
  - NEVER calls Cleaner V2
  - Offline, deterministic, NO database / network / Milvus / MySQL

Responsibility (§3):
  Answers ONLY:
    "What metadata is this? How to normalize? Duplicate? Conflict?"
  Does NOT answer:
    "Should this block be MOVE_METADATA and removed from evidence?"
    (That is Cleaner V2's job.)

Output (§6):
  MetadataNormalizationResult — fully auditable.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Import Frozen Contract from local working directory copy
try:
    from parsed_document_v2 import ParsedDocument  # type: ignore
except ImportError:  # pragma: no cover — fallback for alternate sys.path
    from modules.ingestion.parsed_document_v2 import ParsedDocument  # type: ignore


# =============================================================================
# Enums — status / source priority
# =============================================================================

class FieldStatus(str, Enum):
    RESOLVED = "RESOLVED"
    CONFLICT = "CONFLICT"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"
    MISSING = "MISSING"
    DERIVABLE = "DERIVABLE"
    FUTURE_CHUNK_METADATA_CANDIDATE = "FUTURE_CHUNK_METADATA_CANDIDATE"


class DuplicateType(str, Enum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    NORMALIZED_EQUIVALENT_DUPLICATE = "NORMALIZED_EQUIVALENT_DUPLICATE"
    SEMANTICALLY_DISTINCT = "SEMANTICALLY_DISTINCT"
    CONFLICT = "CONFLICT"


class MetadataSourceTier(int, Enum):
    """
    Generic source reliability tiers (§15).
    Higher number = more reliable.
    """
    BODY_INFERENCE = 10
    DOCUMENT_PROPERTIES = 20
    ADJACENT_KV_PAIR = 30
    HTML_META_PROPERTY = 40
    EXPLICIT_STRUCTURED_PARSER_METADATA = 50


SOURCE_TIER_NAME = {
    MetadataSourceTier.BODY_INFERENCE: "body_inference",
    MetadataSourceTier.DOCUMENT_PROPERTIES: "document_properties",
    MetadataSourceTier.ADJACENT_KV_PAIR: "adjacent_kv_pair",
    MetadataSourceTier.HTML_META_PROPERTY: "html_meta_property",
    MetadataSourceTier.EXPLICIT_STRUCTURED_PARSER_METADATA: "explicit_structured_parser_metadata",
}


class StorageRecommendation(str, Enum):
    MILVUS_SCALAR_CANDIDATE = "MILVUS_SCALAR_CANDIDATE"
    MYSQL_ONLY = "MYSQL_ONLY"
    JSON_METADATA = "JSON_METADATA"
    CITATION_ONLY = "CITATION_ONLY"
    UNDECIDED = "UNDECIDED"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RawMetadataCandidate:
    canonical_field: Optional[str]
    raw_key: str
    raw_value: str
    source_tier: MetadataSourceTier
    source_method: str
    source_block_order: Optional[int] = None
    source_block_type: Optional[str] = None
    page: Optional[int] = None
    provenance_source_file: Optional[str] = None

    def describe(self) -> str:
        blk = f"block#{self.source_block_order}" if self.source_block_order is not None else "n/a"
        return (
            f"[{SOURCE_TIER_NAME[self.source_tier]}|{self.source_method}] "
            f"{self.raw_key!r}={self.raw_value!r} ({blk})"
        )


@dataclass
class FieldResult:
    field_name: str
    raw_values: list[RawMetadataCandidate] = field(default_factory=list)
    normalized_value: Any = None
    status: FieldStatus = FieldStatus.MISSING
    confidence: float = 0.0
    source_block_orders: list[int] = field(default_factory=list)
    source_methods: list[str] = field(default_factory=list)
    warning: Optional[str] = None

    def add_candidate(self, cand: RawMetadataCandidate) -> None:
        self.raw_values.append(cand)
        if cand.source_block_order is not None and cand.source_block_order not in self.source_block_orders:
            self.source_block_orders.append(cand.source_block_order)
        if cand.source_method and cand.source_method not in self.source_methods:
            self.source_methods.append(cand.source_method)


@dataclass
class DuplicateGroup:
    canonical_field: str
    duplicate_type: DuplicateType
    candidate_indices: list[int]
    explanation: str = ""


@dataclass
class MetadataConflict:
    canonical_field: str
    candidate_indices: list[int]
    explanation: str = ""


# =============================================================================
# Output Contract (§6)
# =============================================================================

@dataclass
class MetadataNormalizationResult:
    document_id: str
    source_file: str
    parser_name: str
    parser_version: str

    normalized_metadata: dict[str, Any] = field(default_factory=dict)
    field_results: dict[str, FieldResult] = field(default_factory=dict)
    consumed_block_candidates: list[int] = field(default_factory=list)

    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    conflicts: list[MetadataConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    deferred_candidates: list[RawMetadataCandidate] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "normalized_metadata": dict(self.normalized_metadata),
            "field_results": {},
            "consumed_block_candidates": list(self.consumed_block_candidates),
            "duplicate_groups": [
                {
                    "canonical_field": g.canonical_field,
                    "duplicate_type": g.duplicate_type.value,
                    "candidate_indices": list(g.candidate_indices),
                    "explanation": g.explanation,
                }
                for g in self.duplicate_groups
            ],
            "conflicts": [
                {
                    "canonical_field": c.canonical_field,
                    "candidate_indices": list(c.candidate_indices),
                    "explanation": c.explanation,
                }
                for c in self.conflicts
            ],
            "warnings": list(self.warnings),
            "deferred_candidates_count": len(self.deferred_candidates),
        }
        for fn, fr in self.field_results.items():
            out["field_results"][fn] = {
                "normalized_value": fr.normalized_value,
                "status": fr.status.value,
                "confidence": fr.confidence,
                "source_block_orders": list(fr.source_block_orders),
                "source_methods": list(fr.source_methods),
                "warning": fr.warning,
                "raw_values": [
                    {
                        "canonical_field": rv.canonical_field,
                        "raw_key": rv.raw_key,
                        "raw_value": rv.raw_value,
                        "source_tier": rv.source_tier.value,
                        "source_tier_name": SOURCE_TIER_NAME[rv.source_tier],
                        "source_method": rv.source_method,
                        "source_block_order": rv.source_block_order,
                        "source_block_type": rv.source_block_type,
                        "page": rv.page,
                        "provenance_source_file": rv.provenance_source_file,
                    }
                    for rv in fr.raw_values
                ],
            }
        return out


# =============================================================================
# §7 / §10 — Canonical fields + Alias Mapping (STRICT semantic distinction)
# =============================================================================

FIELD_ALIASES: dict[str, str] = {
    # title
    "title": "title",
    "标　　题": "title",
    "标  题": "title",
    "标题": "title",
    "文件标题": "title",
    # source_org
    "发文机关": "source_org",
    "发布机关": "source_org",
    "制定机关": "source_org",
    "印发机关": "source_org",
    "主办单位": "source_org",
    "来源": "source_org",
    "source_org": "source_org",
    # publish_date — STRICTLY separate from creation_date / effective_date
    "发布日期": "publish_date",
    "发布时间": "publish_date",
    "publish_date": "publish_date",
    "pubdate": "publish_date",
    # creation_date — 成文日期 — SEMANTICALLY DISTINCT from publish_date
    "成文日期": "creation_date",
    "成文时间": "creation_date",
    "created": "creation_date",
    "creation_date": "creation_date",
    # effective_date — 施行/实施/生效 — SEMANTICALLY DISTINCT
    "施行日期": "effective_date",
    "实施日期": "effective_date",
    "生效日期": "effective_date",
    "开始施行日期": "effective_date",
    "effective_date": "effective_date",
    # expiry_date
    "废止日期": "expiry_date",
    "失效日期": "expiry_date",
    "终止日期": "expiry_date",
    "expiry_date": "expiry_date",
    "expiration_date": "expiry_date",
    # document_number
    "发文字号": "document_number",
    "文号": "document_number",
    "文件编号": "document_number",
    "document_number": "document_number",
    # document_type
    "文件类型": "document_type",
    "公文种类": "document_type",
    "document_type": "document_type",
    "doc_type": "document_type",
    # authority_level
    "效力级别": "authority_level",
    "authority_level": "authority_level",
    # legal_status
    "法律状态": "legal_status",
    "现行状态": "legal_status",
    "legal_status": "legal_status",
    "status": "legal_status",
    # jurisdiction / region
    "jurisdiction": "jurisdiction",
    "适用区域": "jurisdiction",
    "region": "region",
    "地域": "region",
    "所在地区": "region",
    # source url / file
    "source_url": "source_url",
    "source_file": "source_file",
    "source_path": "source_file",
    # document id
    "document_id": "document_id",
    "doc_id": "document_id",
    "索引号": "document_index_number",
    # theme / category
    "主题分类": "theme_category",
    "主题": "theme_category",
}


CANONICAL_FIELDS_V1 = {
    "document_id", "logical_document_id",
    "title", "source_org",
    "publish_date", "creation_date", "effective_date", "expiry_date",
    "document_number", "document_index_number",
    "document_type", "authority_level", "legal_status",
    "jurisdiction", "region",
    "source_url", "source_file",
    "theme_category",
}


CHAPTER_SECTION_ARTICLE_LABELS = {"章", "节", "条", "款", "项",
                                   "chapter", "section", "article"}

METADATA_KEY_LABELS: set[str] = set()
for alias_key in FIELD_ALIASES.keys():
    clean = alias_key.strip().rstrip(":：").strip()
    if clean:
        METADATA_KEY_LABELS.add(clean)
        METADATA_KEY_LABELS.add(clean + ":")
        METADATA_KEY_LABELS.add(clean + "：")


KV_KEY_MAX_CHARS = 20
KV_VALUE_MAX_CHARS = 200


def _canonicalize_alias_lookup(raw_key: str) -> Optional[str]:
    """
    Try to map a raw key string to a canonical field name.
    Applies (in order):
      1. Exact match in FIELD_ALIASES
      2. Strip trailing :/： + strip
      3. Collapse internal whitespace + retry against alias keys (for 索 引 号 → 索引号)
    Returns canonical field name or None.
    """
    if not raw_key:
        return None
    # 1. exact
    c = FIELD_ALIASES.get(raw_key)
    if c:
        return c
    # 2. strip colon variants
    s = raw_key.strip().rstrip(":：").strip()
    c = FIELD_ALIASES.get(s)
    if c:
        return c
    # 3. collapse internal whitespace (both in candidate and in alias keys)
    collapsed_key = re.sub(r"\s+", "", s)
    for alias_key, canon in FIELD_ALIASES.items():
        collapsed_alias = re.sub(r"\s+", "", alias_key.rstrip(":：").strip())
        if collapsed_alias == collapsed_key and collapsed_key:
            return canon
    return None


METADATA_KEY_LABELS_NORMALIZED: set[str] = set()
for alias_key in FIELD_ALIASES.keys():
    base = re.sub(r"\s+", "", alias_key.strip().rstrip(":：").strip())
    if base:
        METADATA_KEY_LABELS_NORMALIZED.add(base)


# =============================================================================
# §11 — Date normalization
# =============================================================================

_DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
    re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
    re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"),
    re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})"),
]


_MONTH_DAY_MAX = {
    1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


def normalize_date(raw: Any) -> tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, "empty raw value"
    s = str(raw).strip()
    if not s:
        return None, "empty raw value"
    s_clean = re.split(r"[\sTt\-]\d{1,2}[:：]", s, maxsplit=1)[0].strip()
    for pat in _DATE_PATTERNS:
        m = pat.search(s_clean)
        if not m:
            continue
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12):
            continue
        dmax = _MONTH_DAY_MAX.get(mo, 31)
        if not (1 <= d <= dmax):
            continue
        return f"{y:04d}-{mo:02d}-{d:02d}", None
    return None, f"unrecognized date format: {raw!r}"


# =============================================================================
# §12 — Document number normalization
# =============================================================================

def normalize_document_number(raw: Any) -> tuple[str, str, FieldStatus]:
    if raw is None:
        return "", "", FieldStatus.NORMALIZATION_FAILED
    raw_s = str(raw).strip()
    if not raw_s:
        return "", "", FieldStatus.NORMALIZATION_FAILED
    norm = re.sub(r"\s+", "", raw_s)
    norm = norm.replace("—", "-").replace("－", "-").replace("–", "-")
    return raw_s, norm, FieldStatus.RESOLVED


# =============================================================================
# MetadataNormalizerV1
# =============================================================================

class MetadataNormalizerV1:

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug
        self._all_candidates: list[RawMetadataCandidate] = []

    # ------------------------------------------------------------------ main

    def normalize(self, doc: ParsedDocument) -> MetadataNormalizationResult:
        """Does NOT mutate `doc`."""
        self._all_candidates = []

        result = MetadataNormalizationResult(
            document_id=doc.document_id,
            source_file=doc.source_file,
            parser_name=doc.parser_name,
            parser_version=doc.parser_version,
        )

        field_results: dict[str, FieldResult] = {}
        consumed_block_set: set[int] = set()

        self._gather_from_parser_metadata(doc, field_results, consumed_block_set)
        self._gather_from_metadata_blocks(doc, field_results, consumed_block_set)
        self._gather_from_adjacent_kv_pairs(doc, field_results, consumed_block_set)

        self._gather_deferred_candidates(doc, result)

        self._resolve_field_results(field_results, result)
        self._detect_duplicates_and_conflicts(field_results, result)

        for fn, fr in field_results.items():
            if fr.status is FieldStatus.RESOLVED:
                result.normalized_metadata[fn] = fr.normalized_value

        result.field_results = field_results
        result.consumed_block_candidates = sorted(consumed_block_set)
        return result

    # ================================================================== Phase 1a

    def _gather_from_parser_metadata(
        self,
        doc: ParsedDocument,
        field_results: dict[str, FieldResult],
        consumed_block_set: set[int],
    ) -> None:
        md = doc.metadata or {}
        for raw_key, raw_val in md.items():
            if raw_val is None:
                continue
            raw_val_s = str(raw_val).strip()
            if not raw_val_s:
                continue
            canon = _canonicalize_alias_lookup(raw_key)
            cand = RawMetadataCandidate(
                canonical_field=canon,
                raw_key=raw_key,
                raw_value=raw_val_s,
                source_tier=MetadataSourceTier.EXPLICIT_STRUCTURED_PARSER_METADATA,
                source_method="parser_doc_metadata",
                provenance_source_file=doc.source_file,
            )
            self._register_candidate(cand, field_results)

        identity_candidates = [
            ("document_id", doc.document_id, "doc_attribute_document_id"),
            ("source_file", doc.source_file, "doc_attribute_source_file"),
        ]
        if doc.title and str(doc.title).strip():
            identity_candidates.append(("title", doc.title, "doc_attribute_title"))
        for canon, val, method in identity_candidates:
            cand = RawMetadataCandidate(
                canonical_field=canon,
                raw_key=canon,
                raw_value=str(val).strip(),
                source_tier=MetadataSourceTier.EXPLICIT_STRUCTURED_PARSER_METADATA,
                source_method=method,
            )
            self._register_candidate(cand, field_results)

    # ================================================================== Phase 1b

    def _gather_from_metadata_blocks(
        self,
        doc: ParsedDocument,
        field_results: dict[str, FieldResult],
        consumed_block_set: set[int],
    ) -> None:
        for b in doc.blocks:
            if b.type.name != "METADATA":
                continue
            text = (b.text or "").strip()
            if not text:
                continue
            parts = self._split_single_block_kv(text)
            if parts:
                raw_key, raw_val = parts
            else:
                raw_key = "__METADATA_BLOCK_VALUE__"
                raw_val = text
            canon = _canonicalize_alias_lookup(raw_key)
            cand = RawMetadataCandidate(
                canonical_field=canon,
                raw_key=raw_key,
                raw_value=raw_val,
                source_tier=MetadataSourceTier.ADJACENT_KV_PAIR,
                source_method="metadata_block_kv",
                source_block_order=b.order,
                source_block_type=b.type.name,
                page=b.page,
                provenance_source_file=(
                    b.provenance_stored.source_file
                    if (hasattr(b, "provenance_stored") and b.provenance_stored)
                    else doc.source_file
                ),
            )
            self._register_candidate(cand, field_results)
            consumed_block_set.add(b.order)

    # ================================================================== Phase 1c (§9)

    def _gather_from_adjacent_kv_pairs(
        self,
        doc: ParsedDocument,
        field_results: dict[str, FieldResult],
        consumed_block_set: set[int],
    ) -> None:
        blocks = doc.blocks
        n = len(blocks)
        for i in range(n):
            key_block = blocks[i]
            if key_block.type.name not in {"HEADING", "PARAGRAPH", "METADATA", "UNKNOWN"}:
                continue
            key_text = (key_block.text or "").strip()
            if not key_text or len(key_text) > KV_KEY_MAX_CHARS:
                continue
            clean_key = key_text.rstrip(":：").strip()
            has_colon = key_text.endswith(":") or key_text.endswith("：")
            collapsed_clean = re.sub(r"\s+", "", clean_key)
            in_vocab = collapsed_clean in METADATA_KEY_LABELS_NORMALIZED
            if not (has_colon or in_vocab):
                continue
            val_block_order: Optional[int] = None
            for j in range(i + 1, n):
                vb = blocks[j]
                vb_text = (vb.text or "").strip()
                if not vb_text:
                    continue
                if vb.type.name not in {"PARAGRAPH", "HEADING", "UNKNOWN", "METADATA"}:
                    break
                val_block_order = j
                break
            if val_block_order is None:
                continue
            val_block = blocks[val_block_order]
            val_text = (val_block.text or "").strip()
            if not val_text or len(val_text) > KV_VALUE_MAX_CHARS:
                continue
            canon = _canonicalize_alias_lookup(clean_key)
            cand = RawMetadataCandidate(
                canonical_field=canon,
                raw_key=clean_key,
                raw_value=val_text,
                source_tier=MetadataSourceTier.ADJACENT_KV_PAIR,
                source_method="adjacent_block_kv",
                source_block_order=key_block.order,
                source_block_type=key_block.type.name,
                page=key_block.page,
                provenance_source_file=doc.source_file,
            )
            self._register_candidate(cand, field_results)
            consumed_block_set.add(key_block.order)
            consumed_block_set.add(val_block.order)

    # ================================================================== Phase 2 (§8)

    def _gather_deferred_candidates(
        self,
        doc: ParsedDocument,
        result: MetadataNormalizationResult,
    ) -> None:
        chapter_re = re.compile(r"第[一二三四五六七八九十百千万零〇○0-9]+[章节条]")
        for b in doc.blocks:
            t = (b.text or "").strip()
            if not t:
                continue
            if chapter_re.match(t):
                label = "chapter"
                if "节" in t[:20]:
                    label = "section"
                if "条" in t[:20]:
                    label = "article"
                cand = RawMetadataCandidate(
                    canonical_field=label,
                    raw_key=f"{label}_candidate",
                    raw_value=t[:80],
                    source_tier=MetadataSourceTier.BODY_INFERENCE,
                    source_method="deferred_structure_label",
                    source_block_order=b.order,
                    source_block_type=b.type.name,
                    page=b.page,
                )
                result.deferred_candidates.append(cand)

    # ================================================================== Phase 3

    def _resolve_field_results(
        self,
        field_results: dict[str, FieldResult],
        result: MetadataNormalizationResult,
    ) -> None:
        for fn, fr in field_results.items():
            if not fr.raw_values:
                fr.status = FieldStatus.MISSING
                continue
            normalized_variants: list[tuple[Any, RawMetadataCandidate]] = []
            for rv in fr.raw_values:
                try:
                    norm_val, norm_warn = self._normalize_value_for_field(fn, rv.raw_value)
                except Exception as exc:
                    fr.warning = f"normalization exception: {exc}"
                    norm_val, norm_warn = None, f"exception: {exc}"
                if norm_val is not None:
                    normalized_variants.append((norm_val, rv))
            buckets: dict[Any, list[RawMetadataCandidate]] = {}
            for nval, rv in normalized_variants:
                buckets.setdefault(nval, []).append(rv)
            if not buckets:
                fr.status = FieldStatus.NORMALIZATION_FAILED
                fr.warning = fr.warning or "all candidates failed normalization"
                continue
            if len(buckets) == 1:
                nval = next(iter(buckets.keys()))
                contributors = buckets[nval]
                max_tier = max(c.source_tier.value for c in contributors)
                fr.normalized_value = nval
                fr.status = FieldStatus.RESOLVED
                fr.confidence = min(
                    max_tier / MetadataSourceTier.EXPLICIT_STRUCTURED_PARSER_METADATA.value,
                    1.0,
                )
                continue
            bucket_max_tier = {
                nval: max(c.source_tier.value for c in rvs)
                for nval, rvs in buckets.items()
            }
            top_tier = max(bucket_max_tier.values())
            top_buckets = [nv for nv, t in bucket_max_tier.items() if t == top_tier]
            if len(top_buckets) == 1:
                nval = top_buckets[0]
                fr.normalized_value = nval
                fr.status = FieldStatus.RESOLVED
                fr.confidence = min(
                    top_tier / MetadataSourceTier.EXPLICIT_STRUCTURED_PARSER_METADATA.value,
                    1.0,
                )
                lower = [nv for nv in buckets if nv != nval]
                fr.warning = (
                    f"multiple normalized buckets existed; selected highest-tier "
                    f"(tier={top_tier}). Lower-tier values: {lower}"
                )
                continue
            fr.status = FieldStatus.CONFLICT
            fr.normalized_value = None
            fr.warning = (
                f"conflict: {len(top_buckets)} distinct normalized values at "
                f"tier={top_tier}: {top_buckets}"
            )
            result.conflicts.append(
                MetadataConflict(
                    canonical_field=fn,
                    candidate_indices=list(range(len(fr.raw_values))),
                    explanation=fr.warning or "",
                )
            )

    # ================================================================== Phase 4 (§13/§14)

    def _detect_duplicates_and_conflicts(
        self,
        field_results: dict[str, FieldResult],
        result: MetadataNormalizationResult,
    ) -> None:
        for fn, fr in field_results.items():
            cands = fr.raw_values
            if len(cands) < 2:
                continue
            exact_key: dict[tuple[str, str], list[int]] = {}
            norm_key: dict[str, list[int]] = {}
            for i, rv in enumerate(cands):
                ek = (rv.raw_key, rv.raw_value)
                exact_key.setdefault(ek, []).append(i)
                try:
                    nval, _ = self._normalize_value_for_field(fn, rv.raw_value)
                except Exception:
                    nval = None
                nk = f"{{{fn}}}::{nval}" if nval is not None else f"{{{fn}}}::__RAW__::{rv.raw_value!r}"
                norm_key.setdefault(nk, []).append(i)
            seen: set[tuple[int, int]] = set()
            for indices in exact_key.values():
                if len(indices) >= 2:
                    result.duplicate_groups.append(
                        DuplicateGroup(
                            canonical_field=fn,
                            duplicate_type=DuplicateType.EXACT_DUPLICATE,
                            candidate_indices=list(indices),
                            explanation=(
                                f"Exact raw duplicate in field {fn}: "
                                f"{cands[indices[0]].raw_key!r}={cands[indices[0]].raw_value!r}"
                            ),
                        )
                    )
                    for a in indices:
                        for b in indices:
                            if a < b:
                                seen.add((a, b))
            for indices in norm_key.values():
                if len(indices) < 2:
                    continue
                remaining = [
                    i for i in indices
                    if not any(
                        (min(i, j), max(i, j)) in seen
                        for j in indices if j != i
                    )
                ]
                if len(remaining) >= 2:
                    result.duplicate_groups.append(
                        DuplicateGroup(
                            canonical_field=fn,
                            duplicate_type=DuplicateType.NORMALIZED_EQUIVALENT_DUPLICATE,
                            candidate_indices=list(remaining),
                            explanation=f"Normalized-equivalent candidates in field {fn}",
                        )
                    )
                    for a in remaining:
                        for b in remaining:
                            if a < b:
                                seen.add((a, b))
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    if (i, j) in seen:
                        continue
                    c1, c2 = cands[i], cands[j]
                    if fn in CANONICAL_FIELDS_V1:
                        cls = DuplicateType.CONFLICT
                        expl = (
                            f"Same canonical field {fn} with distinct unresolved values: "
                            f"{c1.describe()} vs {c2.describe()}"
                        )
                        if not any(
                            cc.canonical_field == fn and set(cc.candidate_indices) == {i, j}
                            for cc in result.conflicts
                        ):
                            result.conflicts.append(
                                MetadataConflict(
                                    canonical_field=fn,
                                    candidate_indices=[i, j],
                                    explanation=expl,
                                )
                            )
                    else:
                        cls = DuplicateType.SEMANTICALLY_DISTINCT
                        expl = (
                            f"Field {fn} candidates semantically distinct: "
                            f"{c1.describe()} vs {c2.describe()}"
                        )
                    result.duplicate_groups.append(
                        DuplicateGroup(
                            canonical_field=fn,
                            duplicate_type=cls,
                            candidate_indices=[i, j],
                            explanation=expl,
                        )
                    )
                    seen.add((i, j))

    # ================================================================== Value normalizers

    def _normalize_value_for_field(
        self,
        field_name: str,
        raw_value: Any,
    ) -> tuple[Any, Optional[str]]:
        if raw_value is None:
            return None, "empty"
        s = str(raw_value).strip()
        if not s:
            return None, "empty"
        fn = field_name
        if fn in {"publish_date", "effective_date", "expiry_date", "creation_date"}:
            return normalize_date(s)
        if fn == "document_number":
            raw_s, norm, status = normalize_document_number(s)
            if status is FieldStatus.RESOLVED:
                return norm, None
            return None, "document_number normalization failed"
        if fn == "document_index_number":
            return re.sub(r"\s+", "", s), None
        if fn in {"source_org", "title", "document_type", "authority_level",
                   "legal_status", "jurisdiction", "region", "theme_category"}:
            return re.sub(r"\s+", " ", s).strip(), None
        if fn in {"document_id", "logical_document_id", "source_url", "source_file"}:
            return s.strip(), None
        return s, None

    # ================================================================== Helpers

    @staticmethod
    def _split_single_block_kv(text: str) -> Optional[tuple[str, str]]:
        t = text.strip()
        if not t:
            return None
        for sep in ["：", ":", "\t", "  "]:
            if sep in t:
                idx = t.find(sep)
                if 0 < idx < len(t) - 1:
                    k = t[:idx].strip().rstrip(":：").strip()
                    v = t[idx + len(sep):].strip()
                    if k and v:
                        return k, v
        return None

    def _register_candidate(
        self,
        cand: RawMetadataCandidate,
        field_results: dict[str, FieldResult],
    ) -> None:
        canon = cand.canonical_field
        if canon is None:
            canon = "__UNKNOWN__"
        if canon not in field_results:
            field_results[canon] = FieldResult(field_name=canon)
        field_results[canon].add_candidate(cand)
        self._all_candidates.append(cand)


# =============================================================================
# §17/§18 — Field Matrix (aggregation for retrieval readiness)
# =============================================================================

@dataclass
class FieldMatrixRow:
    field_name: str
    docs_checked: int = 0
    docs_present: int = 0
    docs_resolved: int = 0
    docs_conflict: int = 0
    docs_normalization_failed: int = 0
    distinct_values: int = 0
    avg_confidence: float = 0.0
    example_values: list[str] = field(default_factory=list)
    filter_value_assessment: str = ""
    future_storage_recommendation: StorageRecommendation = StorageRecommendation.UNDECIDED

    def to_csv_row(self) -> list[str]:
        cov = (
            f"{100.0 * self.docs_present / self.docs_checked:.1f}%"
            if self.docs_checked else "n/a"
        )
        return [
            self.field_name,
            str(self.docs_checked),
            str(self.docs_present),
            cov,
            str(self.docs_resolved),
            str(self.docs_conflict),
            str(self.docs_normalization_failed),
            str(self.distinct_values),
            f"{self.avg_confidence:.3f}",
            " | ".join(self.example_values[:5]),
            self.filter_value_assessment,
            self.future_storage_recommendation.value,
        ]


FIELD_MATRIX_CSV_HEADER = [
    "field_name", "docs_checked", "docs_present", "coverage_pct",
    "docs_resolved", "docs_conflict", "docs_normalization_failed",
    "distinct_values", "avg_confidence", "example_values",
    "filter_value_assessment", "future_storage_recommendation",
]


def build_field_matrix(
    results: list[MetadataNormalizationResult],
) -> dict[str, FieldMatrixRow]:
    per_field: dict[str, FieldMatrixRow] = {}
    all_values: dict[str, set[str]] = {}
    confidences: dict[str, list[float]] = {}
    examples: dict[str, list[str]] = {}
    for res in results:
        for fn, fr in res.field_results.items():
            if fn == "__UNKNOWN__":
                continue
            row = per_field.setdefault(fn, FieldMatrixRow(field_name=fn))
            row.docs_checked += 1
            if fr.raw_values:
                row.docs_present += 1
            if fr.status is FieldStatus.RESOLVED:
                row.docs_resolved += 1
            elif fr.status is FieldStatus.CONFLICT:
                row.docs_conflict += 1
            elif fr.status is FieldStatus.NORMALIZATION_FAILED:
                row.docs_normalization_failed += 1
            if fr.confidence > 0:
                confidences.setdefault(fn, []).append(fr.confidence)
            if fr.status is FieldStatus.RESOLVED and fr.normalized_value is not None:
                vs = str(fr.normalized_value)
                all_values.setdefault(fn, set()).add(vs)
                ex = examples.setdefault(fn, [])
                if vs not in ex and len(ex) < 8:
                    ex.append(vs)
    for fn, row in per_field.items():
        vals = all_values.get(fn, set())
        row.distinct_values = len(vals)
        confs = confidences.get(fn, [])
        row.avg_confidence = (sum(confs) / len(confs)) if confs else 0.0
        row.example_values = examples.get(fn, [])
        coverage = (row.docs_present / row.docs_checked) if row.docs_checked else 0.0
        resolved_ratio = (row.docs_resolved / row.docs_present) if row.docs_present else 0.0
        stable = resolved_ratio >= 0.8 and row.docs_conflict == 0
        high_cov = coverage >= 0.7
        if fn in {"title", "logical_document_id", "document_id", "source_file", "source_url"}:
            row.filter_value_assessment = "Provenance / identity — always needed."
            row.future_storage_recommendation = (
                StorageRecommendation.MILVUS_SCALAR_CANDIDATE
                if fn in {"logical_document_id", "document_id", "source_file"}
                else StorageRecommendation.JSON_METADATA
            )
        elif fn in {"document_type", "source_org", "legal_status", "authority_level"}:
            row.filter_value_assessment = "Typical equality/IN filter for retrieval."
            if high_cov and stable:
                row.future_storage_recommendation = StorageRecommendation.MILVUS_SCALAR_CANDIDATE
            elif coverage > 0.3:
                row.future_storage_recommendation = StorageRecommendation.MYSQL_ONLY
            else:
                row.future_storage_recommendation = StorageRecommendation.JSON_METADATA
        elif fn in {"effective_date", "publish_date", "expiry_date", "creation_date"}:
            row.filter_value_assessment = "Temporal range filter — high value if reliable."
            if high_cov and stable and row.avg_confidence >= 0.7:
                row.future_storage_recommendation = StorageRecommendation.MILVUS_SCALAR_CANDIDATE
            elif coverage > 0.3:
                row.future_storage_recommendation = StorageRecommendation.MYSQL_ONLY
            else:
                row.future_storage_recommendation = StorageRecommendation.JSON_METADATA
        elif fn == "document_number":
            row.filter_value_assessment = "Citation / exact lookup candidate."
            if high_cov and stable:
                row.future_storage_recommendation = StorageRecommendation.MILVUS_SCALAR_CANDIDATE
            else:
                row.future_storage_recommendation = StorageRecommendation.CITATION_ONLY
        elif fn in {"jurisdiction", "region", "theme_category"}:
            row.filter_value_assessment = "Regional/category filter candidate."
            if high_cov and stable and row.distinct_values <= 500:
                row.future_storage_recommendation = StorageRecommendation.MILVUS_SCALAR_CANDIDATE
            elif coverage > 0.2:
                row.future_storage_recommendation = StorageRecommendation.MYSQL_ONLY
            else:
                row.future_storage_recommendation = StorageRecommendation.JSON_METADATA
        else:
            row.future_storage_recommendation = StorageRecommendation.UNDECIDED
    return per_field


__all__ = [
    "FieldStatus", "DuplicateType", "MetadataSourceTier", "SOURCE_TIER_NAME",
    "StorageRecommendation",
    "RawMetadataCandidate", "FieldResult", "DuplicateGroup", "MetadataConflict",
    "MetadataNormalizationResult",
    "FIELD_ALIASES", "CANONICAL_FIELDS_V1",
    "MetadataNormalizerV1",
    "normalize_date", "normalize_document_number",
    "FieldMatrixRow", "FIELD_MATRIX_CSV_HEADER", "build_field_matrix",
]
