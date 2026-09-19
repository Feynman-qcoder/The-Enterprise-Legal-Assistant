"""
Chunk Metadata Contract V1 — Stable Schema for Future Chunking Strategies
==========================================================================

ISOLATED. Stdlib only: dataclasses / enum / typing / hashlib / json.

DOES NOT depend on: Parser V2, Cleaner V2.1, MetadataNormalizer V1, QGate V2.4,
                    Milvus, MySQL, Embedding, Re-ingestion.

Design Principles (§2):
  Metadata separated into 5 layers (A..E).  No flattening.

  A. IDENTITY            — chunk + document + source immutable ids
  B. DOCUMENT METADATA   — Frozen projection from Metadata Contract V1 (INHERIT only)
  C. STRUCTURAL METADATA — chapter / section / article / heading_path ...
  D. PROVENANCE          — back to ParsedDocument V2 blocks, mandatory
  E. RETRIEVAL / RUNTIME — table lineage, parent/child, quality flags

Scope:
  - Typed contract dataclasses
  - 16 deterministic validation invariants
  - Deterministic chunk_id generation contract (rules only, no actual chunks)
  - Document metadata projection mapping table (Metadata Contract V1 → Chunk)
  - Embedding vs metadata policy classification
  - Milvus filter/scalar/json auxiliary classification

Version: CHUNK_METADATA_CONTRACT_V1.0
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict, fields as _dc_fields
from enum import Enum
from typing import Any, Optional


# ============================================================================
# §18 — NO UPSTREAM MODIFICATION ASSERTION
# This module NEVER imports or mutates Parser/Cleaner/QGate modules.
# ============================================================================


# ============================================================================
# §2 Layer Enums (explicit 5-layer taxonomy, never flattened)
# ============================================================================

class MetadataLayer(str, Enum):
    """The 5 contract layers. Every field belongs to exactly one layer."""
    A_IDENTITY = "A_IDENTITY"
    B_DOCUMENT_METADATA = "B_DOCUMENT_METADATA"
    C_STRUCTURAL = "C_STRUCTURAL"
    D_PROVENANCE = "D_PROVENANCE"
    E_RETRIEVAL_RUNTIME = "E_RETRIEVAL_RUNTIME"


class SourceFormat(str, Enum):
    """Canonical source formats (matches manifest format values lowercased)."""
    MD = "md"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    OTHER = "other"


# ============================================================================
# §8 PARENT / CHILD — optional, schema-stable, strategy-agnostic values
# ============================================================================

class ChunkLevel(str, Enum):
    """Future-proof values; unknown/new strategies can add without schema break.

    Usage example (not forced by contract now):
      PARENT  — big semantic container (e.g. page-group, section)
      CHILD   — sliding-window / small indexed child inside a PARENT
      ATOMIC  — one-level flat chunking, no parent/child relation applied
    """
    PARENT = "PARENT"
    CHILD = "CHILD"
    ATOMIC = "ATOMIC"


# ============================================================================
# §9 Milvus filter classification — PRE-DECISION taxonomy only
# ============================================================================

class FilterCandidateClass(str, Enum):
    FILTER_SCALAR_CANDIDATE = "FILTER_SCALAR_CANDIDATE"
    JSON_AUXILIARY = "JSON_AUXILIARY"
    NOT_FILTERED = "NOT_FILTERED"


# ============================================================================
# §10 Embedding-text vs metadata policy classification
# ============================================================================

class EmbeddingPolicy(str, Enum):
    """Whether the field value may appear in chunk.text used for embedding.

    A. RETRIEVAL_FILTER_ONLY — Milvus/MySQL filtering only; NEVER into text
    B. CITATION_ONLY        — Citation/provenance only; NEVER into embedding text
    C. OPTIONAL_CONTEXT_PREFIX — Chunking Strategy MAY prepend for contextualized
                                 embedding.  Decision is strategy-owned.
    D. NEVER_EMBED          — Internal / quality status / sentinel values.
                             MUST NOT pollute embedding text EVER.
    """
    RETRIEVAL_FILTER_ONLY = "RETRIEVAL_FILTER_ONLY"        # A
    CITATION_ONLY = "CITATION_ONLY"                          # B
    OPTIONAL_CONTEXT_PREFIX = "OPTIONAL_CONTEXT_PREFIX"      # C
    NEVER_EMBED = "NEVER_EMBED"                               # D


# ============================================================================
# §15 Source owner taxonomy (for field matrix ownership accountability)
# ============================================================================

class FieldSourceOwner(str, Enum):
    """Who computes/owns the canonical value for a given field.

    PARSER                             — Frozen ParsedDocument V2 attribute
    METADATA_NORMALIZER                — Frozen Metadata Contract V1 / Normalizer V1
    CHUNKER                            — Chunker computes from blocks/text (future)
    QGATE                              — Quality Gate V2.4 verdict/signals
    DERIVED_DETERMINISTIC              — Pure deterministic derivation (e.g. SHA)
    """
    PARSER = "PARSER"
    METADATA_NORMALIZER = "METADATA_NORMALIZER"
    CHUNKER = "CHUNKER"
    QGATE = "QGATE"
    DERIVED_DETERMINISTIC = "DERIVED_DETERMINISTIC"


# ============================================================================
# §11 D — Internal sentinels (MUST NOT leak into contract public values)
# ============================================================================

INTERNAL_SENTINEL_VALUES: set[str] = {
    "__METADATA_BLOCK_VALUE__",
    "__UNSET__",
    "__NONE__",
    "__CHUNKER_INTERNAL__",
}
PUBLIC_CONTRACT_FORBIDDEN_SUBSTRINGS: set[str] = set(INTERNAL_SENTINEL_VALUES)


def _contains_forbidden_sentinel(value: Any) -> bool:
    """Return True if any string member contains a forbidden sentinel."""
    if value is None:
        return False
    if isinstance(value, str):
        return any(s in value for s in PUBLIC_CONTRACT_FORBIDDEN_SUBSTRINGS)
    if isinstance(value, list):
        return any(_contains_forbidden_sentinel(x) for x in value)
    if isinstance(value, dict):
        return any(
            _contains_forbidden_sentinel(k) or _contains_forbidden_sentinel(v)
            for k, v in value.items()
        )
    return False


# ============================================================================
# §3 IDENTITY Layer — required, non-empty, deterministic rules
# ============================================================================

# chunk_id deterministic generation contract (RULES ONLY, no actual chunks):
#
#   chunk_id = SHA256(
#       "CMCV1|"                            +  # contract version salt
#       source_sha256                       +  # original source bytes identity
#       "|" + document_id                   +  # ParsedDocument identity
#       "|" + str(chunk_index)              +  # per-chunk 0-based ordinal
#       "|" + str(original_block_order_start) +
#       "|" + str(original_block_order_end)
#   ).hexdigest()[:32]   (32 hex chars == 128-bit collision resistance)
#
# Rationale:
#   * Deterministic — same source + same chunk boundaries → same chunk_id.
#   * Survives re-chunking with same boundaries.
#   * Does NOT depend on chunk text (which might vary by strategy).
#   * Derived from SOURCE IDENTITY + DOCUMENT ID + ORDINAL + PROVENANCE RANGE.

CHUNK_ID_CONTRACT_SALT = "CMCV1"
CHUNK_ID_HEX_LENGTH = 32


def compute_chunk_id_contract(
    *,
    source_sha256: str,
    document_id: str,
    chunk_index: int,
    original_block_order_start: int,
    original_block_order_end: int,
) -> str:
    """Deterministic chunk_id builder per frozen contract rule above.

    NOTE: This is the RULE REFERENCE implementation. Actual Chunker V2 should
    reuse exactly this function or re-implement the byte-by-byte identical
    concatenation + SHA256 + slice.  The value must be byte-deterministic.
    """
    if not source_sha256:
        raise ChunkContractViolation("compute_chunk_id_contract: source_sha256 required")
    if not document_id:
        raise ChunkContractViolation("compute_chunk_id_contract: document_id required")
    if chunk_index < 0:
        raise ChunkContractViolation(f"chunk_index must be >=0, got {chunk_index}")
    if original_block_order_start < 0:
        raise ChunkContractViolation(
            f"original_block_order_start must be >=0, got {original_block_order_start}"
        )
    if original_block_order_end < original_block_order_start:
        raise ChunkContractViolation(
            f"original_block_order_end ({original_block_order_end}) < start "
            f"({original_block_order_start})"
        )
    payload = (
        f"{CHUNK_ID_CONTRACT_SALT}|{source_sha256}|{document_id}|"
        f"{chunk_index}|{original_block_order_start}|{original_block_order_end}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:CHUNK_ID_HEX_LENGTH]


# ============================================================================
# §4 Document Metadata Projection Mapping Table
# ============================================================================
# Metadata Contract V1 fields that have document_to_chunk_inheritance == "INHERIT"
# or otherwise must be projected verbatim into chunk metadata (no overwrite).
#
# Rules:
#   * Chunk inherits → chunk field name = original name (no rename).
#   * Normalizer owns the canonical value; Chunk MUST NOT overwrite.
#   * null stays null.
#   * No Last-Write-Wins EVER (same frozen invariant as Metadata Contract V1).

@dataclass(frozen=True)
class DocumentMetadataProjection:
    """Single row in DOCUMENT → CHUNK projection mapping table."""
    document_metadata_field: str
    chunk_metadata_field: str          # same as doc field for INHERIT
    required: bool                     # required at chunk layer?
    filterable: FilterCandidateClass   # filter candidate classification
    citation_relevant: bool            # used in legal citation output?
    embedding_policy: EmbeddingPolicy


DOCUMENT_METADATA_PROJECTION_TABLE: list[DocumentMetadataProjection] = [
    # --- Frozen Metadata Contract V1 18 fields with INHERIT policy ---
    DocumentMetadataProjection(
        "document_id", "document_id", True,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.CITATION_ONLY,
    ),
    DocumentMetadataProjection(
        "logical_document_id", "logical_document_id", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.CITATION_ONLY,
    ),
    DocumentMetadataProjection(
        "title", "title", True,
        FilterCandidateClass.JSON_AUXILIARY, True,
        EmbeddingPolicy.OPTIONAL_CONTEXT_PREFIX,
    ),
    DocumentMetadataProjection(
        "source_org", "source_org", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.OPTIONAL_CONTEXT_PREFIX,
    ),
    DocumentMetadataProjection(
        "publish_date", "publish_date", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "creation_date", "creation_date", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "effective_date", "effective_date", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "expiry_date", "expiry_date", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "document_number", "document_number", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.CITATION_ONLY,
    ),
    DocumentMetadataProjection(
        "document_index_number", "document_index_number", False,
        FilterCandidateClass.JSON_AUXILIARY, False,
        EmbeddingPolicy.CITATION_ONLY,
    ),
    DocumentMetadataProjection(
        "document_type", "document_type", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "authority_level", "authority_level", False,
        FilterCandidateClass.JSON_AUXILIARY, True,
        EmbeddingPolicy.OPTIONAL_CONTEXT_PREFIX,
    ),
    DocumentMetadataProjection(
        "legal_status", "legal_status", False,
        FilterCandidateClass.JSON_AUXILIARY, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "jurisdiction", "jurisdiction", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, True,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "region", "region", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, False,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "theme_category", "theme_category", False,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, False,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
    DocumentMetadataProjection(
        "source_url", "source_url", False,
        FilterCandidateClass.JSON_AUXILIARY, False,
        EmbeddingPolicy.CITATION_ONLY,
    ),
    DocumentMetadataProjection(
        "source_file", "source_file", True,
        FilterCandidateClass.FILTER_SCALAR_CANDIDATE, False,
        EmbeddingPolicy.RETRIEVAL_FILTER_ONLY,
    ),
]


DOCUMENT_METADATA_PROJECTION_MAP: dict[str, DocumentMetadataProjection] = {
    p.document_metadata_field: p
    for p in DOCUMENT_METADATA_PROJECTION_TABLE
}


# ============================================================================
# Contract-level exception (§13 deterministic validation)
# ============================================================================

class ChunkContractViolation(ValueError):
    """Raised when any Chunk Metadata Contract invariant is broken."""


# ============================================================================
# LAYER C — Structural Metadata (chapter / section / article deferred fields
#           finally promoted at Chunk stage; §5 + Metadata Contract V1 §10)
# ============================================================================

@dataclass
class ChunkStructuralContext:
    """Within-document structural context.  Nullable everywhere.

    Semantics frozen (§5):
      chapter       — 章 / top-level segment, e.g. "第一编 合同"
      section       — 节 / mid-level segment, e.g. "第三章 合同的效力"
      article       — 条 / low-level unit, e.g. "第五百零二条"
      heading_path  — heading hierarchy leading to (and including) this chunk.
                      Example: ["第三编 合同", "第二章 合同的订立"]
                      Must be ordered from outer (top heading) to inner (closest
                      heading that precedes or opens the chunk content).
      heading_level — closest enclosing heading level (1..9) if known, else null.
      block_type    — dominant content block type in the chunk.
                      Values mirror ParsedDocument.BlockType:
                      HEADING | PARAGRAPH | LIST_ITEM | TABLE | METADATA | UNKNOWN
                      If the chunk mixes types, block_type is the most common
                      non-METADATA non-UNKNOWN type, else the dominant type.
      content_type  — coarse semantic classification of the chunk's content:
                      LEGAL_TEXT | HEADING_ONLY | TABLE_ONLY | MIXED_TEXT_TABLE
                      | METADATA_SEGMENT | REVIEW_CONTENT | UNKNOWN
    """

    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    heading_path: list[str] = field(default_factory=list)
    heading_level: Optional[int] = None
    block_type: Optional[str] = None
    content_type: Optional[str] = None

    # --- frozen allowed block_type vocabulary (ParsedDocument V2 block types)
    ALLOWED_BLOCK_TYPES = {
        "HEADING", "PARAGRAPH", "LIST_ITEM", "TABLE", "METADATA", "UNKNOWN", None,
    }
    ALLOWED_CONTENT_TYPES = {
        "LEGAL_TEXT", "HEADING_ONLY", "TABLE_ONLY", "MIXED_TEXT_TABLE",
        "METADATA_SEGMENT", "REVIEW_CONTENT", "UNKNOWN", None,
    }

    def __post_init__(self) -> None:
        # chapter / section / article — null or non-empty (whitespace-only → treat as empty)
        for attr in ("chapter", "section", "article"):
            v = getattr(self, attr)
            if v is None:
                continue
            if not isinstance(v, str):
                raise ChunkContractViolation(
                    f"structural.{attr} must be str or None, got {type(v)}"
                )
            if not v.strip():
                raise ChunkContractViolation(
                    f"structural.{attr} must be non-empty whitespace-free or None"
                )
        # heading_level 1..9 OR None
        if self.heading_level is not None:
            if not isinstance(self.heading_level, int):
                raise ChunkContractViolation(
                    f"heading_level must be int 1..9 or None, got {type(self.heading_level)}"
                )
            if self.heading_level < 1 or self.heading_level > 9:
                raise ChunkContractViolation(
                    f"heading_level out of range 1..9: {self.heading_level}"
                )
        # block_type restricted vocabulary
        if self.block_type not in self.ALLOWED_BLOCK_TYPES:
            raise ChunkContractViolation(
                f"block_type {self.block_type!r} not in allowed set "
                f"{sorted(x for x in self.ALLOWED_BLOCK_TYPES if x is not None)}"
            )
        # content_type restricted vocabulary
        if self.content_type not in self.ALLOWED_CONTENT_TYPES:
            raise ChunkContractViolation(
                f"content_type {self.content_type!r} not in allowed set "
                f"{sorted(x for x in self.ALLOWED_CONTENT_TYPES if x is not None)}"
            )
        # heading_path: must be list of non-empty strings if present
        if not isinstance(self.heading_path, list):
            raise ChunkContractViolation("heading_path must be a list of str")
        for i, s in enumerate(self.heading_path):
            if not isinstance(s, str):
                raise ChunkContractViolation(f"heading_path[{i}] must be str")
            if not s.strip():
                raise ChunkContractViolation(f"heading_path[{i}] must be non-empty")


# ============================================================================
# LAYER D — Provenance (TRACEABILITY to ParsedDocument V2 blocks; MANDATORY)
# ============================================================================

@dataclass
class ChunkProvenance:
    """Mandatory trace chain Chunk → ParsedDocument V2 blocks.

    Invariants (enforced in __post_init__ + ChunkMetadata.validate):
      - source_block_orders         — non-empty, ints, monotonic strictly increasing,
                                      no duplicates.
      - source_block_ids            — optional; if non-empty must have the same
                                      count as source_block_orders (1:1 mapping).
      - original_block_order_start  — min(source_block_orders)
      - original_block_order_end    — max(source_block_orders)
      - page_start ≤ page_end       — when both present.
      - page_numbers subset         — if non-empty all pages must fall within
                                      [page_start, page_end].
    """

    source_block_orders: list[int]
    source_block_ids: list[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    page_numbers: list[int] = field(default_factory=list)
    original_block_order_start: int = -1
    original_block_order_end: int = -1
    parser_name: str = ""
    parser_version: str = ""

    def __post_init__(self) -> None:
        # --- source_block_orders: non-empty list of ints ---
        if not self.source_block_orders:
            raise ChunkContractViolation(
                "ChunkProvenance.source_block_orders cannot be empty (mandatory provenance)"
            )
        if not isinstance(self.source_block_orders, list):
            raise ChunkContractViolation("source_block_orders must be list[int]")
        prev = None
        seen: set[int] = set()
        for v in self.source_block_orders:
            if not isinstance(v, int):
                raise ChunkContractViolation(
                    f"source_block_orders entries must be int, got {v!r}"
                )
            if v < 0:
                raise ChunkContractViolation(
                    f"source_block_orders entries must be >=0, got {v}"
                )
            if prev is not None and v <= prev:
                raise ChunkContractViolation(
                    f"source_block_orders not strictly monotonic increasing: "
                    f"..., {prev}, {v}"
                )
            if v in seen:
                raise ChunkContractViolation(
                    f"duplicate block_order in source_block_orders: {v}"
                )
            seen.add(v)
            prev = v

        # --- start / end derived consistency ---
        expected_start = min(self.source_block_orders)
        expected_end = max(self.source_block_orders)
        if self.original_block_order_start == -1:
            object.__setattr__(self, "original_block_order_start", expected_start)
        if self.original_block_order_end == -1:
            object.__setattr__(self, "original_block_order_end", expected_end)
        if self.original_block_order_start != expected_start:
            raise ChunkContractViolation(
                f"original_block_order_start={self.original_block_order_start} "
                f"≠ min(source_block_orders)={expected_start}"
            )
        if self.original_block_order_end != expected_end:
            raise ChunkContractViolation(
                f"original_block_order_end={self.original_block_order_end} "
                f"≠ max(source_block_orders)={expected_end}"
            )

        # --- source_block_ids: if present 1:1 with block_orders ---
        if self.source_block_ids and len(self.source_block_ids) != len(self.source_block_orders):
            raise ChunkContractViolation(
                f"source_block_ids count {len(self.source_block_ids)} ≠ "
                f"source_block_orders count {len(self.source_block_orders)}; "
                "must be 1:1 or empty"
            )
        for i, s in enumerate(self.source_block_ids):
            if not isinstance(s, str) or not s:
                raise ChunkContractViolation(
                    f"source_block_ids[{i}] must be non-empty str"
                )

        # --- page range ---
        if self.page_start is not None and self.page_start < 1:
            raise ChunkContractViolation(f"page_start must be >=1 or None, got {self.page_start}")
        if self.page_end is not None and self.page_end < 1:
            raise ChunkContractViolation(f"page_end must be >=1 or None, got {self.page_end}")
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ChunkContractViolation(
                    f"page_end ({self.page_end}) < page_start ({self.page_start})"
                )
        # page_numbers must be within [page_start, page_end] when both set
        if self.page_numbers:
            if not isinstance(self.page_numbers, list):
                raise ChunkContractViolation("page_numbers must be list[int]")
            for pn in self.page_numbers:
                if not isinstance(pn, int) or pn < 1:
                    raise ChunkContractViolation(
                        f"page_numbers entries must be int >=1, got {pn!r}"
                    )
                if self.page_start is not None and pn < self.page_start:
                    raise ChunkContractViolation(
                        f"page_numbers entry {pn} < page_start {self.page_start}"
                    )
                if self.page_end is not None and pn > self.page_end:
                    raise ChunkContractViolation(
                        f"page_numbers entry {pn} > page_end {self.page_end}"
                    )


# ============================================================================
# LAYER E — Table Lineage metadata (§7 TABLE support, no table splitting yet)
# ============================================================================

@dataclass
class ChunkTableContext:
    """Table lineage for chunks that intersect TABLE blocks.

    If a future TABLE block is SPLIT into multiple chunks:
      * table_ids / source_table_block_orders still refer to the original table.
      * table_part_index numbers the consecutive pieces (0-based).
      * table_header_preserved must be True on every piece that re-replicates
        the header for standalone readability.
      * table_structure_preserved is True only for chunks that carry the full
        TableData. Chunks that only carry table rows as textual fragments get
        table_structure_preserved=False and become text fallback only.

    For NON-TABLE chunks, leave defaults (contains_table=False) and all-other
    empty/null.  Structured TableData remains authoritative; Markdown is only
    a projection.
    """

    contains_table: bool = False
    table_ids: list[str] = field(default_factory=list)
    source_table_block_orders: list[int] = field(default_factory=list)
    table_header_preserved: bool = False
    table_structure_preserved: bool = False
    table_part_index: Optional[int] = None
    table_parts_total: Optional[int] = None

    def __post_init__(self) -> None:
        if self.contains_table:
            if not self.table_ids and not self.source_table_block_orders:
                raise ChunkContractViolation(
                    "contains_table=True requires non-empty table_ids OR "
                    "source_table_block_orders"
                )
            # monotonic unique block orders
            if self.source_table_block_orders:
                prev = None
                seen: set[int] = set()
                for v in self.source_table_block_orders:
                    if not isinstance(v, int) or v < 0:
                        raise ChunkContractViolation(
                            f"source_table_block_orders invalid entry: {v!r}"
                        )
                    if prev is not None and v <= prev:
                        raise ChunkContractViolation(
                            "source_table_block_orders not strictly monotonic"
                        )
                    if v in seen:
                        raise ChunkContractViolation(
                            f"duplicate in source_table_block_orders: {v}"
                        )
                    seen.add(v)
                    prev = v
            if self.table_ids:
                for i, t in enumerate(self.table_ids):
                    if not isinstance(t, str) or not t:
                        raise ChunkContractViolation(
                            f"table_ids[{i}] must be non-empty str"
                        )
            if self.table_part_index is not None:
                if self.table_part_index < 0:
                    raise ChunkContractViolation(
                        f"table_part_index must be >=0, got {self.table_part_index}"
                    )
            if self.table_parts_total is not None:
                if self.table_parts_total < 1:
                    raise ChunkContractViolation(
                        f"table_parts_total must be >=1, got {self.table_parts_total}"
                    )
                if self.table_part_index is not None and self.table_part_index >= self.table_parts_total:
                    raise ChunkContractViolation(
                        f"table_part_index {self.table_part_index} >= "
                        f"table_parts_total {self.table_parts_total}"
                    )
        else:
            # contains_table=False → all other table fields must be empty/default
            if self.table_ids or self.source_table_block_orders:
                raise ChunkContractViolation(
                    "contains_table=False but table_ids/source_table_block_orders set"
                )
            if self.table_header_preserved:
                raise ChunkContractViolation(
                    "contains_table=False but table_header_preserved=True"
                )
            if self.table_structure_preserved:
                raise ChunkContractViolation(
                    "contains_table=False but table_structure_preserved=True"
                )
            if self.table_part_index is not None or self.table_parts_total is not None:
                raise ChunkContractViolation(
                    "contains_table=False but table_part_index/table_parts_total set"
                )


# ============================================================================
# LAYER E — Quality / Review flags (§11 — metadata, NOT embedded legal text)
# ============================================================================

@dataclass
class ChunkQualityFlags:
    """QGate / review lineage.  Must remain metadata, never legal content.

    quality_verdict values are QGate-owned: PASS | WARNING | POLICY_REVIEW | REVIEW | null
    has_review_content          — True if any Cleaner REVIEW decision lies inside
                                  the provenance of this chunk.
    review_rule_ids             — opaque rule identifiers that fired REVIEW for
                                  this chunk's provenance range (informational).
    """

    quality_verdict: Optional[str] = None
    quality_warnings: list[str] = field(default_factory=list)
    has_review_content: bool = False
    review_rule_ids: list[str] = field(default_factory=list)

    ALLOWED_QUALITY_VERDICTS = {
        "PASS", "WARNING", "POLICY_REVIEW", "REVIEW", None,
    }

    def __post_init__(self) -> None:
        if self.quality_verdict not in self.ALLOWED_QUALITY_VERDICTS:
            raise ChunkContractViolation(
                f"quality_verdict {self.quality_verdict!r} not in allowed set "
                f"{sorted(x for x in self.ALLOWED_QUALITY_VERDICTS if x is not None)}"
            )
        for i, w in enumerate(self.quality_warnings):
            if not isinstance(w, str) or not w:
                raise ChunkContractViolation(
                    f"quality_warnings[{i}] must be non-empty str"
                )
        for i, r in enumerate(self.review_rule_ids):
            if not isinstance(r, str) or not r:
                raise ChunkContractViolation(
                    f"review_rule_ids[{i}] must be non-empty str"
                )


# ============================================================================
# LAYER E — Parent/Child Compatibility (§8 — optional, schema-stable)
# ============================================================================

@dataclass
class ChunkParentChildInfo:
    parent_chunk_id: Optional[str] = None
    chunk_level: Optional[str] = None           # ChunkLevel value or null
    parent_document_id: Optional[str] = None    # cross-doc parent (rare)

    def __post_init__(self) -> None:
        if self.chunk_level is not None:
            allowed = {c.value for c in ChunkLevel}
            if self.chunk_level not in allowed:
                raise ChunkContractViolation(
                    f"chunk_level {self.chunk_level!r} not in allowed set {sorted(allowed)}"
                )
        if self.parent_chunk_id is not None:
            if not isinstance(self.parent_chunk_id, str) or not self.parent_chunk_id:
                raise ChunkContractViolation(
                    "parent_chunk_id must be non-empty str or None"
                )
        if self.parent_document_id is not None:
            if not isinstance(self.parent_document_id, str) or not self.parent_document_id:
                raise ChunkContractViolation(
                    "parent_document_id must be non-empty str or None"
                )


# ============================================================================
# TOP LEVEL — ChunkMetadata (composition of all 5 layers)
# ============================================================================

@dataclass
class ChunkMetadata:
    """Stable typed Chunk Metadata Contract V1.

    Layers:
      A. IDENTITY                — top-level scalar fields: chunk_id / document_id /
                                   source_file / source_sha256 / source_format /
                                   chunk_index.
      B. DOCUMENT METADATA       — projected via Frozen Metadata Contract V1
                                   INHERIT policy (see DOCUMENT_METADATA_PROJECTION_MAP).
                                   Embedded as scalar fields directly on the object
                                   for easy Milvus scalar access.
      C. STRUCTURAL              — ChunkStructuralContext (chapter/section/article ...)
      D. PROVENANCE              — ChunkProvenance (block orders / pages / parser)
      E. RETRIEVAL / RUNTIME     — ChunkTableContext, ChunkParentChildInfo,
                                   ChunkQualityFlags.
    """

    # ------- LAYER A: IDENTITY -------
    chunk_id: str
    document_id: str
    logical_document_id: Optional[str]
    source_file: str
    source_sha256: str
    source_format: str
    chunk_index: int

    # ------- LAYER B: DOCUMENT METADATA PROJECTION -------
    # (Frozen Metadata Contract V1 fields with INHERIT policy, null-preserving)
    title: Optional[str] = None
    source_org: Optional[str] = None
    publish_date: Optional[str] = None
    creation_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    document_number: Optional[str] = None
    document_index_number: Optional[str] = None
    document_type: Optional[str] = None
    authority_level: Optional[str] = None
    legal_status: Optional[str] = None
    jurisdiction: Optional[str] = None
    region: Optional[str] = None
    theme_category: Optional[str] = None
    source_url: Optional[str] = None

    # version tag for contract versioning (never embedded)
    contract_version: str = "CHUNK_METADATA_CONTRACT_V1.0"

    # ------- LAYER C: STRUCTURAL -------
    structural: ChunkStructuralContext = field(
        default_factory=ChunkStructuralContext
    )

    # ------- LAYER D: PROVENANCE -------
    provenance: ChunkProvenance = field(default_factory=lambda: ChunkProvenance(source_block_orders=[-1]))  # placeholder; will fail unless overridden

    # ------- LAYER E: RETRIEVAL / RUNTIME -------
    table_context: ChunkTableContext = field(default_factory=ChunkTableContext)
    parent_child: ChunkParentChildInfo = field(default_factory=ChunkParentChildInfo)
    quality: ChunkQualityFlags = field(default_factory=ChunkQualityFlags)

    # ------- STRICT SCHEMA: unknown extra fields MUST be rejected via this dict -------
    # If a project uses strict schema mode, consumers should check extras dict is empty.
    extras: dict[str, Any] = field(default_factory=dict)

    # ==================================================================
    # §13 Deterministic Validation Invariants (16 total, frozen list)
    # ==================================================================

    def validate(self) -> list[str]:
        """Run ALL frozen validation invariants.

        Returns:
          List[str] of violation messages.  Empty list → PASS.
        Throws:
          ChunkContractViolation on structural malformation that prevents
          invariant evaluation (e.g. provenance not set).
        """
        violations: list[str] = []

        # --- Sub-object validation (__post_init__ already ran at construction,
        #     re-run here for enforce-after-mutate usage). ---
        try:
            self.structural.__post_init__()
        except ChunkContractViolation as exc:
            violations.append(f"STRUCTURAL: {exc}")
        try:
            self.provenance.__post_init__()
        except ChunkContractViolation as exc:
            violations.append(f"PROVENANCE: {exc}")
        try:
            self.table_context.__post_init__()
        except ChunkContractViolation as exc:
            violations.append(f"TABLE_CONTEXT: {exc}")
        try:
            self.parent_child.__post_init__()
        except ChunkContractViolation as exc:
            violations.append(f"PARENT_CHILD: {exc}")
        try:
            self.quality.__post_init__()
        except ChunkContractViolation as exc:
            violations.append(f"QUALITY: {exc}")

        # --- INVARIANT 1: chunk_id non-empty & correct length format ---
        if not self.chunk_id:
            violations.append("INV1: chunk_id cannot be empty")
        elif not isinstance(self.chunk_id, str):
            violations.append("INV1: chunk_id must be str")
        elif len(self.chunk_id) != 32:
            violations.append(
                f"INV1: chunk_id length must be 32 hex chars, got {len(self.chunk_id)}"
            )
        elif any(c not in "0123456789abcdef" for c in self.chunk_id):
            violations.append("INV1: chunk_id must be lowercase hex")

        # --- INVARIANT 2: document identity non-empty ---
        if not self.document_id:
            violations.append("INV2: document_id cannot be empty")

        # --- INVARIANT 3: source identity non-empty ---
        if not self.source_file:
            violations.append("INV3: source_file cannot be empty")
        if not self.source_sha256:
            violations.append("INV3: source_sha256 cannot be empty")
        elif not isinstance(self.source_sha256, str):
            violations.append("INV3: source_sha256 must be str")
        elif len(self.source_sha256) != 64:
            violations.append(
                f"INV3: source_sha256 must be 64 hex chars, got {len(self.source_sha256)}"
            )
        elif any(c not in "0123456789abcdefABCDEF" for c in self.source_sha256):
            violations.append("INV3: source_sha256 must be hex string")

        # source_format enum check
        try:
            SourceFormat(self.source_format.lower())
        except ValueError:
            # "other" falls back; accept any non-empty lower-case-able string
            pass
        if not self.source_format:
            violations.append("INV3: source_format cannot be empty")

        # chunk_index >= 0
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            violations.append(
                f"INV3: chunk_index must be int >=0, got {self.chunk_index!r}"
            )

        # --- INVARIANT 4: source_block_orders non-empty (also in Provenance) ---
        if not self.provenance.source_block_orders:
            violations.append("INV4: provenance.source_block_orders cannot be empty")
        elif self.provenance.source_block_orders == [-1]:
            # placeholder default detected
            violations.append("INV4: provenance.source_block_orders uses placeholder default (-1)")

        # --- INVARIANT 5: block orders monotonic (in Provenance.__post_init__) ---
        # (already validated, but recorded for traceability via exceptions above)

        # --- INVARIANT 6: no duplicate block orders ---
        # (already validated)

        # --- INVARIANT 7: start/end order consistent ---
        # (already validated in Provenance.__post_init__)

        # --- INVARIANT 8: page_start <= page_end when both exist ---
        # (already validated)

        # --- INVARIANT 9: pages consistent with page range ---
        # (already validated)

        # --- INVARIANT 10: article/chapter/section may be null OR non-empty strings ---
        for attr in ("chapter", "section", "article"):
            v = getattr(self.structural, attr)
            if v is None:
                continue
            if not isinstance(v, str):
                violations.append(f"INV10: structural.{attr} must be str or None")
            elif not v.strip():
                violations.append(f"INV10: structural.{attr} must be non-empty or null")

        # --- INVARIANT 11: sentinel values forbidden in any string field ---
        for f_obj in _dc_fields(self):
            key = f_obj.name
            if key in {"structural", "provenance", "table_context",
                       "parent_child", "quality", "extras"}:
                continue
            v = getattr(self, key)
            if _contains_forbidden_sentinel(v):
                violations.append(
                    f"INV11: forbidden sentinel detected in field {key!r}={v!r}"
                )
        # sentinel scan in structural / provenance / table / parent_child / quality
        for sub in (self.structural, self.provenance, self.table_context,
                    self.parent_child, self.quality):
            for f_obj in _dc_fields(sub):
                v = getattr(sub, f_obj.name)
                if _contains_forbidden_sentinel(v):
                    violations.append(
                        f"INV11: forbidden sentinel detected in sub.{type(sub).__name__}."
                        f"{f_obj.name}={v!r}"
                    )
        if _contains_forbidden_sentinel(self.extras):
            violations.append("INV11: forbidden sentinel detected in extras dict")

        # --- INVARIANT 12: unknown extra fields rejected if strict schema ---
        #     (the project convention: extras MUST be empty.  Consumer may
        #      override by adding keys explicitly, but contract flags it.)
        if self.extras:
            violations.append(
                f"INV12: strict schema — unknown extra fields present: "
                f"{sorted(self.extras.keys())}"
            )

        # --- INVARIANT 13: TABLE structure metadata internally consistent ---
        #     (already validated in ChunkTableContext.__post_init__)
        # Additional cross-check: if contains_table, source_table_block_orders
        # must be subset of provenance.source_block_orders
        if self.table_context.contains_table and self.table_context.source_table_block_orders:
            prov_set = set(self.provenance.source_block_orders)
            missing = [
                b for b in self.table_context.source_table_block_orders
                if b not in prov_set
            ]
            if missing:
                violations.append(
                    f"INV13: source_table_block_orders {missing} not subset of "
                    f"provenance.source_block_orders {sorted(prov_set)}"
                )

        # --- INVARIANT 14: parent_chunk_id optional ---
        #     (already validated. chunk_level is Optional.)

        # --- INVARIANT 15: no document metadata semantic overwrite ---
        # Chunk layer MUST NOT invent Normalizer-owned values.
        # Policy: if a field's value would be produced by Normalizer, the Chunk
        # layer either inherits it AS-IS or leaves it null.
        # We enforce a conservative proxy: fields cannot be internal sentinels
        # (already INV11) AND fields claiming to be "N/A" / "UNKNOWN" strings
        # are banned because those constitute fabrication of non-null values.
        FABRICATION_BANNED_VALUES = {"N/A", "NA", "UNKNOWN", "UNSET", "TBD", "NONE"}
        LAYER_B_FIELDS = {
            "title", "source_org", "publish_date", "creation_date",
            "effective_date", "expiry_date", "document_number",
            "document_index_number", "document_type", "authority_level",
            "legal_status", "jurisdiction", "region", "theme_category",
            "source_url",
        }
        for attr in LAYER_B_FIELDS:
            v = getattr(self, attr)
            if v is None:
                continue
            if isinstance(v, str) and v.strip().upper() in FABRICATION_BANNED_VALUES:
                violations.append(
                    f"INV15: fabrication forbidden — document metadata field {attr!r} "
                    f"has pseudo-null value {v!r}; leave as null instead"
                )

        # --- INVARIANT 16: metadata round-trip serialization stable ---
        # to_dict() → from_dict() → to_dict() must be byte-identical.
        try:
            d1 = self.to_dict()
            s1 = json.dumps(d1, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            d2 = ChunkMetadata.from_dict(d1).to_dict()
            s2 = json.dumps(d2, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if s1 != s2:
                violations.append("INV16: to_dict/from_dict round-trip not stable")
        except Exception as exc:
            violations.append(f"INV16: round-trip serialization failed: {type(exc).__name__}: {exc}")

        # --- INVARIANT 17 (bonus): chunk_id matches deterministic contract if ---
        #     all components present (deep check, not required for minimal contract
        #     because chunk_id might come from trusted previous run).
        if (
            not violations  # only if base invariants pass
            and self.source_sha256
            and len(self.source_sha256) == 64
            and self.document_id
            and isinstance(self.chunk_index, int)
            and self.provenance.original_block_order_start >= 0
            and self.provenance.original_block_order_end >= self.provenance.original_block_order_start
        ):
            try:
                expected = compute_chunk_id_contract(
                    source_sha256=self.source_sha256,
                    document_id=self.document_id,
                    chunk_index=self.chunk_index,
                    original_block_order_start=self.provenance.original_block_order_start,
                    original_block_order_end=self.provenance.original_block_order_end,
                )
                if expected != self.chunk_id:
                    violations.append(
                        f"INV17: chunk_id determinism mismatch — "
                        f"expected {expected!r} from contract rule, "
                        f"actual {self.chunk_id!r}"
                    )
            except ChunkContractViolation:
                pass  # inputs not suitable for deterministic check

        return violations

    # ==================================================================
    # Deterministic serialization helpers
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON-safe dict.  No mutation of caller.

        Uses explicit field selection (never vars(self)) so adding new fields
        without updating this method is a deliberate, reviewable change.
        """
        return {
            # ------- LAYER A -------
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "logical_document_id": self.logical_document_id,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "chunk_index": self.chunk_index,
            # ------- LAYER B -------
            "title": self.title,
            "source_org": self.source_org,
            "publish_date": self.publish_date,
            "creation_date": self.creation_date,
            "effective_date": self.effective_date,
            "expiry_date": self.expiry_date,
            "document_number": self.document_number,
            "document_index_number": self.document_index_number,
            "document_type": self.document_type,
            "authority_level": self.authority_level,
            "legal_status": self.legal_status,
            "jurisdiction": self.jurisdiction,
            "region": self.region,
            "theme_category": self.theme_category,
            "source_url": self.source_url,
            "contract_version": self.contract_version,
            # ------- LAYER C -------
            "structural": asdict(self.structural),
            # ------- LAYER D -------
            "provenance": asdict(self.provenance),
            # ------- LAYER E -------
            "table_context": asdict(self.table_context),
            "parent_child": asdict(self.parent_child),
            "quality": asdict(self.quality),
            # extras
            "extras": dict(self.extras) if self.extras else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        """Construct from dict produced by to_dict().  Deterministic."""
        d = dict(data)  # shallow copy, non-mutating caller

        def _sub(cls_: Any, key: str):
            raw = d.pop(key, None)
            if raw is None:
                return cls_()
            if isinstance(raw, cls_):
                return raw
            # dataclass: unpack dict
            flds = {f.name for f in _dc_fields(cls_)}
            filtered = {k: v for k, v in raw.items() if k in flds}
            return cls_(**filtered)

        structural = _sub(ChunkStructuralContext, "structural")
        provenance_raw = d.pop("provenance", None)
        if provenance_raw is None:
            provenance = ChunkProvenance(source_block_orders=[-1])
        elif isinstance(provenance_raw, ChunkProvenance):
            provenance = provenance_raw
        else:
            flds = {f.name for f in _dc_fields(ChunkProvenance)}
            filtered = {k: v for k, v in provenance_raw.items() if k in flds}
            provenance = ChunkProvenance(**filtered)
        table_context = _sub(ChunkTableContext, "table_context")
        parent_child = _sub(ChunkParentChildInfo, "parent_child")
        quality = _sub(ChunkQualityFlags, "quality")

        extras = d.pop("extras", {}) or {}

        # Remaining keys: top-level scalar fields.  Raise on unknown.
        flds = {f.name for f in _dc_fields(cls)} - {
            "structural", "provenance", "table_context",
            "parent_child", "quality", "extras",
        }
        unknown = set(d.keys()) - flds
        if unknown:
            raise ChunkContractViolation(
                f"ChunkMetadata.from_dict: unknown top-level keys: {sorted(unknown)}"
            )

        return cls(
            structural=structural,
            provenance=provenance,
            table_context=table_context,
            parent_child=parent_child,
            quality=quality,
            extras=extras,
            **{k: d[k] for k in d if k in flds},
        )

    # ==================================================================
    # §4 — Document Metadata projection (INHERIT from Metadata Contract V1)
    # ==================================================================

    def apply_document_metadata_projection(
        self,
        normalized_document_metadata: dict[str, Any],
    ) -> "ChunkMetadata":
        """Copy fields FROM frozen MetadataNormalizer output TO Chunk layer AS-IS.

        Rules (strict):
          * Only fields in DOCUMENT_METADATA_PROJECTION_MAP are copied.
          * Chunk layer values already non-null → raise (no overwrite; no LWW).
          * null in source → leave chunk value as-is (null).
          * Sentinels → forbidden (raises).

        Returns a NEW ChunkMetadata instance with inherited fields.
        Does NOT mutate self.
        """
        if not isinstance(normalized_document_metadata, dict):
            raise ChunkContractViolation(
                "apply_document_metadata_projection requires dict input"
            )
        d = self.to_dict()
        for doc_field, projection in DOCUMENT_METADATA_PROJECTION_MAP.items():
            if doc_field not in normalized_document_metadata:
                continue
            src_val = normalized_document_metadata[doc_field]
            if _contains_forbidden_sentinel(src_val):
                raise ChunkContractViolation(
                    f"doc metadata field {doc_field!r} contains forbidden sentinel"
                )
            chunk_field = projection.chunk_metadata_field
            if chunk_field not in d:
                # Unknown chunk target field (shouldn't happen if map is correct)
                raise ChunkContractViolation(
                    f"projection target chunk field {chunk_field!r} not found on ChunkMetadata"
                )
            current = d.get(chunk_field)
            # null source → skip
            if src_val is None:
                continue
            # empty string treated as null (don't overwrite)
            if isinstance(src_val, str) and src_val == "":
                continue
            # chunk value already non-null non-empty → raise (NO OVERWRITE)
            if current is not None and (not isinstance(current, str) or current != ""):
                raise ChunkContractViolation(
                    f"Document metadata projection would overwrite chunk.{chunk_field} "
                    f"(current={current!r}) with {src_val!r} — LWW forbidden."
                )
            d[chunk_field] = src_val
        return ChunkMetadata.from_dict(d)


# ============================================================================
# Contract frozen invariants list (for documentation / matrix)
# ============================================================================

CHUNK_METADATA_CONTRACT_V1_INVARIANTS: list[str] = [
    "INV1_chunk_id_nonempty_length32_lowercase_hex",
    "INV2_document_identity_nonempty",
    "INV3_source_identity_source_sha256_length64_hex_chunk_index_ge0",
    "INV4_source_block_orders_nonempty",
    "INV5_source_block_orders_strictly_monotonic_increasing",
    "INV6_no_duplicate_source_block_orders",
    "INV7_original_block_order_start_end_equal_to_min_max_block_orders",
    "INV8_page_start_le_page_end_when_both_present",
    "INV9_page_numbers_entries_within_page_start_page_end",
    "INV10_chapter_section_article_either_null_or_nonempty_string",
    "INV11_internal_metadata_sentinels_must_not_leak_into_any_public_contract_field",
    "INV12_strict_schema_extras_must_be_empty_dict",
    "INV13_table_source_block_orders_subset_of_provenance_block_orders_and_table_fields_consistent",
    "INV14_parent_chunk_id_and_chunk_level_optional_with_validated_vocabulary",
    "INV15_no_document_metadata_semantic_overwrite_no_lww_no_fabricated_pseudo_null_values",
    "INV16_metadata_to_dict_from_dict_roundtrip_serialization_stable",
    "INV17_chunk_id_matches_deterministic_contract_when_all_components_present",
]


# ============================================================================
# Export symbols (frozen surface)
# ============================================================================

__all__ = [
    # Core classes
    "ChunkMetadata",
    "ChunkStructuralContext",
    "ChunkProvenance",
    "ChunkTableContext",
    "ChunkQualityFlags",
    "ChunkParentChildInfo",
    # Enums
    "MetadataLayer",
    "SourceFormat",
    "ChunkLevel",
    "FilterCandidateClass",
    "EmbeddingPolicy",
    "FieldSourceOwner",
    "ChunkContractViolation",
    # Projection mapping
    "DocumentMetadataProjection",
    "DOCUMENT_METADATA_PROJECTION_TABLE",
    "DOCUMENT_METADATA_PROJECTION_MAP",
    # Identity helpers
    "compute_chunk_id_contract",
    "CHUNK_ID_CONTRACT_SALT",
    "CHUNK_ID_HEX_LENGTH",
    # Validation / sentinel
    "CHUNK_METADATA_CONTRACT_V1_INVARIANTS",
    "INTERNAL_SENTINEL_VALUES",
    "PUBLIC_CONTRACT_FORBIDDEN_SUBSTRINGS",
    "_contains_forbidden_sentinel",
]
