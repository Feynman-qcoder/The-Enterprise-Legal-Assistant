"""
ParsedDocument V2 — FROZEN CONTRACT MODULE (after Freeze Patch)
=================================================================
Pure-stdlib implementation: dataclasses / enum / typing only.

Guarantees:
- Output types compatible with existing V1 pipeline via LegacyAdapter.
- Current chunker can consume legacy adapter output (LegacyParsedDocument).
- MySQL / Milvus schema unchanged.
- Retrieval code unchanged.

Quality impact: MUST be validated by future dry-run + Retrieval Eval.
This module does NOT promise chunk-count diff < X% or performance ± Y%.

FREEZE PATCH CHANGES:
  1. Constructor NO LONGER auto-renumbers invalid block.order.
     Broken order raises ContractViolation immediately.
  2. Block.provenance_stored (optional Provenance) must be consistent
     with Block.order when present:
         block.order == block.provenance_stored.block_order
     renumber_blocks() keeps both in sync (replaces frozen Provenance).
  3. Explicit validate() remains as strict entry point for Parser exit gate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# =============================================================================
# BlockType — 6 types exactly
# =============================================================================

class BlockType(str, Enum):
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"
    METADATA = "METADATA"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# Contract-level exception
# =============================================================================

class ContractViolation(ValueError):
    """Raised when an invariant defined by this Contract is broken."""


# =============================================================================
# TableCell / TableData
# =============================================================================

@dataclass
class TableCell:
    """Minimal table cell: text + merge semantics (colspan/rowspan)."""

    text: str
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False

    def __post_init__(self) -> None:
        if self.colspan < 1:
            raise ContractViolation(f"TableCell.colspan must be >= 1, got {self.colspan}")
        if self.rowspan < 1:
            raise ContractViolation(f"TableCell.rowspan must be >= 1, got {self.rowspan}")
        if self.text is None:
            raise ContractViolation("TableCell.text cannot be None (use '')")
        if not isinstance(self.text, str):
            object.__setattr__(self, "text", str(self.text))

    def display_text(self) -> str:
        t = self.text.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" in t:
            t = t.replace("\n", "<br>")
        if "|" in t:
            t = t.replace("|", "\\|")
        return t


@dataclass
class TableData:
    """Structured 2-D table. Cell-level merge only; no nested tables."""

    headers: list[TableCell]
    rows: list[list[TableCell]]
    caption: Optional[str] = None

    def num_cols(self) -> int:
        return sum(c.colspan for c in self.headers)

    def _validate_rows(self) -> None:
        if not self.headers:
            raise ContractViolation("TableData.headers cannot be empty")
        header_cols = len(self.headers)
        for row_idx, row in enumerate(self.rows):
            if len(row) != header_cols:
                raise ContractViolation(
                    f"TableData row {row_idx} has {len(row)} cells, headers have {header_cols}"
                )

    def __post_init__(self) -> None:
        self._validate_rows()

    def to_markdown(self) -> str:
        """GFM-compatible Markdown table. Caption prefixed as bold heading if present."""
        self._validate_rows()

        lines: list[str] = []
        if self.caption:
            cap = self.caption.strip()
            if cap:
                lines.append(f"**{cap}**")
                lines.append("")

        header_cells = [c.display_text() for c in self.headers]
        lines.append("| " + " | ".join(header_cells) + " |")

        seps = []
        for cell in self.headers:
            width = max(3, len(cell.display_text()))
            seps.append("-" * width)
        lines.append("| " + " | ".join(seps) + " |")

        for row in self.rows:
            row_cells = [c.display_text() for c in row]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)


# =============================================================================
# Provenance — FROZEN minimal triple. KEEP frozen.
# =============================================================================

@dataclass(frozen=True)
class Provenance:
    """Lightest trace tuple: {source_file, page, block_order}."""

    source_file: str
    page: Optional[int]
    block_order: int

    def __post_init__(self) -> None:
        if not self.source_file:
            raise ContractViolation("Provenance.source_file cannot be empty")
        if self.page is not None and self.page <= 0:
            raise ContractViolation(f"Provenance.page must be >=1 or None, got {self.page}")
        if self.block_order < 0:
            raise ContractViolation(
                f"Provenance.block_order must be >=0, got {self.block_order}"
            )

    def with_block_order(self, new_block_order: int) -> "Provenance":
        """Return a new Provenance with updated block_order (frozen-safe copy)."""
        return Provenance(
            source_file=self.source_file,
            page=self.page,
            block_order=new_block_order,
        )


# =============================================================================
# Block
# =============================================================================

@dataclass
class Block:
    """
    Minimal semantic unit inside a document. Order-preserving.

    Invariants (enforced in __post_init__):
    - order >= 0
    - HEADING: level in 1..9
    - TABLE: table_data not None; text == table_data.to_markdown()
    - non-HEADING: level is None
    - non-TABLE: table_data is None
    - IF provenance_stored is not None: provenance_stored.block_order == order
    """

    block_id: str
    type: BlockType
    text: str
    order: int
    level: Optional[int] = None
    table_data: Optional[TableData] = None
    page: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance_stored: Optional[Provenance] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def make_block_id(document_id: str, order: int) -> str:
        return f"{document_id}_b{order:04d}"

    def build_provenance(self, source_file: str) -> Provenance:
        """
        Create a fresh Provenance value-object for this block using the given
        source_file (not stored; for ad-hoc trace construction use provenance_stored).
        """
        return Provenance(
            source_file=source_file,
            page=self.page,
            block_order=self.order,
        )

    def attach_provenance(self, source_file: str) -> Provenance:
        """
        Build Provenance from source_file, store it into provenance_stored,
        and validate that provenance.block_order == order.
        Returns the attached Provenance.
        """
        prov = self.build_provenance(source_file)
        object.__setattr__(self, "provenance_stored", prov)
        # Trigger invariant via __post_init__ on self fields (partial)
        self._check_provenance_consistency()
        return prov

    # ------------------------------------------------------------------ invariants
    def _check_provenance_consistency(self) -> None:
        if self.provenance_stored is None:
            return
        if self.provenance_stored.block_order != self.order:
            raise ContractViolation(
                f"Block.order ({self.order}) != provenance_stored.block_order "
                f"({self.provenance_stored.block_order}) for block_id='{self.block_id}'. "
                "Single source of truth required."
            )

    def __post_init__(self) -> None:
        # --- order ---
        if self.order < 0:
            raise ContractViolation(f"Block.order must be >=0, got {self.order}")

        # --- type coercion ---
        if isinstance(self.type, str):
            try:
                object.__setattr__(self, "type", BlockType(self.type.upper()))
            except ValueError as exc:
                raise ContractViolation(
                    f"Block.type '{self.type}' not in BlockType enum"
                ) from exc

        # --- text ---
        if self.text is None:
            object.__setattr__(self, "text", "")
        if not isinstance(self.text, str):
            object.__setattr__(self, "text", str(self.text))

        # --- heading ---
        if self.type is BlockType.HEADING:
            if self.level is None:
                raise ContractViolation("HEADING block requires level (1..9)")
            if not isinstance(self.level, int):
                raise ContractViolation("HEADING level must be int")
            if self.level < 1 or self.level > 9:
                raise ContractViolation(
                    f"HEADING level must be in 1..9, got {self.level}"
                )
        else:
            if self.level is not None:
                raise ContractViolation(
                    f"level must be None for {self.type.value} block"
                )

        # --- table ---
        if self.type is BlockType.TABLE:
            if self.table_data is None:
                raise ContractViolation("TABLE block requires table_data (TableData)")
            if not isinstance(self.table_data, TableData):
                raise ContractViolation(
                    "TABLE block.table_data must be TableData instance"
                )
            if not self.text.strip():
                object.__setattr__(self, "text", self.table_data.to_markdown())
            else:
                expected = self.table_data.to_markdown()
                if self.text != expected:
                    raise ContractViolation(
                        "TABLE block.text must equal table_data.to_markdown()"
                    )
        else:
            if self.table_data is not None:
                raise ContractViolation(
                    f"table_data must be None for {self.type.value} block"
                )

        # --- provenance consistency ---
        self._check_provenance_consistency()


# =============================================================================
# ParsedDocument
# =============================================================================

@dataclass
class ParsedDocument:
    """
    Parser V2 unified output contract.

    FREEZE BEHAVIOR: Constructor WILL NOT silently auto-renumber blocks.
    If any blocks[i].order != i, constructor raises ContractViolation.
    Use explicit doc.renumber_blocks() BEFORE passing blocks here, or call
    it on the already-built doc ONLY IF you later mutate via unsafe helpers.

    Invariants:
    - source_file / document_id cannot be empty
    - parser_name / parser_version cannot be empty
    - for all i: blocks[i].order == i  (reading order == index, strict)
    - block_id must be unique inside document
    - for any block with provenance_stored:
          blocks[i].order == blocks[i].provenance_stored.block_order
    """

    document_id: str
    source_file: str
    title: str
    metadata: dict[str, Any]
    blocks: list[Block]
    parser_name: str
    parser_version: str
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    def total_chars(self, *, include_metadata_blocks: bool = False) -> int:
        total = 0
        for b in self.blocks:
            if not include_metadata_blocks and b.type is BlockType.METADATA:
                continue
            total += len(b.text)
        return total

    @staticmethod
    def prepare_blocks(document_id: str, blocks: list["Block"]) -> None:
        """
        Pre-constructor explicit renumber utility (Parser side helper).

        Since ParsedDocument constructor STRICTLY requires blocks[i].order == i
        and raises ContractViolation otherwise, Parser implementations should
        call this BEFORE constructing ParsedDocument whenever block orders are
        placeholder / dirty / appended incrementally.

        Mutates the input blocks list in place:
          - blocks[i].order = i
          - blocks[i].block_id = make_block_id(document_id, i)
          - IF blocks[i].provenance_stored exists: replaces frozen Provenance
            with a copy whose block_order == i  (single source of truth)

        Contract: this is an EXPLICIT, opt-in utility only. The ParsedDocument
        constructor itself will NEVER auto-call it.
        """
        for i, b in enumerate(blocks):
            object.__setattr__(b, "order", i)
            object.__setattr__(
                b,
                "block_id",
                Block.make_block_id(document_id, i),
            )
            if b.provenance_stored is not None:
                new_prov = b.provenance_stored.with_block_order(i)
                object.__setattr__(b, "provenance_stored", new_prov)
                b._check_provenance_consistency()

    def renumber_blocks(self) -> None:
        """
        Post-construct explicit renumber.

        Useful only if you later UNSAFELY mutate doc.blocks (append/remove/reorder).
        For Parser-side pre-construction order fixups, prefer the static helper
        ``ParsedDocument.prepare_blocks(doc_id, blocks)``.

        - Reassigns blocks[i].order = i
        - Reassigns blocks[i].block_id = make_block_id(doc_id, i)
        - IF blocks[i].provenance_stored exists: replaces frozen Provenance
          with a copy whose block_order == i  (single source of truth)
        """
        ParsedDocument.prepare_blocks(self.document_id, self.blocks)

    def validate(self) -> None:
        """Strict validation (no auto-fix, no auto-renumber). Re-raises invariant failures."""
        self.__post_init__()

    # ------------------------------------------------------------------ invariants
    def __post_init__(self) -> None:
        # --- identity ---
        if not self.source_file:
            raise ContractViolation("ParsedDocument.source_file cannot be empty")
        if not self.document_id:
            raise ContractViolation("ParsedDocument.document_id cannot be empty")
        if not self.parser_name or not self.parser_version:
            raise ContractViolation("parser_name / parser_version required")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if self.blocks is None:
            object.__setattr__(self, "blocks", [])

        # --- strict reading order: NO implicit repair (FREEZE PATCH #1) ---
        seen_ids: set[str] = set()
        for i, b in enumerate(self.blocks):
            if b.order != i:
                raise ContractViolation(
                    f"blocks[{i}].order == {b.order}, expected {i}. "
                    "Constructor does NOT auto-renumber (Freeze Patch #1). "
                    "Explicitly call doc.renumber_blocks() BEFORE constructing the "
                    "final ParsedDocument (or after unsafely mutating block list)."
                )
            if b.block_id in seen_ids:
                raise ContractViolation(f"Duplicate block_id: {b.block_id}")
            seen_ids.add(b.block_id)

            # --- order == provenance.block_order (FREEZE PATCH #2) ---
            if b.provenance_stored is not None:
                if b.provenance_stored.block_order != b.order:
                    raise ContractViolation(
                        f"blocks[{i}].order == {b.order} but "
                        f"provenance_stored.block_order == {b.provenance_stored.block_order}. "
                        "These must be identical (Freeze Patch #2)."
                    )


# =============================================================================
# Legacy Adapter
# =============================================================================

@dataclass(frozen=True)
class LegacyParsedDocument:
    """
    V1-compatible shape. Field names match modules.ingestion.document_parsing.ParsedDocument:
        text: str
        metadata: dict[str, Any]
        segments: tuple[str, ...]
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: tuple[str, ...] = ()


class LegacyAdapter:
    """Pure helpers: ParsedDocument V2 -> Legacy (V1 chunker consumable)."""

    @staticmethod
    def to_legacy_text(
        doc: "ParsedDocument",
        *,
        include_metadata: bool = False,
    ) -> str:
        parts: list[str] = []
        for b in doc.blocks:
            t = b.type
            if t is BlockType.HEADING:
                level = max(1, min(9, int(b.level or 1)))
                parts.append(f"\n\n{'#' * level} {b.text.strip()}\n\n")
            elif t is BlockType.PARAGRAPH:
                parts.append(f"\n\n{b.text}")
            elif t is BlockType.LIST_ITEM:
                lvl = 0
                md = b.metadata or {}
                if "list_level" in md and isinstance(md["list_level"], int):
                    lvl = max(0, md["list_level"])
                indent = "  " * lvl
                parts.append(f"\n{indent}- {b.text}")
            elif t is BlockType.TABLE:
                md_table = (
                    b.table_data.to_markdown() if b.table_data is not None else b.text
                )
                parts.append(f"\n\n{md_table}\n\n")
            elif t is BlockType.METADATA:
                if include_metadata:
                    parts.append(f"\n\n<!-- METADATA(order={b.order}) -->\n{b.text}\n")
            elif t is BlockType.UNKNOWN:
                parts.append(
                    f"\n\n<!-- UNKNOWN BLOCK(order={b.order}) -->\n{b.text}\n"
                )
            else:  # pragma: no cover
                parts.append(f"\n\n{b.text}")
        return "".join(parts).strip()

    @staticmethod
    def to_legacy_parsed_document(doc: "ParsedDocument") -> LegacyParsedDocument:
        full_text = LegacyAdapter.to_legacy_text(doc)
        meta_shallow = dict(doc.metadata)

        has_page = any(b.page is not None for b in doc.blocks)
        if has_page:
            per_page: dict[int, list[Block]] = {}
            for b in doc.blocks:
                if b.page is None:
                    continue
                per_page.setdefault(b.page, []).append(b)
            max_page = max(per_page.keys()) if per_page else 0
            segments_list: list[str] = []
            for p in range(1, max_page + 1):
                if p not in per_page:
                    segments_list.append("")
                    continue
                # === Freeze Patch fix: operate on COPIES, never mutate caller blocks ===
                # Shallow-copy list, dataclasses.replace each block to get independent objects.
                # This allows us to prepare_blocks(renumber) locally without touching the
                # caller-owned doc.blocks references. Provenance stays frozen per instance.
                pseudo_blocks = [
                    dataclasses.replace(b) for b in per_page[p]
                ]
                # Explicit pre-construction renumber (constructor will NOT auto-renumber).
                ParsedDocument.prepare_blocks(doc.document_id, pseudo_blocks)
                fake_doc = ParsedDocument(
                    document_id=doc.document_id,
                    source_file=doc.source_file,
                    title=doc.title,
                    metadata=dict(doc.metadata),
                    blocks=pseudo_blocks,
                    parser_name=doc.parser_name,
                    parser_version=doc.parser_version,
                )
                try:
                    fake_doc.validate()
                except ContractViolation:
                    # Defensive: any residual ordering issue → re-prepare + validate once
                    ParsedDocument.prepare_blocks(doc.document_id, fake_doc.blocks)
                    fake_doc.validate()
                segments_list.append(LegacyAdapter.to_legacy_text(fake_doc))
            return LegacyParsedDocument(
                text=full_text,
                metadata=meta_shallow,
                segments=tuple(segments_list),
            )

        return LegacyParsedDocument(
            text=full_text,
            metadata=meta_shallow,
            segments=(full_text,),
        )


__all__ = [
    "BlockType",
    "ContractViolation",
    "TableCell",
    "TableData",
    "Provenance",
    "Block",
    "ParsedDocument",
    "LegacyParsedDocument",
    "LegacyAdapter",
]
