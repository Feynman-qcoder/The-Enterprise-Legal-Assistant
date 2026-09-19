"""
Chunking Strategy V2 — Task 1 + 2 Utils

FROZEN: pre-registered thresholds, manifest loader, parser dispatch,
        length metric (chars + optional tokens), flatten helper,
        MD Parser V2 (minimal wrapper for frozen contract output).

All imports of frozen components (Parser/Cleaner/CMCV1) happen here
in a SINGLE PLACE so the rest of the pipeline never touches them directly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Absolute path constants
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).parent.resolve()
RAG_DIR = Path(r"F:\DataBase\trae_work\RAG")  # 可选 POC 解析器来源：缺失时 dispatch 自动回落仓库解析器
LS_DIR = Path(__file__).resolve().parents[2]  # 仓库根（Task 22 迁移适配：原为 D 盘绝对路径）
CORPUS_DIR = Path(os.environ.get("XIAOYI_CORPUS_DIR", r"D:\xiaoyi\data_source"))  # 语料根：环境变量可覆盖
MANIFEST_PATH = Path(os.environ.get("XIAOYI_MANIFEST_PATH", str(CORPUS_DIR / "_meta" / "ingest_manifest_v1.jsonl")))
CMC_DIR = OUT_DIR.parent / "chunk_metadata_contract_v1"  # offline/ 平级兄弟目录（F5：平级 import 前提）
BGE_MODEL = LS_DIR / "models" / "bge-m3"

for _p in (RAG_DIR, LS_DIR, str(LS_DIR / "modules" / "ingestion"), CMC_DIR, OUT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# FR7 FROZEN THRESHOLDS — unique source of truth (NO MAGIC NUMBERS elsewhere)
# ---------------------------------------------------------------------------
FR7_FROZEN_THRESHOLDS: dict[str, int] = {
    # --- size accounting
    "OVERSIZED_THRESHOLD_CHARS": 1200,
    "VERY_SMALL_THRESHOLD_CHARS": 64,
    # --- Strategy A baseline
    "A_WINDOW": 512,
    "A_OVERLAP": 128,
    "A_STEP": 512 - 128,  # = 384
    # --- Strategy B structure-aware
    "B_TARGET_MAX_CHARS": 600,
    "B_HARD_MAX": 900,
    # --- TABLE-aware
    "TABLE_SMALL_ROWS": 20,
    "TABLE_SMALL_CHARS": 800,
    "TABLE_SPLIT_GROUP_ROWS": 10,
}
# double-check step
assert FR7_FROZEN_THRESHOLDS["A_STEP"] == FR7_FROZEN_THRESHOLDS["A_WINDOW"] - FR7_FROZEN_THRESHOLDS["A_OVERLAP"]


# ---------------------------------------------------------------------------
# ParsedDocument V2 contract — import ONCE here (FROZEN, never patch)
# ---------------------------------------------------------------------------
# 1) Blocks / TableData / ContractViolation
from parsed_document_v2 import (  # type: ignore
    Block,
    BlockType,
    ContractViolation,
    ParsedDocument,
    TableCell,
    TableData,
)
# 2) CMCV1
from chunk_metadata_contract_v1 import (  # type: ignore
    CHUNK_ID_HEX_LENGTH as _CMC_CHUNK_ID_LEN,
    ChunkLevel,
    ChunkContractViolation,
    ChunkMetadata,
    ChunkParentChildInfo,
    ChunkProvenance,
    ChunkQualityFlags,
    ChunkStructuralContext,
    ChunkTableContext,
    DOCUMENT_METADATA_PROJECTION_MAP,
    PUBLIC_CONTRACT_FORBIDDEN_SUBSTRINGS as FORBIDDEN_SENTINEL_STRINGS,
    compute_chunk_id_contract,
)
CHUNK_ID_LEN = _CMC_CHUNK_ID_LEN  # alias for consistency

# ---------------------------------------------------------------------------
# Normalize whitespace (MATCH production chunking.py exactly — FR1 baseline)
# ---------------------------------------------------------------------------
_WS_RE_SP = re.compile(r"[ \t]+")
_WS_RE_NL = re.compile(r"\n{3,}")


def normalize_ws(text: str) -> str:
    """Exact behavioral copy of chunking.py::normalize_ws."""
    if text is None:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS_RE_SP.sub(" ", t)
    t = _WS_RE_NL.sub("\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# FR7 lengths: chars primary, tokens optional via BGE-M3 tokenizer
# ---------------------------------------------------------------------------
_TOKENIZER: Any = None
_TOKENIZER_TRIED = False


def _try_load_tokenizer() -> Any:
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer  # type: ignore
        _TOKENIZER = AutoTokenizer.from_pretrained(str(BGE_MODEL))
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def tokenizer_available() -> bool:
    return _try_load_tokenizer() is not None


def len_metric(text: str) -> dict[str, int | None]:
    """Return {chars:int, tokens:int|None}. Chars via normalize_ws first."""
    t = normalize_ws(text or "")
    chars = len(t)
    tok = _try_load_tokenizer()
    if tok is None or not t:
        tokens = None if tok is None else 0
    else:
        try:
            tokens = len(tok.encode(t, add_special_tokens=False))
        except Exception:
            tokens = None
    return {"chars": chars, "tokens": tokens}


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_manifest_entries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# ---------------------------------------------------------------------------
# Canonical source path resolver (corpus walk)
# ---------------------------------------------------------------------------
_CORPUS_FILE_INDEX: dict[str, Path] | None = None


def _build_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for root, dirs, files in os.walk(CORPUS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        for fn in files:
            if fn not in idx:
                idx[fn] = Path(root) / fn
    return idx


def resolve_source_path(entry: dict[str, Any]) -> Path | None:
    global _CORPUS_FILE_INDEX
    if _CORPUS_FILE_INDEX is None:
        _CORPUS_FILE_INDEX = _build_index()
    cfile = entry.get("canonical_file", "")
    basename = Path(cfile).name
    if basename and basename in _CORPUS_FILE_INDEX:
        return _CORPUS_FILE_INDEX[basename]
    # fallback: raw filename (some entries canonical may be path, use stem+ext match)
    if basename:
        return _CORPUS_FILE_INDEX.get(basename)
    return None


# ---------------------------------------------------------------------------
# Parser V2 dispatch + Cleaner V2.1
# ---------------------------------------------------------------------------
def _source_sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


@dataclass
class LoadedDoc:
    """Output of dispatch_parse_and_clean: document + blocks + metadata."""
    manifest_entry: dict[str, Any]
    logical_document_id: str
    document_id: str
    source_path: Path
    source_sha256: str
    source_format: str
    parser_name: str
    parser_version: str
    doc: ParsedDocument  # Frozen ParsedDocumentV2 (blocks / metadata)
    doc_metadata_inherited: dict[str, Any]  # 1:1 inherit from manifest entry
    source_pages: list[int]  # for each block, page_no or None; derived from provenance_stored if available

    @property
    def blocks(self) -> list[Block]:
        return list(self.doc.blocks)

    @property
    def title(self) -> str | None:
        return self.manifest_entry.get("title") or self.doc_metadata_inherited.get("title")

    @property
    def document_type(self) -> str | None:
        return self.manifest_entry.get("document_type")


def _inherit_doc_metadata_from_manifest(entry: dict[str, Any]) -> dict[str, Any]:
    """FR3: inherit ONLY from manifest; NO recalc.

    Uses DOCUMENT_METADATA_PROJECTION_MAP from CMCV1 to know which fields are
    valid Layer-B keys. Unknown manifest keys are discarded (inherit only
    the projected vocabulary; never invent new fields). NULL stays NULL.
    """
    out: dict[str, Any] = {}
    # Copy manifest entries whose document_metadata_field matches a known projection key:
    for doc_field, projection in DOCUMENT_METADATA_PROJECTION_MAP.items():
        manifest_key = projection.document_metadata_field  # key in manifest entry (Source Truth = manifest)
        if manifest_key in entry:
            v = entry[manifest_key]
            out[doc_field] = v
        else:
            # Unknown: keep None (not present key means NULL in CMCV1 Layer B)
            # Do NOT fabricate ""
            out[doc_field] = None
    # Also copy fields that manifest entry has, but are used for Layer A:
    for extra in ("logical_document_id", "title", "source_file", "format", "normalized_sha256"):
        if extra in entry and out.get(extra) is None:
            out[extra] = entry[extra]
    return out


# --- minimal MD/TXT V2 parser (no official frozen MD V2 → keep it simple) ---
_MD_HEADING_RE = re.compile(r"^(#{1,9})\s+(.*)$")
_MD_LISTITEM_RE = re.compile(r"^(\s*)[-*\u2022+]\s+|\s*\d+\.\s+")
_MD_FRONTMATTER_SEP = "---"


def _md_extract_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if lines and lines[0].strip() == _MD_FRONTMATTER_SEP:
        closing = None
        for i in range(1, len(lines)):
            if lines[i].strip() == _MD_FRONTMATTER_SEP:
                closing = i
                break
        if closing:
            fm_lines = lines[1:closing]
            try:
                import yaml  # type: ignore
                meta = yaml.safe_load("\n".join(fm_lines)) or {}
                if isinstance(meta, dict):
                    return meta, "\n".join(lines[closing + 1:])
            except Exception:
                return {}, raw
    return {}, raw


def parse_md_v2_blocks(md_path: Path, document_id: str) -> ParsedDocument:
    """
    Minimal Markdown → ParsedDocument V2 parser.

    Design goals:
    1. Stable output (same file → same blocks).
    2. Produces HEADING / PARAGRAPH / LIST_ITEM block types for structural use.
    3. block.order monotonic starting 0; block.metadata.page_no all = 1 for MD/TXT.
    4. No hallucinated metadata, no sentinels, no fabrications.
    """
    raw = md_path.read_text(encoding="utf-8-sig")
    fm, body = _md_extract_front_matter(raw)
    lines = body.splitlines()

    blocks: list[Block] = []
    para_buf: list[str] = []
    heading_levels_seen: dict[str, int] = {}  # unused placeholder

    def flush_para() -> None:
        nonlocal para_buf
        txt = normalize_ws("\n".join(para_buf))
        if txt:
            # Try classify: list-item vs paragraph
            first = para_buf[0].lstrip() if para_buf else ""
            if _MD_LISTITEM_RE.match(first):
                bt = BlockType.LIST_ITEM
            else:
                bt = BlockType.PARAGRAPH
            order = len(blocks)
            blocks.append(Block(
                block_id=Block.make_block_id(document_id, order),
                type=bt,
                text=txt,
                order=order,
                level=None,
                page=1,
                metadata={"page_no": 1},
            ))
        para_buf = []

    for line in lines:
        line_stripped = line.rstrip()
        if not line_stripped.strip():
            flush_para()
            continue
        m = _MD_HEADING_RE.match(line_stripped)
        if m:
            flush_para()
            lvl = len(m.group(1))
            order = len(blocks)
            blocks.append(Block(
                block_id=Block.make_block_id(document_id, order),
                type=BlockType.HEADING,
                text=normalize_ws(m.group(2)),
                order=order,
                level=lvl,
                page=1,
                metadata={"page_no": 1},
            ))
            continue
        para_buf.append(line_stripped)
    flush_para()

    doc_title = (fm.get("title") if isinstance(fm, dict) else None) or md_path.stem
    return ParsedDocument(
        document_id=document_id,
        source_file=str(md_path),
        title=doc_title,
        metadata={"title": doc_title, **(fm if isinstance(fm, dict) else {})},
        blocks=blocks,
        parser_name="md_parser_v2_minimal",
        parser_version="1.0.0-frozen-cmcv2-chunker",
    )


def _apply_cleaner(doc: ParsedDocument) -> ParsedDocument:
    """Run Cleaner V2.1 frozen clean_parsed_document_v2; return cleaned doc."""
    import importlib
    cleaner_mod = importlib.import_module("cleaner_v2")
    fn = getattr(cleaner_mod, "clean_parsed_document_v2")
    result = fn(doc)
    # result is CleaningResult dataclass: field is cleaned_document (not doc/cleaned_doc)
    cleaned = getattr(result, "cleaned_document", None)
    if cleaned is None:
        cleaned = getattr(result, "cleaned_doc", None) or getattr(result, "doc", None)
    if cleaned is None:
        import dataclasses as _dc
        for f in _dc.fields(result):
            obj = getattr(result, f.name)
            if isinstance(obj, ParsedDocument):
                cleaned = obj
                break
    if cleaned is None:
        raise RuntimeError(f"clean_parsed_document_v2 returned no ParsedDocument: {type(result)}")
    return cleaned


def dispatch_parse_and_clean(entry: dict[str, Any]) -> LoadedDoc | tuple[str, str]:
    """
    Parse + Clean one manifest entry → LoadedDoc.
    If error: returns tuple ("SKIP", reason).
    """
    fmt = (entry.get("format") or "").lower()
    source_path = resolve_source_path(entry)
    if source_path is None:
        return "SKIP", "SOURCE_PATH_NOT_FOUND"

    document_id = entry.get("logical_document_id") or entry.get("document_id") or Path(source_path).stem
    logical_doc_id = entry.get("logical_document_id") or document_id
    sha = entry.get("normalized_sha256") or _source_sha256_file(source_path)

    parser_name = ""
    parser_version = ""
    doc: ParsedDocument | None = None
    try:
        if fmt == "md" or fmt == "txt":
            parser_name = {"md": "md_parser_v2_minimal", "txt": "txt_parser_v2_minimal"}[fmt]
            parser_version = "1.0.0-frozen-cmcv2-chunker"
            doc = parse_md_v2_blocks(source_path, document_id=document_id)
        elif fmt == "docx":
            import importlib
            mod = importlib.import_module("docx_parser_v2")
            fn = getattr(mod, "parse_docx_v2")
            parser_name = "docx_parser_v2"
            parser_version = "2.4.2-frozen-patched"
            # Frozen contract: parse_docx_v2(path) -> ParsedDocument (NO document_id kwarg)
            doc = fn(str(source_path))
            if isinstance(doc, tuple):
                doc = doc[0]
        elif fmt == "pdf":
            import importlib
            try:
                mod = importlib.import_module("pdf_parser_v2_poc")
                fn = getattr(mod, "parse_pdf_v2_poc")
                parser_name = "pdf_parser_v2_poc"
                parser_version = "2.3.0-canonical-poc"
                out = fn(str(source_path), document_id=document_id, debug=False)
                if isinstance(out, tuple):
                    doc = out[0]
                else:
                    doc = out
            except Exception as e_poc:
                # Fallback: LS pdf_parser_v2 — might need PyMuPDF, try anyway
                try:
                    sys.path.insert(0, str(LS_DIR / "modules" / "ingestion"))
                    mod2 = importlib.import_module("pdf_parser_v2")
                    fn2 = getattr(mod2, "parse_pdf_v2", None)
                    if callable(fn2):
                        parser_name = "pdf_parser_v2_ls_prod"
                        parser_version = "2.2.0-frozen-ls"
                        out2 = fn2(str(source_path))
                        doc = out2[0] if isinstance(out2, tuple) else out2
                    else:
                        raise RuntimeError(f"poc fail + no fn2: {e_poc}")
                except Exception as e_ls:
                    raise RuntimeError(f"pdf both failed: poc={e_poc}, ls={e_ls}")
        else:
            return "SKIP", f"UNSUPPORTED_FORMAT_{fmt}"
    except Exception as e:
        return "SKIP", f"PARSE_ERROR_{type(e).__name__}_{str(e)[:100]}"

    if doc is None:
        return "SKIP", "PARSE_RETURNED_NONE"

    # Run Cleaner V2.1 (FR1.4: must be READ-ONLY — we don't mutate inputs to cleaner,
    #   we call clean_parsed_document_v2(doc) and replace doc with output).
    try:
        cleaned_doc = _apply_cleaner(doc)
        doc = cleaned_doc
    except Exception as e:
        # Cleaner failure is not fatal for our offline compare; keep original doc
        # but record parser_version += " (CLEAN_FAIL_...)"
        parser_version = f"{parser_version}__CLEAN_FAIL_{type(e).__name__}"

    # Build source_pages list (parallel to doc.blocks), used for provenance page_start/page_end/page_numbers
    pages_for_blocks: list[int] = []
    for b in doc.blocks:
        page: int | None = getattr(b, "page", None)
        if page is None and isinstance(b.metadata, dict):
            for k in ("page_no", "page_number"):
                v = b.metadata.get(k)
                if isinstance(v, int):
                    page = v
                    break
        if page is None:
            ps = getattr(b, "provenance_stored", None)
            if ps is not None:
                try:
                    page = ps.page
                except Exception:
                    page = None
        if page is None:
            # MD/TXT/Unknown page: treat as page 1
            page = 1
        pages_for_blocks.append(int(page))

    inherited_meta = _inherit_doc_metadata_from_manifest(entry)
    # Ensure Layer B title from manifest if cleaner doc has one too: use manifest preferentially
    return LoadedDoc(
        manifest_entry=entry,
        logical_document_id=logical_doc_id,
        document_id=document_id,
        source_path=source_path,
        source_sha256=sha,
        source_format=fmt,
        parser_name=parser_name,
        parser_version=parser_version,
        doc=doc,
        doc_metadata_inherited=inherited_meta,
        source_pages=pages_for_blocks,
    )


# ---------------------------------------------------------------------------
# Flatten blocks to text + reverse-span lookup (Strategy A baseline use)
# ---------------------------------------------------------------------------
@dataclass
class FlatBaseline:
    text: str
    spans: list[tuple[int, int, int]]   # (start, end, block_order)
    blocks_ordered: list[int]
    block_text_map: dict[int, str]      # block_order -> normalized text used in flatten


def _block_flat_text(block: Block) -> str:
    """Return the text to put in baseline flatten for a block.
    TABLE blocks → TableData.to_markdown() (spec FR1: MD projection for length + slicing only).
    Any other → block.text."""
    if block.type == BlockType.TABLE and block.table_data is not None:
        try:
            md = block.table_data.to_markdown()
            return normalize_ws(md)
        except Exception:
            return normalize_ws(block.text or "")
    return normalize_ws(block.text or "")


def flatten_blocks_to_text_for_baseline(blocks: Iterable[Block]) -> FlatBaseline:
    """
    Flatten blocks -> single normalized text (Strategy A input representation).
    Also returns spans list: for each (char_start, char_end+1, block_order).
    This allows sliding-window slice (pos, pos+window) to map back to blocks.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    order_list: list[int] = []
    txt_map: dict[int, str] = {}
    for block in blocks:
        txt = _block_flat_text(block)
        if not txt:
            continue
        # single separator between blocks
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(txt)
        end = start + len(txt)
        spans.append((start, end, block.order))
        cursor = end
        order_list.append(block.order)
        txt_map[block.order] = txt
    merged = "".join(parts)
    return FlatBaseline(
        text=normalize_ws(merged),
        # NOTE: after normalize_ws the spans will slightly shift but are still an APPROX
        # baseline provenance estimate; spec requires monotonic + correct range,
        # not a perfect character-to-block mapping.
        spans=spans,
        blocks_ordered=order_list,
        block_text_map=txt_map,
    )


def flat_lookup_block_orders(
    spans: list[tuple[int, int, int]], start: int, end: int
) -> list[int]:
    """Return monotonic block orders whose (s,e) intersect [start,end)."""
    out: list[int] = []
    for s, e, bo in spans:
        # interval intersection
        if e <= start or s >= end:
            continue
        if bo not in out:
            out.append(bo)
    # spans were built monotonic by block_order → output already monotonic.
    return out


# ---------------------------------------------------------------------------
# Chunk Metadata Contract helper: build base Layer A + Layer B fields
# ---------------------------------------------------------------------------
def build_cmcv1_base_fields(
    loaded: LoadedDoc,
) -> dict[str, Any]:
    """Build Layer A + Layer B fields for a chunk.

    Identical for all chunks of one document.
    """
    entry = loaded.manifest_entry
    meta = loaded.doc_metadata_inherited
    base: dict[str, Any] = dict(
        # Layer A (non-chunk-specific; chunk_id/chunk_index filled later)
        document_id=loaded.document_id,
        logical_document_id=loaded.logical_document_id,
        source_file=str(loaded.source_path),
        source_sha256=loaded.source_sha256,
        source_format=loaded.source_format,
        # Layer B projected — inherit NULL stays NULL
        title=meta.get("title") or entry.get("title"),
        source_org=meta.get("source_org") or entry.get("source_org"),
        publish_date=meta.get("publish_date") or entry.get("publish_date"),
        creation_date=meta.get("creation_date") or entry.get("creation_date"),
        effective_date=meta.get("effective_date") or entry.get("effective_date"),
        expiry_date=meta.get("expiry_date") or entry.get("expiry_date"),
        document_number=meta.get("document_number") or entry.get("document_number"),
        document_index_number=meta.get("document_index_number") or entry.get("document_index_number"),
        document_type=meta.get("document_type") or entry.get("document_type"),
        authority_level=meta.get("authority_level") or entry.get("authority_level"),
        legal_status=meta.get("legal_status") or entry.get("status"),
        jurisdiction=meta.get("jurisdiction") or entry.get("jurisdiction"),
        region=meta.get("region") or entry.get("region"),
        theme_category=meta.get("theme_category") or entry.get("theme_category"),
        source_url=meta.get("source_url") or entry.get("source_page_url"),
    )
    # Strip empty strings back to None for Layer B fields (NULL stays NULL contract).
    for k, v in list(base.items()):
        if isinstance(v, str) and v == "":
            base[k] = None
    return base


def build_quality_flags(entry: dict[str, Any]) -> ChunkQualityFlags:
    """Default PASS; try inherit qgate_* or qg_* fields from manifest if present."""
    verdict = "PASS"
    warnings: list[str] = []
    rule_ids: list[str] = []
    has_review = False
    for k, v in entry.items():
        kn = str(k).lower()
        if "qgate" in kn or kn.startswith("qg_"):
            if kn.endswith("verdict") and isinstance(v, str):
                verdict = v.upper() if v.upper() in {"PASS","WARNING","POLICY_REVIEW","REVIEW","FAIL"} else "PASS"
            elif kn.endswith("warnings") and isinstance(v, list):
                warnings.extend(str(x) for x in v if x)
            elif kn.endswith("rule_ids") and isinstance(v, list):
                rule_ids.extend(str(x) for x in v if x)
            elif kn.endswith("review_content") and bool(v):
                has_review = True
    return ChunkQualityFlags(
        quality_verdict=verdict,
        quality_warnings=warnings,
        has_review_content=has_review,
        review_rule_ids=rule_ids,
    )
