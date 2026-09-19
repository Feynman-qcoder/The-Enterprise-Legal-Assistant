"""
Task 2: Build A/B corpora (CANON chunks + SYN_CONTRACT_* docx chunks via frozen chunking strategies)
and produce SYN_citation_block_map to enable precise Citation Evidence Hit for A类 queries.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _reval_utils import (  # noqa: E402
    CANON_A_CHUNKS_PATH,
    CANON_B_CHUNKS_PATH,
    CORPUS_A_PATH,
    CORPUS_B_PATH,
    CORPUS_DIR,
    SYN_BLOCK_MAP_PATH,
    SYN_DIR,
    SYN_FACTS_PATH,
    SEED,
    ensure_dirs,
    inject_sys_path,
    read_json,
    read_jsonl,
    set_global_seed,
    sha256_file,
    write_json,
    file_stat,
)
ensure_dirs()
inject_sys_path()
set_global_seed(SEED)


def _normalize_for_corpus(rec: dict[str, Any], strategy: str) -> dict[str, Any]:
    """Canonicalize a chunk record into the minimal shape for embedding + retrieval."""
    text = rec.get("text") or ""
    chunk_id = rec.get("chunk_id") or ""
    meta = rec.get("chunk_metadata") or {}
    # document_id
    doc_id = (
        meta.get("document_id")
        or (meta.get("identity") or {}).get("document_id")
        or (meta.get("document_metadata") or {}).get("document_id")
        or None
    )
    prov = meta.get("provenance") or {}
    sbo = prov.get("source_block_orders") or meta.get("source_block_orders") or []
    if sbo is None:
        sbo = []
    sbo_set = sorted({int(x) for x in sbo if isinstance(x, (int, float)) and x is not None})
    return {
        "chunk_id": chunk_id,
        "text": text,
        "text_len_chars": len(text),
        "strategy": strategy,
        "document_id": doc_id,
        "source_block_orders": sbo_set,
        "chunk_metadata": {
            "document_id": doc_id,
            "format": (meta.get("document_metadata") or {}).get("format"),
            "source_file": (prov.get("source_file") or (meta.get("provenance") or {}).get("source_file")),
            "chunk_level": (meta.get("structural_context") or {}).get("chunk_level"),
        },
    }


def load_canon_chunks(p: Path, strategy: str) -> list[dict[str, Any]]:
    data = read_json(p)
    chunks = data.get("chunks", [])
    out: list[dict[str, Any]] = []
    for c in chunks:
        out.append(_normalize_for_corpus(c, strategy))
    return out


def build_syn_entry(docx_path: Path, syn_meta: dict[str, Any]) -> dict[str, Any]:
    """Create a manifest-style entry to pass to dispatch_parse_and_clean."""
    doc_stem = docx_path.stem  # SYN_CONTRACT_001_DATA_PROC
    # Compute sha from file to obtain unique normalized_sha256
    sha = sha256_file(docx_path)
    logical_doc_id = syn_meta.get("synthetic_contract_id", doc_stem)
    return {
        "logical_document_id": logical_doc_id,
        "document_id": logical_doc_id,
        "format": "docx",
        "source_path": str(docx_path),
        "source_file": docx_path.name,
        "normalized_sha256": sha,
        "ingest_approved": True,
        "synthetic": True,
        "_syn_meta": syn_meta,
    }


def chunk_syn_doc(entry: dict[str, Any]):
    """Return (a_chunks, b_chunks) for one SYN doc."""
    import importlib
    from dataclasses import asdict
    from pathlib import Path

    from _v2_utils import LoadedDoc, DOCUMENT_METADATA_PROJECTION_MAP  # type: ignore
    from _v2_strategies import chunk_strategy_a, chunk_strategy_b  # type: ignore

    # Manually parse DOCX + build LoadedDoc entry to avoid _v2_utils.resolve_source_path
    # which only indexes canonical manifest files (not real_contract dir).
    source_path = Path(entry["source_path"])
    mod = importlib.import_module("docx_parser_v2")
    fn = getattr(mod, "parse_docx_v2")
    doc = fn(str(source_path))
    if isinstance(doc, tuple):
        doc = doc[0]
    document_id = entry["logical_document_id"]

    # Build metadata inheritance like frozen helper does (minimal)
    doc_meta_inherited: dict[str, Any] = {}
    for doc_field, projection in DOCUMENT_METADATA_PROJECTION_MAP.items():
        mk = projection.document_metadata_field
        doc_meta_inherited[doc_field] = entry.get(mk)
    for extra in ("logical_document_id", "title", "source_file", "format", "normalized_sha256"):
        if extra in entry and doc_meta_inherited.get(extra) is None:
            doc_meta_inherited[extra] = entry.get(extra)

    # source_pages: docx blocks — if page attribute absent/None, assign 1 to avoid CMCV1 violation
    pages_from_blocks = [getattr(b, "page", None) for b in doc.blocks]
    source_pages = []
    for p in pages_from_blocks:
        if isinstance(p, int) and p >= 1:
            source_pages.append(p)
        else:
            source_pages.append(1)

    loaded = LoadedDoc(
        manifest_entry=entry,
        document_id=document_id,
        logical_document_id=document_id,
        source_path=source_path,
        source_sha256=entry.get("normalized_sha256") or "",
        source_format=entry.get("format", "docx"),
        parser_name="docx_parser_v2",
        parser_version="2.4.2-frozen-patched",
        doc=doc,
        doc_metadata_inherited=doc_meta_inherited,
        source_pages=source_pages,
    )
    a_chunks_raw, _audit_a = chunk_strategy_a(loaded)
    b_chunks_raw, _audit_b = chunk_strategy_b(loaded)
    # V2Chunk objects have .chunk_id, .text, .chunk_metadata (CMCV1 dataclass)
    def _ch_to_rec(ch, strategy: str) -> dict[str, Any]:
        meta_dict = {}
        cmd = getattr(ch, "chunk_metadata", None)
        if cmd is not None:
            # convert CMCV1 dataclass -> dict (asdict from frozen utils)
            from dataclasses import asdict
            try:
                meta_dict = asdict(cmd)
            except Exception:
                meta_dict = {k: getattr(cmd, k) for k in dir(cmd) if not k.startswith("_") and not callable(getattr(cmd, k))}
        return {
            "chunk_id": getattr(ch, "chunk_id", None) or "",
            "text": getattr(ch, "text", None) or "",
            "chunk_metadata": meta_dict,
            "strategy": strategy,
        }
    a_recs = [_normalize_for_corpus(_ch_to_rec(c, "A"), "A") for c in a_chunks_raw]
    b_recs = [_normalize_for_corpus(_ch_to_rec(c, "B"), "B") for c in b_chunks_raw]
    return a_recs, b_recs


def build_syn_citation_block_map(
    syn_facts: list[dict[str, Any]],
    syn_a_chunks: dict[str, list[dict[str, Any]]],
    syn_b_chunks: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build synthetic_contract_id -> target_field -> value_str -> {A:[chunk_ids],B:[chunk_ids]} mapping.
    target_fields covered: party_a, party_b, amount, sign_date, expiry_date, governing_or_dispute_resolution,
                            contract_number, auto_renewal, tax_rate, payment_terms (top 5 required + 5 bonus)
    """
    CORE_FIELDS = [
        "party_a.name", "party_b.name", "amount", "sign_date", "expiry_date",
        "governing_or_dispute_resolution", "contract_number", "auto_renewal",
        "tax_rate", "payment_terms",
    ]

    def _get_value(syn: dict, fieldpath: str) -> tuple[str, list[str]]:
        """Return (display_value, candidate_strings_to_search_in_text)."""
        parts = fieldpath.split(".")
        obj: Any = syn
        keys_so_far = []
        for p in parts:
            keys_so_far.append(p)
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                # fallback: important_clause_facts.tax_rate
                if fieldpath == "tax_rate":
                    v = syn.get("important_clause_facts", {}).get("tax_rate")
                    if v is None:
                        return "", []
                    return f"{v*100:g}%", [f"{int(v*100)}%", f"{v*100:g}%", f"{v}"]
                return "", []
        # Format the strings to search
        if obj is None:
            return "", []
        if isinstance(obj, bool):
            if fieldpath == "auto_renewal":
                txt = "自动续约" if obj else "不自动续约"
                return txt, [txt, "自动续期" if obj else "不自动续期"]
            return str(obj), [str(obj)]
        if isinstance(obj, (int, float)) and fieldpath == "amount":
            cand = [f"{int(obj):,}元", f"{int(obj):,}", f"{obj:,}", f"{obj}"]
            # also add 汉字 if small
            return cand[0], cand
        if isinstance(obj, (int, float)) and fieldpath == "tax_rate":
            cand = [f"{obj*100:g}%"]
            return cand[0], cand
        s = str(obj).strip()
        if not s:
            return "", []
        cands = [s]
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            y, mo, d = m.groups()
            cands.append(f"{y}年{int(mo)}月{int(d)}日")
        return s, cands

    def _find_chunk_ids(chunks: list[dict[str, Any]], needles: list[str]) -> list[str]:
        ids: list[str] = []
        if not needles:
            return ids
        for ch in chunks:
            t = ch.get("text") or ""
            for n in needles:
                if not n:
                    continue
                if n in t:
                    ids.append(ch.get("chunk_id", ""))
                    break
        return [cid for cid in ids if cid]

    result: dict[str, Any] = {
        "version": 1,
        "by_synthetic_contract_id": {},
    }
    for syn in syn_facts:
        sid = syn["synthetic_contract_id"]
        a_chs = syn_a_chunks.get(sid, [])
        b_chs = syn_b_chunks.get(sid, [])
        field_map: dict[str, Any] = {}
        covered_count = 0
        for fp in CORE_FIELDS:
            display_val, needles = _get_value(syn, fp)
            a_ids = _find_chunk_ids(a_chs, needles)
            b_ids = _find_chunk_ids(b_chs, needles)
            field_map[fp] = {
                "ground_truth_display": display_val,
                "search_needles": needles,
                "A_chunk_ids": sorted(set(a_ids)),
                "B_chunk_ids": sorted(set(b_ids)),
            }
            if a_ids or b_ids:
                covered_count += 1
        result["by_synthetic_contract_id"][sid] = {
            "synthetic_file": syn.get("file_name"),
            "core_fields_covered": covered_count,
            "core_fields_total": len(CORE_FIELDS),
            "fields": field_map,
        }
    return result


def main() -> dict:
    # 1. CANON
    canon_a = load_canon_chunks(CANON_A_CHUNKS_PATH, "A")
    canon_b = load_canon_chunks(CANON_B_CHUNKS_PATH, "B")
    print(f"CANON: A={len(canon_a)}, B={len(canon_b)}")

    # 2. SYN
    syn_facts = read_jsonl(SYN_FACTS_PATH)
    syn_meta_by_filename = {s["file_name"]: s for s in syn_facts}
    syn_docx_files = sorted(SYN_DIR.glob("SYN_CONTRACT_*.docx"))
    assert len(syn_docx_files) == 10

    syn_a_all: list[dict[str, Any]] = []
    syn_b_all: list[dict[str, Any]] = []
    syn_a_by_sid: dict[str, list[dict[str, Any]]] = {}
    syn_b_by_sid: dict[str, list[dict[str, Any]]] = {}
    for f in syn_docx_files:
        meta = syn_meta_by_filename[f.name]
        sid = meta["synthetic_contract_id"]
        entry = build_syn_entry(f, meta)
        print(f"  Chunking {sid} / {f.name} ...")
        a_recs, b_recs = chunk_syn_doc(entry)
        print(f"    -> A={len(a_recs)} chunks, B={len(b_recs)} chunks")
        # Append synthetic doc prefix to document_id for safety
        for r in a_recs:
            if not r.get("document_id"):
                r["document_id"] = sid
        for r in b_recs:
            if not r.get("document_id"):
                r["document_id"] = sid
        syn_a_all.extend(a_recs)
        syn_b_all.extend(b_recs)
        syn_a_by_sid[sid] = a_recs
        syn_b_by_sid[sid] = b_recs

    # 3. Merge
    corpus_a = canon_a + syn_a_all
    corpus_b = canon_b + syn_b_all

    # Strip any chunks with empty text just in case (but report count)
    def _strip_empty(corpus: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        empty = 0
        out = []
        for r in corpus:
            if r.get("text"):
                out.append(r)
            else:
                empty += 1
        return out, empty
    corpus_a, empty_a = _strip_empty(corpus_a)
    corpus_b, empty_b = _strip_empty(corpus_b)

    # Check chunk_id uniqueness within each corpus
    def _dedup_ids(corpus: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        seen = set(); out = []; dups = 0
        for r in corpus:
            cid = r["chunk_id"]
            if cid in seen:
                dups += 1
                # regenerate unique: hash(text + doc_id + idx)
                raw = f"{r['text'][:100]}|{r.get('document_id')}|{len(out)}"
                new_id = "dup_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                r["chunk_id"] = new_id
                cid = new_id
            seen.add(cid)
            out.append(r)
        return out, dups
    corpus_a, dup_a = _dedup_ids(corpus_a)
    corpus_b, dup_b = _dedup_ids(corpus_b)

    write_json(CORPUS_A_PATH, {"count": len(corpus_a), "strategy": "A",
                               "canon_count": len(canon_a), "syn_count": len(syn_a_all),
                               "empty_dropped": empty_a, "duplicates_renamed": dup_a,
                               "chunks": corpus_a})
    write_json(CORPUS_B_PATH, {"count": len(corpus_b), "strategy": "B",
                               "canon_count": len(canon_b), "syn_count": len(syn_b_all),
                               "empty_dropped": empty_b, "duplicates_renamed": dup_b,
                               "chunks": corpus_b})

    # 4. SYN citation block map
    syn_map = build_syn_citation_block_map(syn_facts, syn_a_by_sid, syn_b_by_sid)
    write_json(SYN_BLOCK_MAP_PATH, syn_map)

    stats = {
        "A": {
            "total": len(corpus_a), "canon": len(canon_a), "syn": len(syn_a_all),
            "empty": empty_a, "dups_renamed": dup_a,
            "per_syn": {k: len(v) for k, v in syn_a_by_sid.items()},
            "source_files": {
                "canon_a_chunks": file_stat(CANON_A_CHUNKS_PATH),
            },
        },
        "B": {
            "total": len(corpus_b), "canon": len(canon_b), "syn": len(syn_b_all),
            "empty": empty_b, "dups_renamed": dup_b,
            "per_syn": {k: len(v) for k, v in syn_b_by_sid.items()},
            "source_files": {
                "canon_b_chunks": file_stat(CANON_B_CHUNKS_PATH),
            },
        },
        "syn_block_map": {
            "docs": len(syn_map["by_synthetic_contract_id"]),
            "covered_core_fields_sum": sum(
                v["core_fields_covered"] for v in syn_map["by_synthetic_contract_id"].values()
            ),
            "total_core_fields_sum": sum(
                v["core_fields_total"] for v in syn_map["by_synthetic_contract_id"].values()
            ),
        },
    }
    with open(HERE / "_corpora_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    s = main()
    print(json.dumps({
        "A_total": s["A"]["total"], "B_total": s["B"]["total"],
        "A_per_syn": s["A"]["per_syn"], "B_per_syn": s["B"]["per_syn"],
        "SYN_block_map_coverage": f"{s['syn_block_map']['covered_core_fields_sum']}/{s['syn_block_map']['total_core_fields_sum']}",
    }, ensure_ascii=False, indent=2))
