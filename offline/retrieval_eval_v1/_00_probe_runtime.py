"""
Task 0: Runtime probe for Retrieval Evaluation V1.
Checks that all data sources and dependencies are available, including the Emb STOP gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _reval_utils import (  # noqa: E402
    CANON_A_CHUNKS_PATH,
    CANON_B_CHUNKS_PATH,
    EMB_INDEX_DIR,
    LEGAL_EVAL_PATH,
    LS_DIR,
    MANIFEST_PATH,
    OUT_DIR,
    SYN_DIR,
    SYN_FACTS_PATH,
    ensure_dirs,
    probe_embedding_service,
    read_json,
    read_jsonl,
    snapshot_imported_modules,
    write_json,
)


def main() -> dict:
    ensure_dirs()
    report: dict = {"probe_version": "1.0", "paths": {}}

    # ---- Canon A/B chunks
    for name, p in (("CANON_A", CANON_A_CHUNKS_PATH), ("CANON_B", CANON_B_CHUNKS_PATH)):
        info = {"path": str(p), "exists": p.exists()}
        if info["exists"]:
            try:
                data = read_json(p)
                info["nonempty"] = bool(data.get("chunks")) and (data.get("count", 0) > 0)
                info["count"] = data.get("count", 0)
            except Exception as e:
                info["nonempty"] = False
                info["error"] = str(e)
        report["paths"][name] = info

    # ---- SYN DOCX
    syn_docx = sorted(SYN_DIR.glob("SYN_CONTRACT_*.docx"))
    report["paths"]["SYN_DOCX"] = {
        "path": str(SYN_DIR),
        "count": len(syn_docx),
        "files": [p.name for p in syn_docx],
    }

    # ---- SYN facts
    info_sf = {"path": str(SYN_FACTS_PATH), "exists": SYN_FACTS_PATH.exists()}
    if info_sf["exists"]:
        try:
            rows = read_jsonl(SYN_FACTS_PATH)
            info_sf["rows"] = len(rows)
        except Exception as e:
            info_sf["error"] = str(e)
    report["paths"]["SYN_FACTS"] = info_sf

    # ---- Legal eval
    info_le = {"path": str(LEGAL_EVAL_PATH), "exists": LEGAL_EVAL_PATH.exists()}
    if info_le["exists"]:
        try:
            rows = read_jsonl(LEGAL_EVAL_PATH)
            info_le["rows"] = len(rows)
        except Exception as e:
            info_le["error"] = str(e)
    report["paths"]["LEGAL_EVAL"] = info_le

    # ---- Manifest
    info_mf = {"path": str(MANIFEST_PATH), "exists": MANIFEST_PATH.exists()}
    if info_mf["exists"]:
        try:
            rows = read_jsonl(MANIFEST_PATH)
            info_mf["rows"] = len(rows)
        except Exception as e:
            info_mf["error"] = str(e)
    report["paths"]["MANIFEST"] = info_mf

    # ---- LS dir
    report["paths"]["LS_DIR"] = {"path": str(LS_DIR), "exists": LS_DIR.exists()}

    # ---- Embedding STOP gate probe (REAL)
    emb = probe_embedding_service()
    report["embedding_probe"] = {
        "model_path_exists": emb.model_path_exists,
        "model_path_resolved": emb.model_path_resolved,
        "stop_reason": emb.stop_reason,
        "ping_ok": emb.ping_ok,
        "dimension": emb.dimension,
        "error": emb.error,
    }

    # ---- Forbidden module snapshot (before any heavy import)
    report["forbidden_modules_snapshot"] = snapshot_imported_modules()

    # ---- Derived summary flags
    report["CANON_A_NONEMPTY"] = report["paths"]["CANON_A"].get("nonempty", False)
    report["CANON_B_NONEMPTY"] = report["paths"]["CANON_B"].get("nonempty", False)
    report["SYN_DOCX_COUNT"] = report["paths"]["SYN_DOCX"].get("count", 0)
    report["SYN_FACTS_ROWS"] = report["paths"]["SYN_FACTS"].get("rows", 0)
    report["LEGAL_EVAL_ROWS"] = report["paths"]["LEGAL_EVAL"].get("rows", 0)
    report["EMB_MODEL_PATH_EXISTS"] = emb.model_path_exists
    report["EMB_MODEL_PATH_MISSING"] = (emb.stop_reason == "MODEL_PATH_MISSING")
    report["STOP_REASON"] = emb.stop_reason or ""

    write_json(OUT_DIR / "_probe_runtime.json", report)
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    print("\n--- Summary ---")
    for k in (
        "CANON_A_NONEMPTY",
        "CANON_B_NONEMPTY",
        "SYN_DOCX_COUNT",
        "SYN_FACTS_ROWS",
        "LEGAL_EVAL_ROWS",
        "EMB_MODEL_PATH_EXISTS",
        "STOP_REASON",
    ):
        print(f"  {k} = {r.get(k)}")
    forb = r.get("forbidden_modules_snapshot", {})
    print(f"  FORBIDDEN_IMPORTED = {any(forb.values())}")
