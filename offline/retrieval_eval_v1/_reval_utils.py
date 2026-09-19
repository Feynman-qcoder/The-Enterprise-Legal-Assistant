"""
Retrieval Evaluation V1 — Shared Utilities.

Provides:
  - path constants
  - sys.path injection (Legal_System repo, chunking dir, cmc dir)
  - embedding STOP gate probe
  - canonical chunks reader
  - synthetic facts / legal eval reader
  - determinism seed
  - frozen module import guard helpers

FROZEN boundary: this module NEVER imports Parser/Cleaner/CMCV1/Chunking Strategy
directly. Those imports happen inside individual task scripts via chunking_strategy_v2.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Absolute path constants
# ---------------------------------------------------------------------------
RAG_WORK_DIR: Path = Path(__file__).resolve().parent.parent  # offline/（Task 22 迁移适配：原为 TRAE 工作区绝对路径）
OUT_DIR: Path = (RAG_WORK_DIR / "retrieval_eval_v1").resolve()
EMB_INDEX_DIR: Path = (OUT_DIR / "offline_embedding_index").resolve()

CHUNKING_DIR: Path = (RAG_WORK_DIR / "chunking_strategy_v2").resolve()
CMC_DIR: Path = (RAG_WORK_DIR / "chunk_metadata_contract_v1").resolve()

LS_DIR: Path = Path(__file__).resolve().parents[2]  # 仓库根（Task 22 迁移适配）
LS_MODULES: Path = LS_DIR / "modules"

CORPUS_DIR: Path = Path(os.environ.get("XIAOYI_CORPUS_DIR", r"D:\xiaoyi\data_source")).resolve()  # 环境变量可覆盖（Task 22 迁移适配）
META_DIR: Path = CORPUS_DIR / "_meta"
SYN_DIR: Path = CORPUS_DIR / "real_contract"

MANIFEST_PATH: Path = META_DIR / "ingest_manifest_v1.jsonl"
SYN_FACTS_PATH: Path = SYN_DIR / "synthetic_contract_facts.jsonl"
LEGAL_EVAL_PATH: Path = META_DIR / "legal_eval_v1.jsonl"

CANON_A_CHUNKS_PATH: Path = CHUNKING_DIR / "strategy_a_chunks.json"
CANON_B_CHUNKS_PATH: Path = CHUNKING_DIR / "strategy_b_chunks.json"

BGE_MODEL_ABS: Path = LS_DIR / "models" / "bge-m3"

QUESTIONS_PATH: Path = OUT_DIR / "retrieval_questions.jsonl"
CORPUS_A_PATH: Path = EMB_INDEX_DIR / "corpus_strategy_a.json"
CORPUS_B_PATH: Path = EMB_INDEX_DIR / "corpus_strategy_b.json"
SYN_BLOCK_MAP_PATH: Path = EMB_INDEX_DIR / "SYN_citation_block_map.json"
INDEX_MANIFEST_PATH: Path = EMB_INDEX_DIR / "index_manifest.json"
INDEX_A_NPZ_PATH: Path = EMB_INDEX_DIR / "strategy_a.npz"
INDEX_B_NPZ_PATH: Path = EMB_INDEX_DIR / "strategy_b.npz"
QUERY_EMB_NPZ_PATH: Path = EMB_INDEX_DIR / "query_embeddings.npz"
RETRIEVAL_RESULTS_CSV: Path = OUT_DIR / "retrieval_query_results.csv"
METRICS_CSV: Path = OUT_DIR / "retrieval_ab_metrics.csv"
FAILURE_SAMPLES_JSON: Path = OUT_DIR / "failure_samples.json"
REPORT_MD: Path = OUT_DIR / "retrieval_eval_v1.md"
REPORT_JSON: Path = OUT_DIR / "retrieval_eval_v1.json"
PYTEST_RESULT_JSON: Path = OUT_DIR / "_run_pytest_result.json"
MODULES_SNAPSHOT_JSON: Path = OUT_DIR / "_process_modules_snapshot.json"

# Deterministic seed
SEED: int = 20260821

# Forbidden module prefixes (for T8 import guard)
FORBIDDEN_MODULES: tuple[str, ...] = (
    "modules.milvus_store",
    "modules.database",
    "modules.cache",
    "modules.rerank",
    "modules.rag.hybrid_rrf",
    "pymilvus",
)


def ensure_dirs() -> None:
    for d in (OUT_DIR, EMB_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# sys.path injection — make Legal_System, chunking, CMC importable
# ---------------------------------------------------------------------------
def inject_sys_path() -> None:
    extras: list[str] = [
        str(LS_DIR),
        str(LS_MODULES),
        str(LS_DIR / "modules" / "ingestion"),
        str(CMC_DIR),
        str(CHUNKING_DIR),
    ]
    for p in extras:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Also set cwd-relevant env so config.py reads the right .env if needed
    os.environ.setdefault("EMBEDDING_MODEL_PATH", "models/bge-m3")
    os.environ.setdefault("EMBEDDING_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np  # lazy
        np.random.seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Basic readers
# ---------------------------------------------------------------------------
def read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {"exists": False}
    st = p.stat()
    return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}


# ---------------------------------------------------------------------------
# Manifest reader (for SYN docs — build manifest-style entries)
# ---------------------------------------------------------------------------
def read_canon_manifest() -> list[dict[str, Any]]:
    """Read canonical-53 ingest manifest."""
    return read_jsonl(MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Embedding STOP gate probe
# ---------------------------------------------------------------------------
@dataclass
class EmbProbeResult:
    model_path_exists: bool
    model_path_resolved: str
    stop_reason: str | None  # None = OK, "MODEL_PATH_MISSING" = must STOP
    ping_ok: bool | None
    dimension: int | None
    error: str | None


def probe_embedding_service(fake_model_path: str | None = None) -> EmbProbeResult:
    """
    Probe BGE-M3 availability.
    If fake_model_path is set: patch settings to return non-existent path (for T9).
    Returns EmbProbeResult.
    """
    inject_sys_path()
    try:
        from modules.core.config import get_settings, Settings  # type: ignore
    except Exception as e:
        return EmbProbeResult(False, "", "MODEL_PATH_MISSING", None, None, f"config import err: {e}")

    # Resolve model path the same way LocalEmbedding does
    if fake_model_path:
        resolved = Path(fake_model_path).resolve()
    else:
        settings = get_settings()
        resolved = (LS_DIR / settings.embedding_model_path).resolve()

    exists = resolved.exists()
    if not exists:
        return EmbProbeResult(
            model_path_exists=False,
            model_path_resolved=str(resolved),
            stop_reason="MODEL_PATH_MISSING",
            ping_ok=False,
            dimension=None,
            error=None,
        )

    # Actually ping: import and dimension property
    try:
        # Temporarily swap settings if fake path requested but path exists? Only used in real run.
        from modules.embeddings.local_embedding import LocalEmbeddingService  # type: ignore
        svc = LocalEmbeddingService()
        dim = svc.dimension
        return EmbProbeResult(
            model_path_exists=True,
            model_path_resolved=str(resolved),
            stop_reason=None,
            ping_ok=True,
            dimension=dim,
            error=None,
        )
    except Exception as e:
        return EmbProbeResult(
            model_path_exists=True,
            model_path_resolved=str(resolved),
            stop_reason="MODEL_PATH_MISSING",
            ping_ok=False,
            dimension=None,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Forbidden module snapshot
# ---------------------------------------------------------------------------
def snapshot_imported_modules() -> dict[str, bool]:
    """Return dict mapping each forbidden module name -> is imported."""
    return {mod: any(k.startswith(mod) for k in sys.modules.keys()) for mod in FORBIDDEN_MODULES}


def any_forbidden_imported() -> bool:
    return any(snapshot_imported_modules().values())


# ---------------------------------------------------------------------------
# Chunk record helper
# ---------------------------------------------------------------------------
def extract_chunk_meta(chunk_record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a chunk record dict to a flattened summary."""
    meta = chunk_record.get("chunk_metadata") or {}
    doc_id = meta.get("document_id") or meta.get("logical_document_id") or meta.get("identity", {}).get("document_id")
    prov = meta.get("provenance") or {}
    sbo = prov.get("source_block_orders") or meta.get("source_block_orders") or []
    if isinstance(sbo, list):
        sbo_set = set(int(x) for x in sbo if x is not None)
    else:
        sbo_set = set()
    return {
        "document_id": doc_id,
        "source_block_orders_set": sorted(sbo_set),
        "chunk_id": chunk_record.get("chunk_id"),
        "text": chunk_record.get("text") or "",
        "strategy": chunk_record.get("strategy"),
        "chunk_metadata_raw": meta,
    }
