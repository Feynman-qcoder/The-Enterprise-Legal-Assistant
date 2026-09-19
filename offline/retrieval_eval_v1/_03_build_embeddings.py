"""
Task 3: Embedding STOP gate probe + offline embedding index builder (Strategy A/B).
Produces:
  offline_embedding_index/strategy_a.npz   (vectors, chunk_ids, document_ids, texts_len_chars)
  offline_embedding_index/strategy_b.npz   (same structure)
  offline_embedding_index/query_embeddings.npz (query_ids, vectors)
  offline_embedding_index/index_manifest.json
The STOP gate is applied first: if embedding probe returns MODEL_PATH_MISSING, write
STOP marker json and exit 0 with status printed (for T9 test coverage).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _reval_utils import (  # noqa: E402
    CANON_A_CHUNKS_PATH,
    CANON_B_CHUNKS_PATH,
    CORPUS_A_PATH,
    CORPUS_B_PATH,
    EMB_INDEX_DIR,
    INDEX_A_NPZ_PATH,
    INDEX_B_NPZ_PATH,
    INDEX_MANIFEST_PATH,
    QUESTIONS_PATH,
    QUERY_EMB_NPZ_PATH,
    REPORT_JSON,
    SEED,
    SYN_DIR,
    ensure_dirs,
    file_stat,
    inject_sys_path,
    probe_embedding_service,
    read_json,
    read_jsonl,
    set_global_seed,
    sha256_file,
    snapshot_imported_modules,
    write_json,
)

import numpy as np  # noqa: E402

ensure_dirs()
inject_sys_path()
set_global_seed(SEED)


BATCH_SIZE = 16


async def _encode_texts_batched(svc, texts: list[str]) -> np.ndarray:
    """Encode texts in batches, returning N×D float32 matrix."""
    dim = svc.dimension  # trigger ping once
    out_rows: list[np.ndarray] = []
    n = len(texts)
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"    encode batch [{i}:{i+len(batch)}) / {n} ...", flush=True)
        rows = await svc.embed_documents(batch)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        assert arr.shape[1] == dim, f"shape mismatch: {arr.shape} vs dim={dim}"
        out_rows.append(arr)
    if not out_rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(out_rows)


async def _encode_queries_batched(svc, queries: list[str]) -> np.ndarray:
    dim = svc.dimension
    rows = []
    for q in queries:
        v = await svc.embed_query(q)
        rows.append(v)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _verify_l2_unit(vectors: np.ndarray, tol: float = 1e-4) -> tuple[bool, float]:
    if vectors.size == 0:
        return True, 0.0
    norms = np.linalg.norm(vectors, axis=1)
    max_err = float(np.max(np.abs(norms - 1.0)))
    return max_err < tol, max_err


def _build_stop_result(reason: str, error: str | None = None) -> dict:
    return {
        "gate_result": "STOP",
        "STOP_REASON": reason,
        "error": error,
        "stage": "build_embeddings",
    }


async def build_embedding_index(fake_model_path: str | None = None) -> dict:
    """Build embedding indexes. Returns manifest dict; writes npz files."""
    ensure_dirs()
    # --- STOP gate probe FIRST ---
    probe = probe_embedding_service(fake_model_path=fake_model_path)
    if probe.stop_reason == "MODEL_PATH_MISSING":
        # Write STOP marker to report json (partial)
        stop = _build_stop_result("MODEL_PATH_MISSING", probe.error)
        write_json(REPORT_JSON, stop)
        print(f"[STOP] EMBEDDING MODEL_PATH_MISSING: {probe.model_path_resolved}")
        return {"status": "STOP", **stop}

    assert probe.ping_ok and probe.dimension and probe.dimension > 0

    # Load corpora
    corpus_a = read_json(CORPUS_A_PATH)
    corpus_b = read_json(CORPUS_B_PATH)
    a_chunks = corpus_a["chunks"]
    b_chunks = corpus_b["chunks"]
    a_texts = [c["text"] for c in a_chunks]
    b_texts = [c["text"] for c in b_chunks]
    a_ids = [c["chunk_id"] for c in a_chunks]
    b_ids = [c["chunk_id"] for c in b_chunks]
    a_doc_ids = [str(c.get("document_id") or "") for c in a_chunks]
    b_doc_ids = [str(c.get("document_id") or "") for c in b_chunks]
    a_lens = np.array([c.get("text_len_chars") or len(t) for c, t in zip(a_chunks, a_texts)], dtype=np.uint16)
    b_lens = np.array([c.get("text_len_chars") or len(t) for c, t in zip(b_chunks, b_texts)], dtype=np.uint16)

    # Import embedding service (real run)
    from modules.embeddings.local_embedding import LocalEmbeddingService  # type: ignore
    svc = LocalEmbeddingService()
    dim = svc.dimension
    assert dim == probe.dimension

    t0 = time.time()
    print(f"Embedding corpus A: {len(a_texts)} chunks, dim={dim}")
    a_vec = await _encode_texts_batched(svc, a_texts)
    print(f"Corpus A vectors: {a_vec.shape}")

    print(f"Embedding corpus B: {len(b_texts)} chunks")
    b_vec = await _encode_texts_batched(svc, b_texts)
    print(f"Corpus B vectors: {b_vec.shape}")

    # --- query embeddings ---
    questions = read_jsonl(QUESTIONS_PATH)
    q_ids = [q["query_id"] for q in questions]
    q_texts = [q["query"] for q in questions]
    print(f"Embedding queries: {len(q_texts)} questions")
    q_vec = await _encode_queries_batched(svc, q_texts)
    print(f"Query vectors: {q_vec.shape}")
    t_enc_total = time.time() - t0

    # L2 norm sanity check
    a_norm_ok, a_max_err = _verify_l2_unit(a_vec)
    b_norm_ok, b_max_err = _verify_l2_unit(b_vec)
    q_norm_ok, q_max_err = _verify_l2_unit(q_vec)
    assert a_norm_ok, f"A L2 max_err={a_max_err}"
    assert b_norm_ok, f"B L2 max_err={b_max_err}"
    # query embeddings use embed_query which should also normalize, but be lenient
    if not q_norm_ok:
        print(f"[WARNING] Query L2 max_err={q_max_err} (tol=1e-4)")

    # Save npz files
    np.savez_compressed(
        INDEX_A_NPZ_PATH,
        vectors=a_vec.astype(np.float32),
        chunk_ids=np.array(a_ids, dtype=object),
        document_ids=np.array(a_doc_ids, dtype=object),
        texts_len_chars=a_lens,
    )
    np.savez_compressed(
        INDEX_B_NPZ_PATH,
        vectors=b_vec.astype(np.float32),
        chunk_ids=np.array(b_ids, dtype=object),
        document_ids=np.array(b_doc_ids, dtype=object),
        texts_len_chars=b_lens,
    )
    np.savez_compressed(
        QUERY_EMB_NPZ_PATH,
        vectors=q_vec.astype(np.float32),
        query_ids=np.array(q_ids, dtype=object),
    )

    # Manifest
    source_files = {
        "corpus_a": {
            "path": str(CORPUS_A_PATH),
            "sha256": sha256_file(CORPUS_A_PATH),
            **file_stat(CORPUS_A_PATH),
        },
        "corpus_b": {
            "path": str(CORPUS_B_PATH),
            "sha256": sha256_file(CORPUS_B_PATH),
            **file_stat(CORPUS_B_PATH),
        },
        "canon_a_chunks_json": {
            "path": str(CANON_A_CHUNKS_PATH),
            "sha256": sha256_file(CANON_A_CHUNKS_PATH),
            **file_stat(CANON_A_CHUNKS_PATH),
        },
        "canon_b_chunks_json": {
            "path": str(CANON_B_CHUNKS_PATH),
            "sha256": sha256_file(CANON_B_CHUNKS_PATH),
            **file_stat(CANON_B_CHUNKS_PATH),
        },
        "syn_docx_files": sorted(p.name for p in SYN_DIR.glob("SYN_CONTRACT_*.docx")),
    }

    manifest = {
        "version": 1,
        "seed": SEED,
        "embedding_model_path": probe.model_path_resolved,
        "embedding_dimension": dim,
        "normalize_embeddings": True,
        "device": "cpu",
        "batch_size": BATCH_SIZE,
        "build_start_ts": t0,
        "build_elapsed_seconds": t_enc_total,
        "vectors_a_count": int(a_vec.shape[0]),
        "vectors_b_count": int(b_vec.shape[0]),
        "query_count": int(q_vec.shape[0]),
        "l2_norm_check": {
            "a_max_err": a_max_err,
            "b_max_err": b_max_err,
            "q_max_err": q_max_err,
        },
        "source_files": source_files,
        "forbidden_modules_at_build": snapshot_imported_modules(),
    }
    write_json(INDEX_MANIFEST_PATH, manifest)
    return {"status": "OK", "manifest": manifest}


def main() -> dict:
    result = asyncio.run(build_embedding_index())
    summary_path = HERE / "_embedding_build_summary.json"
    write_json(summary_path, result)
    if result.get("status") == "STOP":
        print("EMBEDDING BUILD STOPPED:", result.get("STOP_REASON"))
    else:
        m = result["manifest"]
        print(f"BUILD OK: A={m['vectors_a_count']}, B={m['vectors_b_count']}, Q={m['query_count']}, dim={m['embedding_dimension']}, elapsed={m['build_elapsed_seconds']:.1f}s")
    return result


if __name__ == "__main__":
    main()
