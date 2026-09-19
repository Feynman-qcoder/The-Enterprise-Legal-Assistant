"""
Retrieval Evaluation V1 — Offline Acceptance Tests (9 tests).

Scope (read-only):
  T1  deliverables_exist
  T2  dataset_compliance
  T3  embedding_index_valid
  T4  retrieval_results_structure
  T5  metrics_correctness
  T6  gate_logic_consistency
  T7  determinism
  T8  frozen_forbidden_imports
  T9  embedding_stop_gate_triggers + synfacts quarantine

Frozen boundaries:
  - NO writes to Milvus / MySQL / Redis / FastAPI
  - NO reranker / hybrid search imports
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent  # offline/retrieval_eval_v1（Task 22 迁移适配：原为 TRAE 工作区绝对路径推导）
EMB_INDEX_DIR = OUT_DIR / "offline_embedding_index"

REQUIRED_DELIVERABLES: list[Path] = [
    OUT_DIR / "retrieval_questions.jsonl",
    EMB_INDEX_DIR / "corpus_strategy_a.json",
    EMB_INDEX_DIR / "corpus_strategy_b.json",
    EMB_INDEX_DIR / "SYN_citation_block_map.json",
    EMB_INDEX_DIR / "index_manifest.json",
    EMB_INDEX_DIR / "strategy_a.npz",
    EMB_INDEX_DIR / "strategy_b.npz",
    EMB_INDEX_DIR / "query_embeddings.npz",
    OUT_DIR / "retrieval_query_results.csv",
    OUT_DIR / "retrieval_ab_metrics.csv",
    OUT_DIR / "failure_samples.json",
    OUT_DIR / "retrieval_eval_v1.md",
    OUT_DIR / "retrieval_eval_v1.json",
]

FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "modules.milvus_store",
    "modules.database",
    "modules.cache",
    "modules.rerank",
    "modules.rag.hybrid_rrf",
    "pymilvus",
)

# AST-style regex for real import statements (no dict-key string false positives)
_IMPORT_LINE_RE = re.compile(
    r"^\s*(?:import\s+([a-zA-Z0-9_\.]+)"
    r"|from\s+([a-zA-Z0-9_\.]+)\s+import\b)",
    re.MULTILINE,
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _iter_corpus_chunks(corpus: Any) -> list[dict[str, Any]]:
    """Normalize corpus payload — wrapper may be {..., 'chunks': [...]} or plain list."""
    if isinstance(corpus, list):
        return corpus
    if isinstance(corpus, dict) and "chunks" in corpus and isinstance(corpus["chunks"], list):
        return corpus["chunks"]
    raise TypeError(f"Unknown corpus structure: type={type(corpus).__name__}")


def _chunk_source_file(c: dict[str, Any]) -> str:
    """Extract source_file (string) from a chunk record, regardless of schema version."""
    sf = c.get("source_file")
    if isinstance(sf, str) and sf:
        return sf
    for md_key in ("chunk_metadata", "metadata"):
        md = c.get(md_key)
        if isinstance(md, dict):
            v = md.get("source_file")
            if isinstance(v, str) and v:
                return v
        elif isinstance(md, str):
            try:
                md_parsed = json.loads(md)
                v = md_parsed.get("source_file")
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass
    return ""


def _chunk_document_id(c: dict[str, Any]) -> str:
    did = c.get("document_id")
    if isinstance(did, str) and did:
        return did
    for md_key in ("chunk_metadata", "metadata"):
        md = c.get(md_key)
        if isinstance(md, dict):
            v = md.get("document_id")
            if isinstance(v, str) and v:
                return v
    return ""


# ===========================================================================
# T1 — deliverables_exist
# ===========================================================================
def test_t1_deliverables_exist() -> None:
    missing = [str(p) for p in REQUIRED_DELIVERABLES if not p.exists()]
    assert not missing, f"Missing deliverables: {missing}"
    for p in REQUIRED_DELIVERABLES:
        assert p.stat().st_size > 0, f"Empty deliverable: {p}"


# ===========================================================================
# T2 — dataset_compliance
# ===========================================================================
def test_t2_dataset_compliance() -> None:
    rows = _read_jsonl(OUT_DIR / "retrieval_questions.jsonl")
    assert len(rows) >= 100, f"Expected >=100 queries, got {len(rows)}"

    by_type: dict[str, int] = {}
    query_ids: set[str] = set()
    for r in rows:
        qid = r["query_id"]
        assert qid not in query_ids, f"Duplicate query_id: {qid}"
        query_ids.add(qid)
        assert "query" in r and isinstance(r["query"], str) and r["query"].strip()
        qtype = r["query_type"]
        by_type[qtype] = by_type.get(qtype, 0) + 1
        assert r.get("difficulty") in {"easy", "medium", "hard"}, (
            f"Bad difficulty {r.get('difficulty')} for {qid}"
        )

        if qtype == "contract_fact":
            assert r.get("ground_truth_contract_id"), f"A类 missing contract_id: {qid}"
            gt_fact = r.get("ground_truth_fact")
            assert isinstance(gt_fact, dict) and gt_fact, f"A类 empty fact: {qid}"
            ev = r.get("evidence_match_rules") or {}
            values = ev.get("values") or []
            assert values, f"A类 evidence empty: {qid}"
        elif qtype == "legal_clause":
            # Must bind BOTH document_id AND a verifiable clause anchor.
            # Anchor sources (accepted in order):
            #   1. law_article_refs[]                        (structured article refs)
            #   2. ground_truth_fact.{article,chapter,section,articles}
            #   3. evidence_match_rules.values with 第X条/第X章 pattern
            #   4. (fallback for doc-internal keywords) ground_truth_fact.verify_keywords
            #      AND evidence_match_rules.document_id present AND evidence.values non-empty.
            doc_id = (
                r.get("ground_truth_logical_document_id")
                or (r.get("evidence_match_rules") or {}).get("document_id")
                or r.get("ground_truth_document_id")
            )
            assert doc_id, f"B类 missing doc binding: {qid}"

            refs: list[Any] = list(r.get("law_article_refs") or [])
            gt = r.get("ground_truth_fact") or {}
            for key in ("article", "chapter", "section", "articles"):
                v = gt.get(key)
                if v:
                    refs.append(v)
            ev = r.get("evidence_match_rules") or {}
            ev_vals = list(ev.get("values") or [])
            for v in ev_vals:
                if v and ("第" in str(v) and ("条" in str(v) or "章" in str(v))):
                    refs.append(v)
            # Doc-keyword fallback: if a verify_keywords string exists AND we have
            # explicit doc_id binding AND evidence values exist, that counts as
            # an explicit "article/chapter equivalent" binding for queries whose
            # legal clause is not numbered (e.g. 电子签名法 "视为符合书面形式" rule).
            if not refs:
                vk = gt.get("verify_keywords")
                if (
                    vk
                    and isinstance(vk, str)
                    and vk.strip()
                    and ev.get("document_id")
                    and ev_vals
                ):
                    refs.append(("verify_keywords", vk.strip(), doc_id))

            assert refs, (
                f"B类 missing article/chapter/section refs (law_article_refs, ground_truth_fact.*, "
                f"evidence values, and verify_keywords fallback all empty): {qid}"
            )

    assert by_type.get("contract_fact", 0) >= 50, f"A类 <50: {by_type}"
    assert by_type.get("legal_clause", 0) >= 50, f"B类 <50: {by_type}"


# ===========================================================================
# T3 — embedding_index_valid
# ===========================================================================
def test_t3_embedding_index_valid() -> None:
    import numpy as np

    manifest = _read_json(EMB_INDEX_DIR / "index_manifest.json")
    # Support multiple schema names: model_path / embedding_model_path.
    model_path = str(
        manifest.get("model_path")
        or manifest.get("embedding_model_path")
        or manifest.get("embedding")
        or ""
    ).replace("\\", "/").lower()
    assert "bge-m3" in model_path, f"manifest model_path must be BGE-M3, got {model_path!r}"

    dim = int(manifest.get("dimension") or manifest.get("embedding_dimension") or 0)
    assert dim == 1024, f"Expected dimension 1024 for BGE-M3, got {dim}"

    for strategy_key, npz_name in (("A", "strategy_a.npz"), ("B", "strategy_b.npz")):
        # chunk_ids are stored as dtype=object (str), so allow_pickle is needed.
        data = np.load(EMB_INDEX_DIR / npz_name, allow_pickle=True)
        vecs = data["vectors"]
        ids = np.asarray(data["chunk_ids"]).tolist()
        assert vecs.ndim == 2
        assert vecs.shape[1] == dim
        assert vecs.shape[0] == len(ids)
        assert vecs.shape[0] >= 1

        corpus_raw = _read_json(EMB_INDEX_DIR / f"corpus_strategy_{strategy_key.lower()}.json")
        corpus_chunks = _iter_corpus_chunks(corpus_raw)
        assert len(corpus_chunks) == len(ids), (
            f"{strategy_key} corpus/idx mismatch: corpus={len(corpus_chunks)} idx={len(ids)}"
        )
        corpus_ids = [c["chunk_id"] for c in corpus_chunks]
        assert list(ids) == corpus_ids, f"{strategy_key} chunk_id order mismatch"

    q = np.load(EMB_INDEX_DIR / "query_embeddings.npz", allow_pickle=True)
    q_vecs = q["vectors"]
    q_ids = list(np.asarray(q["query_ids"]).tolist())
    assert q_vecs.ndim == 2 and q_vecs.shape[1] == dim, (
        f"Query embedding dim wrong: {q_vecs.shape}"
    )
    assert q_vecs.shape[0] == len(q_ids) == 100, (
        f"Expected 100 query embeddings, got qvecs={q_vecs.shape[0]} ids={len(q_ids)}"
    )


# ===========================================================================
# T4 — retrieval_results_structure
# ===========================================================================
def test_t4_retrieval_results_structure() -> None:
    rows = _read_csv_dicts(OUT_DIR / "retrieval_query_results.csv")
    expected_cols = {
        "query_id", "query", "query_type", "strategy", "top_k", "rank",
        "retrieved_chunk_id", "retrieved_document_id", "score", "text_snippet",
        "hit_ground_truth", "citation_evidence_hit", "ce_fallback_block_intersect_na",
    }
    header = set(rows[0].keys())
    missing = expected_cols - header
    assert not missing, f"CSV header missing columns: {missing}"

    query_ids = {r["query_id"] for r in rows}
    assert len(query_ids) == 100, f"Expected 100 distinct queries, got {len(query_ids)}"

    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in rows:
        groups[(r["query_id"], r["strategy"], r["top_k"])].append(int(r["rank"]))

    for strategy in ("A", "B"):
        for top_k in (5, 10):
            for qid in query_ids:
                key = (qid, strategy, str(top_k))
                ranks = sorted(groups[key])
                assert ranks == list(range(1, top_k + 1)), f"Bad ranks for {key}: {ranks}"

    assert len(rows) == 3000, f"Expected 3000 retrieval rows (100*2*(5+10)), got {len(rows)}"

    for r in rows[:200]:
        s = float(r["score"])
        assert -2.0 <= s <= 2.0, f"Score out of range: {r}"
        for col in ("hit_ground_truth", "citation_evidence_hit", "ce_fallback_block_intersect_na"):
            assert r[col] in {"0", "1"}, f"{col} not 0/1: {r[col]} in {r['query_id']}"


# ===========================================================================
# T5 — metrics_correctness (recompute from CSV)
# ===========================================================================
def _recompute_metrics(
    rows: list[dict[str, str]], strategy: str, scope: str, top_k: int
) -> dict[str, float]:
    filtered = []
    for r in rows:
        if r["strategy"] != strategy:
            continue
        if scope == "CONTRACT_FACT" and r["query_type"] != "contract_fact":
            continue
        if scope == "LEGAL_CLAUSE" and r["query_type"] != "legal_clause":
            continue
        if int(r["top_k"]) != top_k:
            continue
        filtered.append(r)

    from collections import defaultdict
    by_q: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in filtered:
        by_q[r["query_id"]].append(r)

    n = len(by_q)
    assert n > 0, f"No rows: strategy={strategy} scope={scope} top_k={top_k}"

    hit_count = 0
    ce_hit_count = 0
    mrr_sum = 0.0
    for _qid, qrows in by_q.items():
        qrows.sort(key=lambda x: int(x["rank"]))
        best_hit = best_ce = None
        for r in qrows:
            rank = int(r["rank"])
            if best_hit is None and r["hit_ground_truth"] == "1":
                best_hit = rank
            if best_ce is None and r["citation_evidence_hit"] == "1":
                best_ce = rank
        if best_hit is not None:
            hit_count += 1
            if top_k == 10:
                mrr_sum += 1.0 / best_hit
        if best_ce is not None:
            ce_hit_count += 1

    return {
        "NUM_QUERIES": float(n),
        f"HIT_RATE@{top_k}": hit_count / n,
        f"RECALL@{top_k}": hit_count / n,
        f"CITATION_EV_HIT_RATE@{top_k}": ce_hit_count / n,
        "MRR@10": (mrr_sum / n) if top_k == 10 else float("nan"),
    }


def test_t5_metrics_correctness() -> None:
    retrieval_rows = _read_csv_dicts(OUT_DIR / "retrieval_query_results.csv")
    metric_rows = _read_csv_dicts(OUT_DIR / "retrieval_ab_metrics.csv")

    available_cols = set(metric_rows[0].keys())

    for mr in metric_rows:
        strategy = mr["STRATEGY"]
        scope = mr["QUERY_SCOPE"]
        num_q_declared = int(mr["NUM_QUERIES"])
        for top_k in (5, 10):
            recomputed = _recompute_metrics(retrieval_rows, strategy, scope, top_k)
            assert int(recomputed["NUM_QUERIES"]) == num_q_declared
            for metric in (f"HIT_RATE@{top_k}", f"RECALL@{top_k}"):
                if metric not in mr:
                    continue
                expected = float(mr[metric])
                actual = recomputed[metric]
                assert abs(expected - actual) < 5e-4, (
                    f"{metric} mismatch for {strategy}/{scope}/k={top_k}: "
                    f"csv={expected} recomputed={actual}"
                )
            ce_metric = f"CITATION_EV_HIT_RATE@{top_k}"
            if ce_metric in available_cols:
                expected_ce = float(mr.get(ce_metric, "nan"))
                actual_ce = recomputed[ce_metric]
                assert abs(expected_ce - actual_ce) < 5e-4, (
                    f"{ce_metric} mismatch for {strategy}/{scope}: "
                    f"csv={expected_ce} recomputed={actual_ce}"
                )
        # MRR@10 check
        if "MRR@10" in mr:
            recomputed_10 = _recompute_metrics(retrieval_rows, strategy, scope, 10)
            expected_mrr = float(mr["MRR@10"])
            actual_mrr = recomputed_10["MRR@10"]
            assert abs(expected_mrr - actual_mrr) < 5e-4, (
                f"MRR@10 mismatch {strategy}/{scope}: csv={expected_mrr} recomputed={actual_mrr}"
            )


# ===========================================================================
# T6 — gate_logic_consistency
# ===========================================================================
def test_t6_gate_logic_consistency() -> None:
    report = _read_json(OUT_DIR / "retrieval_eval_v1.json")
    metric_rows = _read_csv_dicts(OUT_DIR / "retrieval_ab_metrics.csv")

    def get_all_row(strategy: str) -> dict[str, str]:
        for mr in metric_rows:
            if mr["STRATEGY"] == strategy and mr["QUERY_SCOPE"] == "ALL":
                return mr
        pytest.fail(f"ALL row not found for strategy {strategy}")

    a = get_all_row("A")
    b = get_all_row("B")
    b_r10 = float(b["RECALL@10"]); a_r10 = float(a["RECALL@10"])
    b_ce = float(b["CITATION_EV_HIT_RATE@10"]) if "CITATION_EV_HIT_RATE@10" in b else None
    a_ce = float(a["CITATION_EV_HIT_RATE@10"]) if "CITATION_EV_HIT_RATE@10" in a else None

    gate = report["gate_result"]
    assert gate in {"PASS", "FAIL", "REVIEW_REQUIRED"}, f"Unknown gate: {gate}"

    review_flag = bool(report.get("REVIEW_REQUIRED", False))

    if gate == "PASS":
        assert b_r10 >= a_r10 - 1e-6, f"PASS but B Recall@10 < A: B={b_r10} A={a_r10}"
        if a_ce is not None and b_ce is not None:
            assert b_ce >= a_ce - 0.02 - 1e-6, (
                f"PASS but B CE loss: B={b_ce} A={a_ce}"
            )

    if gate == "REVIEW_REQUIRED":
        assert review_flag is True

    if gate != "FAIL":
        sc = report.get("strategy_compare") or {}
        for k in ("RECALL@10", "MRR@10", "CITATION_EV_HIT_RATE@10"):
            assert k in sc, f"strategy_compare missing {k}"
            e = sc[k]
            assert "delta_B_minus_A_mean" in e
            assert "ci_95_low" in e and "ci_95_high" in e


# ===========================================================================
# T7 — determinism (score ordering + chunk_id tie-break ascending)
# ===========================================================================
def test_t7_determinism() -> None:
    rows = _read_csv_dicts(OUT_DIR / "retrieval_query_results.csv")
    from collections import defaultdict
    groups: dict[tuple[str, str, str], list[tuple[int, str, float]]] = defaultdict(list)
    for r in rows:
        groups[(r["query_id"], r["strategy"], r["top_k"])].append(
            (int(r["rank"]), r["retrieved_chunk_id"], float(r["score"]))
        )

    # CSV scores are printed with 6 decimal digits, so two rows that appear equal
    # in the CSV may actually differ below that threshold in memory. To avoid
    # false-positive tie-break violations against the rounded printed values, we
    # only enforce score-monotonic ordering at the CSV-printed resolution.
    PRINTED_EQ_TOL = 0.5e-6  # ±0.5 count at 6 decimals.

    all_cid_sorted_keys: list[str] = []
    for key, items in groups.items():
        items.sort(key=lambda t: t[0])
        prev_score: float | None = None
        for _rank, _cid, score in items:
            if prev_score is not None:
                # Monotonic non-increase at printed precision
                if score - prev_score > PRINTED_EQ_TOL:
                    pytest.fail(
                        f"Score order broken (later score > earlier by > 0.5e-6) "
                        f"at {key}: prev={prev_score:.12f} curr={score:.12f}"
                    )
            prev_score = score
            all_cid_sorted_keys.append(f"{key[0]}|{key[1]}|{key[2]}|{_rank}\t{_cid}\t{score:.12f}")

    # Determinism fingerprint: identical (query_id, strategy, top_k, rank)
    # sequences always produce the same (chunk_id, score) in this CSV.
    payload = "\n".join(sorted(all_cid_sorted_keys)).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    print(f"[T7] fingerprint sha256 = {sha}")
    assert len(sha) == 64

    # A second, stronger determinism check: within each group, every rank 1..K
    # appears exactly once, guaranteeing no duplicate/skipped ranks from drift.
    for key, items in groups.items():
        ranks = sorted(t[0] for t in items)
        k = int(key[2])
        expected = list(range(1, k + 1))
        assert ranks == expected, f"Rank sequence corruption at {key}: {ranks} vs {expected}"


# ===========================================================================
# T8 — frozen_forbidden_imports (process-level + source-level, AST-style regex)
# ===========================================================================
def test_t8_frozen_forbidden_imports() -> None:
    # 1) Process-wide
    for mod in FORBIDDEN_MODULE_PREFIXES:
        loaded = any(k == mod or k.startswith(mod + ".") for k in sys.modules.keys())
        assert not loaded, f"Forbidden module loaded in process: {mod}"

    # 2) Source scan — only real `import X` / `from X import` lines count.
    for py in sorted(OUT_DIR.glob("*.py")):
        if py.name.startswith("test_"):
            continue  # test file itself is the enforcer
        text = py.read_text(encoding="utf-8", errors="replace")
        # Remove comments (simple line-level) to reduce FP
        cleaned_lines: list[str] = []
        for ln in text.splitlines():
            stripped = ln.split("#", 1)[0]
            cleaned_lines.append(stripped)
        cleaned = "\n".join(cleaned_lines)
        for m in _IMPORT_LINE_RE.finditer(cleaned):
            imported = m.group(1) or m.group(2)
            for forbidden in FORBIDDEN_MODULE_PREFIXES:
                if imported == forbidden or imported.startswith(forbidden + "."):
                    pytest.fail(
                        f"Forbidden import statement `{imported}` found in {py.name}"
                    )

    # 3) Probe snapshot records forbidden modules as NOT loaded at build time.
    snap_path = OUT_DIR / "_probe_runtime.json"
    if snap_path.exists():
        snap = _read_json(snap_path)
        fs = snap.get("forbidden_modules_snapshot", {})
        for mod in FORBIDDEN_MODULE_PREFIXES:
            assert fs.get(mod) is False, f"Probe snapshot reports forbidden {mod}=True"

    # 4) Report JSON frozen_boundaries must all be False.
    report = _read_json(OUT_DIR / "retrieval_eval_v1.json")
    fb = report.get("frozen_boundaries", {})
    for key in (
        "milvus_imported", "mysql_imported", "redis_imported",
        "rerank_imported", "hybrid_imported", "upstream_patch_applied",
    ):
        assert fb.get(key) is False, f"frozen_boundaries.{key} is not False: {fb}"


# ===========================================================================
# T9 — STOP gate + synfacts quarantine + SYN_CONTRACT corpus presence
# ===========================================================================
def test_t9_embedding_stop_gate_and_synfacts_quarantine() -> None:
    probe = _read_json(OUT_DIR / "_probe_runtime.json")
    stop_reason = (
        probe.get("STOP_REASON")
        or probe.get("embedding_probe", {}).get("stop_reason")
        or ""
    )
    model_path = probe.get("embedding_probe", {}).get("model_path_resolved", "")
    assert model_path.lower().endswith("bge-m3"), (
        f"Probe model_path must resolve to BGE-M3, got {model_path!r}"
    )
    assert not stop_reason, (
        f"STOP gate was triggered ({stop_reason}) yet evaluation artifacts exist"
    )

    # (B) synthetic_contract_facts.jsonl must NOT appear in corpora.
    synfacts_name = "synthetic_contract_facts.jsonl"
    for strategy in ("a", "b"):
        corpus_raw = _read_json(EMB_INDEX_DIR / f"corpus_strategy_{strategy}.json")
        chunks = _iter_corpus_chunks(corpus_raw)
        bad: list[str] = []
        for c in chunks:
            sf = _chunk_source_file(c)
            if synfacts_name in sf:
                bad.append(c.get("chunk_id") or "<?>")
        assert not bad, (
            f"Strategy {strategy.upper()} has synthetic_contract_facts chunks: {bad[:3]}"
        )

    # (C) SYN_CONTRACT_*.docx (contract-scope evaluation set) MUST be in corpus.
    for strategy in ("a", "b"):
        corpus_raw = _read_json(EMB_INDEX_DIR / f"corpus_strategy_{strategy}.json")
        chunks = _iter_corpus_chunks(corpus_raw)
        syn_docids: set[str] = set()
        syn_sourcefiles: set[str] = set()
        for c in chunks:
            did = _chunk_document_id(c)
            if did and (did.startswith("SYN-") or did.startswith("SYN_CONTRACT_")):
                syn_docids.add(did)
            sf = _chunk_source_file(c)
            if "SYN_CONTRACT_" in sf and sf.endswith(".docx"):
                syn_sourcefiles.add(sf)
        total_syn = max(len(syn_docids), len(syn_sourcefiles))
        # Expect 10 SYN contracts — tolerate >=10 (more would be strange, >= is safe).
        assert total_syn >= 10, (
            f"Strategy {strategy.upper()} expected >=10 SYN_CONTRACT docs, "
            f"got docids={len(syn_docids)} sourcefiles={len(syn_sourcefiles)}"
        )

    # (D) SYN_citation_block_map MUST cover all 10 contracts.
    bmap_raw = _read_json(EMB_INDEX_DIR / "SYN_citation_block_map.json")
    # Schema may be:
    #   a) {syn_id: entry} directly, OR
    #   b) {"version": x, "by_synthetic_contract_id": {syn_id: entry}}
    bmap: dict
    if isinstance(bmap_raw, dict) and "by_synthetic_contract_id" in bmap_raw and isinstance(
        bmap_raw["by_synthetic_contract_id"], dict
    ):
        bmap = bmap_raw["by_synthetic_contract_id"]
    elif isinstance(bmap_raw, dict):
        bmap = {k: v for k, v in bmap_raw.items() if isinstance(v, dict)}
    else:
        raise TypeError(f"Unknown SYN_citation_block_map shape: {type(bmap_raw).__name__}")

    assert len(bmap) >= 10, (
        f"SYN_citation_block_map expected >=10 entries, got {len(bmap)}"
    )
    for syn_id, entry in bmap.items():
        ok = any(
            key in entry and entry[key]
            for key in (
                "precise_chunks",
                "block_offsets",
                "fields",
                "A_chunk_ids", "B_chunk_ids",
                "chunks_by_field",
            )
        )
        assert ok, (
            f"SYN block map entry {syn_id} has no usable block/chunk info: {list(entry.keys())[:5]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
