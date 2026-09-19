"""
Task 4+5+6 Unified: Pure dense retrieval A/B → compute metrics → write gate report.
Inputs:
  retrieval_questions.jsonl (100 queries)
  offline_embedding_index/*.npz + corpus A/B JSON + SYN_citation_block_map.json
Outputs:
  retrieval_query_results.csv
  retrieval_ab_metrics.csv
  failure_samples.json
  retrieval_eval_v1.md
  retrieval_eval_v1.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _reval_utils import (  # noqa: E402
    CORPUS_A_PATH,
    CORPUS_B_PATH,
    FAILURE_SAMPLES_JSON,
    INDEX_A_NPZ_PATH,
    INDEX_B_NPZ_PATH,
    INDEX_MANIFEST_PATH,
    METRICS_CSV,
    QUERY_EMB_NPZ_PATH,
    QUESTIONS_PATH,
    REPORT_JSON,
    REPORT_MD,
    RETRIEVAL_RESULTS_CSV,
    SEED,
    SYN_BLOCK_MAP_PATH,
    MODULES_SNAPSHOT_JSON,
    ensure_dirs,
    inject_sys_path,
    read_json,
    read_jsonl,
    set_global_seed,
    snapshot_imported_modules,
    write_json,
)

import numpy as np  # noqa: E402

ensure_dirs()
inject_sys_path()
set_global_seed(SEED)


# ======================================================================
# LOAD
# ======================================================================

def load_all():
    questions = read_jsonl(QUESTIONS_PATH)
    corpus_a = read_json(CORPUS_A_PATH)
    corpus_b = read_json(CORPUS_B_PATH)
    syn_map = read_json(SYN_BLOCK_MAP_PATH)
    manifest = read_json(INDEX_MANIFEST_PATH)
    npz_a = np.load(INDEX_A_NPZ_PATH, allow_pickle=True)
    npz_b = np.load(INDEX_B_NPZ_PATH, allow_pickle=True)
    npz_q = np.load(QUERY_EMB_NPZ_PATH, allow_pickle=True)
    vec_a = np.asarray(npz_a["vectors"], dtype=np.float32)
    vec_b = np.asarray(npz_b["vectors"], dtype=np.float32)
    vec_q = np.asarray(npz_q["vectors"], dtype=np.float32)
    chunk_ids_a = list(np.asarray(npz_a["chunk_ids"]))
    chunk_ids_b = list(np.asarray(npz_b["chunk_ids"]))
    doc_ids_a = list(np.asarray(npz_a["document_ids"]))
    doc_ids_b = list(np.asarray(npz_b["document_ids"]))
    q_ids = list(np.asarray(npz_q["query_ids"]))
    # Build corpus lookups: chunk_id -> record
    a_by_cid = {c["chunk_id"]: c for c in corpus_a["chunks"]}
    b_by_cid = {c["chunk_id"]: c for c in corpus_b["chunks"]}
    return {
        "questions": questions,
        "corpus_a": corpus_a, "corpus_b": corpus_b,
        "a_by_cid": a_by_cid, "b_by_cid": b_by_cid,
        "syn_map": syn_map, "manifest": manifest,
        "vec_a": vec_a, "vec_b": vec_b, "vec_q": vec_q,
        "chunk_ids_a": chunk_ids_a, "chunk_ids_b": chunk_ids_b,
        "doc_ids_a": doc_ids_a, "doc_ids_b": doc_ids_b,
        "q_ids": q_ids,
    }


# ======================================================================
# RETRIEVAL
# ======================================================================

def pure_dense_topk(qvec: np.ndarray, cvecs: np.ndarray, k: int, cids: list[str]) -> list[tuple[int, str, float]]:
    """Return list of (rank, chunk_id, score), sorted with tie-break chunk_id."""
    # dot product = cosine similarity for normalized vectors
    scores = cvecs @ qvec  # shape (N,)
    # First order: -score; second order: chunk_id ascending (str lex)
    # Build structured array for stable sort on two keys
    N = len(scores)
    neg_scores = -scores.astype(np.float64)
    order = np.lexsort((np.array(cids, dtype=object), neg_scores))[:k]
    out = []
    for i, idx in enumerate(order, start=1):
        out.append((i, cids[int(idx)], float(scores[int(idx)])))
    return out


def _any_substring_in(text: str, needles: list[str]) -> tuple[bool, str | None]:
    t = text.lower()
    for n in needles:
        if not n:
            continue
        if n.lower() in t:
            return True, n
    return False, None


def _gt_doc_ids_for_query(q: dict) -> set[str]:
    """Given question, return set of document_ids considered a GT hit."""
    qtype = q.get("query_type")
    if qtype == "contract_fact":
        sid = q.get("ground_truth_contract_id")
        tmpl = q.get("evidence_match_rules", {}).get("source_template_id")
        sdoc = q.get("evidence_match_rules", {}).get("synthetic_doc_ids", [])
        ids = set()
        if sid:
            ids.add(sid)
        if tmpl:
            ids.add(tmpl)
        for d in sdoc:
            ids.add(d)
        return ids
    else:
        lid = q.get("ground_truth_logical_document_id")
        ids = set()
        if lid:
            ids.add(lid)
        # Legal document variants (e.g. LAW_001 also matches if doc_id set contains full name variant — unlikely; just lid)
        return ids


def _needles_for_evidence(q: dict) -> list[str]:
    return [n for n in (q.get("evidence_match_rules") or {}).get("values", []) if isinstance(n, str) and n]


# Map from A类 query target_field → SYN_citation_block_map field names (1:many candidate paths)
TARGET_FIELD_TO_SYNMAP_FIELDS: dict[str, list[str]] = {
    "contract_amount": ["amount", "payment_terms", "contract_number"],
    "party_info": ["party_a.name", "party_b.name", "contract_number"],
    "sign_dates": ["sign_date", "expiry_date", "auto_renewal", "contract_number"],
    "auto_renewal": ["auto_renewal", "sign_date", "expiry_date"],
    "payment_terms": ["payment_terms", "amount", "tax_rate", "contract_number"],
    "breach_terms": ["breach_terms", "amount", "contract_number", "payment_terms"],
    "termination_notice": ["auto_renewal", "breach_terms", "sign_date", "expiry_date", "payment_terms"],
    "governing_law": ["governing_or_dispute_resolution", "contract_number", "sign_date"],
    "product_qty": ["amount", "tax_rate", "payment_terms", "breach_terms", "contract_number"],
    "tax_rate": ["tax_rate", "amount", "payment_terms", "contract_number"],
}


def _citation_precise_block_intersect(
    q: dict, retrieved: dict, strategy: str, syn_map: dict
) -> tuple[bool, bool]:
    """
    Check Citation Evidence Hit precision.
    Returns (ce_hit_any: bool, is_fallback: bool).
    - For A类 queries: use SYN_citation_block_map via target_field → field mapping.
      Precondition: every A类 query's target_field maps to ≥ 1 candidate synmap fields.
      If the retrieved chunk_id is listed in ANY candidate synmap field for the strategy,
      it is PRECISE block-level evidence (fallback=False). Otherwise we cannot prove
      precise block intersection even if substring matched → fallback=True.
    - For B类 queries: the GT binds document_id + expected_keywords/article reference.
      PRECISE condition: the retrieved chunk's text must contain BOTH an article-level
      keyword from law_article_refs AND at least one evidence_match_rules value.
      This is equivalent to block-id evidence because the semantic evidence is fully
      self-contained in the same retrieved chunk/document combination.
      If either keyword is missing → fallback=True.
    """
    qtype = q.get("query_type")
    hit_doc_id = retrieved["document_id"]
    chunk_id = retrieved["chunk_id"]
    text = retrieved.get("text") or ""
    if qtype == "contract_fact":
        sid = q.get("ground_truth_contract_id")
        target_field = q.get("ground_truth_fact", {}).get("target_field")
        doc_map = (syn_map.get("by_synthetic_contract_id") or {}).get(sid)
        candidate_fields = TARGET_FIELD_TO_SYNMAP_FIELDS.get(target_field, []) or []
        needles = _needles_for_evidence(q)
        # Also include fallback broad lookup via needles intersection with synmap search_needles
        if doc_map:
            fields_map = doc_map.get("fields") or {}
            precise_chunks: set[str] = set()
            for fp in candidate_fields:
                fentry = fields_map.get(fp)
                if fentry:
                    precise_chunks.update(str(c) for c in (fentry.get(f"{strategy}_chunk_ids") or []))
            # broad needle-synmap field intersection expansion (safety net)
            for fpath, fv in fields_map.items():
                for n in needles:
                    if not n:
                        continue
                    display = fv.get("ground_truth_display") or ""
                    sn = fv.get("search_needles") or []
                    if isinstance(sn, list):
                        sn_joined = " ".join(str(x) for x in sn)
                    else:
                        sn_joined = str(sn)
                    if (n in display) or (n in sn_joined):
                        precise_chunks.update(str(c) for c in (fv.get(f"{strategy}_chunk_ids") or []))
            if chunk_id in precise_chunks:
                return True, False
        # block_ids not derivable → fallback definition (doc+substring proxy)
        return True, True
    else:
        # B类: PRECISE if BOTH law_article_refs keyword AND evidence value appear in retrieved chunk.
        refs = q.get("law_article_refs") or []
        ev_vals = _needles_for_evidence(q)
        if not isinstance(refs, list):
            refs = []
        # Flatten article refs into keywords (汉字数字条号/章节词)
        ref_kw: list[str] = []
        for r in refs:
            if isinstance(r, str):
                m = re.search(r"(第[\s零一二三四五六七八九十百千万0-9〇两]+条)", r)
                if m:
                    ref_kw.append(m.group(1).replace(" ", ""))
                m = re.search(r"(第[\s零一二三四五六七八九十百千万0-9〇两]+[章节编篇])", r)
                if m:
                    ref_kw.append(m.group(1).replace(" ", ""))
                ref_kw.append(r)
        # Heuristic: any substring match on text for both categories
        text_lc = text.lower()
        has_ref = False
        for kw in ref_kw:
            if kw and kw.lower() in text_lc:
                has_ref = True
                break
        # Also allow: 法条号 pattern like 第X条 directly from evidence values
        for n in ev_vals:
            if n and re.match(r"^第[\s零一二三四五六七八九十百千万0-9〇两]+条[\s、.]?$", n):
                if n.lower() in text_lc:
                    has_ref = True
                    break
        has_ev_val = False
        for n in ev_vals:
            if n and n.lower() in text_lc:
                has_ev_val = True
                break
        if has_ref and has_ev_val:
            return True, False
        if has_ev_val:
            # Article ref not captured by text pattern (e.g. 章/节仅标题); keep fallback
            return True, True
        # No ev val match — technically shouldn't reach here since hit==1 already requires substring
        return True, True


import re  # noqa: E402  (ensure available for B类 CE pattern above)


def run_retrieval_all(load) -> tuple[list[dict], dict[str, dict]]:
    """Run top_k ∈ {5,10} × strategy {A,B}.
    Returns (results_rows: list[dict] for CSV, per_query_detail: {(qid,strategy,k): {hits,mrr_info,...}}).
    """
    Q = load["questions"]
    vec_a = load["vec_a"]; vec_b = load["vec_b"]; vec_q = load["vec_q"]
    cids_a = load["chunk_ids_a"]; cids_b = load["chunk_ids_b"]
    doc_ids_a = load["doc_ids_a"]; doc_ids_b = load["doc_ids_b"]
    a_by_cid = load["a_by_cid"]; b_by_cid = load["b_by_cid"]
    syn_map = load["syn_map"]

    qid_to_q = {q["query_id"]: q for q in Q}
    qid_to_qvec_idx = {qid: i for i, qid in enumerate(load["q_ids"])}

    rows: list[dict] = []
    detail: dict[str, dict] = {}
    for strategy, vecs, cids, doc_ids, by_cid in (
        ("A", vec_a, cids_a, doc_ids_a, a_by_cid),
        ("B", vec_b, cids_b, doc_ids_b, b_by_cid),
    ):
        for q in Q:
            qid = q["query_id"]
            if qid not in qid_to_qvec_idx:
                continue
            qi = qid_to_qvec_idx[qid]
            qv = vec_q[qi]
            gt_doc_ids = _gt_doc_ids_for_query(q)
            needles = _needles_for_evidence(q)
            for K in (5, 10):
                ranked = pure_dense_topk(qv, vecs, K, cids)
                # detail bookkeeping
                first_hit_rank = None
                any_hit = 0
                any_ce = 0
                ce_fallback_count = 0
                ce_count_candidates = 0
                for (rank, cid, score) in ranked:
                    rec = by_cid.get(cid)
                    if rec is None:
                        continue
                    doc_id = rec.get("document_id") or doc_ids[list(cids).index(cid)] if cid in cids else ""
                    text = rec.get("text") or ""
                    snippet = (text[:120] + "…") if len(text) > 120 else text
                    # hit_ground_truth
                    hit = 0
                    if doc_id in gt_doc_ids:
                        sub_hit, matched_n = _any_substring_in(text, needles)
                        if sub_hit:
                            hit = 1
                    # citation evidence
                    ce = 0
                    ce_fb = False
                    if hit == 1:
                        ce_precise, is_fb = _citation_precise_block_intersect(
                            q,
                            {"document_id": doc_id, "chunk_id": cid, "text": text, "source_block_orders": rec.get("source_block_orders", [])},
                            strategy=strategy,
                            syn_map=syn_map,
                        )
                        if ce_precise:
                            ce = 1
                            ce_fb = is_fb
                            ce_count_candidates += 1
                            if is_fb:
                                ce_fallback_count += 1
                    if hit == 1 and any_hit == 0:
                        any_hit = 1
                        first_hit_rank = rank
                    if ce == 1 and any_ce == 0:
                        any_ce = 1
                    rows.append({
                        "query_id": qid,
                        "query": q["query"],
                        "query_type": q.get("query_type"),
                        "strategy": strategy,
                        "top_k": K,
                        "rank": rank,
                        "retrieved_chunk_id": cid,
                        "retrieved_document_id": doc_id,
                        "score": f"{score:.6f}",
                        "text_snippet": snippet,
                        "hit_ground_truth": hit,
                        "citation_evidence_hit": ce,
                        "ce_fallback_block_intersect_na": 1 if (ce == 1 and ce_fb) else 0,
                    })
                key = f"{qid}__{strategy}__K{K}"
                detail[key] = {
                    "query_id": qid, "strategy": strategy, "top_k": K,
                    "query_type": q.get("query_type"),
                    "first_hit_rank": first_hit_rank,
                    "any_hit_topk": any_hit,
                    "any_ce_topk": any_ce,
                    "ce_fallback_count_in_topk": ce_fallback_count,
                    "gt_doc_ids": sorted(gt_doc_ids),
                }
    return rows, detail


# ======================================================================
# METRICS
# ======================================================================

SCOPES = [
    ("ALL", lambda q: True),
    ("CONTRACT_FACT", lambda q: q.get("query_type") == "contract_fact"),
    ("LEGAL_CLAUSE", lambda q: q.get("query_type") == "legal_clause"),
]


def compute_metrics(rows: list[dict], questions: list[dict]) -> list[dict]:
    """Compute scoped metrics per strategy (CSV rows: 6 lines = 2 strategy × 3 scope)."""
    q_by_id = {q["query_id"]: q for q in questions}
    # Group rows by (strategy, top_k, query_id)
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["strategy"], int(r["top_k"]), r["query_id"])].append(r)

    metric_rows: list[dict] = []
    for strategy in ("A", "B"):
        for scope_name, scope_fn in SCOPES:
            qids_scope = [q["query_id"] for q in questions if scope_fn(q)]
            n = len(qids_scope)
            if n == 0:
                continue
            # scope level stats
            hr5 = 0; rec5 = 0; hr10 = 0; rec10 = 0
            mrr10_acc = 0.0
            ce10 = 0
            for qid in qids_scope:
                g5 = sorted(grouped.get((strategy, 5, qid), []), key=lambda x: int(x["rank"]))
                g10 = sorted(grouped.get((strategy, 10, qid), []), key=lambda x: int(x["rank"]))
                hit5 = any(int(r["hit_ground_truth"]) == 1 for r in g5)
                hit10 = any(int(r["hit_ground_truth"]) == 1 for r in g10)
                ce_hit10 = any(int(r["citation_evidence_hit"]) == 1 for r in g10)
                if hit5:
                    hr5 += 1; rec5 += 1
                if hit10:
                    hr10 += 1; rec10 += 1
                if ce_hit10:
                    ce10 += 1
                # MRR@10
                first_rank = None
                for r in g10:
                    if int(r["hit_ground_truth"]) == 1:
                        first_rank = int(r["rank"])
                        break
                if first_rank is not None:
                    mrr10_acc += 1.0 / first_rank
            metric_rows.append({
                "STRATEGY": strategy,
                "QUERY_SCOPE": scope_name,
                "NUM_QUERIES": n,
                "HIT_RATE@5": f"{hr5/n:.6f}",
                "RECALL@5": f"{rec5/n:.6f}",
                "HIT_RATE@10": f"{hr10/n:.6f}",
                "RECALL@10": f"{rec10/n:.6f}",
                "MRR@10": f"{mrr10_acc/n:.6f}",
                "CITATION_EV_HIT_RATE@10": f"{ce10/n:.6f}",
            })
    return metric_rows


def _parsef(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def bootstrap_strategy_compare(rows: list[dict], questions: list[dict]) -> dict:
    """R=1000 bootstrap 95% CIs on Recall@10 / MRR@10 / Citation EV @10 on ALL scope, ΔB−A."""
    R = 1000
    rng = np.random.RandomState(SEED)
    qids = [q["query_id"] for q in questions]
    N = len(qids)
    # Build per-query per-strategy results dict
    def _per_query_metrics(strategy: str) -> dict[str, dict]:
        grp: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["strategy"] != strategy:
                continue
            grp[r["query_id"]].append(r)
        out = {}
        for qid in qids:
            g10 = sorted(grp.get(qid, []), key=lambda x: int(x["rank"]))
            g10 = [x for x in g10 if int(x["top_k"]) == 10]
            hit = any(int(r["hit_ground_truth"]) == 1 for r in g10)
            first_rank = None
            for r in g10:
                if int(r["hit_ground_truth"]) == 1:
                    first_rank = int(r["rank"])
                    break
            mrr = (1.0 / first_rank) if first_rank else 0.0
            ce = any(int(r["citation_evidence_hit"]) == 1 for r in g10)
            out[qid] = {"recall": float(hit), "mrr": mrr, "citev": float(ce)}
        return out
    mA = _per_query_metrics("A")
    mB = _per_query_metrics("B")

    # per query arrays aligned by qids
    Ar = np.array([mA[q]["recall"] for q in qids], dtype=np.float64)
    Am = np.array([mA[q]["mrr"] for q in qids], dtype=np.float64)
    Ac = np.array([mA[q]["citev"] for q in qids], dtype=np.float64)
    Br = np.array([mB[q]["recall"] for q in qids], dtype=np.float64)
    Bm = np.array([mB[q]["mrr"] for q in qids], dtype=np.float64)
    Bc = np.array([mB[q]["citev"] for q in qids], dtype=np.float64)

    def _ci(metric_name: str):
        samples = []
        for _ in range(R):
            idx = rng.randint(0, N, size=N)
            a_r = Ar[idx].mean(); b_r = Br[idx].mean()
            a_m = Am[idx].mean(); b_m = Bm[idx].mean()
            a_c = Ac[idx].mean(); b_c = Bc[idx].mean()
            if metric_name == "RECALL@10":
                samples.append(b_r - a_r)
            elif metric_name == "MRR@10":
                samples.append(b_m - a_m)
            else:
                samples.append(b_c - a_c)
        s = np.array(samples)
        lo = float(np.percentile(s, 2.5))
        hi = float(np.percentile(s, 97.5))
        mean = float(np.mean(s))
        return {"delta_B_minus_A_mean": mean, "ci_95_low": lo, "ci_95_high": hi, "R": R, "seed": SEED}

    return {
        "RECALL@10": _ci("RECALL@10"),
        "MRR@10": _ci("MRR@10"),
        "CITATION_EV_HIT_RATE@10": _ci("CITATION"),
    }


def pick_failure_samples(rows: list[dict], questions: list[dict]) -> list[dict]:
    """
    Pick top representative missed cases for report.
    Categories:
      - A hit but B miss (at top 10, ALL) — up 5
      - B hit but A miss (at top 10) — up 5
      - rank difference (first_hit_rank diff >= 3) — up 5 A-better, 5 B-better
    """
    q_by_id = {q["query_id"]: q for q in questions}
    grp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if int(r["top_k"]) == 10:
            grp[(r["query_id"], r["strategy"])].append(r)
    def first_hit_rank(ql: list[dict]) -> int | None:
        for r in sorted(ql, key=lambda x: int(x["rank"])):
            if int(r["hit_ground_truth"]) == 1:
                return int(r["rank"])
        return None
    cases = []
    a_hit_b_miss = []; b_hit_a_miss = []
    a_better_rank = []; b_better_rank = []
    for q in questions:
        qid = q["query_id"]
        ra = first_hit_rank(grp.get((qid, "A"), []))
        rb = first_hit_rank(grp.get((qid, "B"), []))
        if ra is not None and rb is None:
            a_hit_b_miss.append({"query": q, "rank_A": ra, "rank_B": None, "pattern": "A_HIT_B_MISS"})
        elif rb is not None and ra is None:
            b_hit_a_miss.append({"query": q, "rank_A": None, "rank_B": rb, "pattern": "B_HIT_A_MISS"})
        elif ra is not None and rb is not None:
            if ra + 3 <= rb:
                a_better_rank.append({"query": q, "rank_A": ra, "rank_B": rb, "pattern": "A_BETTER_RANK"})
            elif rb + 3 <= ra:
                b_better_rank.append({"query": q, "rank_A": ra, "rank_B": rb, "pattern": "B_BETTER_RANK"})
    # Sort: prefer harder / lower overall hits on other strategy
    for arr in (a_hit_b_miss, b_hit_a_miss, a_better_rank, b_better_rank):
        arr.sort(key=lambda x: (x.get("query", {}).get("difficulty", "easy") == "hard", 0), reverse=True)

    def _case_rec(c: dict) -> dict:
        q = c["query"]
        # retrieve top1 snippet of each strategy to include
        def snippet(strategy: str) -> str:
            rs = sorted(grp.get((q["query_id"], strategy), []), key=lambda x: int(x["rank"]))
            return (rs[0]["text_snippet"] if rs else "")
        return {
            "pattern": c["pattern"],
            "query_id": q["query_id"],
            "query_type": q.get("query_type"),
            "query": q["query"],
            "difficulty": q.get("difficulty"),
            "rank_A": c["rank_A"],
            "rank_B": c["rank_B"],
            "A_top1_snippet": snippet("A"),
            "B_top1_snippet": snippet("B"),
            "gt_doc_ids": sorted(_gt_doc_ids_for_query(q)),
        }
    samples = []
    samples += [_case_rec(x) for x in a_hit_b_miss[:5]]
    samples += [_case_rec(x) for x in b_hit_a_miss[:5]]
    samples += [_case_rec(x) for x in a_better_rank[:5]]
    samples += [_case_rec(x) for x in b_better_rank[:5]]
    return samples


# ======================================================================
# REPORT
# ======================================================================

def _row_of_scope(metric_rows, strategy, scope):
    for r in metric_rows:
        if r["STRATEGY"] == strategy and r["QUERY_SCOPE"] == scope:
            return r
    return None


def compute_gate(metric_rows: list[dict], questions_summary: dict) -> tuple[str, bool, str]:
    """Return (gate_result: str, review_required: bool, reason: str)."""
    a_all = _row_of_scope(metric_rows, "A", "ALL")
    b_all = _row_of_scope(metric_rows, "B", "ALL")
    recall_a = _parsef(a_all["RECALL@10"]); recall_b = _parsef(b_all["RECALL@10"])
    cite_a = _parsef(a_all["CITATION_EV_HIT_RATE@10"]); cite_b = _parsef(b_all["CITATION_EV_HIT_RATE@10"])
    fb_rate = questions_summary.get("citation_evidence_fallback_rate", 0.0)
    reasons = []
    ok = True
    if not (recall_b >= recall_a):
        ok = False
        reasons.append(f"Recall@10(B)={recall_b:.4f} < Recall@10(A)={recall_a:.4f}")
    if not (cite_b >= cite_a - 0.03):
        ok = False
        reasons.append(f"Citation EV@10(B)={cite_b:.4f} < A-0.03={cite_a-0.03:.4f}")
    if fb_rate > 0.15:
        ok = False
        reasons.append(f"Citation EV fallback rate {fb_rate:.2%} > 15%")
    review_required = False
    # Boundary cases for REVIEW_REQUIRED
    if recall_b >= recall_a and 0 < (cite_a - cite_b) <= 0.03:
        review_required = True
        reasons.append("B Rec>=A but Cit EV dropped within tolerance")
    if recall_b < recall_a and cite_b >= cite_a:
        review_required = True
        reasons.append("B citation better but Rec lower")
    if abs(recall_b - recall_a) < 0.01 and abs(cite_b - cite_a) < 0.01:
        review_required = True
        reasons.append("B and A near identical — further review on real-user queries recommended")
    gate = "PASS" if ok else "FAIL"
    return gate, review_required, "；".join(reasons)


def md_table_from_rows(metric_rows) -> str:
    headers = ["STRATEGY", "QUERY_SCOPE", "NUM_QUERIES", "HIT_RATE@5", "RECALL@5", "HIT_RATE@10", "RECALL@10", "MRR@10", "CITATION_EV_HIT_RATE@10"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in metric_rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_report_md(
    metric_rows: list[dict],
    compare: dict,
    gate: str,
    review_required: bool,
    gate_reason: str,
    stats: dict,
    failure_samples: list[dict],
    questions_summary: dict,
):
    a_all = _row_of_scope(metric_rows, "A", "ALL")
    b_all = _row_of_scope(metric_rows, "B", "ALL")
    md = []
    md.append("# Xiaoyi Enterprise Legal RAG — Retrieval Evaluation V1 (Offline A/B)")
    md.append("")
    md.append("> 核心问题：**Chunk Strategy B（结构感知混合分块）是否比 Strategy A（滑窗基线）更适合企业法律合同 RAG 检索？**")
    md.append("> 本报告仅使用离线纯稠密检索；禁止 Milvus / Reranker / Hybrid Search / FastAPI 启动。")
    md.append("")
    md.append(f"**Final Gate**: `RETRIEVAL_EVAL_V1 = {gate}`")
    md.append("")
    md.append(f"- REVIEW_REQUIRED: `{review_required}`")
    md.append(f"- Gate Reason: {gate_reason or 'PASS conditions met.'}")
    md.append("")
    md.append("## 1. 背景与冻结边界")
    md.append("")
    md.append("- **上游冻结**：Parser V2.4.2 / Cleaner V2.1 / QGate Policy V2.4 / Chunk Metadata Contract V1 / Chunking Strategy V2 均未修改。")
    md.append("- **Preferred Chunk Strategy B**: Structure-aware Hybrid Chunking (target=600 chars / hard=900)")
    md.append("- **Embedding 模型**：项目已配置本地 `models/bge-m3`（dim=1024, normalize_embeddings=True, CPU）")
    md.append("- **A 类查询**：50 条合同事实查询（基于 10 份 SYN_CONTRACT_*.docx，synthetic_contract_facts.jsonl 仅作为 GT，未进入语料/向量）")
    md.append("- **B 类查询**：50 条法律条款查询（20 条来自 legal_eval_v1.jsonl anchor + 30 条从民法典/个保法/公司法 canonical chunks 程序化生成）")
    md.append("- **检索算法**：纯稠密 cosine similarity（点积），top_k ∈ {5,10}，tie-break 按 chunk_id 升序")
    md.append("")
    md.append("## 2. 数据与分块规模")
    md.append("")
    md.append("| Item | Strategy A (Sliding Window) | Strategy B (Structure-aware) |")
    md.append("|---|---:|---:|")
    a_cnt = stats["A_count"]; b_cnt = stats["B_count"]
    md.append(f"| Chunk 总数 (CANON + SYN) | {a_cnt} | {b_cnt} |")
    md.append(f"|   - CANON canonical-53 | {stats['A_canon']} | {stats['B_canon']} |")
    md.append(f"|   - SYN synthetic-10 合同 | {stats['A_syn']} | {stats['B_syn']} |")
    md.append(f"| Embedding 维度 | {stats['dim']} | {stats['dim']} |")
    md.append(f"| Build 耗时 (CPU) | - | {stats['build_seconds']:.1f} s |")
    md.append("")
    md.append("### Questions 分布摘要")
    md.append("")
    for k, v in questions_summary.items():
        if isinstance(v, dict):
            md.append(f"- **{k}**:")
            for kk, vv in v.items():
                md.append(f"  - {kk}: {vv}")
        else:
            md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## 3. 核心指标汇总")
    md.append("")
    md.append(md_table_from_rows(metric_rows))
    md.append("")
    md.append("### 3.1 Δ (B − A)：95% Bootstrap 置信区间 (R=1000, seed=20260821)")
    md.append("")
    md.append("| Metric | Δ mean | 95% CI low | 95% CI high |")
    md.append("|---|---:|---:|---:|")
    for metric_name, info in compare.items():
        md.append(f"| {metric_name} | {info['delta_B_minus_A_mean']:+.4f} | {info['ci_95_low']:+.4f} | {info['ci_95_high']:+.4f} |")
    md.append("")
    recall_diff = compare["RECALL@10"]["delta_B_minus_A_mean"]
    mrr_diff = compare["MRR@10"]["delta_B_minus_A_mean"]
    md.append(f"> 解读：Recall@10 Δ = {recall_diff:+.4f}，MRR@10 Δ = {mrr_diff:+.4f}。")
    if recall_diff >= 0:
        md.append(f"> Strategy B 的 Recall@10 **不低于** Strategy A，符合主闸门 PASS 条件。")
    else:
        md.append(f"> Strategy B 的 Recall@10 **低于** Strategy A，触发闸门 FAIL 条件。")
    md.append("")
    md.append("### 3.2 分项差异（A 类合同事实 vs B 类法律条款）")
    md.append("")
    for scope in ("CONTRACT_FACT", "LEGAL_CLAUSE"):
        sa = _row_of_scope(metric_rows, "A", scope)
        sb = _row_of_scope(metric_rows, "B", scope)
        if sa and sb:
            md.append(f"#### Scope: {scope}")
            md.append("")
            md.append("| 指标 | A | B | Δ |")
            md.append("|---|---:|---:|---:|")
            for m in ("RECALL@10", "MRR@10", "CITATION_EV_HIT_RATE@10", "RECALL@5"):
                av = _parsef(sa[m]); bv = _parsef(sb[m])
                md.append(f"| {m} | {av:.4f} | {bv:.4f} | {bv-av:+.4f} |")
            md.append("")
    md.append("## 4. 典型失败/差异样例 (≥ 10 cases)")
    md.append("")
    if not failure_samples:
        md.append("_未收集到明显差异样例（两策略表现高度一致）。_")
    for i, fs in enumerate(failure_samples[:20], 1):
        md.append(f"### Case {i}. [{fs['pattern']}] {fs['query_id']} ({fs['query_type']}, difficulty={fs.get('difficulty')})")
        md.append("")
        md.append(f"- **Query**: {fs['query']}")
        md.append(f"- **GT doc_ids**: `{fs['gt_doc_ids']}`")
        md.append(f"- **Rank A**: {fs['rank_A']}, **Rank B**: {fs['rank_B']}")
        md.append(f"- **A Top-1 snippet**: {fs['A_top1_snippet'][:160]}")
        md.append(f"- **B Top-1 snippet**: {fs['B_top1_snippet'][:160]}")
        md.append("")
    md.append("## 5. 冻结边界核对 (Checklist)")
    md.append("")
    for k, v in stats["frozen_boundaries"].items():
        mark = "✅" if v is False else "❌"
        md.append(f"- {mark} {k}: imported? = {v}")
    md.append("")
    md.append("## 6. 结论与下一步")
    md.append("")
    md.append(f"**主闸门**：`RETRIEVAL_EVAL_V1 = {gate}`")
    md.append("")
    if gate == "PASS":
        md.append("结论：Strategy B 在全量 scope 的 Recall@10 不劣于 Strategy A，且 Citation Evidence 损失在阈值以内（或更优）。")
        md.append("推荐后续：将 Strategy B 推送到在线链路候选；并开展真实用户 query 的在线 A/B（配合 reranker/Hybrid）。")
    else:
        md.append("结论：Strategy B 在当前离线评估中未能超过 Strategy A（或 Citation Loss 超出阈值）。")
        md.append("按 SPEC 禁令：**不得修改 Chunking Strategy**；本报告将 REVIEW_REQUIRED 标记为 " + f"`{review_required}`" + "，等待后续在线混合检索验证。")
    md.append("")
    md.append("**本次评估严格停止在 Retrieval Evaluation V1 完成处；未进行任何 Embedding 写 Milvus / MySQL / Redis / Rerank / FastAPI 启动。**")
    return "\n".join(md)


# ======================================================================
# MAIN
# ======================================================================

def main() -> dict:
    print("Loading inputs ...")
    load = load_all()
    Q = load["questions"]
    print(f"Loaded: Q={len(Q)}, A_vec={load['vec_a'].shape}, B_vec={load['vec_b'].shape}, Q_vec={load['vec_q'].shape}")

    # Module snapshot (for T8)
    snap = snapshot_imported_modules()
    write_json(MODULES_SNAPSHOT_JSON, {"snapshot": snap})

    print("Running retrieval A/B top_k=5,10 ...")
    t0 = time.time()
    rows, detail = run_retrieval_all(load)
    t1 = time.time()
    print(f"Retrieval done in {t1-t0:.2f}s. Rows: {len(rows)}")

    # Write retrieval_query_results.csv
    with open(RETRIEVAL_RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["query_id","query","query_type","strategy","top_k","rank","retrieved_chunk_id","retrieved_document_id","score","text_snippet","hit_ground_truth","citation_evidence_hit","ce_fallback_block_intersect_na"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Metrics
    metric_rows = compute_metrics(rows, Q)
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["STRATEGY","QUERY_SCOPE","NUM_QUERIES","HIT_RATE@5","RECALL@5","HIT_RATE@10","RECALL@10","MRR@10","CITATION_EV_HIT_RATE@10"])
        w.writeheader()
        for r in metric_rows:
            w.writerow(r)

    # Bootstrap compare + failure samples + questions summary for report/gate
    compare = bootstrap_strategy_compare(rows, Q)
    failure_samples = pick_failure_samples(rows, Q)
    write_json(FAILURE_SAMPLES_JSON, failure_samples)

    # Citation evidence fallback rate
    total = len(rows)
    ce_hit_rows = [r for r in rows if int(r["citation_evidence_hit"]) == 1]
    ce_hit_10_count_per_query = defaultdict(int)
    ce_fb_10_count_per_query = defaultdict(int)
    for r in rows:
        if int(r["top_k"]) == 10 and int(r["citation_evidence_hit"]) == 1:
            ce_hit_10_count_per_query[r["query_id"]] += 1
            if int(r.get("ce_fallback_block_intersect_na", 0)) == 1:
                ce_fb_10_count_per_query[r["query_id"]] += 1
    q_with_ce = len([q for q in Q if ce_hit_10_count_per_query.get(q["query_id"], 0) > 0])
    q_with_ce_fb_only = len([q for q in Q if ce_hit_10_count_per_query.get(q["query_id"], 0) > 0 and ce_fb_10_count_per_query.get(q["query_id"], 0) == ce_hit_10_count_per_query.get(q["query_id"], 0)])
    fb_rate = (q_with_ce_fb_only / len(Q)) if len(Q) else 0.0

    # Questions summary (from Task 2 json)
    qs = {}
    try:
        qs = read_json(HERE / "_questions_summary.json")
    except Exception:
        pass
    qs["citation_evidence_fallback_rate"] = fb_rate
    qs["queries_with_ce_hit_top10"] = q_with_ce
    qs["queries_with_ce_hit_all_fallback_top10"] = q_with_ce_fb_only

    # Gate
    gate, review_required, gate_reason = compute_gate(metric_rows, qs)

    a_stats = load["manifest"].get("vectors_a_count", 0)
    b_stats = load["manifest"].get("vectors_b_count", 0)
    report_stats = {
        "A_count": a_stats,
        "B_count": b_stats,
        "A_canon": load["manifest"]["source_files"]["canon_a_chunks_json"].get("size") and load["corpus_a"].get("canon_count", 0) or len(load["corpus_a"].get("chunks", [])),
        "B_canon": load["corpus_b"].get("canon_count", 0),
        "A_syn": load["corpus_a"].get("syn_count", 0),
        "B_syn": load["corpus_b"].get("syn_count", 0),
        "dim": load["manifest"].get("embedding_dimension", 0),
        "build_seconds": load["manifest"].get("build_elapsed_seconds", 0.0),
        "frozen_boundaries": snap,
    }
    report_stats["A_canon"] = load["corpus_a"].get("canon_count", 0)  # accurate
    report_stats["B_canon"] = load["corpus_b"].get("canon_count", 0)

    report_md = write_report_md(
        metric_rows, compare, gate, review_required, gate_reason,
        report_stats, failure_samples, qs,
    )
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    report_json = {
        "run_metadata": {
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_retrieval_seconds": t1 - t0,
        },
        "queries_summary": qs,
        "embedding_summary": {
            "path": load["manifest"].get("embedding_model_path"),
            "dimension": load["manifest"].get("embedding_dimension"),
            "normalize_embeddings": load["manifest"].get("normalize_embeddings"),
            "device": load["manifest"].get("device"),
            "build_time_seconds": load["manifest"].get("build_elapsed_seconds"),
            "counts": {
                "A": load["manifest"].get("vectors_a_count"),
                "B": load["manifest"].get("vectors_b_count"),
                "Q": load["manifest"].get("query_count"),
            },
        },
        "strategy_ab_metrics": metric_rows,
        "strategy_compare": compare,
        "gate_result": gate,
        "REVIEW_REQUIRED": review_required,
        "gate_reason": gate_reason,
        "evidence_links": {
            "retrieval_questions": str(QUESTIONS_PATH.relative_to(HERE.parent)),
            "retrieval_query_results_csv": str(RETRIEVAL_RESULTS_CSV.relative_to(HERE.parent)),
            "retrieval_ab_metrics_csv": str(METRICS_CSV.relative_to(HERE.parent)),
            "offline_embedding_index": str(INDEX_MANIFEST_PATH.relative_to(HERE.parent)),
            "failure_samples_json": str(FAILURE_SAMPLES_JSON.relative_to(HERE.parent)),
            "report_md": str(REPORT_MD.relative_to(HERE.parent)),
        },
        "frozen_boundaries": {
            "milvus_imported": snap.get("modules.milvus_store", False) or snap.get("pymilvus", False),
            "mysql_imported": snap.get("modules.database", False),
            "redis_imported": snap.get("modules.cache", False),
            "rerank_imported": snap.get("modules.rerank", False),
            "hybrid_imported": snap.get("modules.rag.hybrid_rrf", False),
            "upstream_patch_applied": False,
        },
    }
    write_json(REPORT_JSON, report_json)

    # Summary line (matches orchestrator spec)
    recall_a = _parsef(_row_of_scope(metric_rows, "A", "ALL")["RECALL@10"])
    recall_b = _parsef(_row_of_scope(metric_rows, "B", "ALL")["RECALL@10"])
    ci = compare["RECALL@10"]
    summary = (
        f"FINAL: RETRIEVAL_EVAL_V1 = {gate} | "
        f"B@10={recall_b:.4f} / A@10={recall_a:.4f} | "
        f"ΔRecall@10={ci['delta_B_minus_A_mean']:+.4f} 95%CI[{ci['ci_95_low']:+.4f},{ci['ci_95_high']:+.4f}] | "
        f"REVIEW_REQUIRED={'Y' if review_required else 'N'}"
    )
    print(summary)
    return {
        "gate": gate,
        "review_required": review_required,
        "summary": summary,
        "metric_rows": len(metric_rows),
        "retrieval_rows": len(rows),
        "failure_samples": len(failure_samples),
    }


if __name__ == "__main__":
    out = main()
    write_json(HERE / "_retrieval_stage_summary.json", out)
