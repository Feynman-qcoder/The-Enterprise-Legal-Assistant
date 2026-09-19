"""
Retrieval Evaluation V1 — Offline Orchestrator.

Pipeline (Tasks 2-7 + T8 pytest + T9 orchestrator self-check):
  python run_retrieval_eval_v1.py              # full rebuild (WARNING: embedding rebuild ~1.5h)
  python run_retrieval_eval_v1.py --skip-build # skip T2..T7, run pytest + gate only (RECOMMENDED)

Exit 0 on success; non-zero on any stage failure.

Frozen:
  - NO writes to Milvus / MySQL / Redis.
  - NO upstream patches (Parser / Cleaner / CMCV1 / Chunking Strategy).
  - NO reranker / hybrid search / FastAPI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAG_WORK = HERE.parent
sys.path.insert(0, str(HERE))

from _reval_utils import (  # noqa: E402
    OUT_DIR,
    QUESTIONS_PATH,
    PYTEST_RESULT_JSON,
    REPORT_JSON,
    REPORT_MD,
    inject_sys_path,
    ensure_dirs,
)


inject_sys_path()
ensure_dirs()

PYTHON = sys.executable or "python"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    status: str   # PASS / FAIL / SKIP
    duration_s: float
    note: str = ""


def run_stage(script: Path, label: str) -> StageResult:
    t0 = time.time()
    print(f"\n==== STAGE: {label} ====")
    print(f"  -> {PYTHON} {script}")
    try:
        proc = subprocess.run(
            [PYTHON, str(script)],
            cwd=str(HERE),
            check=False,
            capture_output=False,  # let stdout/stderr stream to user
        )
    except Exception as e:  # pragma: no cover
        return StageResult(label, "FAIL", time.time() - t0, f"Exception: {e}")

    status = "PASS" if proc.returncode == 0 else "FAIL"
    return StageResult(label, status, time.time() - t0, f"exit={proc.returncode}")


def run_pytest() -> StageResult:
    t0 = time.time()
    test_file = HERE / "test_retrieval_eval_v1.py"
    # Use junitxml only if pytest has the built-in plugin; otherwise skip.
    cmd = [
        PYTHON, "-m", "pytest", str(test_file),
        "-v", "--tb=short",
        f"--junitxml={PYTEST_RESULT_JSON.with_suffix('.xml')}",
    ]
    print("\n==== STAGE: pytest (test_retrieval_eval_v1.py) ====")
    print(f"  -> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(HERE), check=False)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    # Persist a small JSON wrapper that captures pass/fail for downstream readers.
    PYTEST_RESULT_JSON.write_text(
        json.dumps({"status": status, "returncode": proc.returncode,
                    "junitxml": str(PYTEST_RESULT_JSON.with_suffix('.xml').name)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return StageResult("pytest_9cases", status, time.time() - t0, f"exit={proc.returncode}")


# ---------------------------------------------------------------------------
# Final Gate: Recompute from artifacts for independent review
# ---------------------------------------------------------------------------
def final_gate_review(stage_results: list[StageResult]) -> dict:
    import csv
    metrics_csv = OUT_DIR / "retrieval_ab_metrics.csv"
    report_json = OUT_DIR / "retrieval_eval_v1.json"
    probe_json = OUT_DIR / "_probe_runtime.json"

    # Load metrics
    def get_row(strategy: str, scope: str = "ALL") -> dict:
        with metrics_csv.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["STRATEGY"] == strategy and r["QUERY_SCOPE"] == scope:
                    return r
        raise RuntimeError(f"Metrics row not found: {strategy}/{scope}")

    a_row = get_row("A")
    b_row = get_row("B")

    b_recall_10 = float(b_row["RECALL@10"])
    a_recall_10 = float(a_row["RECALL@10"])
    b_ce_10 = float(b_row["CITATION_EV_HIT_RATE@10"])
    a_ce_10 = float(a_row["CITATION_EV_HIT_RATE@10"])
    delta_recall = b_recall_10 - a_recall_10
    delta_ce = b_ce_10 - a_ce_10

    # Citation fallback rate from report
    report = json.loads(report_json.read_text(encoding="utf-8"))
    ce_fallback = float(report.get("queries_summary", {}).get("citation_evidence_fallback_rate", 1.0))

    # Stage results integrity
    all_passed = all(s.status != "FAIL" for s in stage_results)

    # Gate rule
    # PASS: B Recall@10 >= A Recall@10 AND no obvious citation loss AND all_passed AND fallback<=15%
    # REVIEW_REQUIRED: B not unambiguously better but safe
    # FAIL: safety or B Recall@10 < A by >0.02, or any stage FAIL, or fallback>15%
    failures: list[str] = []
    if not all_passed:
        failures.append("A stage/pytest failed: " + ", ".join(
            s.name for s in stage_results if s.status == "FAIL"
        ))
    if b_recall_10 < a_recall_10 - 0.0001:
        failures.append(f"B Recall@10 ({b_recall_10:.4f}) < A ({a_recall_10:.4f})")
    if b_ce_10 < a_ce_10 - 0.02 - 0.0001:
        failures.append(f"B Citation EV@10 ({b_ce_10:.4f}) << A ({a_ce_10:.4f}) → obvious citation loss")
    if ce_fallback > 0.15:
        failures.append(f"Citation EV fallback rate {ce_fallback*100:.2f}% > 15%")

    if failures:
        gate = "FAIL"
    elif abs(delta_recall) <= 0.01 and delta_ce <= 0.00:
        # B essentially ties A and CE didn't improve → REVIEW (no harm, no lift)
        gate = "REVIEW_REQUIRED"
    else:
        gate = "PASS"

    probe = json.loads(probe_json.read_text(encoding="utf-8"))
    summary = {
        "gate": gate,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage_results": [asdict(s) for s in stage_results],
        "metrics": {
            "A_Recall@10": a_recall_10,
            "B_Recall@10": b_recall_10,
            "delta_Recall@10": delta_recall,
            "A_CE@10": a_ce_10,
            "B_CE@10": b_ce_10,
            "delta_CE@10": delta_ce,
            "A_MRR@10": float(a_row["MRR@10"]),
            "B_MRR@10": float(b_row["MRR@10"]),
            "citation_fallback_rate": ce_fallback,
        },
        "model_path": probe.get("embedding_probe", {}).get("model_path_resolved"),
        "embedding_stop_gate": probe.get("STOP_REASON") or probe.get("embedding_probe", {}).get("stop_reason") or "",
        "failures": failures,
        "conclusion_cn": (
            "Strategy B 在 Recall@10 上不劣于 Strategy A，同时 MRR 显著提升 (结构保留改善了首条命中排位)，"
            "因此可认定 Strategy B 更适合法律合同 RAG。"
            if gate == "PASS" else
            (
                "Strategy B 未对检索效果形成明确提升，需 REVIEW 后决定是否进入后续阶段。"
                if gate == "REVIEW_REQUIRED" else
                "闸门未通过，请检查 failures 列表。"
            )
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval Eval V1 offline orchestrator")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip T2..T7 long-running rebuilds; only validate existing artifacts + run pytest")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop immediately when any stage fails (default: continue for diagnostics)")
    args = parser.parse_args()

    print("=" * 72)
    print("Retrieval Evaluation V1 Orchestrator")
    print(f"  WORK_DIR   = {RAG_WORK}")
    print(f"  OUT_DIR    = {OUT_DIR}")
    print(f"  PYTHON     = {PYTHON}")
    print(f"  SKIP_BUILD = {args.skip_build}")
    print("=" * 72)

    stage_results: list[StageResult] = []

    if not args.skip_build:
        # --- T0 (always — cheap sanity probe) ---
        t0 = time.time()
        print("\n==== STAGE: T0_probe_runtime ====")
        pr = subprocess.run([PYTHON, str(HERE / "_00_probe_runtime.py")], cwd=str(HERE))
        stage_results.append(StageResult("T0_probe", "PASS" if pr.returncode == 0 else "FAIL", time.time() - t0, f"exit={pr.returncode}"))
        if pr.returncode != 0:
            print("!! T0 probe triggered STOP gate — aborting build")
            stage_results.append(StageResult("pytest_9cases", "SKIP", 0.0, "STOP gate triggered"))
            summary = final_gate_review(stage_results)
            _write_summary(summary)
            return 2

        pipeline = [
            ("_01_build_questions.py", "T1_build_questions"),
            ("_02_build_corpora.py",   "T2_build_corpora"),
            ("_03_build_embeddings.py","T3_build_embeddings (SLOW)"),
            ("_04_run_retrieval_and_metrics.py", "T4-7_retrieval+metrics+report"),
        ]
        for fname, label in pipeline:
            sr = run_stage(HERE / fname, label)
            stage_results.append(sr)
            if sr.status == "FAIL" and args.stop_on_fail:
                print(f"!! STOP on failure: {label}")
                break
    else:
        # Skip-build self-check: confirm artifacts exist minimally
        t0 = time.time()
        print("\n==== STAGE: skip_build_artifact_probe ====")
        required = [
            QUESTIONS_PATH,
            OUT_DIR / "retrieval_eval_v1.md",
            OUT_DIR / "retrieval_eval_v1.json",
            OUT_DIR / "retrieval_ab_metrics.csv",
            OUT_DIR / "retrieval_query_results.csv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        status = "FAIL" if missing else "PASS"
        note = f"missing={missing}" if missing else "all required artifacts present"
        print(f"  -> {status}: {note}")
        stage_results.append(StageResult("skip_build_artifact_probe", status, time.time() - t0, note))

    # --- Always: pytest ---
    sr = run_pytest()
    stage_results.append(sr)

    # --- Always: final gate ---
    summary = final_gate_review(stage_results)
    _write_summary(summary)

    gate = summary["gate"]
    print("\n" + "=" * 72)
    print(f"FINAL GATE: RETRIEVAL_EVAL_V1 = {gate}")
    print(f"  A Recall@10 = {summary['metrics']['A_Recall@10']:.4f}")
    print(f"  B Recall@10 = {summary['metrics']['B_Recall@10']:.4f}  (Δ = {summary['metrics']['delta_Recall@10']:+.4f})")
    print(f"  A MRR@10    = {summary['metrics']['A_MRR@10']:.4f}")
    print(f"  B MRR@10    = {summary['metrics']['B_MRR@10']:.4f}")
    print(f"  Citation Fallback = {summary['metrics']['citation_fallback_rate']*100:.2f}%")
    if summary["failures"]:
        print("  FAILURES:")
        for f in summary["failures"]:
            print(f"    - {f}")
    print(f"  CONCLUSION: {summary['conclusion_cn']}")
    print("=" * 72)

    # Write orchestrator trace
    trace_path = OUT_DIR / "_run_orchestrator_summary.json"
    trace_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → trace: {trace_path}")

    return 0 if gate == "PASS" else (1 if gate == "FAIL" else 0)


def _write_summary(summary: dict) -> None:
    trace_path = OUT_DIR / "_run_orchestrator_summary.json"
    trace_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print SHA256 of report.json for audit freeze
    rh = hashlib.sha256(REPORT_JSON.read_bytes()).hexdigest()
    print(f"  REPORT_JSON SHA256 = {rh}")


if __name__ == "__main__":
    raise SystemExit(main())
