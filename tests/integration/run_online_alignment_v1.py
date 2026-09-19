# -*- coding: utf-8 -*-
"""
ONLINE RAG ALIGNMENT V1 — Regression Runner (A-E)

A. Canonical retrieval 仍 PASS（dense + source_file 白名单 expr）
B. Legacy-only query 不会把 legacy vectors 带入正式 Legal Retrieval
C. LAW_003 query 的最终 Context 含真实 logical_document_id / version / effective_date
D. Citation Query 只能依据实际 context metadata/正文回答（正向引用 + 知识不足拒答）
E. 原 LIVE RAG/SSE：HTTP 200 + [DONE] + 无 error（抽样原测试集 legal_eval_v1.jsonl）

只读约束：不改 Schema / 不删 Legacy / 不重新入库；除正常聊天历史与答案缓存外无写入。
用法（在仓库根目录运行）：
  python tests/integration/run_online_alignment_v1.py        # 全量：S0/S1/S2 + E（服务可达时）
  python tests/integration/run_online_alignment_v1.py sse    # 仅 E（先启动 uvicorn）
结果落盘：D:/xiaoyi/data_source/_meta/online_rag_alignment_v1_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # .env 与 models/ 均相对仓库根

RESULTS_PATH = r"D:\xiaoyi\data_source\_meta\online_rag_alignment_v1_results.json"
RESULTS_FALLBACK = r"f:\DataBase\trae_work\RAG\align_v1_stage\results\online_rag_alignment_v1_results.json"
EVAL_SET_PATH = r"D:\xiaoyi\data_source\_meta\legal_eval_v1.jsonl"
BASE_URL = os.environ.get("ALIGN_BASE_URL", "http://127.0.0.1:8000")

RESULTS: dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}


def check(name: str, ok: bool, detail: dict | str | None = None) -> bool:
    RESULTS["checks"][name] = {"pass": bool(ok), "detail": detail}
    try:
        brief = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)[:600]
    except Exception:  # noqa: BLE001
        brief = str(detail)[:600]
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {brief}")
    return ok


def _quote(files) -> str:
    return ",".join(json.dumps(f, ensure_ascii=False) for f in sorted(files))


async def s0_scope_sanity() -> tuple[object, dict]:
    from modules.rag.corpus_scope import get_canonical_scope

    scope = get_canonical_scope()
    n = len(scope.source_files)
    lid_to_file = {m.get("logical_document_id"): sf for sf, m in scope.metadata.items()}
    law003_sf = lid_to_file.get("LAW_003")
    m3 = scope.metadata.get(law003_sf, {}) if law003_sf else {}
    header3 = scope.format_header(law003_sf) or ""
    ok = (
        n == 53
        and bool(law003_sf)
        and m3.get("version") == "2025_amended"
        and m3.get("effective_date") == "2026-01-01"
        and "[Source Metadata]" in header3
        and "logical_document_id: LAW_003" in header3
        and "[Evidence]" not in header3  # header 本身不含 Evidence 段（Evidence 在 pipeline 拼接）
    )
    check(
        "S0_manifest_scope",
        ok,
        {
            "canonical_files": n,
            "law003_source_file": law003_sf,
            "law003_version": m3.get("version"),
            "law003_effective_date": m3.get("effective_date"),
            "expr_length": len(scope.build_source_file_expr()),
        },
    )
    return scope, {"lid_to_file": lid_to_file}


async def s1_dense_isolation(scope) -> dict:
    from pymilvus import Collection

    from modules.embeddings.local_embedding import LocalEmbeddingService
    from modules.milvus_store.client import ensure_milvus
    from modules.milvus_store.collections import COLLECTION_LEGAL_CHILD
    from modules.rag.pipeline import _parse_milvus_hits

    ensure_milvus()
    col = Collection(COLLECTION_LEGAL_CHILD)
    col.load()
    whitelist = scope.source_files
    expr_in = scope.build_source_file_expr()
    expr_not_in = f"source_file not in [{_quote(whitelist)}]"

    def _query(expr: str, fields: list[str], limit: int):
        return [
            e.to_dict() if hasattr(e, "to_dict") else dict(e)
            for e in col.query(expr=expr, output_fields=fields, limit=limit)
        ]

    def _search(vec: list[float], limit: int, expr: str | None):
        return col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=limit,
            expr=expr or "",
            output_fields=["source_file", "parent_id"],
        )

    emb = LocalEmbeddingService()

    # ---- Legacy 快照（证明 Legacy 仍在库中、未被删除）----
    legacy_rows = await asyncio.to_thread(_query, expr_not_in, ["source_file", "text"], 64)
    legacy_files = sorted({r.get("source_file", "") for r in legacy_rows})
    print(f"  legacy children(sampled)={len(legacy_rows)} files={legacy_files}")
    check("B0_legacy_still_present", len(legacy_rows) > 0, {"sampled_children": len(legacy_rows), "legacy_files": legacy_files})

    sample = max(legacy_rows, key=lambda r: len(r.get("text") or ""))
    legacy_sf = sample["source_file"]
    probe = (sample.get("text") or "")[:200]

    # ---- B(1/2) 无过滤时该 legacy 文本可召回自身（证明探针确实命中 legacy 向量）----
    v = await emb.embed_query(probe)
    raw_free = await asyncio.to_thread(_search, v, 10, None)
    free_srcs = [e.get("source_file", "") for _, _, e in _parse_milvus_hits(raw_free)]
    check("B1_unfiltered_probe_recalls_legacy", legacy_sf in free_srcs, {"legacy_source_file": legacy_sf, "top10": free_srcs})

    # ---- B(2/2) 加白名单后同一探针不再召回任何 legacy ----
    raw_f = await asyncio.to_thread(_search, v, 10, expr_in)
    f_hits = _parse_milvus_hits(raw_f)
    f_srcs = [e.get("source_file", "") for _, _, e in f_hits]
    b2 = bool(f_srcs) and all(s in whitelist for s in f_srcs) and (legacy_sf not in f_srcs)
    check("B2_filtered_probe_excludes_legacy", b2, {"top10": f_srcs})

    # ---- A canonical 查询（dense+白名单）----
    probes = [
        ("公司法 有限责任公司 股东认缴的出资额 最迟什么时候缴足", "LAW_008"),
        ("劳动合同法 经济补偿金 按什么标准计算", "LAW_009"),
        ("网络安全法 2025修正 人工智能 条款", "LAW_003"),
    ]
    lid_map = {m.get("logical_document_id"): sf for sf, m in scope.metadata.items()}
    a_all = True
    a_detail = {}
    for qtext, lid in probes:
        vec = await emb.embed_query(qtext)
        raw = await asyncio.to_thread(_search, vec, 60, expr_in)
        hits = _parse_milvus_hits(raw)
        srcs = [e.get("source_file", "") for _, _, e in hits]
        expect_sf = lid_map.get(lid, "")
        sub_ok = bool(srcs) and all(s in whitelist for s in srcs) and (expect_sf in srcs[:10])
        a_all = a_all and sub_ok
        a_detail[lid] = {
            "query": qtext,
            "n_hits": len(srcs),
            "all_canonical": all(s in whitelist for s in srcs),
            "expected_in_top10": expect_sf in srcs[:10],
            "top3": srcs[:3],
        }
        print(f"  A[{lid}] n={len(srcs)} all_canonical={all(s in whitelist for s in srcs)} expected_top10={expect_sf in srcs[:10]}")
    check("A_canonical_dense_retrieval", a_all, a_detail)
    return {"legacy_files": legacy_files}


async def s2_pipeline(scope, legacy_files: list[str]) -> None:
    import modules.rag.pipeline as pl
    from modules.rag.pipeline import RagPipeline

    whitelist = scope.source_files
    captured: list[dict] = []
    orig = pl.build_user_message
    orig_intent = pl.is_professional_query

    async def _always_professional(question: str) -> bool:
        return True  # 回归聚焦检索/上下文层：排除意图分层的偶发噪声（观察项单独记录）

    def spy(question, contexts, memory_snippet=None):
        captured.append({"question": question, "contexts": list(contexts)})
        return orig(question, contexts, memory_snippet)

    # 观察项：记录意图模型对 C 查询的真实判定（不改生产行为）
    q_c_probe = "2025年修正后的网络安全法从什么时候开始施行？现行有效的版本是哪一次修正？"
    try:
        intent_probe = await orig_intent(q_c_probe)
    except Exception:  # noqa: BLE001
        intent_probe = None
    RESULTS["intent_probe_professional"] = intent_probe
    print(f"  intent_probe(q_c) professional={intent_probe}")

    pl.build_user_message = spy
    pl.is_professional_query = _always_professional
    ts = int(time.time())
    pipe = RagPipeline()

    async def ask(qtext: str, uid: str) -> str:
        buf: list[str] = []
        async for piece in pipe.stream_chat(qtext, uid):
            buf.append(piece)
        return "".join(buf)

    try:
        # ---- C: LAW_003 context metadata ----
        q_c = "2025年修正后的网络安全法从什么时候开始施行？现行有效的版本是哪一次修正？"
        ans_c = await ask(q_c, f"align_c_{ts}")
        entry = next((e for e in reversed(captured) if e["question"] == q_c), None)
        ctxs = entry["contexts"] if entry else []
        law003_ctx = next((c for c in ctxs if "logical_document_id: LAW_003" in c), "")
        header_sources = re.findall(r"^source_file: (.+)$", "\n".join(ctxs), re.M)
        legacy_leak = [s for s in legacy_files if any(s in c for c in ctxs)]
        c_ok = (
            bool(ans_c)
            and bool(law003_ctx)
            and "version: 2025_amended" in law003_ctx
            and "effective_date: 2026-01-01" in law003_ctx
            and "[Evidence]" in law003_ctx
            and all(s in whitelist for s in header_sources)
            and not legacy_leak
        )
        check(
            "C_law003_context_metadata",
            c_ok,
            {
                "query": q_c,
                "n_contexts": len(ctxs),
                "looks_like_faq_branch": any(c.startswith("问答参考") for c in ctxs),
                "context_heads": [c[:90].replace("\n", " | ") for c in ctxs],
                "header_source_files": header_sources,
                "legacy_leak": legacy_leak,
                "law003_header_ok": bool(law003_ctx)
                and "version: 2025_amended" in law003_ctx
                and "effective_date: 2026-01-01" in law003_ctx,
                "law003_ctx_head": law003_ctx[:260].replace("\n", " | "),
                "answer_head": ans_c[:160],
            },
        )

        # ---- D1: 正向 citation（只能来自 context metadata/正文）----
        q_d1 = "数据安全法自哪一天起施行？"
        ans_d1 = await ask(q_d1, f"align_d1_{ts}")
        e_d1 = next((e for e in reversed(captured) if e["question"] == q_d1), None)
        ctxs_d1 = e_d1["contexts"] if e_d1 else []
        d1_ctx_meta = any("logical_document_id: LAW_002" in c and "effective_date: 2021-09-01" in c for c in ctxs_d1)
        d1_ok = ("2021" in ans_d1) and (("9月1日" in ans_d1) or ("2021-09-01" in ans_d1))
        check("D1_citation_grounded", d1_ok, {"query": q_d1, "context_has_law002_meta": d1_ctx_meta, "answer_head": ans_d1[:160]})

        # ---- D2: 知识不足（语料中不存在的规定，不得编造条文）----
        q_d2 = "《企业湿地保护管理条例》第十四条具体规定了什么内容？"
        ans_d2 = await ask(q_d2, f"align_d2_{ts}")
        d2_ok = bool(re.search(r"未检索到|未找到|无相关|不足以|无法|未收录|没有找到", ans_d2))
        check("D2_no_fabrication", d2_ok, {"query": q_d2, "answer_head": ans_d2[:200]})
    finally:
        pl.build_user_message = orig
        pl.is_professional_query = orig_intent


async def s3_sse() -> None:
    import httpx

    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        try:
            h = await client.get(f"{BASE_URL}/health")
            if h.status_code != 200:
                check("E_live_sse", False, f"health={h.status_code}")
                return
        except Exception as exc:  # noqa: BLE001
            check("E_live_sse", False, f"server unreachable: {exc}")
            return

        queries: list[str] = []
        try:
            with open(EVAL_SET_PATH, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line)["query"])
                    if len(queries) >= 3:
                        break
        except OSError:
            queries = ["公司法股东出资期限", "劳动合同经济补偿标准"]
        queries.append("《企业湿地保护管理条例》第十四条具体规定了什么内容？")

        ts = int(time.time())

        async def _sse_once(q: str, uid: str) -> dict:
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/chat/stream",
                json={"message": q, "user_external_id": uid},
            ) as resp:
                status = resp.status_code
                body = ""
                async for chunk in resp.aiter_text():
                    body += chunk
            done = body.rstrip().endswith("data: [DONE]")
            no_err = '"error"' not in body
            ok = status == 200 and done and no_err and len(body) > len("data: [DONE]")
            err_line = next((ln.strip()[:300] for ln in body.splitlines() if '"error"' in ln), "")
            return {
                "ok": ok,
                "status": status,
                "done": done,
                "no_error": no_err,
                "bytes": len(body),
                "error_line": err_line,
                "body_tail": "" if ok else body[-300:],
            }

        e_all = True
        detail = []
        for i, q in enumerate(queries):
            try:
                r = await _sse_once(q, f"align_e{i}_{ts}")
                if not r["ok"]:
                    print(f"  E[{i}] attempt1 FAIL :: {r['error_line'] or r['body_tail']!r}")
                    await asyncio.sleep(3)  # 冷启动/瞬态错误：等待后重试一次
                    r2 = await _sse_once(q, f"align_e{i}_{ts}r")
                    if r2["ok"]:
                        print(f"  E[{i}] retry PASS")
                        r2["retried"] = True
                        r = r2
                    else:
                        print(f"  E[{i}] retry FAIL :: {r2['error_line'] or r2['body_tail']!r}")
                        r["retry"] = {k: r2[k] for k in ("status", "done", "no_error", "bytes", "error_line", "body_tail")}
                e_all = e_all and r["ok"]
                row = {"i": i, "query": q, "status": r["status"], "done": r["done"], "no_error": r["no_error"], "bytes": r["bytes"]}
                if not r["ok"]:
                    row["error_line"] = r["error_line"]
                    row["body_tail"] = r["body_tail"]
                if r.get("retried"):
                    row["retried"] = True
                detail.append(row)
                print(f"  E[{i}] status={r['status']} done={r['done']} no_error={r['no_error']} bytes={r['bytes']} retried={r.get('retried', False)}")
            except Exception as exc:  # noqa: BLE001
                e_all = False
                detail.append({"i": i, "query": q, "exception": str(exc)})
        check("E_live_sse", e_all, detail)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all", choices=["all", "sse"])
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    if args.mode == "all":
        scope, _ = await s0_scope_sanity()
        legacy_info = await s1_dense_isolation(scope)
        await s2_pipeline(scope, legacy_info["legacy_files"])
        await s3_sse()
    else:
        await s3_sse()

    RESULTS["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    written = None
    for path in (RESULTS_PATH, RESULTS_FALLBACK):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(RESULTS, f, ensure_ascii=False, indent=2)
            written = path
            break
        except OSError as exc:
            print(f"WARN: cannot write results to {path}: {exc}")
    print(f"results -> {written}")

    # ---- 汇总 ----
    c = RESULTS["checks"]

    def all_of(*names: str) -> bool:
        return all(c.get(n, {}).get("pass") for n in names)

    isolation = all_of("S0_manifest_scope", "A_canonical_dense_retrieval", "B0_legacy_still_present", "B1_unfiltered_probe_recalls_legacy", "B2_filtered_probe_excludes_legacy")
    header = all_of("C_law003_context_metadata")
    regression = isolation and header and all_of("D1_citation_grounded", "D2_no_fabrication", "E_live_sse")

    print()
    print(f"ONLINE CANONICAL ISOLATION = {'PASS' if isolation else 'FAIL'}")
    print(f"CONTEXT METADATA HEADER = {'PASS' if header else 'FAIL'}")
    print(f"RAG REGRESSION = {'PASS' if regression else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
