#!/usr/bin/env python3
# =============================================================================
# 思考模式 A/B 消融实验（AutoDL 私有化部署）
# -----------------------------------------------------------------------------
# 目的：用同一批法律问题，对比「关思考 / 开思考」在【真实 RAG 链路】上的
#       质量与延迟差异，用数据回答"该不该关思考"，而不是凭感觉。
#
# 关键设计：
#   1) 每题使用独立的 user_external_id —— 项目 exact 缓存的 key 含 user_scope，
#      不复用同一 scope 才能拿到真实推理延迟（否则秒回 0.01s 假数据）。
#   2) 直接打后端 /api/chat/stream（SSE），度量：
#        首字延迟 TTFB / 总耗时 / 回答字数 / 是否异常
#   3) 完整回答落盘成 JSON + Markdown，便于人工比对质量与留档。
#
# 用法（实例上，仓库根目录）：
#   .venv/bin/python deploy/autodl/thinking_ab.py --label A_off --out /root/autodl-tmp/ab_A.json
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# 分层问题集：简单条文型（答案可直接检索到）vs 复杂适用型（需要跨条款推理）
QUESTIONS: list[tuple[str, str, str]] = [
    ("S1", "简单·条文", "处理敏感个人信息应当取得什么样的同意？"),
    ("S2", "简单·条文", "经营者不得实施哪些混淆行为？"),
    ("H1", "复杂·适用", "合同双方约定的违约金明显高于实际损失时，法院可以如何处理？依据是什么？"),
    ("H2", "复杂·多跳", "网络服务提供者未经用户同意向第三方提供其个人信息，可能承担哪些法律责任？"),
]


def sse_chat(base_url: str, question: str, user_id: str, timeout: int = 300) -> dict:
    """POST /api/chat/stream 并聚合 SSE，返回延迟与回答。"""
    url = base_url.rstrip("/") + "/api/chat/stream"
    payload = json.dumps({"message": question, "user_external_id": user_id},
                         ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.time()
    ttfb = None
    chunks: list[str] = []
    err = None
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            while True:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if ttfb is None:
                    ttfb = time.time() - t0
                if body == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    chunks.append(body)
                    continue
                if "error" in obj:
                    err = str(obj["error"])
                    break
                if "chunk" in obj:
                    chunks.append(str(obj["chunk"]))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    total = time.time() - t0
    answer = "".join(chunks)
    return {
        "ttfb": round(ttfb if ttfb is not None else total, 2),
        "total": round(total, 2),
        "chars": len(answer),
        "done": done,
        "error": err,
        "answer": answer,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="思考模式 A/B 消融实验")
    ap.add_argument("--label", required=True, help="本组标签，如 A_off / B_on / C_low")
    ap.add_argument("--base-url", default="http://127.0.0.1:6006")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    out_path = args.out or f"/root/autodl-tmp/ab_{args.label}.json"
    rows = []
    print("=" * 88)
    print(f"思考模式 A/B · 组={args.label}  base={args.base_url}")
    print("=" * 88)

    for idx, (qid, kind, q) in enumerate(QUESTIONS, 1):
        # ★ 独立 user_external_id：绕开 exact 缓存，拿到真实推理延迟
        user_id = f"ab-{args.label}-{qid}-{int(time.time())}"
        print(f"\n[{idx}/{len(QUESTIONS)}] {qid} ({kind})")
        print(f"    Q: {q}")
        r = sse_chat(args.base_url, q, user_id, args.timeout)
        status = "OK" if (r["done"] and not r["error"] and r["chars"] > 0) else "FAIL"
        print(f"    -> {status}  首字 {r['ttfb']}s / 总耗时 {r['total']}s / {r['chars']} 字")
        if r["error"]:
            print(f"    error: {r['error'][:200]}")
        else:
            head = r["answer"][:220].replace("\n", " ")
            print(f"    答: {head}…")
        rows.append({"qid": qid, "kind": kind, "question": q, **r})

    print("\n" + "=" * 88)
    print(f"{'题号':<5}{'类型':<10}{'首字(s)':<10}{'总耗时(s)':<11}{'字数':<7}状态")
    print("-" * 88)
    for r in rows:
        st = "OK" if (r["done"] and not r["error"] and r["chars"] > 0) else "FAIL"
        print(f"{r['qid']:<5}{r['kind']:<10}{r['ttfb']:<10}{r['total']:<11}{r['chars']:<7}{st}")
    ok = [r for r in rows if r["done"] and not r["error"] and r["chars"] > 0]
    if ok:
        print("-" * 88)
        print(f"均值：首字 {sum(r['ttfb'] for r in ok)/len(ok):.2f}s | "
              f"总耗时 {sum(r['total'] for r in ok)/len(ok):.2f}s | "
              f"字数 {sum(r['chars'] for r in ok)/len(ok):.0f}")
    print("=" * 88)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"label": args.label, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"结果已保存：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
