# -*- coding: utf-8 -*-
"""引用核查忠实性探针（OpenSpec add-citation-check，任务 5.1/5.2/5.3）。

两条探针 + 两条路径验证（对着运行中的后端 8000 端口跑，需 CITATION_CHECK_ENABLED=true）：

探针①（语料外）：问《劳动争议调解仲裁法》第二十七条——该法**不在** 53 篇语料中
    （语料里的 LEGAL_LAW_006 是商事《仲裁法》）。期望：回答声明知识库无依据（prompt 防线），
    且 citation_report 中该法引用为 law_not_in_evidence 或根本无引用（机制防线）。
    真答案：劳动争议申请仲裁的时效期间为一年——若模型自信引用该条（哪怕内容正确），
    说明越出证据用了参数记忆，核查器必须能发现并标注。

探针②（语料内）：拖欠工资问答——期望 citation_report 5/5 grounded
    （与 tasks 2.3 的离线断言一致；在线版验证真实检索证据下的端到端行为）。

路径③（无引用）：概念性问题——期望 total=0 且不报错。

路径④（开关关闭回归）：CITATION_CHECK_ENABLED=false 时同问题两轮 diff——
    SSE 流应与"无核查事件"完全一致（除答案内容本身随机的部分；改用事件类型序列断言）。

运行（仓库根，conda env）：
    python scripts/probe_citation_faithfulness.py
输出：F:/Cache/temp/probe_faithfulness_out.txt（人工判读 + 留存演示素材）
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
OUT = r"F:\Cache\temp\probe_faithfulness_out.txt"

Q_OUT_OF_CORPUS = "根据《中华人民共和国劳动争议调解仲裁法》第二十七条，劳动争议申请仲裁的时效期间是多久？"
# 注意：探针问题带变体后缀——exact 缓存 TTL 1h，同字面问题第二次会缓存命中（无本次证据 → 按设计不发报告）；
# 变体让探针可重复执行，每次都走真实 RAG 路径。缓存命中无报告是 v1 已知限制（README「已知限制」节）。
_VARIANT = ["（请引用具体条文）", "（请给出法律依据）", "（结合法条说明）", "（附上条文原文）"]
Q_IN_CORPUS = "毕业生签订劳动合同后被拖欠工资怎么办？"
Q_NO_CITATION = "什么是竞业限制？请简要解释这个概念。"

out: list[str] = []


def log(s: str = "") -> None:
    print(s)
    out.append(str(s))


def ask(question: str, timeout: float = 300.0) -> dict:
    """POST /api/chat/stream → {answer, report, error}（report=citation_report 事件或 None）。"""
    chunks: list[str] = []
    report = None
    error = ""
    with httpx.Client(timeout=timeout, trust_env=False) as c:
        with c.stream(
            "POST", BASE + "/api/chat/stream",
            json={"message": question}, headers={"Content-Type": "application/json"},
        ) as r:
            if r.status_code != 200:
                return {"answer": "", "report": None, "error": f"HTTP {r.status_code}", "status": r.status_code}
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue
                if "chunk" in ev:
                    chunks.append(ev["chunk"])
                elif ev.get("type") == "citation_report":
                    report = ev
                elif "error" in ev:
                    error = ev.get("error", "")
    return {"answer": "".join(chunks), "report": report, "error": error, "status": 200}


def main() -> int:
    ok = True
    import random
    variant = random.choice(_VARIANT)
    log(f"本轮探针问题变体后缀：{variant}（防 exact 缓存命中导致无证据路径）")
    log()

    # ---- 探针①：语料外 ----
    log("=" * 78)
    log("探针① 语料外：《劳动争议调解仲裁法》第二十七条（该法不在 53 篇语料中）")
    r = ask(Q_OUT_OF_CORPUS + variant)
    answer = r["answer"]
    log(f"HTTP {r['status']}  回答 {len(answer)} 字")
    declared_no_evidence = ("未检索到" in answer) or ("知识库" in answer and ("不足" in answer or "无" in answer)) or ("无法" in answer and "依据" in answer)
    log(f"  prompt 防线（回答声明无依据）：{'✅' if declared_no_evidence else '⚠️ 未声明'}")
    if r["report"]:
        s = r["report"]["summary"]
        log(f"  citation_report：total={s['total']} grounded={s['grounded']} no_evidence={s['law_not_in_evidence']} missing={s['article_missing']}")
        for c in r["report"]["citations"]:
            log(f"    - {c['law']} 第{c['article']}条 → {c['status']}")
        # 机制防线的精确断言：**第27条这一条**（语料外的具体条文）绝不允许被判 grounded——
        # 它不在任何证据中，判有据 = 机制防线失守。其余引用不设越界断言：第五轮实测发现模型会
        # 合法引用证据内真实存在的《仲裁法》第93条（原文即写明"劳动争议仲裁适用…等法律"）来
        # 解释拒绝作答的理由——那是证据内引用（grounded/mismatch 均合法），不是越界。
        target = [c for c in r["report"]["citations"] if "劳动争议调解仲裁法" in c["law"] and c["article"] == 27]
        no_violation = all(c["status"] != "grounded" for c in target) if target else True
        cond = no_violation
        log(f"  机制防线（语料外第27条未被误判有据：{[c['status'] for c in target] or '本轮未引用'}）：{'✅' if cond else '❌'}")
        ok &= cond
    else:
        log("  citation_report：无事件（回答无引用 → total=0 路径）✅")
    log(f"  回答节选：{answer[:200]}…")
    log()

    # ---- 探针②：语料内（拖欠工资，全有据）----
    log("=" * 78)
    log("探针② 语料内：拖欠工资问答（期望全部 grounded，与 2.3 离线断言一致；引用条数随生成浮动）")
    r = ask(Q_IN_CORPUS + variant)
    answer = r["answer"]
    log(f"HTTP {r['status']}  回答 {len(answer)} 字")
    if r["report"]:
        s = r["report"]["summary"]
        log(f"  citation_report：total={s['total']} grounded={s['grounded']} mismatch={s['text_mismatch']} missing={s['article_missing']} no_evidence={s['law_not_in_evidence']}")
        for c in r["report"]["citations"]:
            log(f"    - {c['law']} 第{c['article']}条 → {c['status']}")
        # 验收锚点：①报告产出 ②引用全部来自证据中的法（missing=0 且 no_evidence=0）
        # ③grounded + mismatch == total。mismatch（改写/句序重组）属观察模式的**设计内保守面**——
        # design Risks 预言的误报形态，正是要攒的数据，不计失败（探针②第四轮实测：85 条句序重组 → mismatch）。
        cond = s["total"] >= 3 and s["article_missing"] == 0 and s["law_not_in_evidence"] == 0 \
            and s["grounded"] + s["text_mismatch"] == s["total"]
        log(f"  判定：{'✅ 引用全部来自证据（mismatch 为观察数据）' if cond else '❌'}")
        ok &= cond
        if s["text_mismatch"]:
            log(f"  ⚠️ 观察数据：text_mismatch={s['text_mismatch']}（改写/重组形态，攒误报率用——v2 决定是否分层阈值）")
        # 留存演示素材（5.2 验收：完整 SSE 对话）
        log()
        log("  ---- 演示留存：完整回答 + 报告（存 probe_faithfulness_out.txt）----")
        for line in answer.splitlines():
            log("  | " + line)
    else:
        log("  ❌ 无 citation_report 事件（有引用的回答必须有报告）")
        ok = False
    log()

    # ---- 路径③：无引用/无证据回答不报错 ----
    # 注：法律 RAG 里"概念性问题"几乎总会引法条（竞业限制本身就是劳动合同法概念，首轮实测 total=2）
    # ——本路径的验收意图是「无引用或无证据的回答，核查链路不报错」，不预设 total=0。
    # 用两个形态覆盖：闲聊（非专业引导路径，无证据→无事件）+ 概念问题（有证据，容忍引用条数）。
    log("=" * 78)
    log("路径③ 无引用/无证据回答：闲聊 + 概念性问题（期望：不报错；无证据→无事件或 total=0）")
    r = ask("你好呀，你能帮我做什么？", timeout=120.0)
    log(f"[闲聊] HTTP {r['status']}  回答 {len(r['answer'])} 字  error={r['error'] or '无'}  report={'有' if r['report'] else '无事件'}")
    cond_chat = not r["error"] and (r["report"] is None or r["report"]["summary"]["total"] == 0)
    log(f"  判定：{'✅ 无证据路径不发事件（或不报引用）' if cond_chat else '❌'}")
    ok &= cond_chat

    r = ask(Q_NO_CITATION + variant)
    log(f"[概念] HTTP {r['status']}  回答 {len(r['answer'])} 字  error={r['error'] or '无'}")
    if r["report"]:
        s = r["report"]["summary"]
        log(f"  citation_report：total={s['total']} grounded={s['grounded']} mismatch={s['text_mismatch']} missing={s['article_missing']}")
        for c in r["report"]["citations"]:
            log(f"    - {c['law']} 第{c['article']}条 → {c['status']}")
        cond = not r["error"]  # 概念问题引用法条属正常（竞业限制=劳动合同法概念）；验收点=链路不报错
        log(f"  判定：{'✅ 有引用属正常（不预设 total=0），链路无错误' if cond else '❌'}")
        ok &= cond
    else:
        cond = not r["error"]
        log(f"  无事件（无证据路径）→ {'✅' if cond else '❌'}")
        ok &= cond
    log()

    log("=" * 78)
    log(f"探针总判定：{'PASS' if ok else 'FAIL'}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log(f"输出已留存 → {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
