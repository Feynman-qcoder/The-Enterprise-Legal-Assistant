#!/usr/bin/env python3
# =============================================================================
# AutoDL 私有化部署 · 冒烟验收测试
# -----------------------------------------------------------------------------
# 目的：用一条命令回答「这次私有化部署到底通没通」，而不是靠肉眼看日志。
#
# 链路覆盖（对应当前系统的真实调用顺序）：
#   ① backend /health                → 进程存活 + DB/Milvus 连接成功（启动即建表/连库）
#   ② vLLM   /v1/models              → 本地大模型服务在跑，且模型名对得上
#   ③ POST   /api/chat/stream        → 走完整 RAG：意图 → 嵌入 → Milvus → BM25 → RRF
#                                      → 父文档 → 重排 → 本地千问生成 → SSE 流式
#
# 用法（仓库根目录）：
#   python deploy/autodl/smoke_test.py
#   python deploy/autodl/smoke_test.py --question "经营者不得实施哪些混淆行为？"
#   python deploy/autodl/smoke_test.py --skip-llm        # 只想验后端链路
#
# 退出码：0 = 全部 PASS；1 = 有 FAIL（可据此判断部署是否成功）
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_FALLBACK = "抱歉，回答生成服务暂时不可用，请稍后重试。"
# 语料含《个人信息保护法》canonical 文件，用作默认提问
DEFAULT_QUESTIONS = [
    "处理敏感个人信息应当取得什么样的同意？",
]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_dotenv(root: str) -> dict:
    """极简 .env 解析（不引入额外依赖）：只取 KEY=VALUE，忽略注释与空行。"""
    env = {}
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def http_get_json(url: str, timeout: int, api_key: str = "") -> tuple[int, object]:
    """GET 并解析 JSON。api_key 非空时带 Bearer 头（vLLM 用 --api-key 时必须带）。"""
    headers = {"User-Agent": "xiaoyi-smoke/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace"))


def sse_chat(base_url: str, question: str, timeout: int) -> dict:
    """POST /api/chat/stream 并按 SSE 协议聚合答案。

    后端实际产出格式（见 backend/app/api/chat.py）：
        data: {"chunk": "片段"}\\n\\n     逐片输出
        data: [DONE]\\n\\n              流结束
        data: {"error": "..."}\\n\\n    异常分支
    """
    url = base_url.rstrip("/") + "/api/chat/stream"
    payload = json.dumps({"message": question}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.time()
    ttfb = None
    chunks: list[str] = []
    err: str | None = None
    done = False

    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
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

    total = time.time() - t0
    return {
        "ttfb": ttfb if ttfb is not None else total,
        "total": total,
        "done": done,
        "error": err,
        "answer": "".join(chunks),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="AutoDL 私有化部署冒烟测试")
    ap.add_argument("--root", default=None, help="仓库根目录（缺省自动推断）")
    ap.add_argument("--base-url", default=None, help="backend 地址，默认读 BACKEND_URL 或 127.0.0.1:6006")
    ap.add_argument("--llm-url", default=None, help="vLLM 地址，默认读 DASHSCOPE_BASE_URL")
    ap.add_argument("--question", action="append", default=None, help="可重复；不传则用内置默认问题")
    ap.add_argument("--timeout", type=int, default=180, help="单个请求超时秒数")
    ap.add_argument("--skip-llm", action="store_true", help="跳过 vLLM 直连检查")
    ap.add_argument("--skip-backend", action="store_true", help="跳过 backend 检查")
    args = ap.parse_args()

    root = args.root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dotenv = load_dotenv(root)

    base_url = args.base_url or os.environ.get("BACKEND_URL") or "http://127.0.0.1:6006"
    llm_url = args.llm_url or dotenv.get("DASHSCOPE_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") \
        or "http://127.0.0.1:8001/v1"
    served_name = dotenv.get("LLM_MODEL") or os.environ.get("LLM_MODEL") or "Qwen3.8-27B-FP8"
    # vLLM 侧 --api-key 对应的 key；查 /v1/models 时必须带上，否则 401
    llm_api_key = dotenv.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    fallback = dotenv.get("LLM_FALLBACK_MESSAGE") or os.environ.get("LLM_FALLBACK_MESSAGE") or DEFAULT_FALLBACK
    questions = args.question or DEFAULT_QUESTIONS

    print("=" * 78)
    print("AutoDL 私有化部署 · 冒烟验收")
    print("=" * 78)
    print(f"仓库根    : {root}")
    print(f"backend   : {base_url}")
    print(f"vLLM      : {llm_url}")
    print(f"期望模型名: {served_name}")
    print(f"NO_PROXY  : {os.environ.get('NO_PROXY') or dotenv.get('NO_PROXY') or '(未设置 ⚠️)'}")
    print("=" * 78)

    results: list[tuple[str, bool, str]] = []

    # ---------- ① backend /health ----------
    if not args.skip_backend:
        try:
            status, body = http_get_json(base_url.rstrip("/") + "/health", args.timeout)
            ok = bool(isinstance(body, dict) and body.get("ok"))
            results.append(("/health", ok, f"HTTP {status} {body}"))
        except Exception as exc:  # noqa: BLE001
            results.append(("/health", False, f"{type(exc).__name__}: {exc}"))

    # ---------- ② vLLM /v1/models ----------
    if not args.skip_llm:
        try:
            status, body = http_get_json(llm_url.rstrip("/") + "/models", args.timeout, llm_api_key)
            ids = [m.get("id") for m in (body or {}).get("data", [])] if isinstance(body, dict) else []
            hit = served_name in ids
            results.append(("vLLM /v1/models", hit,
                            f"HTTP {status} 已加载={ids or '[]'}"
                            + ("" if hit else f" ⚠️ 与 LLM_MODEL({served_name}) 不一致")))
        except Exception as exc:  # noqa: BLE001
            results.append(("vLLM /v1/models", False, f"{type(exc).__name__}: {exc}"))

    # ---------- ③ 端到端 RAG 问答 ----------
    if not args.skip_backend:
        for i, q in enumerate(questions, 1):
            label = f"RAG 问答#{i}"
            try:
                r = sse_chat(base_url, q, args.timeout)
                ans = r["answer"]
                if r["error"]:
                    results.append((label, False, f"流内错误: {r['error']}"))
                elif not r["done"]:
                    results.append((label, False, "未收到 [DONE]，流被截断"))
                elif not ans.strip():
                    results.append((label, False, "回答为空"))
                elif fallback and fallback in ans:
                    results.append((label, False, f"命中兜底文案（LLM 不可用）: {ans[:60]}"))
                else:
                    results.append((label, True,
                                    f"首字 {r['ttfb']:.2f}s / 总耗时 {r['total']:.2f}s / {len(ans)} 字"))
                    print(f"\n--- 问题{i}：{q}\n--- 回答（前 300 字）：\n{ans[:300]}\n")
            except Exception as exc:  # noqa: BLE001
                results.append((label, False, f"{type(exc).__name__}: {exc}"))

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print(f"{'检查项':<22} {'结果':<6} 说明")
    print("-" * 78)
    for name, ok, detail in results:
        print(f"{name:<22} {'PASS' if ok else 'FAIL':<6} {detail}")
    print("=" * 78)

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n❌ 冒烟测试未通过，失败项：{', '.join(failed)}")
        print("   排查顺序建议：")
        print("     1) vLLM 是否起在 8001？curl $DASHSCOPE_BASE_URL/models")
        print("     2) NO_PROXY 是否包含 127.0.0.1？否则本地调用会被代理劫持")
        print("     3) MySQL/Milvus 是否就绪？docker compose ps")
        print("     4) 是否已执行 run_v2_ingest.py --precomputed（库空则检索不到东西）")
        return 1

    print("\n✅ 冒烟测试全部通过：私有化千问 + 本地嵌入/重排 + RAG 全链路可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
