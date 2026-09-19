"""Phase 0 风险验证：确认多模态模型能否把合同照片/法条截图识别成文本。

对应 openspec/changes/add-image-query 的任务 1.1（止损闸门）。

用法：
    python scripts/test_vision.py data_corpus/picture/建筑合同.png
    python scripts/test_vision.py data_corpus/picture/*.png       # 多张

设计要点（都是踩过的坑，别改）：
1. **mime 按 magic bytes 探测，不信扩展名** —— 实测 data_corpus/picture/学校承包合同.png
   扩展名是 .png 但内容其实是 WEBP。按扩展名传 mime 会被 API 拒。
2. **trust_env=False** —— 本机环境注入了 HTTP_PROXY 等变量，会让对 DashScope 的请求走坏代理。
3. 只打印识别文本与耗时，**不打印 API Key**。
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def resolve_config() -> tuple[str, str]:
    """复用应用自己的配置解析（modules.core.config），不另起一套默认值。

    这样本脚本验证的就是应用真正会用的 base_url / api_key。
    """
    try:
        from modules.core.config import get_settings  # noqa: PLC0415

        s = get_settings()
        return s.dashscope_api_key, s.dashscope_base_url
    except Exception as exc:  # noqa: BLE001 — 回退到裸环境变量，便于排查
        print("提示: get_settings() 不可用（%s），回退到环境变量" % type(exc).__name__, file=sys.stderr)
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return (
            os.environ.get("DASHSCOPE_API_KEY", ""),
            os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )


# ---- magic bytes 探测真实图片类型 ----
def detect_mime(data: bytes, filename: str) -> tuple[str, str]:
    """返回 (mime, 说明)。看不懂就返回空 mime，由调用方拒绝。"""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "WEBP"
    if data[:4] == b"%PDF":
        return "application/pdf", "PDF(非图片)"
    return "", "未知格式"


def extract_prompt() -> str:
    """提取提示词：只抽取原文，不总结；印章签名用占位符，不猜测。"""
    return (
        "提取这张图片中的全部文字内容，遵守以下规则：\n"
        "1. 保持原有的段落与条款编号结构（例如「第七条」「（二）」「1.1」等），不要合并或省略编号；\n"
        "2. 表格转换为 Markdown 表格；\n"
        "3. 印章、手写签名处标注为 [印章] 或 [签名]，不要猜测其内容；\n"
        "4. 只输出图片中已有的文字，不要解释、总结或推断；\n"
        "5. 如果图片中没有可识别的文字，只输出「无文字」。"
    )


def describe_image(client: httpx.Client, base_url: str, model: str, path: Path, extra_body: dict | None = None) -> dict:
    data = path.read_bytes()
    mime, kind = detect_mime(data, path.name)

    result = {
        "file": path.name,
        "bytes": len(data),
        "declared_ext": path.suffix.lower(),
        "detected_mime": mime or "(无法识别)",
        "kind": kind,
        "ok": False,
        "text": "",
        "elapsed": 0.0,
        "error": "",
    }
    if not mime or mime == "application/pdf":
        result["error"] = "不是受支持的图片格式（探测结果：%s）" % kind
        return result

    ext_mismatch = (
        (path.suffix.lower() in (".jpg", ".jpeg") and mime != "image/jpeg")
        or (path.suffix.lower() == ".png" and mime != "image/png")
        or (path.suffix.lower() == ".webp" and mime != "image/webp")
    )
    result["ext_mismatch"] = ext_mismatch

    b64 = base64.b64encode(data).decode()
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extract_prompt()},
                    {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
                ],
            }
        ],
    }
    if extra_body:
        payload.update(extra_body)  # 任务 2.4：关闭思考模式的开关实验（一次只加一种）
        result["extra_body"] = extra_body
    t0 = time.time()
    try:
        r = client.post(base_url.rstrip("/") + "/chat/completions", json=payload)
        result["http_status"] = r.status_code
        if r.status_code != 200:
            result["error"] = "HTTP %d: %s" % (r.status_code, r.text[:300])
            return result
        body = r.json()
        result["text"] = (body["choices"][0]["message"].get("content") or "").strip()
        result["usage"] = body.get("usage")
        result["ok"] = len(result["text"]) >= 20 and result["text"] != "无文字"
        if not result["ok"]:
            result["error"] = "识别文本过短（%d 字符）：%r" % (len(result["text"]), result["text"][:60])
    except Exception as exc:  # noqa: BLE001 — 验证脚本要看清所有失败
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        result["elapsed"] = time.time() - t0
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    api_key, base_url = resolve_config()
    # 模型优先级：命令行 --model 覆盖 > 环境变量 VISION_MODEL > 默认值
    model = os.environ.get("VISION_MODEL") or "deepseek-v4.1-flash"
    argv = sys.argv[1:]
    if "--model" in argv:
        i = argv.index("--model")
        model = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    # 任务 2.4：--no-thinking [strategy] 关闭思考模式实验。
    #   et（默认）= {"enable_thinking": false}      DashScope Qwen 系开关
    #   re        = {"reasoning_effort": "minimal"} OpenAI 风格开关
    # 一次只加一种：报 400（参数不识别）就换下一种。
    extra_body: dict | None = None
    full_text = "--full" in argv
    if full_text:
        i = argv.index("--full")
        argv = argv[:i] + argv[i + 1:]
    if "--no-thinking" in argv:
        i = argv.index("--no-thinking")
        argv = argv[:i] + argv[i + 1:]  # 先删 flag 本身
        strategy = "et"
        if i < len(argv) and argv[i] in ("et", "re"):
            strategy = argv[i]
            argv = argv[:i] + argv[i + 1:]  # 再删策略值
        extra_body = (
            {"enable_thinking": False} if strategy == "et"
            else {"reasoning_effort": "minimal"}
        )
    files = argv

    if not api_key:
        sys.exit("DASHSCOPE_API_KEY 为空：请在项目 .env 中配置")
    if not base_url:
        sys.exit("DASHSCOPE_BASE_URL 解析为空：检查 config.py 默认值或 .env")
    if not files:
        print(__doc__)
        return 2

    print("=" * 78)
    print("Phase 0 风险验证 · 多模态图片识别")
    print("=" * 78)
    print("  endpoint : %s" % base_url)
    print("  model    : %s" % model)
    print("  api_key  : %s…（长度 %d，不打印明文）" % (api_key[:4], len(api_key)))
    if extra_body:
        print("  thinking : 关闭实验开关 %s" % extra_body)
    print()

    headers = {"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"}
    # trust_env=False：绕开本机注入的 HTTP_PROXY / HTTPS_PROXY
    with httpx.Client(headers=headers, timeout=120.0, trust_env=False) as client:
        results = [describe_image(client, base_url, model, Path(p), extra_body) for p in files]

    n_ok = 0
    for r in results:
        print("-" * 78)
        print("文件     : %s  (%.1f KB)" % (r["file"], r["bytes"] / 1024))
        print("格式探测 : %s   扩展名 %s%s" % (
            r["detected_mime"], r["declared_ext"],
            "   ⚠️ 扩展名与实际不符（已按实际格式发送）" if r.get("ext_mismatch") else ""))
        if r["ok"]:
            n_ok += 1
            print("结果     : ✅ 识别成功  耗时 %.2fs  %d 字符" % (r["elapsed"], len(r["text"])))
            if r.get("usage"):
                print("用量     : %s" % r["usage"])
            shown = r["text"] if full_text else r["text"][:600]
            print("识别文本%s：" % ("（完整）" if full_text else "（前 600 字）"))
            for line in shown.splitlines():
                print("   | " + line)
        else:
            print("结果     : ❌ 失败  耗时 %.2fs" % r["elapsed"])
            print("原因     : %s" % r["error"])
        print()

    print("=" * 78)
    print("GATE: %d/%d 张识别成功" % (n_ok, len(results)))
    if n_ok == len(results):
        print("结论: PASS —— 多模态通道可用，可继续任务 1.2 / 2.x")
    elif n_ok > 0:
        print("结论: PARTIAL —— 部分图失败，先看失败原因再决定是否继续")
    else:
        print("结论: FAIL —— 执行任务 1.2：把 VISION_MODEL 换成 qwen-vl-max 重跑本脚本")
    print("=" * 78)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
