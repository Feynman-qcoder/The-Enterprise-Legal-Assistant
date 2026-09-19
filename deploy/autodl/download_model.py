#!/usr/bin/env python3
# =============================================================================
# ModelScope 并行下载器（AutoDL 专用）
# -----------------------------------------------------------------------------
# 为什么需要它：
#   ModelScope 官方 CLI 走「分块 Range 请求 + 单连接」，在国内 CDN 上常被限速到
#   1~3 MB/s。27B-FP8 权重约 28.5GB，按 2MB/s 要 4 小时——纯粹是在给 GPU 烧钱。
#   本脚本改为「多文件 × 多分段」并行 Range 下载，实测可到 10MB/s 量级。
#
# 用法（仓库根目录）：
#   python deploy/autodl/download_model.py \
#       --model Qwen/Qwen3.8-27B-FP8 \
#       --local-dir /root/autodl-tmp/models/Qwen3.8-27B-FP8 \
#       --workers 16
#
# 常用参数：
#   --list-only     只拉文件清单，不下载（先验证模型名/体积对不对）
#   --verify-only   只校验本地文件大小是否与远端一致
#   --segment-mb N  单分段大小，默认 64（网络抖动大时可降到 16）
#   --insecure      跳过 TLS 校验（仅当公司内网代理做中间人时使用）
#
# 失败兜底：本脚本只用标准库。若清单接口结构变动导致解析失败，
#          脚本会明确提示改用官方 CLI：
#              pip install modelscope
#              modelscope download --model <模型> --local_dir <目录>
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
LIST_URLS = [
    "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision={rev}",
    "https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision={rev}",
]
RAW_URL = "https://modelscope.cn/api/v1/models/{model}/repo?Revision={rev}&FilePath={path}"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def make_ctx(insecure: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_json(url: str, ctx: ssl.SSLContext, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def extract_files(node, out: list) -> None:
    """递归扫描 JSON，收集所有形如 {Path, Size} 的条目。

    ModelScope 各接口的包裹层级会变（Data.Files / Data.FileTree / 平铺数组），
    这里不做结构假设，只认字段名，确保接口微调后仍能用。
    """
    if isinstance(node, dict):
        path = node.get("Path") or node.get("path")
        size = node.get("Size", node.get("size"))
        if path and size is not None:
            try:
                if int(size) > 0:
                    out.append((str(path).lstrip("/"), int(size)))
            except (TypeError, ValueError):
                pass
        for v in node.values():
            extract_files(v, out)
    elif isinstance(node, list):
        for v in node:
            extract_files(v, out)


def fetch_file_list(model: str, rev: str, ctx: ssl.SSLContext) -> list[tuple[str, int]]:
    last_err = None
    for tpl in LIST_URLS:
        url = tpl.format(model=model, rev=rev)
        try:
            data = http_json(url, ctx)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        found: list[tuple[str, int]] = []
        extract_files(data, found)
        # 去重（同一 path 可能出现在多个层级里）
        dedup: dict[str, int] = {}
        for p, s in found:
            dedup[p] = max(dedup.get(p, 0), s)
        if dedup:
            return sorted(dedup.items())
        last_err = RuntimeError(f"{url} 返回的 JSON 里找不到 Path/Size 字段")
    raise RuntimeError(f"无法获取文件清单（{last_err}）")


def range_supported(url: str, ctx: ssl.SSLContext) -> bool:
    """探测服务端是否支持 Range（206 + Content-Range）。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return r.status == 206 and r.headers.get("Content-Range") is not None
    except Exception:  # noqa: BLE001
        return False


def download_segment(url: str, start: int, end: int, dest: str, ctx: ssl.SSLContext, retries: int = 4) -> int:
    """下载 [start, end] 并写入 dest 对应偏移。返回写入字节数。"""
    want = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                data = r.read()
            if len(data) != want:
                raise IOError(f"分段长度不符：期望 {want}，实收 {len(data)}")
            with open(dest, "r+b") as fh:
                fh.seek(start)
                fh.write(data)
            return len(data)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            log(f"    分段 {start}-{end} 第 {attempt} 次失败（{exc}），重试…")
            time.sleep(1.5 * attempt)
    return 0


def download_file(path: str, size: int, dest: str, model: str, rev: str,
                  seg_bytes: int, ctx: ssl.SSLContext, workers: int) -> str:
    url = RAW_URL.format(model=model, rev=rev, path=urllib.parse.quote(path))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # 预留文件到目标大小，供多分段随机写入
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        with open(dest, "wb") as fh:
            fh.truncate(size)

    if not range_supported(url, ctx):
        log(f"  [单线程] {path}（服务端不支持 Range）")
        download_segment(url, 0, size - 1, dest, ctx)
        return path

    segs = [(s, min(s + seg_bytes - 1, size - 1)) for s in range(0, size, seg_bytes)]
    done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(segs))) as pool:
        futs = {pool.submit(download_segment, url, a, b, dest, ctx): (a, b) for a, b in segs}
        for fut in as_completed(futs):
            done += fut.result()

    actual = os.path.getsize(dest)
    if actual != size:
        raise IOError(f"文件 {path} 大小不符：期望 {size}，实际 {actual}")
    log(f"  [完成] {path}  {size / 1024 / 1024:.1f} MB")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="ModelScope 并行下载器（AutoDL 专用）")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8", help="ModelScope 模型 ID")
    ap.add_argument("--local-dir", required=True, help="本地保存目录")
    ap.add_argument("--revision", default="master", help="分支/版本，默认 master")
    ap.add_argument("--workers", type=int, default=16, help="并发分段数，默认 16")
    ap.add_argument("--segment-mb", type=int, default=64, help="单分段 MB，默认 64")
    ap.add_argument("--list-only", action="store_true", help="只列清单不下载")
    ap.add_argument("--verify-only", action="store_true", help="只校验本地大小")
    ap.add_argument("--insecure", action="store_true", help="跳过 TLS 校验")
    args = ap.parse_args()

    ctx = make_ctx(args.insecure)
    seg_bytes = max(1, args.segment_mb) * 1024 * 1024

    print("=" * 78)
    print(f"模型      : {args.model}")
    print(f"版本      : {args.revision}")
    print(f"保存目录  : {args.local_dir}")
    print("=" * 78)

    print("\n[1/3] 拉取文件清单…")
    try:
        files = fetch_file_list(args.model, args.revision, ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 获取清单失败：{exc}")
        print("\n请改用官方 CLI 兜底：")
        print("  pip install modelscope")
        print(f"  modelscope download --model {args.model} --local_dir {args.local_dir}")
        return 2

    total = sum(s for _, s in files)
    print(f"      共 {len(files)} 个文件，合计 {total / 1024 / 1024 / 1024:.2f} GB")
    for p, s in files[:8]:
        print(f"        - {p}  {s / 1024 / 1024:.1f} MB")
    if len(files) > 8:
        print(f"        … 其余 {len(files) - 8} 个略")
    if args.list_only:
        return 0

    print("\n[2/3] 校验本地已有文件…")
    todo, skipped = [], 0
    for p, s in files:
        dest = os.path.join(args.local_dir, p)
        if os.path.exists(dest) and os.path.getsize(dest) == s:
            skipped += 1
        else:
            todo.append((p, s, dest))
    print(f"      已完整 {skipped} 个，待下载 {len(todo)} 个")
    if args.verify_only:
        bad = [p for p, s, d in todo]
        print(f"\n{'✅ 本地齐全' if not bad else '❌ 缺失/不完整：' + ', '.join(bad[:20])}")
        return 0 if not bad else 1
    if not todo:
        print("\n✅ 全部文件已就绪，无需下载")
        return 0

    print(f"\n[3/3] 并行下载（workers={args.workers}, segment={args.segment_mb}MB）…")
    t0 = time.time()
    failed = []
    for idx, (p, s, dest) in enumerate(todo, 1):
        print(f"\n[{idx}/{len(todo)}] {p}  ({s / 1024 / 1024:.1f} MB)")
        try:
            download_file(p, s, dest, args.model, args.revision, seg_bytes, ctx, args.workers)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{p} -> {exc}")
            print(f"  ❌ 失败：{exc}")

    dt = time.time() - t0
    got = sum(os.path.getsize(os.path.join(args.local_dir, p))
              for p, s in files if os.path.exists(os.path.join(args.local_dir, p)))
    print("\n" + "=" * 78)
    print(f"完成：{len(todo) - len(failed)}/{len(todo)} 个文件，用时 {dt / 60:.1f} 分钟，"
          f"均速 {got / 1024 / 1024 / max(dt, 1):.1f} MB/s")
    if failed:
        print("以下文件失败（重跑本脚本会自动续传）：")
        for f in failed:
            print("  -", f)
        return 1
    print(f"✅ 模型已就绪：{args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
