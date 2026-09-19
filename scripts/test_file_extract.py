"""验证「文件素材」路径是否真的可用。

回答两个问题：
1. `parse_document()` 能不能吃下 data_corpus 里的真实文件？提取质量如何？
2. 提取出的文本，**适合直接当检索 query 吗**？（这是我最怀疑的地方）

用法：
    python scripts/test_file_extract.py                # 扫 data_corpus 全量（每类取几个）
    python scripts/test_file_extract.py <文件路径>...   # 指定文件
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modules.ingestion.document_parsing import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    DocumentParseError,
    parse_document,
    parser_available,
)

# BGE-M3 建库时实际用的 token 上限（取自 frozen_assets/embedding_manifest.json 的 max_token_length）
EMBED_MAX_TOKENS = 492
# BGE-M3 的硬上限
BGE_M3_HARD_LIMIT = 8192


def approx_tokens(text: str) -> int:
    """中文粗估：1 token ≈ 1.5 汉字；英文 1 token ≈ 4 字符。取两者折中的保守估计。"""
    return max(1, int(len(text) / 1.5))


def main() -> int:
    argv = sys.argv[1:]
    if argv:
        targets = [Path(p) for p in argv]
    else:
        corpus = REPO / "data_corpus"
        by_ext: dict[str, list[Path]] = defaultdict(list)
        for p in sorted(corpus.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                by_ext[p.suffix.lower()].append(p)
        targets = []
        for ext, files in sorted(by_ext.items()):
            targets.extend(files[:3])  # 每类抓 3 个做样本

    print("=" * 96)
    print("文件素材路径验证")
    print("=" * 96)
    print("  支持的扩展名 : %s" % sorted(SUPPORTED_EXTENSIONS))
    print("  解析器可用性 :")
    for ext in sorted(SUPPORTED_EXTENSIONS):
        print("      %-7s %s" % (ext, "✅ 可用" if parser_available(ext) else "❌ 不可用（缺依赖）"))
    print("  建库时的 token 上限 : %d（BGE-M3 硬上限 %d）" % (EMBED_MAX_TOKENS, BGE_M3_HARD_LIMIT))
    print("  待测文件数 : %d" % len(targets))
    print()

    rows = []
    for path in targets:
        row = {"file": path.name, "ext": path.suffix.lower(), "bytes": 0, "chars": 0,
               "tok": 0, "sec": 0.0, "ok": False, "err": "", "head": ""}
        try:
            row["bytes"] = path.stat().st_size
            t0 = time.time()
            doc = parse_document(path)
            row["sec"] = time.time() - t0
            text = getattr(doc, "text", None) or getattr(doc, "content", "") or ""
            row["chars"] = len(text)
            row["tok"] = approx_tokens(text)
            row["head"] = text[:120].replace("\n", " ⏎ ")
            row["ok"] = len(text) >= 20
            if not row["ok"]:
                row["err"] = "提取文本过短"
        except DocumentParseError as exc:
            row["err"] = "DocumentParseError: %s" % exc
        except Exception as exc:  # noqa: BLE001
            row["err"] = "%s: %s" % (type(exc).__name__, exc)
        rows.append(row)

    print("-" * 96)
    print("%-46s %6s %9s %8s %7s %s" % ("文件", "类型", "字符数", "≈token", "耗时s", "状态"))
    print("-" * 96)
    for r in rows:
        status = "✅" if r["ok"] else "❌ " + r["err"][:40]
        print("%-46s %6s %9d %8d %7.2f %s" % (r["file"][:46], r["ext"], r["chars"], r["tok"], r["sec"], status))

    print()
    print("=" * 96)
    print("关键判定：提取文本能否直接作为检索 query？")
    print("=" * 96)

    ok_rows = [r for r in rows if r["ok"]]
    print("  解析成功 : %d/%d" % (len(ok_rows), len(rows)))
    if not ok_rows:
        print("  结论：解析层不可用，需先解决依赖问题")
        return 1

    over_embed = [r for r in ok_rows if r["tok"] > EMBED_MAX_TOKENS]
    over_hard = [r for r in ok_rows if r["tok"] > BGE_M3_HARD_LIMIT]
    print("  超出建库上限(%d token) : %d/%d" % (EMBED_MAX_TOKENS, len(over_embed), len(ok_rows)))
    print("  超出 BGE-M3 硬上限(%d)  : %d/%d" % (BGE_M3_HARD_LIMIT, len(over_hard), len(ok_rows)))
    print()
    for r in ok_rows:
        flag = ""
        if r["tok"] > BGE_M3_HARD_LIMIT:
            flag = "  ⚠️ 超硬上限，必然被截断"
        elif r["tok"] > EMBED_MAX_TOKENS:
            flag = "  ⚠️ 超建库上限，语义会被稀释"
        print("    %-44s ≈%7d token%s" % (r["file"][:44], r["tok"], flag))
    print()
    print("  抽样正文开头：")
    for r in ok_rows[:3]:
        print("    --- %s ---" % r["file"][:50])
        print("        %s" % r["head"])
    print()
    print("=" * 96)
    if over_hard:
        print("结论：❌ 文件提取文本【不适合直接作为检索 query】")
        print("      文档动辄数万字符，远超 BGE-M3 的 8192 硬上限与建库时的 492 上限。")
        print("      直接当 query 会被截断成前若干字符 → 检索到的内容与文档主题无关。")
    elif over_embed:
        print("结论：⚠️ 部分文件超出建库上限，语义会被稀释，需截断或提炼后再检索")
    else:
        print("结论：✅ 提取文本长度适合直接作为检索 query")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
