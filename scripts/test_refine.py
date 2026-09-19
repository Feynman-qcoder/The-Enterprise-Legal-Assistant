"""任务 2.9 实测：对 3 个真实文件跑 refine_retrieval_query，验证提炼 prompt 质量。

文件（覆盖 docx / md / pdf 三种格式）：
    data_corpus/ENT_CONTRACT_010_保管合同（市场监管总局2025版）.docx
    data_corpus/ENT_POLICY_001_中央企业合规管理办法.md
    data_corpus/LEGAL_LAW_001_中华人民共和国民法典.pdf（提炼输入截断至 4000 字符，属预期）

验收：输出人工判读为「可用于检索该文档主题的法律要点」，且无「以下是…」前言。

运行（在仓库根）：
    python scripts/test_refine.py

注意：需要 PyMuPDF 解析民法典 PDF——本机用户级 site-packages 里有一个坏 DLL 的 pymupdf，
请以 PYTHONNOUSERSITE=1 运行，让 conda env xiaoyi_rag 的 1.28.2 生效：
    PYTHONNOUSERSITE=1 python scripts/test_refine.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modules.ingestion.document_parsing import parse_document  # noqa: E402
from modules.rag.query_refine import _REFINE_INPUT_CHARS, refine_retrieval_query  # noqa: E402

FILES = [
    "data_corpus/ENT_CONTRACT_010_保管合同（市场监管总局2025版）.docx",
    "data_corpus/ENT_POLICY_001_中央企业合规管理办法.md",
    "data_corpus/LEGAL_LAW_001_中华人民共和国民法典.pdf",
]


async def main() -> int:
    for rel in FILES:
        p = REPO / rel
        print("=" * 78)
        if not p.exists():
            print("文件不存在：%s" % p)
            continue
        t0 = time.time()
        try:
            doc = parse_document(p)
        except Exception as exc:  # noqa: BLE001 — 实测脚本要看清所有失败
            print("文件     : %s" % p.name)
            print("解析失败 : %s: %s" % (type(exc).__name__, exc))
            continue
        text = (getattr(doc, "text", "") or "").strip()
        parse_s = time.time() - t0

        t1 = time.time()
        query = await refine_retrieval_query(text)
        refine_s = time.time() - t1

        print("文件     : %s" % p.name)
        print("解析     : %.2fs  提取 %d 字符（送提炼前截断至 %d）" % (parse_s, len(text), _REFINE_INPUT_CHARS))
        print("提炼     : %.2fs  返回 %d 字符" % (refine_s, len(query))
              + ("  ⚠️ 空结果（失败回退）" if not query else ""))
        print("检索 query: %r" % query)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
