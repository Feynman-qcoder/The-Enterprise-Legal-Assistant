# =============================================================================
# -----------------------------------------------------------------------------
# 输入：data_corpus/ 中一篇文档（--doc 文件名或 logical_document_id）+ 冻结清单。
# 输出：Task 19 文档包 JSON（--out 或 stdout）：{document_identity, source_file, parents[]}。
# 被谁调用：运维手工执行；产物交给 run_v2_doc_sync.py --package 灌入。
# =============================================================================
"""
全链编排（Task 22 L1）：data_corpus 原始语料 → Parser V2 + Cleaner V2 →
Strategy B 切块 → Chunk Metadata Contract V1 校验（fail-fast）→ 文档包构造。

复用关系（禁止第二份解析/清洗逻辑）：
- 解析/清洗：offline/chunking_strategy_v2/_v2_utils.dispatch_parse_and_clean（内部动态调
  docx/pdf/md parser 与 cleaner_v2，均随包或仓库既有）
- 切块：_v2_strategies.chunk_strategy_b（结构感知）
- 校验：CMC V1 ChunkMetadata.validate()

用法（仓库根目录、xiaoyi_rag 环境）：
  python offline/scripts/build_doc_package.py --doc ENT_POLICY_001_中央企业合规管理办法.md \
      --identity-suffix _RECHUNK_SMOKE --out pkg.json
  --identity-suffix：非空时在 logical_document_id 后追加，得到独立身份（冒烟/重切块对照用，
  不覆盖生产同名文档）；默认空 = 生产身份（replace 语义）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHUNKING_DIR = REPO / "offline" / "chunking_strategy_v2"

parser = argparse.ArgumentParser(
    description="Build Task-19 document package from data_corpus (full offline chain)"
)
parser.add_argument("--doc", required=True, help="canonical_file 文件名或 logical_document_id")
parser.add_argument("--manifest", default=str(REPO / "frozen_assets" / "ingest_manifest_v1.jsonl"))
parser.add_argument("--corpus-dir", default=str(REPO / "data_corpus"))
parser.add_argument("--identity-suffix", default="", help="追加到 logical_document_id 的后缀（冒烟隔离用）")
parser.add_argument("--out", default=None, help="输出 JSON 路径（缺省 stdout）")
parser.add_argument("--max-parent-chars", type=int, default=8000, help="父段内容上限（在线契约）")
args = parser.parse_args()

# 语料根注入须在 import _v2_utils 之前（模块级常量读取环境变量）
os.environ["XIAOYI_CORPUS_DIR"] = str(Path(args.corpus_dir).resolve())
os.environ["XIAOYI_MANIFEST_PATH"] = str(Path(args.manifest).resolve())
sys.path.insert(0, str(CHUNKING_DIR))
sys.path.insert(0, str(REPO))

# The migrated offline Cleaner is the real implementation; modules/ingestion/cleaner_v2.py
# is intentionally a zero-byte placeholder.  Bind both dynamic-import names explicitly so
# the package builder never falls through to that placeholder (or to an unrelated workspace).
from modules.ingestion import parsed_document_v2 as _parsed_document_v2  # noqa: E402

sys.modules["parsed_document_v2"] = _parsed_document_v2
_cleaner_path = CHUNKING_DIR / "cleaner_v2.py"
_cleaner_spec = importlib.util.spec_from_file_location("cleaner_v2", _cleaner_path)
if _cleaner_spec is None or _cleaner_spec.loader is None:
    raise RuntimeError(f"cannot load Cleaner V2 from {_cleaner_path}")
_cleaner_module = importlib.util.module_from_spec(_cleaner_spec)
sys.modules["cleaner_v2"] = _cleaner_module
_cleaner_spec.loader.exec_module(_cleaner_module)

from _v2_strategies import chunk_strategy_b  # noqa: E402
from _v2_utils import dispatch_parse_and_clean, load_manifest_entries  # noqa: E402

MAX_PARENT = args.max_parent_chars


def _section_path(chunk) -> list[str]:
    """child/parent 共用的展示路径：heading_path 优先，回落 chapter/section/article/title。"""
    s = chunk.metadata.structural
    if s.heading_path:
        return [p for p in s.heading_path if p]
    cand = [s.chapter, s.section, s.article]
    return [c for c in cand if c] or ["全文"]


def _group_parents(chunks: list) -> list[dict]:
    """按 (chapter, section, article, heading_path) 分组连续 chunk → 父段；超限切 part。"""
    parents: list[dict] = []
    groups: list[tuple[tuple, list]] = []
    for c in chunks:
        s = c.metadata.structural
        key = (s.chapter or "", s.section or "", s.article or "", tuple(s.heading_path or ()))
        if groups and groups[-1][0] == key:
            groups[-1][1].append(c)
        else:
            groups.append((key, [c]))
    for key, children in groups:
        sp = _section_path(children[0])
        buf: list[str] = []
        buf_children: list = []
        part = 0
        for c in children:
            t = c.text or ""
            if buf and len("\n\n".join(buf + [t])) > MAX_PARENT:
                parents.append(_mk_parent(sp, buf, buf_children, part))
                part += 1
                buf, buf_children = [], []
            buf.append(t)
            buf_children.append(c)
        parents.append(_mk_parent(sp, buf, buf_children, part))
    return parents


def _mk_parent(section_path: list[str], texts: list[str], children: list, part: int) -> dict:
    sp = section_path if part == 0 else section_path + [f"(part {part + 1})"]
    return {
        "section_path": sp,
        "content": "\n\n".join(texts),
        "metadata": {"parent_part": part, "child_count": len(children)},
        "children": [
            {
                "content": c.text or "",
                "section_path": _section_path(c),
                "metadata": {
                    "chunk_id": c.metadata.chunk_id,
                    "chunk_index": c.metadata.chunk_index,
                    "level": c.metadata.parent_child.chunk_level,
                    "contains_table": bool(c.metadata.table_context.contains_table),
                },
                "child_id": c.metadata.chunk_id,  # Task 20 契约：显式 chunk_id 优先（一致性对照锚点）
            }
            for c in children
        ],
    }


def main() -> int:
    entries = load_manifest_entries()
    entry = next(
        (e for e in entries
         if e.get("logical_document_id") == args.doc
         or Path(e.get("canonical_file", "")).name == args.doc),
        None,
    )
    if entry is None:
        print(f"[FAIL] manifest 中未找到文档: {args.doc}", file=sys.stderr)
        return 2

    loaded = dispatch_parse_and_clean(entry)
    if isinstance(loaded, tuple):
        print(f"[FAIL] 解析/清洗跳过: {loaded}", file=sys.stderr)
        return 2
    if "__CLEAN_FAIL_" in loaded.parser_version:
        print(
            f"[FAIL] stage=cleaner Cleaner V2 失败，禁止使用 fallback: {loaded.parser_version}",
            file=sys.stderr,
        )
        return 4

    chunks, _audit = chunk_strategy_b(loaded)

    violations: list[str] = []
    for c in chunks:
        vs = c.metadata.validate() or []
        violations.extend(f"{c.metadata.chunk_id}: {v}" for v in vs)
    if violations:
        print(f"[FAIL] Chunk Contract V1 校验失败 {len(violations)} 项（fail-fast）:", file=sys.stderr)
        for v in violations[:10]:
            print(f"  - {v}", file=sys.stderr)
        return 3

    logical_id = entry.get("logical_document_id") or Path(entry["canonical_file"]).stem
    # Never substitute the raw-file digest for the Cleaner-generated normalized hash.
    # The one-click wrapper computes this field through the established cleaner contract.
    sha = entry.get("normalized_sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in sha)
    ):
        print("[FAIL] stage=manifest 缺少有效 SHA256，拒绝构造 document_identity", file=sys.stderr)
        return 5
    sha = sha.lower()
    identity = f"{logical_id}{args.identity_suffix}@sha256:{sha}"
    package = {
        "document_identity": identity,
        "source_file": entry.get("canonical_file", ""),
        "identity_sha256": sha,
        "identity_sha256_kind": "normalized_sha256",
        "parents": _group_parents(chunks),
    }
    payload = json.dumps(package, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)

    n_children = sum(len(p["children"]) for p in package["parents"])
    print(
        f"[OK] {identity} | parents={len(package['parents'])} children={n_children} "
        f"chunks={len(chunks)} contract=PASS",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
