# =============================================================================
# V2 文档级增量同步编排脚本（Task 19 Part A：A3）。
# -----------------------------------------------------------------------------
# 输入：单文档 JSON 包路径（--package）或待删 document_identity（--delete）。
# 输出：统计 JSON（MySQL/Milvus 各计数 + 新 corpus_version）打印到 stdout。
# 被谁调用：运维手工执行（python offline/scripts/run_v2_doc_sync.py --package xxx.json）。
# 用法（在仓库根目录、xiaoyi_rag conda 环境下）：
#   python offline/scripts/run_v2_doc_sync.py --package path/to/doc_package.json
#   python offline/scripts/run_v2_doc_sync.py --delete <document_identity>
# =============================================================================
"""
「更新一篇文档」= 跑一次本脚本（文档级 replace：先删后插 + 版本 +1）。

包格式（JSON）：
{
  "document_identity": "TEST_DOC_001@sha256:...",   # 必填，唯一
  "source_file": "TEST_DOC_001.md",                  # 必填
  "parents": [                                        # 必填，≥1
    {
      "section_path": ["第一章", "第一条"],            # 必填，list[str]
      "content": "父段全文...",                        # 必填，≤8000 字符（在线契约）
      "metadata": {"title": "第一条", ...},           # 可选 dict
      "children": [                                   # 必填，≥1
        {
          "child_id": "0123456789abcdef0123456789abcdef", # 可选；有值时保留
          "content": "子块文本", "section_path": [...], "metadata": {...}
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 仓库根进 sys.path

from modules.core.config import get_settings  # noqa: E402
from modules.ingestion.v2_incremental import (  # noqa: E402
    V2ChildSpec,
    V2DocumentPackage,
    V2ParentSpec,
    delete_document,
    ensure_corpus_meta_table,
    read_corpus_version,
    replace_document,
)

MAX_PARENT_CONTENT_CHARS = 8000  # 与在线契约 v2_parent_repository.MAX_V2_PARENT_CONTENT_CHARS 对齐


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"package field {field_name!r} must be a non-empty string")
    return value


def _require_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"package field {field_name!r} must be a non-empty list of strings")
    return value


def _optional_child_id(value: Any, field_name: str) -> str | None:
    """Validate an optional persisted child identity without generating a new ID."""
    if value is None:
        return None
    child_id = _require_str(value, field_name)
    if len(child_id) != 32 or any(char not in "0123456789abcdef" for char in child_id):
        raise ValueError(
            f"package field {field_name!r} must be a lowercase 32-character hex string"
        )
    return child_id


def parse_package(raw: dict[str, Any]) -> V2DocumentPackage:
    """解析并校验文档包结构；任何字段非法立即抛 ValueError（fail-fast）。"""
    document_identity = _require_str(raw.get("document_identity"), "document_identity")
    source_file = _require_str(raw.get("source_file"), "source_file")
    parents_raw = raw.get("parents")
    if not isinstance(parents_raw, list) or not parents_raw:
        raise ValueError("package field 'parents' must be a non-empty list")

    parents: list[V2ParentSpec] = []
    for p_idx, parent_raw in enumerate(parents_raw):
        if not isinstance(parent_raw, dict):
            raise ValueError(f"parents[{p_idx}] must be an object")
        section_path = _require_str_list(
            parent_raw.get("section_path"), f"parents[{p_idx}].section_path"
        )
        content = _require_str(parent_raw.get("content"), f"parents[{p_idx}].content")
        if len(content) > MAX_PARENT_CONTENT_CHARS:
            raise ValueError(
                f"parents[{p_idx}].content exceeds {MAX_PARENT_CONTENT_CHARS} chars "
                "(online contract)"
            )
        metadata = parent_raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"parents[{p_idx}].metadata must be an object")
        children_raw = parent_raw.get("children")
        if not isinstance(children_raw, list) or not children_raw:
            raise ValueError(f"parents[{p_idx}].children must be a non-empty list")
        children: list[V2ChildSpec] = []
        for c_idx, child_raw in enumerate(children_raw):
            if not isinstance(child_raw, dict):
                raise ValueError(f"parents[{p_idx}].children[{c_idx}] must be an object")
            child_content = _require_str(
                child_raw.get("content"),
                f"parents[{p_idx}].children[{c_idx}].content",
            )
            child_section = _require_str_list(
                child_raw.get("section_path"), f"parents[{p_idx}].children[{c_idx}].section_path"
            )
            child_metadata = child_raw.get("metadata") or {}
            if not isinstance(child_metadata, dict):
                raise ValueError(f"parents[{p_idx}].children[{c_idx}].metadata must be an object")
            child_id = _optional_child_id(
                child_raw.get("child_id"),
                f"parents[{p_idx}].children[{c_idx}].child_id",
            )
            children.append(
                V2ChildSpec(
                    content=child_content,
                    section_path=child_section,
                    metadata=child_metadata,
                    child_id=child_id,
                )
            )
        parents.append(
            V2ParentSpec(
                section_path=section_path,
                content=content,
                metadata=metadata,
                children=children,
            )
        )
    return V2DocumentPackage(
        document_identity=document_identity, source_file=source_file, parents=parents
    )


async def sync_package(
    package: V2DocumentPackage,
    collection_name: str,
    *,
    embedding_service: Any | None = None,
) -> dict[str, Any]:
    """Reuse the document-sync write path and return versioned operation statistics."""
    await ensure_corpus_meta_table()
    version_before = await read_corpus_version()
    stats = await replace_document(package, collection_name, embedding_service)
    stats["version_before"] = version_before
    stats["collection"] = collection_name
    return stats


async def sync_delete(document_identity: str, collection_name: str) -> dict[str, Any]:
    """Reuse the document-sync delete path, including a corpus-version bump."""
    await ensure_corpus_meta_table()
    version_before = await read_corpus_version()
    stats: dict[str, Any] = await delete_document(document_identity, collection_name)
    from modules.ingestion.v2_incremental import bump_corpus_version

    stats["new_corpus_version"] = await bump_corpus_version()
    stats["version_before"] = version_before
    stats["collection"] = collection_name
    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description="V2 document-level incremental sync")
    parser.add_argument(
        "--package", type=str, help="path to document package JSON (replace semantics)"
    )
    parser.add_argument("--delete", type=str, help="document_identity to delete")
    args = parser.parse_args()
    if bool(args.package) == bool(args.delete):
        parser.error("exactly one of --package / --delete is required")

    settings = get_settings()
    collection_name = settings.retrieval_v2_collection

    if args.delete:
        stats = await sync_delete(args.delete, collection_name)
    else:
        with open(args.package, encoding="utf-8") as fh:
            raw = json.load(fh)
        package = parse_package(raw)
        stats = await sync_package(package, collection_name)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
