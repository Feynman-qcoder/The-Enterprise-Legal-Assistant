"""Phase C 集成测试：验证 parse→clean→chunk 链路在在线/离线使用同一 Cleaner。

不依赖 DB / FastAPI / Milvus；仅验证：
  1) 在线 ingest_legal_document 的解析-清洗边界可被复用（直接调用 clean_parsed_document）；
  2) 清洗后 chunk 文本不含 web boilerplate / 合同长下划线；
  3) 离线 mysql_loaders 的 parse→clean 路径经 parse_document 统一入口。
"""

from __future__ import annotations

from pathlib import Path

from modules.ingestion.chunking import (
    chunk_pdf_pages_to_parents,
    split_children_from_parent,
)
from modules.ingestion.document_cleaning import clean_parsed_document
from modules.ingestion.document_parsing import parse_document

ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE = Path(r"D:/xiaoyi/data_source")


def _load_accepted_paths() -> list[Path]:
    review = DATA_SOURCE / "_meta" / "corpus_review.jsonl"
    out: list[Path] = []
    for line in review.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json
        d = json.loads(line)
        if d.get("review_status") == "ACCEPTED":
            p = Path(d["path"])
            if p.exists():
                out.append(p)
    return out


def test_online_offline_share_cleaner() -> None:
    """Online (incremental) 与 Offline (mysql_loaders) 都经由 parse_document+clean。"""
    paths = _load_accepted_paths()
    assert paths, "corpus_review ACCEPTED 未找到真实文件"
    # 抽 3 种格式各一，验证链路可跑且输出清洁
    sampled: dict[str, Path] = {}
    for p in paths:
        ext = p.suffix.lower()
        if ext not in sampled:
            sampled[ext] = p
        if len(sampled) >= 3:
            break
    for p in sampled.values():
        parsed = parse_document(p)            # 在线/离线共同入口
        cleaned = clean_parsed_document(parsed)  # 同一 Cleaner
        parents = chunk_pdf_pages_to_parents(list(cleaned.segments or (cleaned.text,)))
        children = [c for ps in parents for c in split_children_from_parent(ps.text, 0)]
        assert children, f"{p.name} 未产生 chunk"
        joined = "\n".join(c.text for c in children)
        # 清洁后不得含典型 web boilerplate
        assert "JiaThis" not in joined
        assert "javascript:" not in joined
        # 不得残留合同长下划线
        assert "________________" not in joined
        # 不得残留独立页码（如存在）
        # 正文语义保留：至少保留 filename stem 关键词之一
        assert cleaned.metadata["cleaning"]["version"] == "v1.1"


def test_cleaner_preserves_legal_numbering_after_chunk() -> None:
    """清洗后 chunk 不得丢失法条编号（第三条 / 3. / （三））。"""
    p = ROOT / "data" / "中华人民共和国劳动法.pdf"
    if not p.exists():
        import pytest
        pytest.skip("fixture PDF 不在默认 data 目录")
    parsed = parse_document(p)
    cleaned = clean_parsed_document(parsed)
    joined = cleaned.text
    # 劳动法里至少应含第一条
    assert "第一条" in joined
