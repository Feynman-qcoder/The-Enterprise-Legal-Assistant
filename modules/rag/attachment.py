"""素材问答的分派与预算层（OpenSpec add-attachment-query）。

职责（按顺序）：
1. 按「magic bytes + 扩展名」判定素材类型：图片 → vision.describe_image；文档 → document_parsing.parse_document
2. 「无有效正文」判定：解析结果若主要由扫描件/纯图页占位符构成，判为提取失败（fail-closed，
   与 pdf_parser_v2 的 "no text fabricated" 原则同源——占位符长度可能超过阈值，不判会把垃圾送进提示词）
3. 上下文预算：提取文本超过 attachment_context_max_chars 时截断，并返回截断标记（供 transcription 透出）

不修改 document_parsing.py 的任何现有行为——本模块只调用它。
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from modules.core.config import Settings, get_settings
from modules.ingestion.document_parsing import (
    SUPPORTED_EXTENSIONS,
    DocumentParseError,
    parse_document,
    parser_available,
)
from modules.rag.vision import ExtractionError, describe_image, detect_image_mime

logger = logging.getLogger(__name__)

# 图片扩展名（用于前端 accept 与第一层白名单；真实格式仍按 magic bytes 判）
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# 素材总白名单 = 图片 + 文档
ATTACHMENT_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_EXTENSIONS)

# 「无有效正文」判据：解析结果中占位符特征行占比过高即视为无有效正文。
# pdf_parser_v2 对扫描件/纯图页插入的占位符含这些特征（见 W_SCANNED / "[Placeholder: page ... appears scanned"）。
_PLACEHOLDER_MARKERS = re.compile(r"Placeholder.*scanned|image_only_page|SCANNED_OR_IMAGE_ONLY", re.IGNORECASE)


@dataclass
class ExtractedAttachment:
    """一次素材提取的完整结果（供端点组装 transcription 事件与提示词）。"""

    text: str                 # 提取出的文本（已按预算截断）
    kind: str                 # "image" | "document"
    detected_mime: str        # 图片的真实 mime（文档类为空串）
    ext: str                  # 上传文件的扩展名（小写）
    original_chars: int       # 截断前的字符数
    truncated: bool           # 是否因超预算被截断
    used_queries: list[str]   # 实际用于检索的 query（由端点填充；本模块不管检索）


def _has_effective_body(text: str, min_chars: int) -> bool:
    """判定解析结果是否含有效正文（排除占位符主导的结果）。

    规则：去掉占位符特征行后，剩余有效字符数仍须 ≥ min_chars。
    防止「扫描件 PDF 的占位符文本长度超过阈值」被误判为有效提取（实测占位符是完整英文句子）。
    """
    lines = text.splitlines()
    effective = "\n".join(l for l in lines if not _PLACEHOLDER_MARKERS.search(l))
    return len(effective.strip()) >= min_chars


def _truncate_for_context(text: str, max_chars: int) -> tuple[str, bool]:
    """按上下文预算截断；返回 (截断后文本, 是否截断)。"""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


async def extract_text(filename: str, content: bytes, settings: Settings | None = None) -> ExtractedAttachment:
    """素材 → 提取文本（统一入口）。

    入参:
        filename: 上传文件名（只取扩展名做第一层分派参考；图片真实格式按 magic bytes）
        content: 文件原始字节
        settings: 可选配置单例
    返回:
        ExtractedAttachment（已按上下文预算截断，含截断标记）
    抛出:
        ExtractionError: 各类 fail-closed 失败（格式不支持/无有效正文/识别失败等）
    """
    s = settings or get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in ATTACHMENT_EXTENSIONS:
        raise ExtractionError("UNSUPPORTED_FORMAT", "不支持的文件类型：%s（支持图片 jpg/png/webp 与文档 pdf/docx/doc/txt/md）" % ext)

    # ---- 图片路径：magic bytes 优先 ----
    detected_mime = detect_image_mime(content)
    if detected_mime:
        # 扩展名与实际不符时按实际格式处理（实测：.png 后缀的 WEBP）
        if ext in IMAGE_EXTENSIONS and ext != ".webp" and detected_mime != "image/" + ext.lstrip("."):
            logger.info("attachment ext/mime mismatch: ext=%s actual=%s（按实际格式处理）", ext, detected_mime)
        text = await describe_image(content, detected_mime, settings=s)
        original = len(text)
        final, truncated = _truncate_for_context(text, s.attachment_context_max_chars)
        return ExtractedAttachment(
            text=final, kind="image", detected_mime=detected_mime, ext=ext,
            original_chars=original, truncated=truncated, used_queries=[],
        )

    # 非图片扩展名但内容探测为图片：也走图片路径（否则会被文档解析器当垃圾拒掉）
    if ext in IMAGE_EXTENSIONS:
        # 扩展名像图片但 magic bytes 认不出 → 内容损坏或非图片
        raise ExtractionError("BAD_IMAGE", "图片文件无法识别（内容与扩展名 %s 不符且探测不到已知图片格式）" % ext)

    # ---- 文档路径：复用现有解析器门面（不修改 document_parsing.py）----
    if ext in SUPPORTED_EXTENSIONS and not parser_available(ext):
        raise ExtractionError("PARSER_UNAVAILABLE", "该格式（%s）的解析依赖当前不可用（如 .doc 需要 LibreOffice）" % ext)

    with tempfile.TemporaryDirectory(prefix="attachment_") as td:
        tmp = Path(td) / ("upload" + ext)
        tmp.write_bytes(content)
        try:
            doc = parse_document(tmp)
        except DocumentParseError as exc:
            raise ExtractionError("DOCUMENT_PARSE_FAILED", "文档解析失败：%s" % exc) from exc
        text = getattr(doc, "text", "") or ""
        text = text.strip()

    if len(text) < s.attachment_min_text_chars:
        raise ExtractionError(
            "DOCUMENT_TEXT_TOO_SHORT",
            "未能从该文档提取到有效文字（%d 字符）" % len(text),
        )
    if not _has_effective_body(text, s.attachment_min_text_chars):
        raise ExtractionError(
            "NO_EFFECTIVE_BODY",
            "该 PDF 可能是扫描件（无文本层），暂无可提取的文字内容",
        )

    original = len(text)
    final, truncated = _truncate_for_context(text, s.attachment_context_max_chars)
    return ExtractedAttachment(
        text=final, kind="document", detected_mime="", ext=ext,
        original_chars=original, truncated=truncated, used_queries=[],
    )
