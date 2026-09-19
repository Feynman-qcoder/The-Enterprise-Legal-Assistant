"""素材问答端点（OpenSpec add-attachment-query）。

POST /api/chat/stream-with-attachment   (multipart: file + 可选 question)

链路：
  上传素材 → attachment.extract_text() → 上下文预算截断
    → SSE 推 transcription 事件（提取文本/类型/规模/截断标记/检索 query/实际问题）
    → （有提问用提问，无提问用缺省指令）作为 query 走现有检索
    → pipeline.stream_chat(query, attachment_context=提取文本)（检索链路零改动）
    → SSE 推回答分片

设计约束（spec 固化，勿违反）：
- ATTACHMENT_ENABLED=false → 503（feature flag，随时回滚）
- 素材不进对话历史（仅提取文本进）；不写 Milvus/MySQL
- 提取失败 fail-closed：明确错误提示，不进入生成
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.api.chat import persist_user_turn  # 复用现有聊天记录持久化
from backend.app.deps import enforce_rate_limit, get_pipeline  # 限流 + 单例 RagPipeline（与 /stream 同源）
from modules.core.config import get_settings  # 配置单例（含 attachment_* / citation_check_* 字段）
from modules.rag.attachment import ATTACHMENT_EXTENSIONS, extract_text
from modules.rag.citation_check import run_citation_check  # 引用核查（观察模式，fail-open；素材路径证据=检索上下文+素材文本）
from modules.rag.query_refine import refine_retrieval_query  # 无提问时的检索要点提炼（spec D10）
from modules.rag.vision import ExtractionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat-attachment"])

# 用户未提问时的缺省分析指令（spec：只上传不提问 → 用明确指令驱动生成，且透出给用户）
DEFAULT_QUESTION = "请概述这份材料的主要内容、关键条款与主要法律风险点。"


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# 入库侧的上限（incremental.MAX_UPLOAD_BYTES=20MB）比素材侧 attachment_max_bytes(默认10MB) 宽，
# 且 knowledge._read_limited 从不抛 413（超限只静默截断）——直接复用会让 10~20MB 素材绕过限制。
# 故按 HANDOFF T1 的修法在本地实现：逻辑照抄 _read_limited，换成 attachment_max_bytes 并真正抛 413。
async def _read_attachment_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """分块读取素材字节；超过 max_bytes 抛 413（spec：超限必须显式拒绝，不能静默截断）。"""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            await upload.close()
            raise HTTPException(
                status_code=413,
                detail=f"文件过大：素材不得超过 {max_bytes // (1024 * 1024)} MB",
            )
    await upload.close()
    return b"".join(chunks)


@router.post("/stream-with-attachment", dependencies=[Depends(enforce_rate_limit)])
async def chat_stream_with_attachment(
    file: UploadFile = File(...),
    question: str = Form(default=""),
    pipeline=Depends(get_pipeline),
) -> StreamingResponse:
    """素材问答：上传图片/文件 + 可选问题 → 提取文本作上下文 → 现有检索 + 生成。"""
    settings = get_settings()
    if not settings.attachment_enabled:  # feature flag：默认关，随时回滚
        raise HTTPException(status_code=503, detail="图片/文件功能未开启")

    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in ATTACHMENT_EXTENSIONS):
        raise HTTPException(status_code=415, detail="不支持的文件类型；支持图片 jpg/jpeg/png/webp 与文档 pdf/docx/doc/txt/md")

    content = await _read_attachment_limited(file, settings.attachment_max_bytes)  # 超限 → 413

    async def gen():
        # ---- 提取（fail-closed）----
        try:
            extracted = await extract_text(filename, content, settings=settings)
        except ExtractionError as exc:
            yield _sse({"error": exc.message, "code": exc.code})
            return
        except Exception as exc:  # noqa: BLE001 — 兜底：任何提取异常都转成明确错误
            logger.exception("attachment extraction failed")
            yield _sse({"error": "素材处理失败：%s" % exc, "code": "EXTRACT_FAILED"})
            return

        # ---- 确定问题与检索 query（spec D10：素材全文绝不作检索 query）----
        user_question = (question or "").strip()
        if user_question:
            # 有提问 → query = 用户问题（自然语言，天然合规）
            effective_question = user_question
            retrieval_query = user_question
            used_default = False
        else:
            # 无提问 → 缺省指令驱动生成；检索 query 先尝试从素材提炼要点，
            # 提炼失败则回退缺省问题本身（跳过检索比用全文检索更可取——spec D10）
            effective_question = DEFAULT_QUESTION
            refined = await refine_retrieval_query(extracted.text, settings=settings)
            retrieval_query = refined or DEFAULT_QUESTION
            used_default = True

        # ---- transcription 事件：先透出提取结果，供用户核对 ----
        yield _sse({
            "type": "transcription",
            "kind": extracted.kind,                       # image | document
            "mime": extracted.detected_mime,              # 图片真实格式（文档为空）
            "chars": extracted.original_chars,            # 截断前字符数
            "truncated": extracted.truncated,             # 是否超预算截断
            "truncation_note": "文档较长，仅使用前部分内容" if extracted.truncated else "",
            "text_preview": extracted.text[:2000],        # 提取文本预览（完整文本进提示词，不整段刷屏）
            "retrieval_query": retrieval_query,           # 实际用于检索的 query（无提问时为提炼要点或缺省问题）
            "effective_question": effective_question,     # 实际使用的问题（缺省时让用户看到）
            "used_default_question": used_default,
        })

        # ---- 生成：检索链路零改动，素材只进提示词 ----
        # 引用核查证据快照（add-citation-check）：开关关闭时 None（行为与改造前逐字节一致）
        evidence: list[str] | None = [] if settings.citation_check_enabled else None
        buf: list[str] = []
        try:
            async for piece in pipeline.stream_chat(
                retrieval_query,
                None,
                attachment_context=extracted.text,
                evidence_out=evidence,
            ):
                buf.append(piece)
                yield _sse({"chunk": piece})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"error": str(exc), "code": "GENERATION_FAILED"})
            return

        # ---- 引用核查（观察模式；spec 双路径覆盖：证据 = 检索上下文 + 素材文本）----
        # 素材文本 append 进证据列表 → 回答里的合同条款引用对素材验证（spec「素材条款引用」场景）；
        # fail-open（红线 2）：核查器异常只记日志跳过事件，答案流已完整输出，不受影响。
        if evidence is not None:
            evidence.append(extracted.text)
            try:
                report, _elapsed_ms = run_citation_check(
                    "".join(buf), evidence, settings.citation_check_text_threshold,
                )
                if report is not None:
                    yield _sse(report.to_event())
            except Exception:  # noqa: BLE001 — fail-open
                logger.warning("citation check failed on attachment path (fail-open, event skipped)", exc_info=True)
        yield "data: [DONE]\n\n"

        # 聊天记录：存提取文本（不存素材原始数据——spec 铁律）
        full_answer = "".join(buf)
        try:
            await persist_user_turn(None, effective_question, full_answer)
        except Exception:
            logger.exception("persist attachment chat history failed")

    return StreamingResponse(gen(), media_type="text/event-stream")
