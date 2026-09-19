# =============================================================================
# 教学说明：本文件在整体链路中的位置
# -----------------------------------------------------------------------------
# 输入：HTTP POST JSON 体（`ChatRequest`：`message`、`user_external_id` 可选）。
# 输出：`StreamingResponse`，`media_type=text/event-stream`，body 为 SSE 文本流。
# 被谁调用：浏览器或前端 `fetch("/api/chat/stream")`；路由由 `main.py` 挂载到 `/api` 前缀下。
# =============================================================================
"""
核心作用：
接收前端的聊天请求 → 调用 RAG（检索增强生成）管线流式返回回答 → 回答流结束后把聊天记录存入 MySQL → 若过程出错则给前端返回错误信息。
"""

from __future__ import annotations

import json  # dumps 把 dict 变成 JSON 字符串嵌入 SSE
import logging

from fastapi import APIRouter, Depends  # APIRouter 分组路由；Depends 注入依赖
from fastapi.responses import StreamingResponse  # 流式响应类

from backend.app.deps import enforce_rate_limit, get_pipeline  # 单例 RagPipeline 提供者 + 限流 dependency（Task 21 Part 2）
from backend.app.schemas import ChatHistoryItem, ChatHistoryResponse, ChatRequest  # Pydantic：校验 JSON 字段类型与长度
from modules.database.session import get_session_factory  # 历史回显只读会话（方案 A）
from modules.memory.service import fetch_chat_history  # 只读历史查询（勿用 resolve_user_id：其会自动建档，违反"不存在→空列表"契约）
from modules.core.config import get_settings  # 引用核查开关与阈值（add-citation-check）
from modules.rag.citation_check import run_citation_check  # 观察模式核查：只报告不改答案，异常 fail-open
from sqlalchemy import select  # 用户存在性检查（不存在则空列表）

from modules.database.models import UserTab  # users_tab ORM（存在性探测）
from modules.memory.service import persist_user_turn  # 写 his_chat_tab
from modules.rag.pipeline import RagPipeline  # 类型注解

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])  # 最终路径 = /api + /chat + /stream


@router.post("/stream", dependencies=[Depends(enforce_rate_limit)])  # Task 21 Part 2：滑动窗口限流（fail-open；/health 不挂载天然豁免）
async def chat_stream(
    body: ChatRequest,
    pipeline: RagPipeline = Depends(get_pipeline), # 依赖注入：`pipeline` 由框架调用 `get_pipeline()` 填入。
) -> StreamingResponse:
    """
    异步生成器 `gen`：迭代 `pipeline.stream_chat`，把每个片段包装成 SSE 行。

    依赖注入：`pipeline` 由框架调用 `get_pipeline()` 填入。

    入参:
        body: 已校验的聊天请求体（message、可选 user_external_id）。
        pipeline: 注入的 RAG 管线单例。
    返回:
        `StreamingResponse`，MIME 为 `text/event-stream`，body 来自内部 `gen()`。
    """

    async def gen():
        """
        定义内部异步生成器，负责调用 RAG 管线流式获取回答、包装成 SSE 格式返回、处理异常、存聊天记录。。

        入参:
            无（使用外层 `body` 与 `pipeline`）。
        返回:
            异步迭代器，逐项产出 `data: ...\\n\\n` 形式的 SSE 字符串；结束时尝试持久化聊天记录。
        """
        buf: list[str] = []  # 创建空列表，用于累积 RAG 管线返回的每一段回答（比如 AI 先返回 “你”，再返回 “好”，buf 会存 ["你", "好"]）；
        # 引用核查证据快照（add-citation-check，design D2）：开关关闭时为 None——
        # 不收集、不核查、无事件，SSE 输出与改造前逐字节一致（红线 4/5）
        settings = get_settings()
        evidence: list[str] | None = [] if settings.citation_check_enabled else None
        try:
            #pipeline.stream_chat 是 RAG 管线的流式聊天方法（内部会先检索知识库，再逐段生成回答）
            #piece：每一次迭代拿到的 “回答片段”（比如一个字、一个词）
            async for piece in pipeline.stream_chat(body.message, body.user_external_id, evidence_out=evidence):
                buf.append(piece)  # 把片段存到列表，后续拼接完整回答；
                #json.dumps(...)：把 {"chunk": "回答片段"} 转成 JSON 字符串（确保中文不转义，ensure_ascii=False）；
                #yield：生成器关键字，每执行一次 yield，就会把这一行 SSE 格式的字符串返回给前端；
                #格式要求：data: ...\n\n 是 SSE 协议强制的，前端的 EventSource 才能识别；
                yield f"data: {json.dumps({'chunk': piece}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 — 任意异常转成 JSON 事件给前端展示
            err = {"error": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"  #把错误返回给前端，前端可以显示 “服务出错”；
            return  # 出错不再 persist，避免存半截答案
        full_answer = "".join(buf)  # 完整助手回复，把之前累积的每一段回答片段拼接成完整的回答文本；
        # ---- 引用核查（观察模式，spec 时序：…chunk… → citation_report → [DONE]）----
        # 整体 try/except fail-open（红线 2）：核查器任何异常 → 日志 + 跳过事件，绝不能影响答案流。
        # evidence 为空列表（缓存命中等无本次证据的路径）→ run_citation_check 返回 None → 不发事件
        #（对没有本次证据的答案发报告只会产生系统性误导——design D1 的 grounding 定义所限）。
        if evidence is not None:
            try:
                report, _elapsed_ms = run_citation_check(
                    full_answer, evidence, settings.citation_check_text_threshold,
                )
                if report is not None:
                    yield f"data: {json.dumps(report.to_event(), ensure_ascii=False)}\n\n"
            except Exception:  # noqa: BLE001 — fail-open：跳过事件，流照常收尾
                logger.warning("citation check failed (fail-open, event skipped)", exc_info=True)
        yield "data: [DONE]\n\n"  # 所有回答片段与核查事件返回完后，给前端发一个 [DONE] 标识，前端收到后就知道流结束了
        try:
            await persist_user_turn(body.user_external_id, body.message, full_answer)  # 存聊天记录，无 external_id 时内部 return
        except Exception:
            logger.exception("persist chat history failed")  # 落库失败打栈，但不影响已发送的 SSE，前端会显示 “服务出错”；

    #把异步生成器 gen() 传给 StreamingResponse，FastAPI 会自动迭代 gen()，把每一次 yield 的内容返回给前端；
    #media_type="text/event-stream"：告诉浏览器这是 SSE 格式的流式响应，浏览器会用 EventSource 或 fetch 逐行接收。
    return StreamingResponse(gen(), media_type="text/event-stream")  


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(
    user_external_id: str | None = None,
    limit: int = 50,
) -> ChatHistoryResponse:
    """
    历史回显只读接口（方案 A，插入任务）：返回该用户最近聊天记录（时间升序）。

    契约：
    - user_external_id 缺失 / 用户不存在 → 200 + 空列表（不报错，与匿名现状对齐）
    - limit 钳制 [1, 200]，默认 50
    - 不挂限流（只读轻接口，与 /health 同哲学）
    - 查询异常不自行兜底：冒泡给 Task 21 全局异常处理器（结构化 500）

    入参:
        user_external_id: query 参数，前端 localStorage UUID。
        limit: 最多返回条数（默认 50，上限 200）。
    返回:
        `ChatHistoryResponse`：{items: [{question, answer, created_at}...], count}。
    """
    ext = (user_external_id or "").strip()
    if not ext:
        return ChatHistoryResponse(items=[], count=0)  # 匿名：空历史，200
    limit = max(1, min(limit, 200))  # 钳制上限

    factory = get_session_factory()
    async with factory() as session:
        res = await session.execute(select(UserTab).where(UserTab.external_id == ext))
        row = res.scalar_one_or_none()
        if row is None:
            return ChatHistoryResponse(items=[], count=0)  # 用户不存在：空历史，200（不自动建档）
        items = await fetch_chat_history(session, int(row.id), limit)
    return ChatHistoryResponse(items=[ChatHistoryItem(**item) for item in items], count=len(items))
