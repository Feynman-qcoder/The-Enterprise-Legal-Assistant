"""素材问答的图片识别模块（OpenSpec add-attachment-query）。

把用户上传的图片（合同照片、法条截图）交给多模态模型，提取出图片中已有的文字。

设计要点（都有实测依据，别改）：
1. **mime 按 magic bytes 探测，不信扩展名**——实测存在扩展名 .png 但内容为 WEBP 的文件
   （data_corpus/picture/学校承包合同.png），按扩展名传 mime 会被 API 拒。
2. **httpx 加 trust_env=False**——本机环境可能注入 HTTP(S)_PROXY，会让对 DashScope 的请求走坏代理。
3. **提示词只提取原文不总结**——提取文本要作为检索/上下文材料，总结会丢法条编号等命中信号；
   印章/手写签名标注占位符而不猜测内容（与 pdf_parser_v2 的 "no text fabricated" 原则同源）。
4. **可重试错误重试 1 次**——超时/5xx/429；已产出的场景不适用（本函数非流式，无此问题）。
5. **短文本判失败**——提取结果 < settings.attachment_min_text_chars 时抛 ExtractionError（fail-closed）。
"""

from __future__ import annotations

import base64
import logging
import re

import httpx

from modules.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 图片格式白名单：按 magic bytes 探测出的 mime 才允许进入识别
SUPPORTED_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})

# 前言剥离（任务 2.3 追加）：实测思考模式下模型仍会输出「以下是图片中提取的文字内容：」
# 这类开场白（A/B 三图中 2 张出现），prompt 层禁令不够强，必须在代码层兜底——
# 前言流进检索/提示词就是脏数据。
_PREAMBLE_RE = re.compile(r"^以下(是|为).{0,40}[:：]\s*$")

# 提取提示词：只取原文，不总结；禁止前言（实测模型会输出「以下是图片中提取的文字内容：」）
_EXTRACT_PROMPT = (
    "提取这张图片中的全部文字内容，遵守以下规则：\n"
    "1. 保持原有的段落与条款编号结构（例如「第七条」「（二）」「1.1」等），不要合并或省略编号；\n"
    "2. 表格转换为 Markdown 表格；\n"
    "3. 印章、手写签名处标注为 [印章] 或 [签名]，不要猜测其内容；\n"
    "4. 只输出图片中已有的文字，不要解释、总结、推断，也不要输出任何开场白（如「以下是…」）；\n"
    "5. 如果图片中没有可识别的文字，只输出「无文字」。"
)


class ExtractionError(ValueError):
    """素材提取失败（fail-closed）。code 供 API 层映射为可操作的错误提示。

    两个属性都必须有：端点层按 `exc.message` / `exc.code` 组装 SSE 错误事件
    （T4 实测曾因漏存 message 属性导致生成器崩溃、连接中断——见 test_endpoint_extraction_error_yields_sse_error）。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def detect_image_mime(data: bytes) -> str:
    """按 magic bytes 探测图片格式；非受支持图片返回空串。

    刻意不信任扩展名：实测存在 .png 后缀但内容为 WEBP 的文件。
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _is_retryable(exc: Exception) -> bool:
    """与 pipeline._is_retryable_llm_error 同口径：超时/5xx/429/连接失败才重试。"""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


def _strip_preamble(text: str) -> str:
    """剥离模型输出的开场白行（如「以下是图片中提取的文字内容：」）及其后的空行。

    只剥首行且必须匹配前言模式——正文中合法的「以下是…」句（如合同条款引用）不受影响。
    """
    lines = text.splitlines()
    if lines and _PREAMBLE_RE.match(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


async def describe_image(image_bytes: bytes, mime: str, settings: Settings | None = None) -> str:
    """图片 → 文本。调多模态模型（复用 DashScope 通道），失败重试 1 次。

    入参:
        image_bytes: 图片原始字节。
        mime: 由 detect_image_mime 探测出的真实 mime（不要用扩展名猜）。
        settings: 可选配置单例（默认 get_settings()）。
    返回:
        图片中已有的文字内容（保留条款编号结构）。
    抛出:
        ExtractionError: 识别失败（短文本/超时/非 200），fail-closed。
    """
    s = settings or get_settings()
    if mime not in SUPPORTED_IMAGE_MIMES:
        raise ExtractionError("UNSUPPORTED_IMAGE", "不支持的图片格式（按文件内容探测为 %s）" % (mime or "未知"))

    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": s.vision_model,
        "max_tokens": 2048,
        "temperature": 0,
        # 任务 2.4 A/B 结论（2026-09-19，三图对比）：**不开** enable_thinking=False——
        # 关思考虽省 18~54% 耗时且无前言，但印章图丢失 2 处 [签名] 占位（感知质量降级），
        # 按「质量不降级才关闭」规则保持默认（思考开）。数据与决策见 tasks.md 2.4 注记。
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _EXTRACT_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": "Bearer %s" % s.dashscope_api_key,
        "Content-Type": "application/json",
    }
    url = s.dashscope_base_url.rstrip("/") + "/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(2):  # 可重试错误重试 1 次
        try:
            # trust_env=False：绕开环境注入的代理变量，直连 DashScope
            async with httpx.AsyncClient(timeout=s.attachment_timeout_seconds, trust_env=False) as client:
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                raise httpx.HTTPStatusError(
                    "HTTP %d: %s" % (r.status_code, r.text[:200]),
                    request=r.request,
                    response=r,
                )
            body = r.json()
            text = (body["choices"][0]["message"].get("content") or "").strip()
            text = _strip_preamble(text)  # 前言剥离（任务 2.3 追加，实测思考模式下 prompt 禁令不够强）
            break
        except Exception as exc:  # noqa: BLE001 — 统一容错后按可重试性分派
            last_exc = exc
            if attempt == 0 and _is_retryable(exc):
                logger.warning("vision extract retrying after retryable error: %s", exc)
                continue
            raise ExtractionError("VISION_CALL_FAILED", "图片识别失败：%s" % exc) from exc

    if not text or text == "无文字" or len(text) < s.attachment_min_text_chars:
        raise ExtractionError(
            "IMAGE_TEXT_TOO_SHORT",
            "图片可能不清晰，未能提取到足够文字（%d 字符），请重新拍摄" % len(text or ""),
        )
    return text
