"""素材问答（add-attachment-query）的单元测试。

核心断言（任务 3.1 的验收标准）：
1. build_user_message 不传 attachment_context 时，输出与改造前【逐字节一致】
2. attachment_context 非空时注入【用户上传的材料】段
3. attachment._has_effective_body 对扫描件占位符判失败
4. attachment._truncate_for_context 预算截断
5. vision.detect_image_mime 的 magic bytes 探测（含 WEBP 伪装 PNG 的实测场景）
6. 端点层的 feature flag / 格式白名单（503 / 415）

运行：pytest tests/test_attachment_query.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.rag.prompts import build_user_message  # noqa: E402


# ---------------------------------------------------------------------------
# 1) 逐字节一致（向后兼容红线）
# ---------------------------------------------------------------------------

def test_build_user_message_unchanged_without_attachment():
    """不传 attachment_context（默认空）时，输出必须与改造前的实现逐字节一致。"""
    question = "违约金比例的上限是多少？"
    contexts = ["[片段原文A] 第七条 违约金……", "[片段原文B] 第五百八十五条……"]
    memory = "用户此前问过保管合同的注意事项。"

    # —— 改造前实现的逐字复刻（作为黄金基准；与 git 历史中的旧版一致）——
    q = question + "\n\n" + (
        "【以下为该用户在系统中的最近若干条问答记录（按时间从早到晚），本轮回复均可作为上下文参考（含闲聊引导与专业作答）。"
        "若用户询问与往期对话相关的内容请据实依据记录；勿编造记录中不存在的内容。】\n" + memory
    )
    blocks = "\n\n".join(f"[片段{i+1}]\n{c}" for i, c in enumerate(contexts))
    golden = (
        f"用户问题：{q}\n\n"
        "作答提示：若参考资料中已有可直接回答该问题的原文或条文，请优先忠实引用该部分，"
        "避免冗长铺垫与过度归纳；仅在必要时用一两句话补充。\n\n"
        f"参考资料：\n{blocks}"
    )

    actual = build_user_message(question, contexts, memory)  # 不传 attachment_context
    assert actual == golden, "默认路径的提示词与改造前不一致——违反向后兼容红线"


def test_build_user_message_unchanged_without_attachment_no_memory():
    """无记忆、无素材时同样逐字节一致。"""
    actual = build_user_message("问题", ["片段A"])
    golden = (
        "用户问题：问题\n\n"
        "作答提示：若参考资料中已有可直接回答该问题的原文或条文，请优先忠实引用该部分，"
        "避免冗长铺垫与过度归纳；仅在必要时用一两句话补充。\n\n"
        "参考资料：\n[片段1]\n片段A"
    )
    assert actual == golden


def test_build_user_message_injects_attachment():
    """attachment_context 非空时注入【用户上传的材料】段，且位于作答提示与参考资料之间。"""
    out = build_user_message("这份合同违约金怎么约的", ["片段A"], attachment_context="第八条 违约金为合同总价的 20%。")
    assert "【用户上传的材料】" in out
    assert "第八条 违约金为合同总价的 20%。" in out
    # 顺序：用户问题 < 作答提示 < 用户上传的材料 < 参考资料
    assert out.index("用户问题：") < out.index("作答提示：") < out.index("【用户上传的材料】") < out.index("参考资料：")


def test_build_user_message_attachment_empty_string_same_as_default():
    """显式传空串 == 不传（空串拼接不产生任何额外字符）。"""
    assert build_user_message("q", ["c"], attachment_context="") == build_user_message("q", ["c"])


# ---------------------------------------------------------------------------
# 2) 无有效正文判定（扫描件占位符）
# ---------------------------------------------------------------------------

def test_has_effective_body_rejects_placeholder_only():
    """解析结果全是扫描件占位符 → 判无有效正文（实测占位符是完整英文句子，长度可超阈值）。"""
    from modules.rag.attachment import _has_effective_body

    placeholder_text = (
        "[Placeholder: page 1 appears scanned/image-only (V1 no OCR)]\n"
        "[Placeholder: page 2 appears scanned/image-only (V1 no OCR)]\n"
        "[Placeholder: page 3 appears scanned/image-only (V1 no OCR)]\n"
    ) * 5  # 长度远超 20，但没有一行是有效正文
    assert _has_effective_body(placeholder_text, 20) is False


def test_has_effective_body_accepts_normal_text():
    """正常文档（有效行为主）→ 判有正文。"""
    from modules.rag.attachment import _has_effective_body

    normal = "第一条 合同标的。\n第二条 价款与支付方式。\n第三条 交付时间与地点。"
    assert _has_effective_body(normal, 20) is True


def test_has_effective_body_mixed_placeholder_minority():
    """占位符是少数、正文为主 → 仍判有正文（一本 PDF 个别页是扫描件不应整体判死）。"""
    from modules.rag.attachment import _has_effective_body

    mixed = (
        "第一条 合同标的。双方就生鲜乳购销事宜达成如下协议。\n"
        "第二条 价款与支付方式：按月结算，货到付款。\n"
        "[Placeholder: page 3 appears scanned/image-only (V1 no OCR)]\n"
        "第四条 违约责任：任何一方违约应承担相应赔偿责任。\n"
        "第五条 争议解决：提交仲裁委员会仲裁。\n"
    )
    assert _has_effective_body(mixed, 20) is True


# ---------------------------------------------------------------------------
# 3) 上下文预算截断
# ---------------------------------------------------------------------------

def test_truncate_no_op_under_budget():
    from modules.rag.attachment import _truncate_for_context

    text = "短文本"
    out, truncated = _truncate_for_context(text, 60000)
    assert out == text and truncated is False


def test_truncate_cuts_over_budget():
    from modules.rag.attachment import _truncate_for_context

    text = "字" * 100
    out, truncated = _truncate_for_context(text, 60)
    assert len(out) == 60 and truncated is True


# ---------------------------------------------------------------------------
# 4) magic bytes 探测（含实测场景：.png 后缀的 WEBP）
# ---------------------------------------------------------------------------

def test_detect_image_mime_formats():
    from modules.rag.vision import detect_image_mime

    assert detect_image_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 16) == "image/jpeg"
    assert detect_image_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16) == "image/png"
    assert detect_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8) == "image/webp"
    assert detect_image_mime(b"%PDF-1.7") == ""            # PDF 不是图片
    assert detect_image_mime(b"plain text bytes") == ""    # 普通文本不是图片


# ---------------------------------------------------------------------------
# 5) 配置字段与默认值
# ---------------------------------------------------------------------------

def test_attachment_settings_defaults():
    from modules.core.config import get_settings

    s = get_settings()
    assert s.attachment_enabled is False                       # 默认关（feature flag）
    assert s.vision_model == "deepseek-v4.1-flash"
    assert s.attachment_max_bytes == 10 * 1024 * 1024
    assert s.attachment_min_text_chars == 20
    assert s.attachment_context_max_chars == 60000


# ---------------------------------------------------------------------------
# 6) 端点层：feature flag 与格式白名单（不起服务，直接调函数）
# ---------------------------------------------------------------------------

def test_endpoint_disabled_returns_503():
    """ATTACHMENT_ENABLED=False → 503。

    显式强制 flag=False（T4 实测教训：若依赖 .env 的默认值，验收窗口期 .env=true 会让本测试
    失效并连带暴露伪 UploadFile 的无限 read 问题——测试的意图是「关→503」，必须自带开关状态）。
    """
    from fastapi import HTTPException

    from backend.app.api import attachment_chat

    import asyncio
    import modules.core.config as cfg

    class _F:  # 伪造 UploadFile 最小接口
        filename = "合同.png"

        def __init__(self) -> None:
            self._sent = False

        async def read(self, n=-1):
            if self._sent:
                return b""  # EOF：一次性返回，避免无限流触发超限分支
            self._sent = True
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

        async def close(self):
            return None

    s = cfg.get_settings()
    orig = s.attachment_enabled
    s.attachment_enabled = False  # 强制关：本测试验证的就是「关 → 503」
    try:
        async def _call():
            return await attachment_chat.chat_stream_with_attachment(file=_F(), question="", pipeline=None)

        try:
            asyncio.run(_call())
            raise AssertionError("应当抛出 503 HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 503
    finally:
        s.attachment_enabled = orig  # 恢复环境值（T4 窗口期为 true，验收结束为 false）


def test_endpoint_rejects_unsupported_extension_415():
    from fastapi import HTTPException

    from backend.app.api import attachment_chat

    import asyncio

    async def _call():
        class _F:
            filename = "virus.exe"
            async def read(self, n=-1):
                return b"MZ\x90\x00"
        return await attachment_chat.chat_stream_with_attachment(file=_F(), question="", pipeline=None)

    # 先开 flag 才能走到 415 分支（503 在前）
    import modules.core.config as cfg
    s = cfg.get_settings()
    orig = s.attachment_enabled
    s.attachment_enabled = True
    try:
        try:
            asyncio.run(_call())
            raise AssertionError("应当抛出 415 HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 415
    finally:
        s.attachment_enabled = orig  # 恢复，避免污染其他测试


# ---------------------------------------------------------------------------
# 7) T1 新增：超限 413（本地读取器按 attachment_max_bytes 拦截）
# ---------------------------------------------------------------------------

def test_endpoint_oversized_returns_413():
    """超过 attachment_max_bytes（默认 10MB）→ 413。

    T1 核查结论：knowledge._read_limited 按 MAX_UPLOAD_BYTES(20MB) 截断且从不抛 413，
    10~20MB 素材会绕过素材侧限制——故端点改用本地 _read_attachment_limited，本测试守护其行为。
    """
    from fastapi import HTTPException

    from backend.app.api import attachment_chat

    import asyncio
    import modules.core.config as cfg

    MB = 1024 * 1024

    class _BigFile:
        """伪造超大 UploadFile：总量 11MB（> 默认 10MB 上限），分块按请求字节数返回。"""

        filename = "big.png"

        def __init__(self) -> None:
            self._left = 11 * MB

        async def read(self, n=-1):
            if n is None or n < 0:
                n = self._left
            n = min(n, self._left)
            if n <= 0:
                return b""
            self._left -= n
            return b"\x89" + b"\x00" * (n - 1)

        async def close(self):  # 本地读取器超限时会调用 close
            return None

    s = cfg.get_settings()
    orig_enabled = s.attachment_enabled
    orig_max = s.attachment_max_bytes
    s.attachment_enabled = True
    s.attachment_max_bytes = 10 * MB  # 显式固定，防止环境变量漂移影响断言
    try:
        async def _call():
            return await attachment_chat.chat_stream_with_attachment(
                file=_BigFile(), question="", pipeline=None
            )

        try:
            asyncio.run(_call())
            raise AssertionError("应当抛出 413 HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 413
    finally:
        s.attachment_enabled = orig_enabled
        s.attachment_max_bytes = orig_max


# ---------------------------------------------------------------------------
# 8) T1 新增：扩展名 .png 但内容为 WEBP → 按 magic bytes 走图片路径（tasks 1.4/5.1 前置）
# ---------------------------------------------------------------------------

def test_extract_text_png_ext_webp_content_dispatches_by_magic():
    """扩展名与实际内容不符（实测场景：学校承包合同.png 实为 WEBP）。

    断言三点：走图片路径（kind=image）、按真实格式处理（detected_mime=webp）、
    describe_image 收到的是 magic bytes 探测出的 mime 而非扩展名推断的 png。
    """
    import asyncio

    import modules.rag.attachment as att

    # 手工构造 WEBP magic bytes（RIFF....WEBPVP8），不读取 data_corpus 真实文件
    webp_bytes = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32
    captured: dict = {}

    async def fake_describe_image(content, mime, settings=None):
        captured["mime"] = mime
        captured["size"] = len(content)
        return "第一条 收购人与销售人就生鲜乳购销事宜订立本合同，共同遵守。"  # > 20 字符，绕开短文本 fail-closed

    orig = att.describe_image
    att.describe_image = fake_describe_image  # 单测不打真实 API
    try:
        result = asyncio.run(att.extract_text("学校承包合同.png", webp_bytes))
    finally:
        att.describe_image = orig

    assert result.kind == "image"
    assert result.detected_mime == "image/webp"   # 按实际内容，而非扩展名
    assert result.ext == ".png"
    assert captured["mime"] == "image/webp"       # 分派依据是 magic bytes
    assert captured["size"] == len(webp_bytes)
    assert "第一条" in result.text
    assert result.truncated is False


# ---------------------------------------------------------------------------
# 9) T2 新增（任务 2.3 追加/2.4 A/B 依据）：前言剥离后处理
# ---------------------------------------------------------------------------

def test_strip_preamble_removes_leading_preamble_line():
    """思考模式下实测输出的「以下是图片中提取的文字内容：」开场白须被剥离。"""
    from modules.rag.vision import _strip_preamble

    text = "以下是图片中提取的文字内容：\n\n第五条 检验方式\n1. 收购人对销售人提供的生鲜乳进行抽样检验。"
    out = _strip_preamble(text)
    assert out.startswith("第五条")
    assert "以下是" not in out.splitlines()[0]


def test_strip_preamble_keeps_body_sentence_starting_with_yixia():
    """正文中合法的「以下是…」句（非首行开场白）不受影响。"""
    from modules.rag.vision import _strip_preamble

    text = "第三条 说明义务\n甲方应当向乙方说明以下内容的真实情况。"
    out = _strip_preamble(text)
    assert out == text  # 原样保留


def test_strip_preamble_noop_for_clean_text():
    from modules.rag.vision import _strip_preamble

    text = "第一条 合同标的。"
    assert _strip_preamble(text) == text


# ---------------------------------------------------------------------------
# 10) T4 回归：ExtractionError 契约 + 端点错误路径真的产出 SSE 错误事件
#     （T4 实测发现：端点按 exc.message 组装错误事件，但类只存了 code →
#      生成器 AttributeError → SSE 连接中断。13 个既有单测全绿却漏测此路径。）
# ---------------------------------------------------------------------------

def test_extraction_error_has_message_and_code_attributes():
    from modules.rag.vision import ExtractionError

    exc = ExtractionError("BAD_IMAGE", "图片文件无法识别")
    assert exc.code == "BAD_IMAGE"
    assert exc.message == "图片文件无法识别"   # 端点按 .message 组装 SSE 事件（曾缺失 → 崩溃）
    assert str(exc) == "图片文件无法识别"


def test_endpoint_extraction_error_yields_sse_error():
    """端点错误路径端到端：损坏图片 → SSE {"error":..., "code":"BAD_IMAGE"}，而非连接中断。"""
    import asyncio
    import json

    from backend.app.api import attachment_chat
    import modules.core.config as cfg

    class _BrokenImage:
        """扩展名 .png 但内容探测不到已知图片格式 → BAD_IMAGE（真实 ExtractionError 路径）。"""

        filename = "broken.png"

        def __init__(self) -> None:
            self._sent = False

        async def read(self, n=-1):
            if self._sent:
                return b""  # EOF：一次性返回（无限重复会撑爆 10MB 上限走进 413 分支）
            self._sent = True
            return b"not an image at all" + b"\x00" * 16

        async def close(self):
            return None

    s = cfg.get_settings()
    orig_enabled = s.attachment_enabled
    s.attachment_enabled = True
    try:
        resp = asyncio.run(
            attachment_chat.chat_stream_with_attachment(file=_BrokenImage(), question="", pipeline=None)
        )

        events: list[str] = []

        async def consume():
            async for chunk in resp.body_iterator:
                events.append(chunk)

        asyncio.run(consume())
        joined = "".join(events)
        assert '"error"' in joined, f"应产出错误事件，实际：{joined[:200]}"
        assert "BAD_IMAGE" in joined
        assert "图片" in joined          # 人话文案在事件里，而非 500/连接中断
        payload = json.loads(joined.split("data: ", 1)[1].strip())
        assert payload["code"] == "BAD_IMAGE"
    finally:
        s.attachment_enabled = orig_enabled
