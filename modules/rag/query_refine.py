"""素材问答的检索要点提炼（OpenSpec add-attachment-query，任务 2.8）。

仅在「用户未提供问题」时使用：从素材文本提炼出一条检索 query，
用于走现有检索链路补充法规依据。

两条硬规则（spec D10，勿违反）：
1. 素材全文 MUST NOT 直接作为检索 query——实测 data_corpus 9/9 文件超出建库
   492 token 上限（民法典 110,057 token），全文检索必然得到无关结果。
2. 提炼失败（超时/空结果）返回空串，由调用方跳过检索或回退缺省问题——
   退化为把全文当 query 比不检索更差（会返回看似相关的无关结果）。

实现说明：提炼为【一条】 query（要点合并），而非多条——现有
`stream_chat(question)` 是单 query 接口，不为此改检索链路。
"""

from __future__ import annotations

import logging

import httpx

from modules.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_REFINE_PROMPT = (
    "以下是一份法律材料（合同/法规/政策）的内容节选。"
    "请从中提炼出用于检索相关中国法律法规的检索要点：\n"
    "1. 提取 1~3 个最能代表该材料法律主题的要点（如合同类型、核心法律问题、涉及的法律领域）；\n"
    "2. 输出为一行检索查询文本（要点之间用空格分隔），总长不超过 100 字；\n"
    "3. 只输出查询文本本身，不要任何解释、编号或开场白。\n\n"
    "材料内容节选：\n"
)

# 提炼输入的截断：判断材料主题不需要全文，取前部即可（控成本与时延）
_REFINE_INPUT_CHARS = 4000


async def refine_retrieval_query(material_text: str, settings: Settings | None = None) -> str:
    """从素材文本提炼一条检索 query。失败返回空串（调用方决定回退策略）。

    入参:
        material_text: 素材提取文本（将截断至 _REFINE_INPUT_CHARS 再送模型）
        settings: 可选配置单例
    返回:
        检索 query 字符串；任何失败返回 ""（绝不抛出——提炼是增强项，不是关键路径）。
    """
    s = settings or get_settings()
    if not material_text or len(material_text) < s.attachment_min_text_chars:
        return ""  # 素材太短，无从提炼

    payload = {
        "model": s.vision_model,  # 复用同一个轻量模型（提炼是短文本任务，flash 档够用）
        "max_tokens": 200,
        "temperature": 0,
        # 任务 2.4/2.9：提炼是纯文本任务（非感知任务，与 vision 提取的决策语境不同——
        # 见 tasks.md 2.4 背景注）。关思考无感知质量风险，且 2.9 三文件实测正是
        # 在此开关下通过验收（1.0~1.7s，要点质量达标）——保留关闭。
        "enable_thinking": False,
        "messages": [
            {
                "role": "user",
                "content": _REFINE_PROMPT + material_text[:_REFINE_INPUT_CHARS],
            }
        ],
    }
    headers = {
        "Authorization": "Bearer %s" % s.dashscope_api_key,
        "Content-Type": "application/json",
    }
    url = s.dashscope_base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=s.attachment_timeout_seconds, trust_env=False) as client:
            r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            logger.warning("refine query failed: HTTP %d", r.status_code)
            return ""
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001 — 提炼失败不致命：返回空串走回退
        logger.warning("refine query error: %s", exc)
        return ""

    # 清理：去掉可能的编号/引号/换行，压成一行
    for prefix in ("以下是", "检索要点：", "查询："):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().strip('"“”').replace("\n", " ")
    if len(text) < 6:  # 太短视为无效
        return ""
    return text[:120]
