# =============================================================================
# -----------------------------------------------------------------------------
# 输入：用户问题纯文本 `question`；隐式读取 `.env` 中 DASHSCOPE_API_KEY 与意图模型名。
# 输出：异步函数返回 `bool`：True 表示走专业检索+RAG，False 表示仅闲聊引导（仍可调 LLM）。
# 被谁调用：`modules/rag/pipeline.py` 中 `stream_chat` 在记忆加载之后、向量检索之前调用。
# =============================================================================
"""
用小模型（如 qwen-turbo）做二分类：输出 JSON `{"professional": true|false}`。

无 API Key 或解析失败时默认 True，避免整个问答服务不可用。
"""

from __future__ import annotations

import json  # 把模型输出中的 JSON 子串解析成 dict
import logging  # 记录异常与降级原因
import re  # 从模型自然语言中抠出 {...} 片段
from functools import lru_cache  # 意图用 ChatOpenAI 单例

from langchain_core.messages import HumanMessage, SystemMessage  # 两条消息组成对话
from langchain_openai import ChatOpenAI  # OpenAI 兼容多轮 chat

from modules.core.config import get_settings  # api_key、base_url、intent_model
from modules.rag.dashscope_http import get_dashscope_async_client, get_dashscope_sync_client  # 自定义 httpx

logger = logging.getLogger(__name__)


@lru_cache
def _intent_llm() -> ChatOpenAI:
    """
    构造专用于意图分类的 ChatOpenAI：低温、短超时，与主生成模型分离。

    入参:
        无。
    返回:
        绑定 DashScope 兼容接口与自定义 httpx 客户端的 `ChatOpenAI` 实例。
    """
    s = get_settings()
    return ChatOpenAI(
        model=s.intent_model,  # 如 qwen-turbo，比 qwen-max 便宜
        temperature=0.0,  # 分类任务要确定性，温度置 0
        api_key=s.dashscope_api_key,
        base_url=s.dashscope_base_url,
        timeout=30,  # 意图应快，30 秒超时
        http_socket_options=(),  # 禁用 LangChain 自定义 transport，保留 httpx 环境代理探测
        http_client=get_dashscope_sync_client(),  # LangChain 内部偶发同步请求用
        http_async_client=get_dashscope_async_client(),  # ainvoke 主路径
    )


def clear_intent_llm_cache() -> None:
    """释放意图客户端引用，供应用关闭后在同一进程内安全再次启动。"""
    _intent_llm.cache_clear()


_JSON_FENCE = re.compile(r"\{[^{}]*\}")  # 非贪婪匹配最外层一对花括号内无嵌套花括号的 JSON（简单场景够用）


async def is_professional_query(question: str) -> bool:
    """
    调用意图模型；解析失败返回 True（保守：宁可多检索也不要误杀专业问题）。

    入参:
        question: 用户本轮自然语言问题。
    返回:
        True 表示按「专业税法/劳动法等咨询」处理并走检索；False 表示仅闲聊引导分支。
        无 API Key 或解析异常时默认 True。
    """
    if not question.strip():  # 全空白
        return False  # 空问题不做检索
    s = get_settings()
    if not s.dashscope_api_key:  # 未配置密钥
        logger.warning("DASHSCOPE_API_KEY 未配置，跳过意图识别，默认 professional=True")
        return True  # 降级：直接进入检索

    sys = SystemMessage(
        content=(
            "你是意图分类器。判断用户问题是否属于一般法律专业咨询，包括但不限于"
            "税法、劳动法、社会保险、企业与劳动者权利义务、交通事故与侵权，"
            "以及企业内部制度、合同审批和知识库文档问答等。"
            "仅输出 JSON：{\"professional\": true|false}，不要其他文字。"
        ),
    )  # 系统角色：约束输出格式
    human = HumanMessage(content=f"问题：{question.strip()}")  # 用户内容：只带问题文本
    try:
        resp = await _intent_llm().ainvoke([sys, human])  # 异步调用 DashScope
        text = str(resp.content)  # 模型返回可能是 str 或其它，统一转 str
        m = _JSON_FENCE.search(text)  # 在整段输出里找 JSON 片段
        if not m:  # 模型没按格式输出
            return True  # 保守当专业
        data = json.loads(m.group(0))  # 解析 JSON 字符串为 dict
        return bool(data.get("professional", True))  # 缺 key 时默认 True
    except Exception as exc:  # noqa: BLE001 — 网络/JSON/模型错误
        logger.exception("intent classify failed: %s", exc)
        return True  # 失败则走专业链路
