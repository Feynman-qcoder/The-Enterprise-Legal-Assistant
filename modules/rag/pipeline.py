# =============================================================================
# -----------------------------------------------------------------------------
# 这是在线问答“总编排文件”：把缓存、记忆、意图、检索、融合、重排、生成串成一条链。
# 你可以把它理解为“问答总调度器”。
#
# 输入：
# - question：用户本轮问题（字符串）
# - user_external_id：可选用户标识（用于“用户隔离缓存 + 历史记忆”）
#
# 输出：
# - AsyncIterator[str]：异步文本片段流（供 SSE 逐块输出）
#
# 主要调用方：
# - backend/app/api/chat.py 中的 pipeline.stream_chat(...)
#
# 关键特点：
# - 先缓存短路，再进入复杂链路；
# - FAQ 高置信可直达，不走大模型；
# - 法律问题走“向量检索 + BM25 + RRF + 重排 + LLM”。
# - LLM 流式容错（Task 19 B3）：首包前失败且可重试（超时/5xx/429）重试 1 次；
#   已产出分片后失败或重试耗尽 → 不再重试（防重复输出），yield 兜底文案后正常结束流；
#   兜底/中断路径的结果绝不写缓存（completed 标志位门控，防半截答案污染一小时）。
# - 语义缓存（Task 23 两级缓存 L2）：exact miss 后 embed 提前 → 语义命中直出并回填
#   exact；双 miss 走原链路，完整成功后写 exact + 语义。flag 默认关闭，关闭时
#   执行路径与现状逐字节等价（embed 仍在意图分流后原位置调用）。
# =============================================================================

"""在线 RAG 主流程（含缓存、记忆、检索、生成）的教学注释版实现。"""

from __future__ import annotations  # 允许把类型注解延迟求值，避免前向引用问题

import asyncio  # 用于 to_thread：把同步阻塞代码放线程池
import logging  # 模块日志
import re  # 正则分词（供 BM25）
from collections.abc import AsyncIterator  # 标注“异步迭代器”返回类型
from pathlib import PureWindowsPath
from typing import Any  # 标注动态返回值

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage  # LangChain 消息类型
from langchain_openai import ChatOpenAI  # OpenAI 兼容客户端（本项目对接 DashScope 兼容接口）
from pymilvus import Collection  # Milvus 集合对象（load/search/insert）
from sqlalchemy import select  # SQLAlchemy 查询构造器

from modules.cache.redis_client import RedisCache, cache_key_for_query, get_redis  # Redis 缓存能力
from modules.cache.semantic_cache import SemanticCache  # 语义缓存（两级缓存 L2，Task 23）
from modules.core.config import ASSISTANT_NAME, get_settings  # 全局配置 + 助手名称
from modules.database.models import LegalTab  # 法律文档 ORM 模型（回查父文档用）
from modules.database.session import get_session_factory  # AsyncSession 工厂
from modules.database.v2_parent_repository import V2ParentRepository
from modules.embeddings.local_embedding import LocalEmbeddingService  # 向量编码服务
from modules.memory.service import (
    DEFAULT_MEMORY_CONTEXT_LINES,  # 默认读取的历史条数
    fetch_recent_chat_lines,  # 读最近历史问答
    format_chat_history_for_prompt,  # 历史行格式化成 prompt 文本
    resolve_user_id,  # external_id -> user_id
)
from modules.milvus_store.client import ensure_milvus  # 确保 Milvus 连接
from modules.milvus_store.collections import COLLECTION_FAQ, COLLECTION_LEGAL_CHILD  # Milvus 集合名
from modules.rag.corpus_scope import CanonicalScope, get_canonical_scope  # canonical 53 篇白名单 + 元数据 lookup
from modules.rag.dashscope_http import (
    get_dashscope_async_client,  # 异步 httpx 客户端
    get_dashscope_sync_client,  # 同步 httpx 客户端
)
from modules.rag.hybrid_rrf import reciprocal_rank_fusion  # RRF 融合算法
from modules.rag.intent import is_professional_query  # 意图分类（专业/非专业）
from modules.rag.prompts import (
    GUIDE_NON_PROFESSIONAL,  # 非专业引导提示词
    RAG_SYSTEM,  # 专业 RAG 系统提示词
    augment_question_with_memory,  # 把历史拼到问题里
    build_user_message,  # 构造最终用户消息
)
from modules.rag.retrieval_contract import (
    LEGACY_RETRIEVAL_CONTRACT,
    V2_RETRIEVAL_CONTRACT,
    LegalRetrievalHit,
    ParentReference,
    RetrievalContractError,
)
from modules.rag.retrieval_v2_adapter import V2DenseRetrievalAdapter

logger = logging.getLogger(__name__)  # 当前模块 logger


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """
    判断 LLM 流式异常是否可安全重试（Task 19 B3）。

    可重试集合（按指令裁决）：超时 / 5xx / 429 / 连接失败。
    采用「异常名 + status_code 属性」的宽松匹配，避免硬依赖 openai/httpx 具体异常类型。

    入参:
        exc: llm.astream 过程中抛出的异常。
    返回:
        bool：True 表示首包前失败时允许重试 1 次。
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):  # 内置超时（含 asyncio）
        return True
    name = type(exc).__name__  # 异常类名匹配（openai/httpx 跨版本稳定）
    if name in {"APITimeoutError", "APIConnectionError", "ConnectError", "ReadTimeout", "RemoteProtocolError"}:
        return True
    status = getattr(exc, "status_code", None)  # openai 风格的状态码属性
    if status is None:
        status = getattr(exc, "code", None)  # httpx/其他风格
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):  # 429 / 5xx
        return True
    return False


def _tokenize(text: str) -> list[str]:  # 定义分词函数：把原始文本切成 BM25 可用 token 列表
    """
    把输入文本粗分词（给 BM25 用）。

    参数：
    - text: 原始文本（可能是用户问题或法律子块文本）

    返回：
    - list[str]: token 列表（中文单字 + 英文词 + 数字）
    """
    # text.lower()：统一小写，减少英文大小写差异噪声
    # 正则说明：
    # - [\u4e00-\u9fff]：单个中文字符
    # - [a-zA-Z]+：连续英文单词
    # - [0-9]+：连续数字
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+", text.lower())  # 返回分词结果列表


def _entity_to_dict(entity: Any) -> dict:  # 定义实体归一化函数：把不同类型 entity 统一为 dict
    """
    兼容不同 pymilvus 版本的 entity 结构，统一转成 dict。

    参数：
    - entity: 可能是 dict、Row-like、或带 to_dict 方法的对象

    返回：
    - dict: 可 .get(...) 的标准字典；失败时返回空字典
    """
    if entity is None:  # 命中无附加字段
        return {}  # 空实体直接返回空字典
    if isinstance(entity, dict):  # 已经是 dict
        return entity  # 已经是字典则原样返回
    if hasattr(entity, "to_dict"):  # 新版本对象通常有 to_dict
        return entity.to_dict()  # type: ignore[no-any-return]  # 调用对象自带 to_dict 转换
    try:
        return dict(entity)  # 尝试把可迭代键值对对象强制转成 dict
    except Exception:  # noqa: BLE001
        return {}  # 兜底为空，不影响主流程


def _parse_milvus_hits(raw_hits: Any) -> list[tuple[int, float, dict]]:  # 定义命中解析函数：把 Milvus 原始结果转统一结构
    """
    解析 Milvus 搜索返回，统一成三元组列表。

    参数：
    - raw_hits: Collection.search(...) 的原始返回

    返回：
    - list[(id, distance_or_similarity, entity_dict)]
    """
    out: list[tuple[int, float, dict]] = []  # 输出容器
    if not raw_hits:  # 没有任何结果
        return out  # 无命中时返回空列表
    if not raw_hits[0]:  # 第一条查询（本代码每次只查1条）无命中
        return out  # 第一查询无命中时返回空列表

    for hit in raw_hits[0]:  # 遍历命中列表
        ent = _entity_to_dict(getattr(hit, "entity", None))  # 规范化 entity
        out.append((int(hit.id), float(hit.distance), ent))  # 统一类型后写入

    return out  # 返回结构化命中列表


def _parse_legacy_legal_hits(raw_hits: Any) -> list[LegalRetrievalHit]:
    """Adapt the Legacy INT64/text/parent_id schema to the shared legal hit contract."""

    adapted: list[LegalRetrievalHit] = []
    for child_id, similarity, entity in _parse_milvus_hits(raw_hits):
        content = str(entity.get("text", ""))
        parent_id = entity.get("parent_id")
        if not content.strip() or parent_id is None:
            raise RetrievalContractError(
                f"Legacy child {child_id} missing content or parent_id",
            )
        adapted.append(
            LegalRetrievalHit(
                chunk_id=child_id,
                content=content,
                parent_reference=int(parent_id),
                metadata={
                    "retrieval_contract": LEGACY_RETRIEVAL_CONTRACT,
                    "source_file": str(entity.get("source_file", "")),
                },
                similarity=similarity,
            ),
        )
    return adapted


def _with_source_metadata(scope: CanonicalScope, source_file: str, text: str) -> str:  # 给单篇父文档正文加元数据头
    """
    在父文档正文前拼接 [Source Metadata] 轻量头（Online RAG Alignment V1）。

    - 只使用 manifest 中实际存在的字段，lookup 失败时保持裸文本（不编造）。
    - 在重排之后调用：重排器看到的仍是裸父文本，排序行为不受 header 影响。
    """
    header = scope.format_header(source_file)  # manifest 只读 lookup
    if not header:  # source_file 缺失或不在 manifest 中
        return text  # 保持裸文本
    return f"{header}\n\n[Evidence]\n{text}"  # 轻量两段式：元数据头 + 证据正文


class RagPipeline:  # 定义在线问答总编排类：封装缓存、检索、重排、生成全流程
    """在线问答业务总编排类（建议进程内单例）。"""

    def __init__(self) -> None:  # 初始化管线对象，挂载配置/embedding/缓存等组件
        """
        初始化总编排对象。

        成员说明：
        - self.settings: 全局配置对象
        - self._emb: 向量编码服务
        - self._rerank: 重排服务（懒加载）
        - self._cache: Redis 缓存服务（exact 层）
        - self._sem: 语义缓存服务（L2 层，Task 23；flag 默认关闭）
        """
        self.settings = get_settings()  # 读取全局配置（单例）
        self._emb = LocalEmbeddingService()  # 初始化 embedding 服务
        self._rerank = None  # 暂不加载重排模型，等用到时再加载
        self._cache = RedisCache(get_redis())  # 初始化缓存封装
        self._sem = SemanticCache(get_redis(), self.settings)  # 语义缓存（与 exact 层共享连接池）
        self._v2_dense = V2DenseRetrievalAdapter(self.settings.retrieval_v2_collection)
        self._v2_parents = V2ParentRepository()

    async def _cache_scope(self, question: str, user_external_id: str | None) -> tuple[str, int]:
        """Namespace cached answers by retrieval contract + corpus version (Task 19 A6).

        version 进入 key：同步成功 → corpus_version+1 → 新请求 key 全新 →
        旧缓存整体自然 miss → TTL 3600 兜底回收。
        生效延迟上限 = version 读取策略间隔（60s TTL，月更场景无感）。
        Legacy 合同固定 v0（key 格式与 A6 前仅差版本段，回滚 A6 即恢复旧格式）。

        Task 23：返回值改为 (scope, version) 元组——exact 层继续用 scope 拼 key，
        语义层用 version 拼 L2 key（同一次请求只读一次 version，语义检查与回填
        共用，避免双读；fetch_corpus_version 本身有 60s TTL 缓存兜底）。
        """

        user_scope = (user_external_id or "").strip()
        if self.settings.retrieval_contract == V2_RETRIEVAL_CONTRACT:
            version = await self._v2_parents.fetch_corpus_version()  # 60s TTL，容错返回 0
        else:
            version = 0
        return f"{self.settings.retrieval_contract}:v{version}:{user_scope}:{question}", version

    def _llm(self) -> ChatOpenAI:  # 构建并返回一个流式 ChatOpenAI 客户端
        """
        构造一个 LLM 客户端实例（流式）。

        返回：
        - ChatOpenAI: 已绑定 DashScope 配置 + httpx 客户端
        """
        s = self.settings  # 局部变量缩短书写
        return ChatOpenAI(
            model=s.llm_model,  # 主模型名称（配置项）
            temperature=0.2,  # 降低随机性，减少胡编
            api_key=s.dashscope_api_key,  # API Key
            base_url=s.dashscope_base_url,  # 兼容接口地址
            streaming=True,  # 打开流式输出（SSE必须）
            timeout=120,  # 请求超时秒数
            http_socket_options=(),  # 禁用 LangChain 自定义 transport，保留 httpx 环境代理探测
            http_client=get_dashscope_sync_client(),  # 同步客户端
            http_async_client=get_dashscope_async_client(),  # 异步客户端
        )

    def _reranker(self):  # 获取重排器：首次调用时懒加载，后续复用
        """
        懒加载重排模型（仅法律复杂链路会用到）。

        返回：
        - LocalRerankService: 重排服务单例
        """
        if self._rerank is None:  # 首次调用才初始化
            from modules.rerank.local_rerank import LocalRerankService  # 延迟导入，减少冷启动

            self._rerank = LocalRerankService()  # 实例化重排服务
        return self._rerank  # 返回已初始化对象

    async def _milvus_search(  # 异步检索包装：把同步 pymilvus search 放在线程池执行
        self,
        collection: str,  # 要检索的集合名（FAQ或法律子块）
        vector: list[float],  # 查询向量（单条）
        limit: int,  # 返回 top-k 条数
        output_fields: list[str],  # 需要返回的标量字段
        expr: str | None = None,  # 可选标量过滤表达式（canonical source_file 白名单）
    ) -> Any:
        """
        在线程中执行 Milvus 同步检索，避免阻塞 asyncio 事件循环。

        参数：
        - collection: 集合名
        - vector: 查询向量
        - limit: top-k
        - output_fields: 附带返回的字段
        - expr: 标量过滤表达式；None/空 表示不过滤（FAQ 集合即不过滤）
        """

        def _run() -> Any:  # 定义同步闭包：线程中真正执行 Milvus 检索
            """同步闭包：真正调用 pymilvus search。"""
            ensure_milvus()  # 先确保连接存在
            col = Collection(collection)  # 绑定集合对象
            col.load()  # 加载到内存，避免 search 失败
            return col.search(
                data=[vector],  # 单条查询，包装成 batch 形式
                anns_field="embedding",  # 向量字段名
                param={"metric_type": "COSINE", "params": {"ef": 128}},  # 检索参数
                limit=limit,  # top-k
                expr=expr or "",  # 标量过滤：法律集合传 canonical 白名单，FAQ 为空串
                output_fields=output_fields,  # 返回字段
            )

        return await asyncio.to_thread(_run)  # 在线程池执行同步检索

    async def _fetch_parents(self, ids: list[int]) -> dict[int, str]:  # 批量回查父文档：输入 parent_id 列表，输出 id->正文映射
        """
        按 parent id 批量回查父文档全文。

        参数：
        - ids: 父文档 id 列表

        返回：
        - dict[parent_id, parent_text]
        """
        if not ids:  # 空输入直接返回空映射
            return {}

        factory = get_session_factory()  # 取会话工厂
        async with factory() as session:  # 打开会话
            res = await session.execute(
                select(LegalTab).where(LegalTab.id.in_(ids)),  # 按 id in 批量查询
            )
            rows = list(res.scalars().all())  # 提取 ORM 列表

        # 保护上下文长度：每篇父文档最多取 8000 字符
        return {r.id: (r.content or "")[:8000] for r in rows}

    async def stream_chat(  # 在线主入口：按业务链路逐步产出回答分片
        self,
        question: str,  # 用户问题（本轮输入）
        user_external_id: str | None = None,  # 用户外部 ID（用于记忆和缓存隔离）
        attachment_context: str = "",  # 素材提取文本（上传图片/文件转出的内容）；空 = 纯文字问答，行为与改造前一致
        evidence_out: list | None = None,  # 引用核查证据快照（add-citation-query）；None = 不收集，行为与改造前一致（红线）
    ) -> AsyncIterator[str]:
        """
        对外主入口：返回异步文本片段流。

        主流程：
        1) 查缓存（exact → 语义 L2）
        2) 读取用户历史记忆
        3) 意图分流（非专业走引导）
        4) 专业链路：FAQ 检索 -> 法律检索 -> 融合 -> 重排 -> 生成

        素材模式（attachment_context 非空，OpenSpec add-attachment-query）：
        - 答案依赖素材内容而缓存 key 只含问题 → 读写两级缓存都跳过（防串答案）
        - 跳过意图分流：用户上传了材料，按专业问题处理
        - 跳过 FAQ 分支：罐头答案无视素材内容
        - 检索仍用「用户问题」作 query（RAG 链路零改动），素材只进提示词

        证据快照（evidence_out，OpenSpec add-citation-check，design D2）：
        - 传入 list 时，RAG 生成子流程**完整成功**后把本次 contexts append 进去；
          中断/兜底/缓存命中等路径不写入（没有可靠的"本次证据"）
        - 调用方（API 层）仅在流结束后读取做引用核查——只读快照，禁止当作业务通道写入
        """
        # ---------------------------------------------------------------------
        # 步骤1：缓存短路（exact 层）
        # ---------------------------------------------------------------------
        # retrieval contract + corpus version + 用户ID + 问题共同隔离缓存，
        # 避免 Legacy/V2 答案互相污染，并支持版本翻面整体失效（A6）
        # 素材模式：答案依赖素材内容，按问题缓存会把 A 的素材答案串给 B —— 跳过读
        scope, sem_version = await self._cache_scope(question, user_external_id)
        key = cache_key_for_query(scope)  # 生成缓存 key
        cached = None if attachment_context else await self._cache.get_json(key)  # 读缓存（素材模式跳过）
        if isinstance(cached, dict) and cached.get("answer"):  # 判断缓存对象合法且包含 answer 字段
            # 命中缓存：直接输出答案并结束，跳过后续所有链路
            yield str(cached["answer"])
            return  # 缓存命中后结束主流程

        # ---------------------------------------------------------------------
        # 步骤1.5：语义缓存检查（Task 23 两级缓存 L2，flag 默认关闭）
        # ---------------------------------------------------------------------
        # exact miss 后、记忆/意图之前：embed 提前 + 向量相似度命中直出。
        # 命中 → 回填 exact（下次字面相同问题走 exact 层）+ 分块直出；
        # flag 关闭时本块整体跳过，embed 仍在步骤4 原位置调用（行为等价红线）。
        qvec: list[float] | None = None
        if self.settings.semantic_cache_enabled and not attachment_context:  # 素材模式跳过语义缓存读
            qvec = await self._emb.embed_query(question)  # embed 提前（≠修改 embedding 逻辑）
            hit = await self._sem.lookup(qvec, sem_version)
            if hit is not None:
                answer, sim, route = hit
                logger.info("semantic cache hit sim=%.4f route=%s", sim, route)  # 命中审计：分数 + 原 route
                await self._cache.set_json(
                    key,  # 回填 exact 层（本问题字面再问时直接短路）
                    {"answer": answer, "route": "semantic"},
                    self.settings.cache_ttl_seconds,
                )
                # 与 FAQ 直达一致的“流式观感”分块输出
                for i in range(0, len(answer), 40):
                    yield answer[i : i + 40]
                return  # 语义命中后结束主流程

        # ---------------------------------------------------------------------
        # 步骤2：用户历史记忆（仅有 user_external_id 时启用）
        # ---------------------------------------------------------------------
        memory_snippet: str | None = None  # 默认无记忆
        if user_external_id and user_external_id.strip():  # 仅在用户ID有效时启用记忆链路
            factory = get_session_factory()  # 会话工厂
            async with factory() as session:  # 打开数据库会话
                uid = await resolve_user_id(session, user_external_id.strip())  # external_id -> user_id
                rows = await fetch_recent_chat_lines(
                    session,  # 当前数据库会话（用于查询历史）
                    uid,  # 当前用户内部主键
                    DEFAULT_MEMORY_CONTEXT_LINES,  # 最近N条
                )
                memory_snippet = format_chat_history_for_prompt(rows)  # 格式化为 prompt 文本
                await session.commit()  # 提交事务，结束本轮记忆读取会话

        # ---------------------------------------------------------------------
        # 步骤3：意图分流（非专业 -> 引导）
        # ---------------------------------------------------------------------
        if not attachment_context and not await is_professional_query(question):  # 意图判定为非专业问题（素材模式强制走专业链路）
            async for piece in self._stream_simple_llm(
                [
                    SystemMessage(content=GUIDE_NON_PROFESSIONAL),  # 系统提示：引导回专业问题
                    HumanMessage(content=augment_question_with_memory(question, memory_snippet)),  # 人类消息附带记忆
                ],
            ):
                yield piece  # 逐片返回
            return  # 非专业路径返回，不进入专业检索

        # ---------------------------------------------------------------------
        # 步骤4：专业链路先做问题向量化
        # ---------------------------------------------------------------------
        # Task 23：语义缓存开启时 qvec 已在步骤1.5 算好，此处复用（避免重复 embed）；
        # flag 关闭时 qvec 为 None，仍在此处调用——调用位置与改造前逐字节一致。
        qvec = qvec if qvec is not None else await self._emb.embed_query(question)  # 将用户问题编码为检索向量

        # ---------------------------------------------------------------------
        # 步骤5：FAQ 检索分支
        # ---------------------------------------------------------------------
        faq_raw = await self._milvus_search(
            COLLECTION_FAQ,  # FAQ 集合
            qvec,  # 查询向量
            limit=10,  # 取前10条
            output_fields=["question", "answer"],  # 返回 question/answer
        )
        faq_parsed = _parse_milvus_hits(faq_raw)  # 解析命中结果

        # 配置阈值：代码里用“相似度阈值”，配置里是“距离阈值”
        th_direct = self.settings.faq_direct_distance_threshold  # FAQ 直达阈值（距离）
        th_llm = self.settings.faq_llm_distance_threshold  # FAQ 进入 LLM 阈值（距离）
        sim_direct = 1.0 - th_direct  # 转相似度阈值
        sim_llm = 1.0 - th_llm  # 转相似度阈值

        if faq_parsed and not attachment_context:  # FAQ 集合存在命中候选（素材模式跳过：罐头答案无视素材内容）
            _best_id, best_sim, ent = faq_parsed[0]  # 取最相似的一条 FAQ 命中

            # 5.1 FAQ 高置信：直接输出，不走 LLM
            if best_sim >= sim_direct and ent.get("answer"):  # 满足高置信阈值且有标准答案
                ans = str(ent["answer"])  # 标准答案文本
                await self._cache.set_json(
                    key,  # 当前问题缓存 key
                    {"answer": ans, "route": "faq_direct"},  # 标记为 FAQ 直达路径
                    self.settings.cache_ttl_seconds,  # 过期时间
                )
                # Task 23：FAQ 直达也是“完整成功”的答案，写语义层供改述问题命中
                if self.settings.semantic_cache_enabled and qvec is not None:
                    await self._sem.store(question, qvec, ans, "faq_direct", sem_version)
                # 为了前端“流式观感”，把长文本分块输出
                for i in range(0, len(ans), 40):
                    yield ans[i : i + 40]
                return  # FAQ 直达输出后结束

            # 5.2 FAQ 中等相似：把 FAQ 答案作为上下文交给 LLM
            close = [x for x in faq_parsed if x[1] >= sim_llm][: self.settings.faq_top_k_for_llm]
            if close and close[0][1] >= sim_llm:  # 存在可用于 LLM 参考的中等相似 FAQ
                ctx: list[str] = []  # FAQ 上下文片段容器
                for _, d, e in close:  # 遍历近似 FAQ，逐条构建参考上下文
                    if e.get("answer"):  # 只取有答案的项
                        ctx.append(f"问答参考（相似度={d:.4f}）：{e['answer']}")  # 拼接上下文
                if ctx:  # 有上下文才调用 LLM
                    async for p in self._rag_stream_llm(
                        question=question,  # 原问题
                        contexts=ctx,  # FAQ 参考上下文
                        memory_snippet=memory_snippet,  # 用户历史
                        user_external_id=user_external_id,  # 缓存隔离
                        qvec=qvec,  # Task 23：透传问题向量供语义层写回
                        evidence_out=evidence_out,  # 引用核查证据快照透传（FAQ+LLM 路径的证据 = FAQ 参考上下文）
                    ):
                        yield p
                    return  # FAQ+LLM 路径结束

        # ---------------------------------------------------------------------
        # 步骤6：法律检索分支（FAQ 不足时）
        # ---------------------------------------------------------------------
        dense_limit = self.settings.hybrid_dense_candidate_k  # dense 候选数量
        canonical_scope = get_canonical_scope()  # 冻结的 53 篇 canonical 范围（进程内缓存，fail-closed）
        retrieval_contract = self.settings.retrieval_contract
        if retrieval_contract == V2_RETRIEVAL_CONTRACT:
            document_identities = await self._v2_parents.fetch_document_identities()
            legal_hits = await self._v2_dense.search(
                qvec,
                limit=dense_limit,
                document_identities=document_identities,
            )
        else:
            legal_raw = await self._milvus_search(
                COLLECTION_LEGAL_CHILD,  # Legacy 法律子块集合
                qvec,  # 查询向量
                limit=dense_limit,  # 候选数量
                output_fields=["text", "parent_id", "source_file"],
                expr=canonical_scope.build_source_file_expr(),
            )
            legal_hits = _parse_legacy_legal_hits(legal_raw)

        if not legal_hits:  # 法律子块检索无命中
            async for p in self._stream_simple_llm(
                [
                    SystemMessage(
                        content=f"你是{ASSISTANT_NAME}。知识库暂无法律片段命中，请诚实说明并给出通用建议。",
                    ),
                    HumanMessage(content=augment_question_with_memory(question, memory_snippet)),
                ],
            ):
                yield p
            return  # 执行兜底回复后结束

        # 把命中结果拆成常用结构
        child_ids = [hit.chunk_id for hit in legal_hits]  # 子块 id 列表（Legacy int / V2 str）
        id_to_text = {hit.chunk_id: hit.content for hit in legal_hits}  # 统一 content contract
        hit_by_id = {hit.chunk_id: hit for hit in legal_hits}
        dense_ranked = list(child_ids)  # dense 路排序结果

        # ---------------------------------------------------------------------
        # 步骤7：可选 BM25 + RRF 融合
        # ---------------------------------------------------------------------
        if self.settings.legal_hybrid_bm25_enabled and len(child_ids) > 1:  # 开关开启且候选足够时启用 BM25 融合
            from rank_bm25 import BM25Okapi  # 延迟导入，减少不必要开销

            tokenized_corpus = [_tokenize(id_to_text[i]) for i in child_ids]  # 子块语料分词
            tokenized_q = _tokenize(question)  # 问题分词
            bm25 = BM25Okapi(tokenized_corpus)  # 基于候选语料构建 BM25 打分器
            scores = bm25.get_scores(tokenized_q)  # 计算查询对每个候选的 BM25 分数
            bm25_order = [
                child_ids[i]
                for i in sorted(
                    range(len(child_ids)),
                    key=lambda k: scores[k],  # 按 BM25 分数排序
                    reverse=True,
                )
            ]
            ranked_lists = [
                dense_ranked,  # dense 排序
                bm25_order[: self.settings.hybrid_bm25_candidate_k],  # BM25 截断排序
            ]
        else:
            ranked_lists = [dense_ranked]  # 未启用 BM25 时只保留 dense 排序

        fused = reciprocal_rank_fusion(
            ranked_lists,  # 多路排序输入
            k=self.settings.hybrid_rrf_k,  # RRF 衰减参数
        )
        top_child_ids = [doc for doc, _ in fused[:30]]  # 融合后取前30个子块

        # ---------------------------------------------------------------------
        # 步骤8：子块 -> 父文档回溯（去重保持顺序）
        # ---------------------------------------------------------------------
        parent_ids_ordered: list[ParentReference] = []  # Legacy int / V2 external string
        seen: set[ParentReference] = set()  # 去重集合
        pid_to_source: dict[ParentReference, str] = {}
        for cid in top_child_ids:  # 按融合顺序遍历子块
            hit = hit_by_id[cid]
            pid = hit.parent_reference
            source_file = hit.source_file
            if retrieval_contract == V2_RETRIEVAL_CONTRACT and source_file:
                source_file = PureWindowsPath(source_file).name
            if source_file:  # metadata header 使用 canonical manifest 文件名
                pid_to_source.setdefault(pid, source_file)
            if pid not in seen:  # 未出现过才加入
                seen.add(pid)
                parent_ids_ordered.append(pid)

        parent_texts_map: dict[ParentReference, str]
        if retrieval_contract == V2_RETRIEVAL_CONTRACT:
            v2_parent_ids = [
                parent_id
                for parent_id in parent_ids_ordered
                if isinstance(parent_id, str)
            ]
            if len(v2_parent_ids) != len(parent_ids_ordered):
                raise RetrievalContractError("V2 retrieval emitted a non-string parent reference")
            v2_records = await self._v2_parents.fetch_parents(v2_parent_ids)
            parent_texts_map = {
                parent_id: record.content
                for parent_id, record in v2_records.items()
            }
        else:
            legacy_parent_ids = [
                parent_id
                for parent_id in parent_ids_ordered
                if isinstance(parent_id, int)
            ]
            if len(legacy_parent_ids) != len(parent_ids_ordered):
                raise RetrievalContractError("Legacy retrieval emitted a non-integer parent reference")
            parent_texts_map = await self._fetch_parents(legacy_parent_ids)
        passages: list[str] = []  # 父文档正文列表（重排输入：保持裸文本，不混入 header）
        passage_sources: list[str] = []  # 与 passages 一一对应的 source_file（供 header 注入）
        for pid in parent_ids_ordered:  # 保持融合顺序
            if not parent_texts_map.get(pid):  # 过滤空文本
                continue
            passages.append(parent_texts_map[pid])
            passage_sources.append(pid_to_source.get(pid, ""))

        # Rerank 候选池截断（Settings 化，Config C 转正 V2）：按 RRF 融合顺序取前 N 篇父文档。
        # settings.rerank_pool_top_n 为 None 或 <=0 = 现状（全部父文档进 rerank）。
        rerank_pool_top_n = self.settings.rerank_pool_top_n or 0
        if rerank_pool_top_n > 0 and len(passages) > rerank_pool_top_n:
            passages = passages[:rerank_pool_top_n]
            passage_sources = passage_sources[:rerank_pool_top_n]

        if not passages:  # 回查后父文档正文为空或缺失
            async for p in self._stream_simple_llm(
                [
                    SystemMessage(content=RAG_SYSTEM),  # 用 RAG 系统提示
                    HumanMessage(content=augment_question_with_memory(question, memory_snippet)),
                ],
            ):
                yield p
            return  # 执行兜底路径后结束

        # ---------------------------------------------------------------------
        # 步骤9：重排（CrossEncoder）——超时降级（Task 21 Part 3）
        # ---------------------------------------------------------------------
        reranker = self._reranker()  # 获取重排服务（懒加载）
        rerank_timeout = self.settings.rerank_timeout_seconds
        scores: list[float] | None
        try:
            # 仅在 rank 调用外围加超时包装；rank 内部实现与模型加载零改动（红线）
            scores = await asyncio.wait_for(
                reranker.rank(question, passages),
                timeout=rerank_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — 超时/异常统一降级（含 TimeoutError）
            logger.warning(
                "rerank degraded after %.3fs (candidates=%d), falling back to RRF order top-%d: %s",
                rerank_timeout,
                len(passages),
                self.settings.legal_rerank_top_n,
                exc,
            )
            scores = None  # 降级：passages 本就按 RRF 融合顺序排列，直接取前 top_n 进 LLM
        if scores is None:
            ranked_idx = list(range(len(passages)))  # RRF 顺序原样保留（降级路径）
        else:
            ranked_idx = sorted(
                range(len(passages)),
                key=lambda i: scores[i],  # 按分数排序
                reverse=True,
            )
        top_n = self.settings.legal_rerank_top_n  # 取前N条
        final_idx = ranked_idx[:top_n]  # 最终选中的父文档下标
        final_ctx = [passages[i] for i in final_idx]  # 最终上下文（裸父文本，重排基准不变）
        final_sources = [passage_sources[i] for i in final_idx]  # 对应 source_file

        # ---------------------------------------------------------------------
        # 步骤9.5：Retrieved Context Metadata Header（Online RAG Alignment V1）
        # ---------------------------------------------------------------------
        # 重排完成后、送入 GLM 前，为每篇父文档注入 manifest 元数据轻量头；
        # lookup 失败保持裸文本，绝不编造字段。BM25/RRF/父回溯/重排已自然继承
        # canonical-only 候选（dense 阶段白名单过滤），本步不改变任何检索行为。
        final_ctx = [
            _with_source_metadata(canonical_scope, sf, text)
            for sf, text in zip(final_sources, final_ctx)
        ]

        # ---------------------------------------------------------------------
        # 步骤10：RAG 生成
        # ---------------------------------------------------------------------
        async for p in self._rag_stream_llm(
            question=question,  # 原问题
            contexts=final_ctx,  # 重排后（含元数据头）的上下文
            memory_snippet=memory_snippet,  # 用户历史记忆
            user_external_id=user_external_id,  # 用户标识（缓存隔离）
            qvec=qvec,  # Task 23：透传问题向量供语义层写回
            attachment_context=attachment_context,  # 素材文本注入提示词（空 = 行为不变）
            evidence_out=evidence_out,  # 引用核查证据快照透传（主路径证据 = 重排后的父文档）
        ):
            yield p  # 逐片返回

    async def _rag_stream_llm(  # RAG 生成子流程：带参考资料的流式输出并写缓存
        self,
        question: str,  # 用户问题
        contexts: list[str],  # 检索得到的参考资料列表
        *,
        memory_snippet: str | None = None,  # 可选：历史记忆文本
        user_external_id: str | None = None,  # 可选：用户标识（用于缓存 scope）
        qvec: list[float] | None = None,  # Task 23：问题向量（语义层写回用；None = 不写语义层）
        attachment_context: str = "",  # 素材文本（空 = 与改造前行为一致）
        evidence_out: list | None = None,  # 引用核查证据快照（None = 不收集；非 None 时成功路径 append contexts）
    ) -> AsyncIterator[str]:
        """
        带检索上下文的流式生成路径。

        行为：
        - 构造 RAG 系统消息 + 用户消息（attachment_context 非空时含【用户上传的材料】段）
        - 流式产出模型文本
        - 仅在「流完整结束」时写入缓存（route=rag_llm）；
          兜底/中断路径绝不写缓存（B3：防半截答案污染缓存一小时）
        - 素材模式（attachment_context 非空）不写两级缓存：答案依赖素材内容，
          按 question 建缓存会把 A 的素材答案串给问同样问题的 B

        重试语义（Task 19 B3）：
        - 首个分片产出前异常 且 错误 ∈ {超时/5xx/429} → 重试 1 次（log 记录）
        - 已产出分片后异常，或重试仍失败 → 不再重试（防重复输出），
          yield 兜底文案（settings.llm_fallback_message）后正常结束流

        语义缓存（Task 23）：
        - completed 门控与 exact 层一致：只有完整成功的答案才写语义层；
          qvec 为 None（flag 关闭 / 调用方未传）时跳过语义写回。
        """
        scope, version = await self._cache_scope(question, user_external_id)  # contract + version + 用户隔离作用域
        key = cache_key_for_query(scope)  # 根据作用域生成缓存键
        llm = self._llm()  # 创建本次生成使用的 LLM 客户端
        messages = [
            SystemMessage(content=RAG_SYSTEM),  # 专业 RAG 系统提示
            HumanMessage(content=build_user_message(question, contexts, memory_snippet, attachment_context)),  # 用户消息（问题+素材上下文+参考资料+记忆）
        ]

        buf: list[str] = []  # 收集所有分片文本，最终拼成完整答案
        completed = False  # B3：成功标志位——只有完整结束的流才允许写缓存
        for attempt in range(2):  # 最多 2 次尝试（首包前失败可重试 1 次）
            produced = False  # 本次尝试内是否已产出分片（重试许可判据）
            try:
                async for chunk in llm.astream(messages):  # 异步流式接收 token chunk
                    if isinstance(chunk, AIMessageChunk) and chunk.content:  # 过滤空 chunk
                        text_piece = str(chunk.content)  # 统一转字符串
                        buf.append(text_piece)  # 累积完整答案
                        produced = True  # 已对外产出分片：此后失败禁止重试（防重复输出）
                        yield text_piece  # 向上游逐片输出
                completed = True  # 流自然耗尽 = 完整成功
                break  # 跳出重试循环
            except Exception as exc:  # noqa: BLE001 — 流式出口统一容错
                retryable = _is_retryable_llm_error(exc)  # 超时/5xx/429/连接失败才可重试
                if produced or not retryable or attempt == 1:  # 已产出 / 不可重试 / 重试耗尽
                    logger.warning(
                        "LLM rag stream aborted (attempt=%d produced=%s retryable=%s): %s",
                        attempt + 1,
                        produced,
                        retryable,
                        exc,
                    )  # 中断审计日志
                    if not produced:  # 用户一个字都没收到时才补兜底文案
                        yield self.settings.llm_fallback_message  # 兜底输出，流正常收尾
                    return  # 中断/兜底路径：completed 保持 False，绝不写缓存
                logger.warning(
                    "LLM rag stream retrying after pre-first-chunk failure (attempt=%d): %s",
                    attempt + 1,
                    exc,
                )  # 重试审计日志（此时 buf 必为空，无重复输出风险）

        # 流式完整结束后写缓存，便于同问题命中（B3：completed 门控）
        # 素材模式不写：答案依赖素材内容，按 question 建缓存会串答案（见 docstring）
        if completed and buf and not attachment_context:
            answer = "".join(buf)  # 完整答案（exact 与语义层共用）
            await self._cache.set_json(
                key,
                {"answer": answer, "route": "rag_llm"},
                self.settings.cache_ttl_seconds,
            )
            # Task 23：完整成功的答案同步写语义层（改述问题可命中）；兜底/中断路径不会走到这里
            if self.settings.semantic_cache_enabled and qvec is not None:
                await self._sem.store(question, qvec, answer, "rag_llm", version)

        # 引用核查证据快照（add-citation-check，design D2）：只在「流完整成功」时 append
        # 本次 contexts——与缓存写入同一门控语义（B3：中断/兜底路径没有可靠的"本次证据"）。
        # evidence_out 仅供 API 层在流结束后只读消费，禁止调用方写入（docstring 已注明）。
        if completed and evidence_out is not None:
            evidence_out.extend(contexts)

    async def _stream_simple_llm(  # 简单生成子流程：无检索上下文，仅做引导/兜底回复
        self,
        messages: list,  # 调用方传入的 LangChain 消息列表
    ) -> AsyncIterator[str]:
        """
        不带检索上下文的简单生成路径（闲聊引导 / 无命中兜底）。

        说明：
        - 本路径默认不写“专业问答同 key”缓存，避免污染专业缓存。
        - 重试/兜底语义与 _rag_stream_llm 一致（Task 19 B3）：
          首包前可重试错误重试 1 次；中断输出兜底文案后正常结束。
        """
        llm = self._llm()  # 创建 LLM 客户端用于流式生成
        buf: list[str] = []  # 收集分片文本（当前仅为占位，不用于缓存写入）
        for attempt in range(2):  # 最多 2 次尝试
            produced = False  # 本次尝试内是否已产出分片
            try:
                async for chunk in llm.astream(messages):  # 流式生成
                    if isinstance(chunk, AIMessageChunk) and chunk.content:  # 过滤空 chunk
                        text_piece = str(chunk.content)  # 统一转字符串
                        buf.append(text_piece)  # 累积
                        produced = True  # 已产出分片：此后失败禁止重试
                        yield text_piece  # 输出
                return  # 正常结束（本路径无缓存写入）
            except Exception as exc:  # noqa: BLE001 — 流式出口统一容错
                retryable = _is_retryable_llm_error(exc)  # 判断可重试性
                if produced or not retryable or attempt == 1:  # 已产出 / 不可重试 / 重试耗尽
                    logger.warning(
                        "LLM simple stream aborted (attempt=%d produced=%s retryable=%s): %s",
                        attempt + 1,
                        produced,
                        retryable,
                        exc,
                    )  # 中断审计日志
                    if not produced:  # 用户一个字都没收到时才补兜底文案
                        yield self.settings.llm_fallback_message  # 兜底输出
                    return  # 结束流（本路径本就不写缓存）
                logger.warning(
                    "LLM simple stream retrying after pre-first-chunk failure (attempt=%d): %s",
                    attempt + 1,
                    exc,
                )  # 重试审计日志
        _ = buf  # 显式占位，强调此路径不写专业缓存
