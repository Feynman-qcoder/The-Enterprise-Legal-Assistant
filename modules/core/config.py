# =============================================================================
# 这个文件解决的是“全项目配置从哪里来、怎么统一读”的问题。  
# 如果没有它，后面每个文件都要自己解析环境变量，代码会非常乱。
# “后面你看到任何模块拿配置，都是 `get_settings()`，这就是统一配置入口。”
# -----------------------------------------------------------------------------
# 输入：进程环境变量、项目根目录下的 `.env` 文件（键名与下方 Field 的 alias 一致）。
# 输出：`Settings` 数据类实例（字段已校验、类型已转换）；`get_settings()` 返回进程内单例。
# 被谁调用：几乎全项目——`session.py`（MySQL DSN）、`redis_client`、`milvus`、
#          `RagPipeline`、`ChatOpenAI` 构造、CORS、`lifespan` 热更新间隔等。
# =============================================================================
"""
本模块用 pydantic-settings 把「.env + 环境变量」变成类型安全的 Python 对象。

技巧：`@lru_cache` 包住 `get_settings()`，保证全进程只解析一次磁盘，避免每次请求都读文件。
"""

from functools import lru_cache  # 装饰器：把「无参函数」的返回值缓存起来，重复调用直接返回旧值

from pydantic import Field, field_validator  # Field：声明字段默认值与 env 别名；field_validator：自定义校验
from pydantic_settings import BaseSettings, SettingsConfigDict  # BaseSettings：可自动从环境变量填充；SettingsConfigDict：模型级配置


class Settings(BaseSettings):
    """
    一条配置 = 一个类属性；`alias` 必须与 `.env` 里的大写变量名一致，pydantic 才会自动映射。
    """

    model_config = SettingsConfigDict(
        env_file=".env",  # 告诉 pydantic：启动时尝试从「当前工作目录」下的 .env 读入变量
        env_file_encoding="utf-8",  # .env 文件按 UTF-8 解码，避免中文路径或注释乱码
        extra="ignore",  # .env 里若有多余键（例如你只临时 export 了别的变量），不报错直接忽略
    )

    # ----- DashScope（阿里云 OpenAI 兼容接口）：意图模型 + 主生成模型 -----
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")  # 空字符串表示未配置：意图模块会降级跳过 API
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )  # LangChain ChatOpenAI 的 base_url，指向 DashScope 兼容 OpenAI SDK 的入口
    llm_model: str = Field(default="deepseek-v4-pro-0813", alias="LLM_MODEL")  # 流式回答用的大模型名
    intent_model: str = Field(default="deepseek-v4-pro-0813", alias="INTENT_MODEL")  # 只做 true/false 分类，用小模型省成本

    # ----- LLM 流式容错（Task 19 B3）：中断/重试耗尽时输出的兜底文案 -----
    llm_fallback_message: str = Field(
        default="抱歉，回答生成服务暂时不可用，请稍后重试。",
        alias="LLM_FALLBACK_MESSAGE",
    )  # 兜底路径的结果不写缓存，仅保证流式出口有礼貌输出

    # ----- MySQL：异步驱动 aiomysql 的连接四要素 -----
    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")  # 数据库主机，本机常用 127.0.0.1
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")  # 端口，MySQL 默认 3306
    mysql_user: str = Field(default="root", alias="MYSQL_USER")  # 登录用户
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")  # 数据库密码；空串表示无密码（仅开发环境）
    mysql_database: str = Field(default="xiaoyi_rag", alias="MYSQL_DATABASE")  # 库名，与建库 SQL 一致

    # ----- Redis：字符串 URL，内含库号 / 密码（若有）-----
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")  # 默认本机 6379，数据库编号 0
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")  # 问答缓存过期时间，单位秒

    # ----- 语义缓存（Task 23 两级缓存 L2）：exact 缓存的向量泛化 -----
    semantic_cache_enabled: bool = Field(default=False, alias="SEMANTIC_CACHE_ENABLED")  # 总开关；False 时执行路径与现状逐字节等价
    semantic_cache_threshold: float = Field(default=0.97, alias="SEMANTIC_CACHE_THRESHOLD")  # 余弦命中阈值；标定依据 offline/scripts/calibrate_semantic_threshold.py：gold 集两两 max_offdiag=0.9868 + 0.03 余度被 cap 至 0.97（0.92 初值不够防「章/节」近义陷阱）
    semantic_cache_max_entries: int = Field(default=200, alias="SEMANTIC_CACHE_MAX_ENTRIES")  # 单版本 Hash 最大条目数；超出按写入时间驱逐最旧

    # ----- 限流（Task 21 Part 2）：Redis ZSET 滑动窗口，fail-open -----
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")  # 总开关；关闭即完全放行
    rate_limit_per_minute: int = Field(default=12, alias="RATE_LIMIT_PER_MINUTE")  # 窗口内最大请求数；依据 E2E C@VU5 自然速率 ~7 req/min，12 只拦滥用突发
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")  # 滑动窗口宽度（秒）

    # ----- Milvus：向量库 gRPC 地址 -----
    milvus_host: str = Field(default="127.0.0.1", alias="MILVUS_HOST")  # Docker 映射到本机时常为 127.0.0.1
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")  # Milvus 默认监听端口
    # 【新增开关·向后兼容】非空则改用 uri 连接：支持 Milvus Lite 本地文件
    # （如 /root/autodl-tmp/milvus.db）或任意完整 URI。
    # 留空 = 沿用上方 host/port 连接，行为与改造前逐字节一致。
    # 动机：AutoDL 等嵌套容器环境无法运行 Docker，装不了 Milvus standalone。
    #
    # ⚠️ 别名刻意使用 MILVUS_LITE_URI，**不要**改成 MILVUS_URI：
    #    MILVUS_URI 是 pymilvus 自己的环境变量（Config.MILVUS_URI），且 pymilvus 会在
    #    import 阶段从「当前工作目录的 .env」读取并校验它，只接受 http(s):// 形式。
    #    若用同名 key 传本地文件路径，进程连 `import pymilvus` 都会抛
    #    ConnectionConfigException: Illegal uri（已实测踩坑）。
    milvus_uri: str = Field(default="", alias="MILVUS_LITE_URI")
    milvus_user: str = Field(default="", alias="MILVUS_USER")  # 单机无鉴权时留空字符串
    milvus_password: str = Field(default="", alias="MILVUS_PASSWORD")  # 同上，空表示不传密码给 pymilvus

    # ----- 本地 Transformer 权重目录（相对「仓库根」的路径字符串）-----
    embedding_model_path: str = Field(default="models/bge-m3", alias="EMBEDDING_MODEL_PATH")  # BGE-M3 句向量模型文件夹
    rerank_model_path: str = Field(
        default="models/bge-reranker-large",
        alias="RERANK_MODEL_PATH",
    )  # CrossEncoder 重排模型文件夹
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")  # 推理设备：cpu 或 cuda:0 等

    # ----- Rerank 实验开关（Settings 化，Config C 转正 V2）：默认值 = 原行为 -----
    rerank_device: str = Field(default="cpu", alias="RERANK_DEVICE")  # reranker 设备：cpu=现状 / cuda=GPU FP16（Config C）
    rerank_pool_top_n: int | None = Field(  # rerank 候选池按 RRF 顺序截断的前 N 篇父文档；None = 全量（现状）
        default=None,
        alias="RERANK_POOL_TOP_N",
    )
    rerank_timeout_seconds: float = Field(  # rerank 调用超时（Task 21 Part 3）：超时/异常降级为 RRF 序 top-N
        default=30.0,
        alias="RERANK_TIMEOUT_SECONDS",
    )  # 默认 30s：CPU top-12 模式 rerank mean ~13s 不误触发；GPU 316ms 正常永不超时

    # ----- 热更新：定时把 MySQL 全量刷到 Milvus（简化版一致性）-----
    hot_update_enabled: bool = Field(default=True, alias="HOT_UPDATE_ENABLED")  # True：lifespan 里起后台任务
    hot_update_interval_seconds: int = Field(default=0, alias="HOT_UPDATE_INTERVAL_SECONDS")  # <=0 时由 effective_* 回退为 60

    # ----- 素材问答（OpenSpec add-attachment-query）：上传图片/文件 → 提取文本 → 作为上下文 -----
    # 全部默认关闭（attachment_enabled=False → 新端点返回 503），留空/关闭时现有链路行为不变。
    attachment_enabled: bool = Field(default=False, alias="ATTACHMENT_ENABLED")  # 素材问答总开关；False 时端点 503
    vision_model: str = Field(default="deepseek-v4.1-flash", alias="VISION_MODEL")  # 图片识别用的多模态模型（复用 DASHSCOPE_BASE_URL/KEY）
    attachment_max_bytes: int = Field(default=10 * 1024 * 1024, alias="ATTACHMENT_MAX_BYTES")  # 单文件上传上限（字节）；10MB（实测 9MB 图可识别）
    attachment_timeout_seconds: float = Field(default=60.0, alias="ATTACHMENT_TIMEOUT_SECONDS")  # 提取（VLM/解析）超时；Phase 0 实测 2.4~13.1s，留余量
    attachment_min_text_chars: int = Field(default=20, alias="ATTACHMENT_MIN_TEXT_CHARS")  # 提取文本低于此字符数判失败（fail-closed）
    attachment_context_max_chars: int = Field(default=60000, alias="ATTACHMENT_CONTEXT_MAX_CHARS")  # 素材文本注入提示词的上限；超出截断并在 transcription 告知（民法典 16.5 万字符会触发）

    # ----- 引用核查（OpenSpec add-citation-check，观察模式：只报告不改答案）-----
    citation_check_enabled: bool = Field(default=False, alias="CITATION_CHECK_ENABLED")  # 总开关；false 时 SSE 与现状逐字节一致（红线）
    citation_check_text_threshold: float = Field(default=0.8, alias="CITATION_CHECK_TEXT_THRESHOLD")  # 文本相似度阈值；数字守卫先于此生效（金额/比例漂移直接 mismatch）

    # ----- Canonical Corpus Scope：53 篇正式语料白名单（在线隔离 + 元数据 header 数据源）-----
    canonical_manifest_path: str = Field(
        default="frozen_assets/ingest_manifest_v1.jsonl",
        alias="CANONICAL_MANIFEST_PATH",
    )  # 只读加载；缺失时 corpus_scope 会 fail-closed 报错，而不是静默放开过滤。
    # Task 20：默认值相对仓库根（由 corpus_scope 锚定解析），绝对路径仅经 .env 覆盖。

    # ----- Legal Retrieval Contract：Legacy / Frozen V2 显式双轨，默认不自动切换 -----
    retrieval_contract: str = Field(default="legacy_v1", alias="RETRIEVAL_CONTRACT")
    retrieval_v2_collection: str = Field(
        default="xiaoyi_legal_child_v2",
        alias="RETRIEVAL_V2_COLLECTION",
    )

    # ----- 法律混合检索：稠密 + BM25 + RRF 的规模参数 -----
    legal_hybrid_bm25_enabled: bool = Field(default=True, alias="LEGAL_HYBRID_BM25_ENABLED")  # False 则只做向量一路排序
    hybrid_dense_candidate_k: int = Field(default=60, alias="HYBRID_DENSE_CANDIDATE_K")  # Milvus 向量检索先取 Top-K 子块
    hybrid_bm25_candidate_k: int = Field(default=60, alias="HYBRID_BM25_CANDIDATE_K")  # BM25 排序后截断，再与稠密路做 RRF
    hybrid_rrf_k: int = Field(default=60, alias="HYBRID_RRF_K")  # RRF 公式里的 k--RRF 衰减系数，越大高名次衰减越慢

    # ----- FAQ：COSINE 相似度与「距离阈值」的换算在 pipeline 里完成 -----
    faq_direct_distance_threshold: float = Field(default=0.01, alias="FAQ_DIRECT_DIST_THRESH")  # 直达答案：相似度 >= 1-该值
    faq_llm_distance_threshold: float = Field(default=0.15, alias="FAQ_LLM_DIST_THRESH")  # 拼进 LLM：相似度 >= 1-该值
    faq_top_k_for_llm: int = Field(default=3, alias="FAQ_TOP_K_FOR_LLM")  # 最多几条 FAQ 片段进上下文
    legal_rerank_top_n: int = Field(default=5, alias="LEGAL_RERANK_TOP_N")  # 父文档重排后取前 N 篇全文进 LLM

    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )  # 浏览器 Origin 白名单，逗号分隔；默认允许本地开发的前端（5173 端口）调用接口，避免跨域报错

    @property
    def mysql_dsn_async(self) -> str:
        """
        SQLAlchemy 异步 URL：协议头必须是 mysql+aiomysql，后面跟用户名密码主机库名。

        入参:
            无（读取当前 `Settings` 实例字段）。
        返回:
            `mysql+aiomysql://...` 形式的异步数据库连接 URL 字符串。
        """
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"  # 用户名密码中的特殊字符需 URL 编码（此处未编码，密码勿含 @ 等）
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"  # 主机:端口/库名
        )

    @property
    def effective_hot_update_interval_seconds(self) -> int:
        """
        若用户把间隔配成 0 或负数，这里统一回退 60 秒，避免 while 忙等打满 CPU。

        入参:
            无。
        返回:
            实际使用的热更新间隔秒数（正整数）。
        """
        if self.hot_update_interval_seconds <= 0:  # 非法或非正间隔
            return 60  # 安全默认值：每分钟最多全量同步一次
        return self.hot_update_interval_seconds  # 否则尊重用户配置

    def cors_origin_list(self) -> list[str]:
        """
        CORSMiddleware 需要 Python 列表；把逗号分隔字符串拆成去空格后的列表。

        入参:
            无。
        返回:
            非空的 Origin 字符串列表，供 CORS 白名单。
        """
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]  # 过滤空段，避免白名单里出现 ""

    #这是 Pydantic v2 的写法，在 Settings 从环境变量/.env 填好字段之后、正式生效之前，对指定字段再做一道检查。
    @field_validator("faq_direct_distance_threshold", "faq_llm_distance_threshold") #这两个字段的值都会交给下面的函数校验：不允许负数，否则相似度换算会乱。
    @classmethod
    def positive_thresh(cls, v: float) -> float:
        """
        阈值在业务上表示「与完全匹配的偏差」，不允许负数，否则相似度换算会乱。

        入参:
            cls: Pydantic 校验器约定的类对象。
            v: 待校验的阈值原始浮点值。
        返回:
            校验通过后的同一浮点值；若 v<0 则抛出 ValueError。
        """
        if v < 0:  # 负数无物理意义
            raise ValueError("threshold must be non-negative")  # 启动时直接失败，强迫修正 .env
        return v  # 校验通过原样返回

    @field_validator("semantic_cache_threshold")
    @classmethod
    def valid_semantic_cache_threshold(cls, v: float) -> float:
        """
        语义缓存余弦阈值必须落在 (0, 1]：0 会吞掉所有条目、>1 永不命中且暴露配置错误。

        入参:
            cls: Pydantic 校验器约定的类对象。
            v: 待校验的阈值原始浮点值。
        返回:
            校验通过后的同一浮点值；非法区间抛 ValueError（启动即失败）。
        """
        if not 0 < v <= 1:
            raise ValueError("semantic_cache_threshold must be in (0, 1]")
        return v

    @field_validator("semantic_cache_max_entries")
    @classmethod
    def valid_semantic_cache_max_entries(cls, v: int) -> int:
        """
        容量上限至少为 1：0 或负数会让每次写入都触发全量驱逐，缓存形同虚设。

        入参:
            cls: Pydantic 校验器约定的类对象。
            v: 待校验的最大条目数。
        返回:
            校验通过后的同一整数；<1 抛 ValueError（启动即失败）。
        """
        if v < 1:
            raise ValueError("semantic_cache_max_entries must be >= 1")
        return v

    @field_validator("retrieval_contract")
    @classmethod
    def valid_retrieval_contract(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"legacy_v1", "v2"}:
            raise ValueError("retrieval_contract must be 'legacy_v1' or 'v2'")
        return normalized


@lru_cache
def get_settings() -> Settings:
    """
    全项目统一入口：第一次调用时构造 Settings()，之后永远返回同一对象实例。

    入参:
        无。
    返回:
        进程内缓存的 `Settings` 单例。
    """
    return Settings()  # 触发 pydantic 从环境变量 + .env 填充字段


ASSISTANT_NAME = "小意"  # 常量：prompts 与兜底文案里引用，修改一处即可改助手昵称
