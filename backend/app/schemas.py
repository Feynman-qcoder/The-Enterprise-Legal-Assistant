"""
Pydantic 模型：
核心作用是用 Pydantic 做数据校验，这是 FastAPI 项目的最佳实践 —— 前端传过来的参数、后端返回的结果，
都要通过 Pydantic 模型约束，既保证数据合法性，又能自动生成 OpenAPI 文档。
"""

from pydantic import BaseModel, Field  # BaseModel：声明数据类；Field：字段约束、默认值与 OpenAPI 文档


class ChatRequest(BaseModel):
    """
    前端 POST /api/chat/stream 的 JSON 体。

    - `message`：本轮用户输入。
    - `user_external_id`：可选；用于绑定 `users_tab` / `his_chat_tab`。不传则等同匿名纯 RAG（不落库、不按用户查历史）。
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="用户自然语言问题",
    )  # ... 表示必填；长度限制防止超大正文撑爆上下文
    user_external_id: str | None = Field(
        default=None,
        max_length=128,
        description="客户端稳定用户标识（建议 UUID）；缺省时不写入聊天历史、不检索记忆",
    )  # None：匿名；非空：启用记忆持久化与自述历史类查询


class HealthResponse(BaseModel):
    """GET /health 返回的简单状态（便于负载均衡或探活）。
    是/health接口的响应模型，固定返回ok=True表示进程存活，assistant字段是对外展示的名称，和前端约定一致即可。
    """

    ok: bool = True  # 固定为 True 表示进程存活（尚未对接 DB/Milvus 深度检查）
    assistant: str = "小意"  # 对外展示名称，可与前端约定一致


class IngestionFileResponse(BaseModel):
    filename: str
    status: str
    knowledge_type: str
    parents: int = 0
    chunks: int = 0
    rows: int = 0
    parent_ids: list[int] = Field(default_factory=list)
    record_ids: list[int] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None
    size_bytes: int = 0
    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    embedding_ms: float = 0.0
    mysql_ms: float = 0.0
    milvus_ms: float = 0.0
    total_ms: float = 0.0


class IngestionBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    files: list[IngestionFileResponse]


class ChatHistoryItem(BaseModel):
    """GET /api/chat/history 单条历史记录（方案 A 历史回显，插入任务）。"""

    question: str = Field(default="", description="用户当轮问题")
    answer: str = Field(default="", description="助手完整回复")
    created_at: str | None = Field(default=None, description="落库时间（ISO 8601 字符串）")


class ChatHistoryResponse(BaseModel):
    """GET /api/chat/history 响应体：历史列表 + 回显条数。"""

    items: list[ChatHistoryItem] = Field(default_factory=list, description="按时间升序的历史记录")
    count: int = 0


class RegisterRequest(BaseModel):
    """
    POST /api/auth/register 的 JSON 体（方案 B 登录注册）。

    - `username`：3-32 位字母/数字/下划线/连字符（与前端同一正则，前后端双校验，裁决 D7）。
    - `password`：8-72 字符（72 上限对齐行业惯例，纯防滥用）。
    - `bind_external_id`：可选；前端旧匿名 UUID，命中无密码行则原地升级（聊天历史无缝保留，裁决 D4）。
    """

    username: str = Field(
        ...,
        min_length=3,
        max_length=32,
        pattern=r"^[a-zA-Z0-9_\-]{3,32}$",
        description="注册用户名（同时将成为该账号的 external_id）",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="明文密码（服务端只存 pbkdf2 哈希，不落日志）",
    )
    bind_external_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选：旧匿名 UUID，绑定升级后聊天历史无缝保留",
    )


class LoginRequest(BaseModel):
    """
    POST /api/auth/login 的 JSON 体。

    刻意宽松校验（只做防滥用边界）：格式不合法的用户名/密码不可能注册成功，
    统一按凭证错误 401 处理，不向前端泄露「格式对不对」之外的信息。
    """

    username: str = Field(..., min_length=1, max_length=128, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="明文密码")


class AuthResponse(BaseModel):
    """
    注册(201)/登录(200)共用响应（方案 B）。

    前端把 `user_external_id` 写回 localStorage['xiaoyi_user_external_id']（裁决 D2）：
    身份锚点不变，下游聊天/历史/记忆链路零改动。
    """

    user_external_id: str = Field(..., description="该账号的 external_id（= 用户名）")
    username: str = Field(..., description="回显用户名")
