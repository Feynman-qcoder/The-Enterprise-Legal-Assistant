# =============================================================================
# -----------------------------------------------------------------------------
# 输入：HTTP POST JSON 体（`RegisterRequest` / `LoginRequest`）。
# 输出：`AuthResponse`（注册 201 / 登录 200）：{user_external_id, username}。
# 被谁调用：浏览器登录/注册卡片 `fetch("/api/auth/...")`；路由由 `main.py` 挂载到 `/api` 前缀下。
# =============================================================================
"""
方案 B 认证路由：注册 / 登录。身份锚点仍为 external_id（裁决 D2），前端拿回后写 localStorage。

- 两者均挂 enforce_rate_limit（裁决 D6）：body 无 user_external_id → 自动 IP 回落（deps.py 现有实现零改动复用）。
- 错误语义（裁决 D7）：409 用户名已存在；401 统一文案「用户名或密码错误」（防用户名枚举）。
- 除上述业务语义外，其余异常不自行兜底：冒泡给 Task 21 全局处理器（结构化 500）。
- 红线：密码/哈希不落日志（本文件不打任何含凭证的日志）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from backend.app.deps import enforce_rate_limit  # 限流 dependency（Task 21 Part 2，直接复用）
from backend.app.schemas import AuthResponse, LoginRequest, RegisterRequest  # Pydantic 校验
from modules.auth.service import UsernameTakenError, authenticate_user, register_user  # 认证服务层
from modules.database.session import get_session_factory  # 异步会话工厂

router = APIRouter(prefix="/auth", tags=["auth"])  # 完整路径 = /api + /auth + /register|/login

USERNAME_TAKEN_MSG = "用户名已被占用，请换一个。"
BAD_CREDENTIALS_MSG = "用户名或密码错误。"  # 统一文案：不区分「用户不存在 / 密码错」


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit)],
)
async def register(body: RegisterRequest) -> AuthResponse:
    """
    注册（可选匿名升级绑定，裁决 D4）。

    契约：
    - 成功 → 201 {user_external_id, username}；前端写回 localStorage 后进入聊天页；
    - 用户名冲突 → 409（预检查 + 唯一约束兜底双层防护）；
    - bind_external_id 命中无密码老行 → 原行升级，聊天历史无缝保留（对前端透明）。

    入参:
        body: 已校验的注册请求体（username / password / 可选 bind_external_id）。
    返回:
        `AuthResponse`。
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            user = await register_user(session, body.username, body.password, body.bind_external_id)
            await session.commit()  # service 层只 flush，事务边界收口在路由层
        except UsernameTakenError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=USERNAME_TAKEN_MSG)
        except IntegrityError:  # 并发同名注册穿过预检查：DB 唯一约束兜底转 409
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=USERNAME_TAKEN_MSG)
    return AuthResponse(user_external_id=user["external_id"], username=body.username)


@router.post(
    "/login",
    response_model=AuthResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def login(body: LoginRequest) -> AuthResponse:
    """
    登录。

    契约：
    - 成功 → 200 {user_external_id, username}；
    - 用户不存在 / 匿名行 / 密码错误 → 401 统一文案（防枚举）；
    - 超限 → 429 + Retry-After（enforce_rate_limit，IP 回落）。

    入参:
        body: 登录请求体（username / password）。
    返回:
        `AuthResponse`。
    """
    factory = get_session_factory()
    async with factory() as session:
        user = await authenticate_user(session, body.username, body.password)  # 只读，无需 commit
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=BAD_CREDENTIALS_MSG)
    return AuthResponse(user_external_id=user["external_id"], username=body.username)
