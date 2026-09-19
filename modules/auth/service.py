# =============================================================================
# -----------------------------------------------------------------------------
# 输入：明文密码 / `pbkdf2_sha256$...` 存储串；注册登录的 (username, password, bind_external_id)。
# 输出：`hash_password` / `verify_password` 纯函数；`register_user` / `authenticate_user` 异步 DB 操作。
# 被谁调用：`backend/app/api/auth.py`（POST /api/auth/register 与 /api/auth/login）。
# =============================================================================
"""
账号密码认证服务（方案 B）：哈希纯函数 + users_tab 读写，与 modules/memory/service.py 同构分层。

架构裁决落地（任务裁定，勿再讨论）：
- D1 零新依赖：hashlib.pbkdf2_hmac + hmac.compare_digest；禁 pyjwt/bcrypt/passlib。
- D2 身份复用：登录/注册只返回 external_id，由前端写回 localStorage，下游聊天/历史/记忆链路零改动。
- D4 匿名升级：bind_external_id 命中且该行无密码 → 原行改 external_id（user_id 不变 → 历史无缝）。

安全红线：任何日志不得出现明文密码或哈希串（本模块不打日志）。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from modules.database.models import UserTab

PBKDF2_ITERATIONS = 100_000  # OWASP 对 PBKDF2-SHA256 的推荐量级；调参只需改此常量（存储串自描述迭代数）
SALT_BYTES = 16  # 128-bit 随机盐：同一密码每次哈希结果不同（防预计算彩虹表）

# 用户不存在/匿名行时也跑一次同代价校验，抹平「用户名存在性」时序侧信道（与 401 统一文案配套防枚举）
_DUMMY_HASH = ""  # 延迟初始化：首次用到时生成，避免 import 期白付 100k 轮迭代


class UsernameTakenError(Exception):
    """注册用户名已被占用（users_tab.external_id 唯一冲突；API 层转 409）。"""


def hash_password(password: str) -> str:
    """
    明文密码 → 自描述存储串。

    入参:
        password: 明文密码（调用方已完成 8-72 长度校验）。
    返回:
        形如 `pbkdf2_sha256$100000$<salt_hex>$<hash_hex>` 的字符串。
    """
    salt = secrets.token_bytes(SALT_BYTES)  # 每次随机盐
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """
    校验明文密码与存储串是否匹配（防时序攻击）。

    入参:
        password: 用户本次输入的明文。
        stored: 库中 `pbkdf2_sha256$...` 存储串。
    返回:
        匹配 True；格式非法/不匹配 False（不抛异常，失败语义统一交给调用方）。
    """
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False  # 未知算法前缀：按不匹配处理，不升级降级
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), hash_hex)  # 常数时间比较，防逐字节时序探测
    except (ValueError, AttributeError):
        return False  # 存储串字段数不足 / 非 str：按不匹配处理


def _dummy_verify(password: str) -> None:
    """恒定耗时的无效校验：用户不存在或匿名行时调用，让响应时间不泄露用户名存在性。"""
    global _DUMMY_HASH
    if not _DUMMY_HASH:
        _DUMMY_HASH = hash_password("xiaoyi-timing-equalizer")  # 仅用于耗时对齐，非真实凭证
    verify_password(password, _DUMMY_HASH)


async def register_user(
    session: AsyncSession,
    username: str,
    password: str,
    bind_external_id: str | None = None,
) -> dict:
    """
    注册新用户；可选把既有匿名 UUID 行原地升级（D4，历史保留）。

    语义（按序）：
    1. username 已被占用（external_id 唯一）→ raise UsernameTakenError（API 层 409）；
    2. bind_external_id 命中且该行 password_hash IS NULL → UPDATE 原行：
       external_id 改为 username、写入哈希；user_id 不变 → his_chat_tab 历史无缝保留；
    3. bind 未命中 / 该行已有密码 → 常规 INSERT 新行。

    入参:
        session: 异步 ORM 会话（本函数只 flush，commit 由路由层负责）。
        username: 已通过 schema 正则校验的用户名（将成为 external_id）。
        password: 明文密码（路由层已校验 8-72 长度）。
        bind_external_id: 前端旧匿名 UUID，可选。
    返回:
        {user_id, external_id}。
    """
    # 1) 用户名占用预检查（注册用户名与匿名 UUID 共用 external_id 命名空间）
    res = await session.execute(select(UserTab).where(UserTab.external_id == username))
    if res.scalar_one_or_none() is not None:
        raise UsernameTakenError(username)

    password_hash = hash_password(password)  # 先算哈希，UPDATE/INSERT 两路共用

    # 2) 匿名升级：命中且无密码 → 原行 UPDATE（历史保留）
    bind = (bind_external_id or "").strip()
    if bind:
        res = await session.execute(select(UserTab).where(UserTab.external_id == bind))
        row = res.scalar_one_or_none()
        if row is not None and row.password_hash is None:
            row.external_id = username  # UUID → 用户名（目标名冲突已由步骤 1 排除）
            row.password_hash = password_hash
            await session.flush()
            return {"user_id": int(row.id), "external_id": username}
        # 未命中或已有密码：走常规 INSERT（老账号不能被他人 bind 劫持）

    # 3) 常规 INSERT 新行
    u = UserTab(external_id=username, password_hash=password_hash)
    session.add(u)
    await session.flush()
    return {"user_id": int(u.id), "external_id": username}


async def authenticate_user(session: AsyncSession, username: str, password: str) -> dict | None:
    """
    登录校验：用户名+密码 → 用户信息；任一环节失败返回 None（不区分原因，防枚举）。

    入参:
        session: 异步 ORM 会话（只读使用，不 commit）。
        username: 用户名。
        password: 明文密码。
    返回:
        匹配 → {user_id, external_id}；用户不存在 / 匿名行（无密码）/ 密码错 → None。
    """
    res = await session.execute(select(UserTab).where(UserTab.external_id == username))
    row = res.scalar_one_or_none()
    if row is None or row.password_hash is None:
        _dummy_verify(password)  # 与真实校验等耗时：抹平用户名存在性时序信号
        return None
    if not verify_password(password, row.password_hash):
        return None
    return {"user_id": int(row.id), "external_id": row.external_id}
