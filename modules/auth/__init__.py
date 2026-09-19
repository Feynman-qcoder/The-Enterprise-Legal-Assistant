"""
认证子包（方案 B）：账号密码注册 / 登录。

导入策略：
- `from modules.auth.service import hash_password, verify_password` —— 纯函数，无 DB 依赖（单测友好）。
- `from modules.auth.service import register_user, authenticate_user, UsernameTakenError` —— 依赖 SQLAlchemy 与 `users_tab`。

零新依赖：pbkdf2（标准库 hashlib）实现，禁 pyjwt/bcrypt/passlib（任务红线）。
"""

from modules.auth.service import (
    UsernameTakenError,
    authenticate_user,
    hash_password,
    register_user,
    verify_password,
)

__all__ = [
    "UsernameTakenError",
    "authenticate_user",
    "hash_password",
    "register_user",
    "verify_password",
]
