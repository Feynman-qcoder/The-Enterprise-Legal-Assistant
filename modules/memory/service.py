# =============================================================================
# 用户聊天数据的数据库访问层，核心围绕 users_tab（用户表）和 his_chat_tab（聊天记录表）做
# 用户 ID 解析、聊天记录查询、记录格式化、聊天记录落库 4 个核心操作。
# -----------------------------------------------------------------------------
# 输入：`AsyncSession` + `external_id` / `user_id`；或 SSE 结束后的 `question`/`answer` 文本。
# 输出：`resolve_user_id` 返回 int；`fetch_recent_chat_lines` 返回 ORM 行列表；`persist_user_turn` 无返回（落库）。
# 被谁调用：`pipeline.stream_chat`（只读+commit）；`api/chat.py` 的 `persist_user_turn`（写 his_chat）。
# =============================================================================
"""
用户与聊天历史的数据访问层：与 `users_tab` / `his_chat_tab` 表一一对应。

`resolve_user_id` 故意不 commit，便于与同一 session 内其它查询共事务。

方案 B 追加（用户要求）：`persist_user_turn` 落库后按用户裁剪历史，
每用户仅保留最近 `HISTORY_RETAIN_LIMIT=50` 条对话（更早的删除）；游客/匿名（无 external_id）零落库。
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from modules.database.models import HisChatTab, UserTab

DEFAULT_MEMORY_CONTEXT_LINES = 10  # 与 pipeline 里 fetch_recent_chat_lines 的 limit 一致

HISTORY_RETAIN_LIMIT = 50  # 每用户历史保留上限（用户要求：最多保留 50 条对话；与前端回显 limit=50 对齐）


async def resolve_user_id(session: AsyncSession, external_id: str) -> int:
    """
    主要作用：根据前端传的「用户外部标识（external_id）」，在数据库里找对应的内部主键 ID；如果用户是新的（没查到），
    就自动创建一条用户记录并返回新生成的 ID。

    SELECT users_tab by external_id；无则 INSERT 一行并 flush 得到自增 id。
    不在此 commit：调用方负责 commit（pipeline 在读完记忆后 commit 一次）。

    入参:
        session: 异步 ORM 会话。
        external_id: 前端传入的用户外部唯一标识（非空字符串）。
    返回:
        `users_tab.id` 整型主键；`external_id` 为空时抛出 ValueError。
    """
    ext = external_id.strip()  # 去空白
    if not ext:  # 空串非法
        raise ValueError("external_id empty")
    res = await session.execute(select(UserTab).where(UserTab.external_id == ext))  # 异步查询
    row = res.scalar_one_or_none()  # 0 行 None，多行会抛异常（unique 约束下不应发生）
    if row is not None:  # 已注册过
        return int(row.id)  # 返回内部主键
    u = UserTab(external_id=ext)  # 新建 ORM 对象
    session.add(u)  # 挂到 session
    await session.flush()  # 发 INSERT 并回填 u.id，但不结束事务
    return int(u.id)  # 新用户 id


async def fetch_recent_chat_lines(
    session: AsyncSession,
    user_id: int,
    limit: int = DEFAULT_MEMORY_CONTEXT_LINES,
) -> list[HisChatTab]:
    """
    主要作用：拿着 user_id 去数据库里查最近 limit 条聊天记录，并按时间正序排列。
    ORDER BY created_at DESC LIMIT n，再在 Python 里 reverse，使列表按时间正序（旧→新）。

    正序便于 Prompt 里写「从早到晚」。

    入参:
        session: 异步 ORM 会话。
        user_id: `users_tab` 主键。
        limit: 最多返回的记录条数（默认与管线常量一致）。
    返回:
        按时间升序排列的 `HisChatTab` ORM 对象列表。
    """
    res = await session.execute(
        select(HisChatTab)
        .where(HisChatTab.user_id == user_id)  # 只查该用户
        .order_by(HisChatTab.created_at.desc())  # 新的在前
        .limit(limit),  # 最多 n 条
    )
    rows = list(res.scalars().all())  # materialize
    rows.reverse()  # 原地反转为时间升序
    return rows


def format_chat_history_for_prompt(rows: list[HisChatTab]) -> str:
    """
    主要作用：把查到的聊天记录列表，转换成可以直接拼到 AI Prompt 里的中文字符串（有固定格式）；如果没记录，返回固定提示句。
    无记录时返回固定提示句；有记录则格式化为「序号. 用户问：… 助手答：…」多行文本。

    对超长 question/answer 做截断，避免撑爆模型上下文。

    入参:
        rows: 已按时间排序的聊天记录 ORM 行列表。
    返回:
        可直接拼入 Prompt 的多行中文说明字符串；无记录时为固定占位句。
    """
    if not rows:  # 新用户从未对话
        return "（当前尚无已存储的聊天记录。）"
    lines: list[str] = []
    for i, r in enumerate(rows, start=1):  # 展示序号从 1 起
        q = (r.question or "").replace("\n", " ").strip()[:2000]  # 换行压空格，限制长度
        a = (r.answer or "").replace("\n", " ").strip()[:4000]
        lines.append(f"{i}. 用户问：{q}\n   助手答：{a}")  # 两行一条记录，缩进对齐助手行
    return "\n".join(lines)  # 单个大字符串


async def _prune_history(session: AsyncSession, user_id: int, keep: int = HISTORY_RETAIN_LIMIT) -> int:
    """
    历史保留上限裁剪（方案 B 用户要求）：删除该用户第 keep 条之前的旧记录，返回删除行数。

    按 id 倒序保留最新 keep 条（id 自增单调，与写入顺序一致，避免 created_at 同秒并列的歧义）。
    命中 (user_id, created_at) 复合索引前缀，单用户行数 ≤ keep+1，代价可忽略。

    入参:
        session: 异步 ORM 会话（独立事务，自带 commit）。
        user_id: `users_tab` 主键。
        keep: 保留条数（默认 50）。
    返回:
        实际删除的行数（0 = 未超限，无 DELETE 发出）。
    """
    res = await session.execute(
        select(HisChatTab.id)
        .where(HisChatTab.user_id == user_id)
        .order_by(HisChatTab.id.desc())
        .offset(keep)  # 跳过最新 keep 条，剩下的都是待删旧记录
    )
    stale_ids = [row[0] for row in res.all()]
    if not stale_ids:
        return 0
    await session.execute(delete(HisChatTab).where(HisChatTab.id.in_(stale_ids)))
    await session.commit()
    return len(stale_ids)


async def persist_user_turn(external_id: str | None, question: str, answer: str) -> None:
    """
    主要作用：把用户本轮的「问题 + 回答」写入数据库的聊天记录表；如果用户 ID 为空或回答为空，就不写入（避免无效数据）。
    独立开 session：INSERT his_chat_tab 一行并 commit。

    无 external_id 或空答案则 no-op，避免写入无意义行。
    方案 B 追加：落库成功后调用 _prune_history 裁剪到最近 50 条（用户要求：登录用户最多保留 50 条对话消息）。

    入参:
        external_id: 用户外部 id；None 或空白时不写入（游客/匿名 → 零落库）。
        question: 用户本轮问题（会按长度截断后入库）。
        answer: 助手完整回复（会按长度截断后入库）；空白时不写入。
    返回:
        无。
    """
    if not external_id or not external_id.strip() or not (answer or "").strip():  # 任一条件不满足
        return  # 直接返回，不发 SQL
    from modules.database.session import get_session_factory  # 函数内导入，避免循环 import

    factory = get_session_factory()
    async with factory() as session:  # 新会话 = 新事务，与 pipeline 内 session 隔离
        uid = await resolve_user_id(session, external_id.strip())  # 确保 users 表有主键
        session.add(
            HisChatTab(
                user_id=uid,  # 外键
                question=question[:8000],  # 与 API 校验上限对齐
                answer=answer[:65000],  # 极长 SSE 拼接仍留余量
            ),
        )
        await session.commit()  # 立即可被后续 fetch_recent_chat_lines 读到
        await _prune_history(session, uid)  # 超过 50 条时删除更早记录


async def fetch_chat_history(
    session: AsyncSession,
    user_id: int,
    limit: int = 50,
) -> list[dict]:
    """
    历史回显专用只读查询（方案 A，插入任务）：返回该用户最近 limit 条对话的字典列表。

    与 fetch_recent_chat_lines 的关系：同构（时间升序）但不复用——
    后者 limit=10 语义属 prompt 记忆，与前端回显的 50/200 上限语义勿混用。

    入参:
        session: 异步 ORM 会话（只读使用，不 commit）。
        user_id: `users_tab` 主键。
        limit: 最多返回条数（调用方负责上限钳制）。
    返回:
        [{question, answer, created_at}, ...] 按时间升序；无记录返回空列表。
    """
    res = await session.execute(
        select(HisChatTab)
        .where(HisChatTab.user_id == user_id)
        .order_by(HisChatTab.created_at.desc())  # 新的在前
        .limit(limit),
    )
    rows = list(res.scalars().all())
    rows.reverse()  # 升序（旧→新），与 fetch_recent_chat_lines 同构
    return [
        {
            "question": r.question or "",
            "answer": r.answer or "",
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
