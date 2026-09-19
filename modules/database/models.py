# =============================================================================
# 这个文件定义“业务数据长什么样”，也就是 FAQ、法律文档、用户、聊天历史的表结构。  
# 后面无论离线还是在线，所有数据库操作都绕不开这里。


# -----------------------------------------------------------------------------
# 输入：无（类定义即「表结构」）；运行时由 SQLAlchemy 根据类生成 DDL 或与已有表映射。
# 输出：`Base` 元类、`FaqTab` / `LegalTab` / `UserTab` / `HisChatTab` ORM 类，供 CRUD 与 `create_all` 使用。
# 被谁调用：`mysql_loaders`（写入 FAQ/法律）、`milvus_sync`（读取行）、`pipeline`（按 id 查父文档）、
#          `memory.service`（用户与历史）、`auth.service`（注册/登录）、`lifespan`（`Base.metadata.create_all`）。
# =============================================================================
"""
SQLAlchemy 2.0 声明式 ORM：一张 Python 类 ≈ 一张 MySQL 表，属性 ≈ 列。

`Mapped[类型]` + `mapped_column(...)` 是推荐写法，IDE 能推断列类型。
"""

from __future__ import annotations

from datetime import datetime  # `default=datetime.utcnow` 插入时使用 UTC 时间

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, Text  # 列类型与索引
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship  # ORM 基类与关系 API


class Base(DeclarativeBase):
    """
    所有业务表的「共同基类」：`Base.metadata` 持有全部表结构，供 `create_all` 一次性建表。
    """
    pass  # 无需额外字段；DeclarativeBase 已提供 metadata 等机制


class FaqTab(Base):
    """
    高频问答表：离线从 Excel 导入；在线 FAQ 向量检索命中后读 answer。
    """

    __tablename__ = "faq_tab"  # 物理表名，与手写 SQL、Milvus 文档说明保持一致

    id: Mapped[int] = mapped_column(
        BigInteger,  # 主键用 BIGINT，与 Milvus INT64 id 对齐
        primary_key=True,  # 主键约束
        autoincrement=True,  # MySQL 自增；插入时可不传 id
    )  # 一行一个 id；Milvus FAQ 集合主键与此相同，便于全量重建向量

    question: Mapped[str] = mapped_column(
        Text,  # 长文本类型
        nullable=False,  # 不允许 NULL
    )  # 用户问题文本；向量索引与检索都基于该字段的语义

    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )  # 标准答案；高相似度时可不经 LLM 直接返回

    is_high_frequency: Mapped[bool] = mapped_column(
        Boolean,
        default=True,  # 新插入行默认 True
        nullable=False,
    )  # False 表示不参与 Milvus 同步（下架或内部题）

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,  # 插入时自动填当前 UTC 时间
    )  # 审计：首次入库时间

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,  # 任意 UPDATE 时自动刷新该列
    )  # 审计：最后修改时间


class LegalTab(Base):
    """
    法律文档表：同一表存 parent（长文）与 child（切片）；child.parent_id 指向 parent.id。
    """

    __tablename__ = "legal_tab"

    __table_args__ = (
        Index("ix_legal_tab_source_file", "source_file", mysql_length=191),  # 前缀索引：utf8mb4 下整列索引易超长
    )  # 按文件名筛选用

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )  # parent 行、child 行各有独立 id；仅 child 行 id 进入 Milvus

    source_file: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )  # PDF 文件名，便于展示来源

    doc_role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,  # 单列索引：按 parent/child 过滤
    )  # 约定字符串 "parent" 或 "child"

    parent_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("legal_tab.id", ondelete="CASCADE"),  # 删父行时数据库自动删子行
        nullable=True,  # parent 行自身无父，填 NULL
        index=True,
    )  # child 行必填：指向本表父行主键

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )  # 同一父下子块序号；parent 行恒为 0

    title: Mapped[str] = mapped_column(
        String(1024),
        default="",
    )  # 如「第3-5页」摘要标题

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )  # parent 存合并长文；child 存短切片

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    children: Mapped[list["LegalTab"]] = relationship(
        "LegalTab",  # 目标类名（字符串避免循环引用解析问题）
        back_populates="parent",  # 与下方 parent 关系互指
        foreign_keys="LegalTab.parent_id",  # 本关系用哪一列做外键
    )  # 从父导航到子列表

    parent: Mapped["LegalTab | None"] = relationship(
        "LegalTab",
        back_populates="children",
        remote_side="LegalTab.id",  # 指明「父」端是 id 这一侧
        foreign_keys="LegalTab.parent_id",
    )  # 从子导航回父


class UserTab(Base):
    """
    终端用户表：一行对应一个 `user_external_id`（前端 UUID 等）。
    """

    __tablename__ = "users_tab"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)  # 内部主键
    external_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,  # 唯一约束：同一 external_id 只能一行
        index=True,
    )  # 与 API 请求体字段对应
    password_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,  # NULL = 匿名用户（不可登录）；存量 49 用户零影响
    )  # 方案 B：格式 pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>，自描述单列
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    chats: Mapped[list["HisChatTab"]] = relationship(
        "HisChatTab",
        back_populates="user",
        cascade="all, delete-orphan",  # 删用户时级联删其所有聊天记录
    )


class HisChatTab(Base):
    """
    聊天历史表：每轮用户问 + 助手答一行；按 user_id 与 created_at 查询最近 N 条。
    """

    __tablename__ = "his_chat_tab"
    __table_args__ = (Index("ix_his_chat_user_created", "user_id", "created_at"),)  # 复合索引加速「某用户最近记录」

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users_tab.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )  # 外键指向 users_tab.id
    question: Mapped[str] = mapped_column(Text, nullable=False)  # 用户本轮问题（落库时可能截断）
    answer: Mapped[str] = mapped_column(Text, nullable=False)  # 助手完整回复（SSE 拼接后）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 时间序排序用

    user: Mapped["UserTab"] = relationship("UserTab", back_populates="chats")  # 多对一：多行聊天属于一个用户
