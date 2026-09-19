# =============================================================================
# 这个文件定义“Milvus 里有什么集合、字段长什么样、索引怎么建”。  
# 它是向量库的数据结构层。
# “后面你看到任何模块拿 Milvus 集合，都是 `Collection(name)`，这就是统一集合访问入口。”

# -----------------------------------------------------------------------------
# 输入：向量维度 `dim`（与 BGE-M3 输出长度一致）；依赖已连上的 Milvus。
# 输出：在 Milvus 中创建或加载集合 `xiaoyi_faq_highfreq`、`xiaoyi_legal_child`；或删后重建。
# 被谁调用：`milvus_sync._blocking_milvus_full_write`、`lifespan`（若启动时预建）、
#          任何需要先保证集合 schema 存在的代码路径。
# =============================================================================
"""
集合 schema 与索引：字段名、顺序、类型必须与 `milvus_sync` 里 `insert` 的列顺序严格一致。

索引类型 HNSW + 度量 COSINE 与在线 `Collection.search` 的 param 一致。
"""

from __future__ import annotations

import logging  # 记录创建/加载/删除集合

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility  # 高层 API：集合是否存在、删表、建 schema

from modules.milvus_store.client import ensure_milvus  # 先保证 TCP 连接存在

logger = logging.getLogger(__name__)

COLLECTION_FAQ = "xiaoyi_faq_highfreq"  # FAQ 向量集合常量，全项目引用同一名字避免拼写错误
COLLECTION_LEGAL_CHILD = "xiaoyi_legal_child"  # 法律子块向量集合名


def _faq_fields(dim: int) -> list[FieldSchema]:
    """
    定义 FAQ 集合各列：主键 id + 文本冗余字段 + 向量列。

    入参:
        dim: 向量列 embedding 的维度。
    返回:
        Milvus `FieldSchema` 列表，顺序与 insert 列顺序一致。
    """
    return [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,  # 64 位整数主键
            is_primary=True,  # 声明为主键列
            auto_id=False,  # False：插入时由业务显式提供 id（与 MySQL 对齐）
            description="与 MySQL faq_tab.id 对齐，便于幂等 upsert/热更新",
        ),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=4096),  # 可变长字符串，上限 4096
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=32768),  # 答案可能很长
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,  # 浮点向量类型
            dim=dim,  # 向量维度，须与模型输出一致
        ),
    ]


def _legal_child_fields(dim: int) -> list[FieldSchema]:
    """
    法律子块集合：id + 回溯字段 + 文本 + 向量。

    入参:
        dim: 向量列 embedding 的维度。
    返回:
        法律子块集合的 `FieldSchema` 列表。
    """
    return [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="与 MySQL legal_tab.id（子块行）对齐",
        ),
        FieldSchema(name="parent_id", dtype=DataType.INT64, description="父文档 legal_tab.id"),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),  # 子块正文
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]


def _create_index(collection: Collection) -> None:
    """
    在向量列上建 HNSW 近似最近邻索引；`metric_type` 与 search 时 COSINE 一致。

    入参:
        collection: 已创建且含 embedding 列的 Milvus 集合。
    返回:
        无。
    """
    index = {
        "index_type": "HNSW",  # 分层小世界图，检索快
        "metric_type": "COSINE",  # 余弦相似度：向量已归一化时常用
        "params": {"M": 16, "efConstruction": 200},  # HNSW 建索引图时的超参，越大建索引越慢、图越密
    }  # 字典传给 Milvus
    collection.create_index(field_name="embedding", index_params=index)  # 只对 embedding 列建索引


def _drop_if_exists(name: str) -> None:
    """
    若集合存在则物理删除（数据清空）；用于全量重建前清理。

    入参:
        name: Milvus 集合名字符串。
    返回:
        无；删除成功时打 INFO 日志。
    """
    if utility.has_collection(name):  # 布尔：是否存在该名的 collection
        utility.drop_collection(name)  # 删除元数据与数据文件（视 Milvus 版本与存储而定）
        logger.info("dropped milvus collection: %s", name)  # 审计日志


def create_collections_if_not_exist(dim: int) -> None:
    """
    幂等：集合已存在则 `load` 进内存；不存在则建 schema、建索引、load。

    入参:
        dim: 向量维度，与 FAQ/法律子块两套集合共用。
    返回:
        无。
    """
    ensure_milvus()  # 无连接则先连
    for name, fields_fn in (
        (COLLECTION_FAQ, _faq_fields),  # 元组：集合名 + 字段生成函数
        (COLLECTION_LEGAL_CHILD, _legal_child_fields),
    ):  # 依次处理两个集合
        if utility.has_collection(name):  # 已存在：可能是上次运行留下的
            col = Collection(name)  # 绑定已有集合
            col.load()  # 把 segment 加载到 query node，否则 search 报错
            logger.info("milvus collection exists (loaded): %s", name)  # 打日志
            continue  # 跳过创建分支

        fields = fields_fn(dim)  # 调用 _faq_fields(dim) 或 _legal_child_fields(dim)
        schema = CollectionSchema(fields=fields, description=f"xiaoyi::{name}")  # 封装成 schema 对象
        collection = Collection(name=name, schema=schema)  # 在 Milvus 里注册新集合
        _create_index(collection)  # 建向量索引
        collection.load()  # 加载以便插入与检索
        logger.info("created milvus collection: %s dim=%s", name, dim)


def recreate_collections(dim: int) -> None:
    """
    破坏性操作：先删两个集合再调用 `create_collections_if_not_exist`，保证与当前代码 schema 一致。

    入参:
        dim: 向量维度。
    返回:
        无。
    """
    ensure_milvus()  # 先连接，否则 utility 无法工作
    _drop_if_exists(COLLECTION_FAQ)  # 删 FAQ 集合
    _drop_if_exists(COLLECTION_LEGAL_CHILD)  # 删法律集合
    create_collections_if_not_exist(dim)  # 按当前 dim 重建
