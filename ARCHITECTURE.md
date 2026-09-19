# 项目目录与模块说明

本仓库的代码布局一览（本地磁盘目录名常为 `Legal_System`）。
说明：`__pycache__`、`node_modules`、`.pytest_cache`、`.venv`、日志等为运行生成物或本地环境，不随仓库分发；
本地开发环境可能另有 `openspec/`、`.codebuddy/`、`data/`、`models/`、`项目用到的软件及工具/` 等工具或数据目录，同样不纳入仓库。

```text
Legal_System/                                # 仓库根目录
├── README.md                                # 项目介绍 / 功能 / 架构 / 启动与部署
├── ARCHITECTURE.md                          # 本文：目录与模块说明
├── PROJECT_ARCHITECTURE.md                  # 架构图（在线问答链路 / V2 入库链路 / Docker 拓扑）
├── 启动和部署.md                             # 启动顺序、依赖服务与数据入库指引
├── 软件工具及版本.md                          # 版本核对清单（运行环境、基础镜像）
├── LICENSE                                  # MIT
├── .env.example                             # 环境变量模板（复制为 .env 后填写；真实 .env 不入库）
├── .gitignore / .dockerignore
├── requirements.txt / requirements-dev.txt  # 运行时依赖 / 开发依赖（pytest、ruff）
├── pyproject.toml / uv.lock                 # 工具与依赖说明、锁定文件
│
├── backend/                                 # HTTP API 层（FastAPI），与领域逻辑解耦
│   └── app/
│       ├── main.py                          # 应用装配：CORS、/health、挂载 /api、全局异常兜底
│       ├── lifespan.py                      # 启动钩子：异步建表、Milvus ensure、可选后台同步
│       ├── deps.py                          # 依赖注入：RagPipeline 进程内单例、滑动窗口限流
│       ├── schemas.py                       # 请求 / 响应模型（ChatRequest、HealthResponse 等）
│       └── api/
│           ├── chat.py                      # POST /api/chat/stream（SSE 主问答）+ 历史回显
│           ├── attachment_chat.py           # POST /api/chat/stream-with-attachment（素材问答）
│           ├── knowledge.py                 # Legacy 上传入口（不写入 V2 集合）
│           └── auth.py                      # 注册 / 登录
│
├── modules/                                 # 领域模块（被 backend 与 offline 脚本共用）
│   ├── core/config.py                       # 全局 Settings（.env 驱动：阈值、路径、连接串）
│   ├── auth/service.py                      # 口令哈希与鉴权
│   ├── cache/
│   │   ├── redis_client.py                  # Redis 异步客户端 + 答案缓存（exact 层）
│   │   ├── semantic_cache.py                # 语义缓存（L2，可开关）
│   │   └── rate_limit.py                    # Redis 滑动窗口限流
│   ├── database/
│   │   ├── models.py                        # ORM：FAQ / Legacy / 用户 / 会话 / V2 父子块与映射
│   │   ├── session.py                       # 异步引擎与会话工厂
│   │   └── v2_parent_repository.py          # V2 父块与映射仓储（进程内缓存 + 冻结校验）
│   ├── embeddings/local_embedding.py        # 本地 BGE-M3 句向量（1024 维）
│   ├── rerank/local_rerank.py               # 本地 BGE-Reranker Cross-Encoder 重排（GPU 并发闸门）
│   ├── milvus_store/
│   │   ├── client.py                        # PyMilvus 连接
│   │   └── collections.py                   # 集合 schema、创建 / 确保存在
│   ├── memory/                              # 对话记忆：历史意图正则 + 读写与格式化
│   ├── ingestion/                           # 离线数据管道（入口脚本在 offline/scripts）
│   │   ├── document_parsing.py              # 解析器门面：按扩展名分派（pdf/docx/html/md/txt）
│   │   ├── pdf_parser_v2.py / docx_parser_v2.py / html_parser_v2.py
│   │   ├── cleaner_v2.py / document_cleaning.py      # 清洗（失败即停）
│   │   ├── chunking.py                      # 父子分块
│   │   ├── metadata_normalizer_v1.py / parsed_document_v2.py
│   │   ├── v2_incremental.py / incremental.py         # V2 增量入库
│   │   ├── milvus_sync.py / mysql_loaders.py          # Legacy 全量同步与装载
│   │   └── *.PRE_*_V1 / *_poc.py / *_backup.py        # 历史快照与 POC（备份用途）
│   └── rag/                                 # 在线 RAG 核心
│       ├── pipeline.py                      # 端到端编排：缓存 → 意图 → 检索 → 重排 → 生成
│       ├── prompts.py                       # 系统提示与用户消息拼装
│       ├── intent.py                        # 意图分流（专业问题 / 闲聊引导）
│       ├── hybrid_rrf.py                    # 稠密 + BM25 混合召回与 RRF 融合
│       ├── retrieval_contract.py            # 检索契约（Legacy / Frozen V2 双轨）
│       ├── retrieval_v2_adapter.py          # V2 稠密检索与父块适配
│       ├── corpus_scope.py                  # Canonical 白名单与元数据头（fail-closed）
│       ├── dashscope_http.py                # DashScope HTTP 客户端（连接池、代理处理）
│       ├── vision.py                        # 图片 → 文本（多模态识别，magic-byte 判格式）
│       ├── attachment.py                    # 素材分派 + 扫描件判死 + 上下文预算截断
│       ├── query_refine.py                  # 未提问时的检索要点提炼
│       └── citation_check.py                # 引用核查（四级证据比对，零 LLM）
│
├── offline/                                 # 离线流水线与评测（业务实现在 modules/ingestion）
│   ├── scripts/                             # 入口：V2 单文件 / 批量入库、冻结资产恢复、同步
│   ├── chunking_strategy_v2/                # 切块策略 A/B（脚本、产物、评审记录）
│   ├── chunk_metadata_contract_v1/          # Chunk 元数据契约（失败即停）
│   ├── retrieval_eval_v1/                   # 检索评测（问题集、离线索引、指标与失败样本）
│   ├── benchmarks/                          # 并发 / 端到端 / 质量基准
│   ├── audit/                               # 语料噪声审计
│   ├── tests/                               # 离线侧单测（pytest 收集路径见 pyproject.toml）
│   └── MIGRATION_C1_SHA256.md               # 迁移与校验记录
│
├── frozen_assets/                           # 冻结检索基线（clone 后可直接复现，无需重解析语料）
│   ├── strategy_b_chunks.json               # 4134 个 Strategy B 子块
│   ├── parent_sections.json                 # 2217 个父块（覆盖 49 个检索文档）
│   ├── child_parent_mapping.json            # 4134 条子块 → 父块映射
│   ├── embeddings.npy                       # 4134 × 1024 的 BGE-M3 预计算向量
│   ├── embedding_manifest.json              # 子块与向量位置的映射清单
│   ├── ingest_manifest_v1.jsonl             # 53 条 canonical 元数据（运行时白名单数据源）
│   └── SHA256_MANIFEST.md                   # 冻结资产完整性校验值
│
├── data_corpus/                             # 公开法律语料（法规、司法解释、合同示范文本）
│   └── picture/                             # 素材问答演示样例（中华人民共和国公司法）
├── data/ + models/                          # 本地数据源与模型权重（体积大，不入库；模型按 README 下载）
│
├── deploy/
│   ├── docker-compose.yml                   # Redis / etcd / MinIO / Milvus，及 app / app-gpu 两个 profile
│   ├── Dockerfile.backend                   # 后端镜像（CPU / GPU 由构建参数区分）
│   ├── docker-daemon-cn.json                # Docker 镜像加速配置片段（可选合并）
│   └── autodl/                              # 私有化部署（AutoDL + vLLM），见其 README
│
├── scripts/                                 # 运维与验证脚本
│   ├── init_mysql.sql / migrate_202608_add_password.sql
│   └── test_vision.py / test_file_extract.py / test_refine.py / probe_citation_faithfulness.py
│
├── docs/                                    # 设计与复盘文档
│   ├── README.md                            # 文档索引
│   ├── autodl-private-deploy.md             # 私有化部署实录（含三层校验与延迟数据）
│   ├── frozen_artifact_integration.md       # 冻结资产接入说明
│   └── thinking_ab_report.md / thinking_ab_raw_answers.md   # 思考模式 A/B 报告与原始回答
│
├── tests/                                   # 单测与集成测试
│   ├── document_parsing/                    # 各解析器单测 + 合成 fixture（PDF / DOCX / XLSX）
│   ├── ingestion/                           # 清洗、增量、V2 入库、知识 API
│   ├── integration/                         # 在线链路联调、入库冒烟、检索 V2 实测
│   ├── rag/ + unit/                         # 检索适配器、语义缓存
│   ├── test_attachment_query.py             # 素材问答（20 项）
│   ├── test_citation_check.py               # 引用核查（31 项，含真实样本回归）
│   └── frozen_artifact_integration_report.md / live_ingestion_smoke_v1.md
│
└── frontend/                                # Vue 3 + Vite 对话前端（SSE）
    ├── index.html                           # 挂载 #app
    ├── package.json / package-lock.json / vite.config.js   # 依赖、脚本与开发代理（5173 → 8000）
    ├── nginx.conf / Dockerfile / .dockerignore             # 容器构建与生产入口（8080）
    └── src/
        ├── main.js                          # createApp、全局样式
        ├── App.vue                          # 对话 UI：SSE 消费、素材上传、引用核查徽标
        └── styles.css                       # 主题与页面样式
```

> 完整架构图（在线问答链路、V2 文档入库链路、Docker 部署拓扑）见 [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)；
> 启动与部署步骤见 [`README.md`](README.md)。
