# Enterprise Legal RAG 项目架构图

本文描述 Enterprise Legal RAG 的在线问答、V2 文档入库、数据存储与 Docker 部署关系。
主检索链路使用 RETRIEVAL_CONTRACT=v2；Legacy 上传入口与 V2 入库链路保持隔离。

## 1. 总体架构

```mermaid
flowchart LR
    USER(["企业用户 / 管理员"])

    subgraph ACCESS["访问与交付层"]
        FE["Vue 3 前端<br/>Vite 开发 / Nginx 部署"]
        API["FastAPI<br/>REST API + SSE 流式响应"]
    end

    subgraph APP["应用与 RAG 编排层"]
        AUTH["认证与会话<br/>注册、登录、历史记录"]
        LIMIT["Redis 滑动窗口限流"]
        RAG["RagPipeline<br/>缓存 → 意图 → 检索 → 重排 → 生成"]
        LEGACY["Legacy 上传入口<br/>/api/knowledge/documents/upload"]
    end

    subgraph MODEL["模型层"]
        EMB["BGE-M3<br/>1024 维 Query / Child Embedding"]
        RERANK["BGE-Reranker-Large<br/>Parent Cross-Encoder 重排"]
        DASHSCOPE["Alibaba Cloud DashScope<br/>意图识别 + LLM 流式生成"]
    end

    subgraph DATA["数据与基础设施层"]
        REDIS[("Redis<br/>答案缓存、语义缓存、限流")]
        MYSQL[("MySQL<br/>用户、会话、FAQ<br/>V2 Parent / Mapping / corpus_version")]
        MILVUS[("Milvus<br/>xiaoyi_faq_highfreq<br/>xiaoyi_legal_child_v2")]
        ETCD[("etcd<br/>Milvus 元数据")]
        MINIO[("MinIO<br/>Milvus 对象存储")]
        MANIFEST[("Canonical Manifest<br/>冻结清单 / 运行态清单")]
        LEGACY_STORE[("Legacy 数据<br/>legal_tab / xiaoyi_legal_child")]
    end

    subgraph INGEST["离线 V2 入库层"]
        RAW["data_corpus<br/>PDF / DOCX / MD / TXT"]
        CORPUS["run_v2_corpus_ingest.py<br/>批量编排"]
        FILE["run_v2_file_ingest.py<br/>单文件总编排"]
        PACKAGE["build_doc_package.py<br/>Parser → Cleaner V2 → Strategy B<br/>Metadata Contract → Parent/Child Package"]
        SYNC["run_v2_doc_sync.py<br/>modules.ingestion.v2_incremental"]
        FROZEN["frozen_assets<br/>Chunks / Parents / Mapping / Embeddings"]
        BASELINE["run_v2_ingest.py --precomputed<br/>冻结基线恢复"]
        VERIFY["写入后验证<br/>Parent、Mapping、Child、Orphan、Dimension"]
        LIVE["ingest_manifest_v2_live.jsonl<br/>成功后原子更新"]
    end

    USER --> FE
    FE <-->|"HTTP / SSE"| API
    API --> AUTH
    API --> LIMIT --> RAG
    API -. "Legacy only" .-> LEGACY

    AUTH <--> MYSQL
    LIMIT <--> REDIS
    RAG <--> REDIS
    RAG <--> MYSQL
    RAG <--> MILVUS
    RAG --> EMB
    RAG --> RERANK
    RAG <--> DASHSCOPE
    MANIFEST -. "CanonicalScope 过滤与元数据" .-> RAG

    MILVUS --> ETCD
    MILVUS --> MINIO
    LEGACY -.-> LEGACY_STORE

    RAW --> FILE
    RAW --> CORPUS --> FILE
    FILE --> PACKAGE --> SYNC
    SYNC --> MYSQL
    SYNC --> EMB --> MILVUS
    SYNC --> VERIFY
    MYSQL --> VERIFY
    MILVUS --> VERIFY
    VERIFY --> LIVE --> MANIFEST

    FROZEN --> BASELINE
    BASELINE --> MYSQL
    BASELINE --> MILVUS
    FROZEN --> MANIFEST

    classDef access fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e;
    classDef app fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;
    classDef model fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef data fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef ingest fill:#ffe4e6,stroke:#e11d48,color:#881337;

    class USER,FE,API access;
    class AUTH,LIMIT,RAG,LEGACY app;
    class EMB,RERANK,DASHSCOPE model;
    class REDIS,MYSQL,MILVUS,ETCD,MINIO,MANIFEST,LEGACY_STORE data;
    class RAW,CORPUS,FILE,PACKAGE,SYNC,FROZEN,BASELINE,VERIFY,LIVE ingest;
```

## 2. 在线问答链路

```mermaid
flowchart TD
    REQ["POST /api/chat/stream"]
    RATE["身份识别与 Redis 限流"]
    EXACT{"Exact Cache 命中？"}
    SEMANTIC{"Semantic Cache 命中？"}
    MEMORY["读取 MySQL 最近会话记忆"]
    INTENT{"DashScope 意图识别"}
    SIMPLE["非专业问题引导 Prompt"]
    QEMBED["BGE-M3 Query Embedding<br/>1024 dimensions"]
    FAQ{"Milvus FAQ 检索"}
    V2DENSE["Milvus V2 Child Dense Retrieval<br/>xiaoyi_legal_child_v2"]
    BM25["对 Dense 候选执行 BM25"]
    RRF["Reciprocal Rank Fusion"]
    PARENT["external_parent_id 回查<br/>MySQL Parent Contract"]
    RR["BGE-Reranker Parent 重排"]
    META["Canonical 元数据头<br/>拼装 RAG Prompt"]
    LLM["DashScope LLM 流式生成"]
    WRITE["写入 Exact / Semantic Cache<br/>持久化聊天记录"]
    SSE["SSE 分片返回前端"]

    SCOPE[("Canonical Manifest")]
    REDIS[("Redis")]
    MYSQL[("MySQL")]

    REQ --> RATE --> EXACT
    RATE <--> REDIS
    EXACT -->|"Hit"| SSE
    EXACT -->|"Miss"| SEMANTIC
    SEMANTIC -->|"Hit"| SSE
    SEMANTIC -->|"Miss / Disabled"| MEMORY
    MEMORY <--> MYSQL
    MEMORY --> INTENT
    INTENT -->|"非专业"| SIMPLE --> LLM
    INTENT -->|"专业"| QEMBED --> FAQ
    FAQ -->|"高置信标准答案"| WRITE
    FAQ -->|"中等置信 FAQ 上下文"| LLM
    FAQ -->|"FAQ 不足"| V2DENSE
    SCOPE -. "白名单过滤" .-> V2DENSE
    V2DENSE --> BM25 --> RRF --> PARENT
    PARENT <--> MYSQL
    PARENT --> RR --> META
    SCOPE -. "Source Metadata" .-> META
    META --> LLM --> WRITE
    WRITE --> REDIS
    WRITE --> MYSQL
    WRITE --> SSE
```

## 3. V2 文档入库链路

```mermaid
flowchart LR
    subgraph NEW["新增文件"]
        RAW["data_corpus 文件"]
        BATCH["run_v2_corpus_ingest.py<br/>全批次预检 + 串行编排"]
        ONE["run_v2_file_ingest.py<br/>单文件入口"]
        SELECT["Manifest 身份选择<br/>重复与冲突保护"]
        BUILD["build_doc_package.py"]
        PARSER["Parser"]
        CLEANER["Cleaner V2<br/>失败即停止"]
        CHUNK["Strategy B Chunking"]
        CONTRACT["Chunk Metadata Contract<br/>失败即停止"]
        PACKAGE["Parent / Child Package<br/>保留稳定 child_id"]
        MODE{"运行模式"}
        DRY["Dry-run 完成<br/>不连接、不写数据库"]
        CHECK["Apply Preflight<br/>MySQL、Milvus、Schema、BGE、Identity"]
        MYSQLWRITE["MySQL Parent + Mapping<br/>corpus_version 递增"]
        EMBED["BGE-M3 Child Embedding<br/>1024 dimensions"]
        MILVUSWRITE["Milvus Insert<br/>xiaoyi_legal_child_v2"]
        POST["写入后核验<br/>Count、Lookup、Orphan=0、Dimension=1024"]
        COMMIT["成功后提交 Live Manifest<br/>提示重启后端刷新 CanonicalScope"]
        PASS["INGESTION_STATUS=PASS"]
    end

    subgraph BASE["冻结资产复现"]
        FROZEN["frozen_assets<br/>4134 Children / 2217 Parents<br/>4134 × 1024 Embeddings"]
        RESTORE["run_v2_ingest.py --precomputed"]
        BASEMYSQL["MySQL V2 Tables"]
        BASEMILVUS["Milvus V2 Collection<br/>直接使用预计算 Embeddings"]
        FINAL["FINAL_CHECK=PASS"]
    end

    RAW --> ONE
    RAW --> BATCH --> ONE
    ONE --> SELECT --> BUILD
    BUILD --> PARSER --> CLEANER --> CHUNK --> CONTRACT --> PACKAGE --> MODE
    MODE -->|"默认 / --dry-run"| DRY
    MODE -->|"--apply"| CHECK
    CHECK --> MYSQLWRITE --> EMBED --> MILVUSWRITE --> POST --> COMMIT --> PASS

    FROZEN --> RESTORE
    RESTORE --> BASEMYSQL --> FINAL
    RESTORE --> BASEMILVUS --> FINAL
```

## 4. Docker 部署拓扑

```mermaid
flowchart LR
    BROWSER["Browser"]
    FRONTEND["frontend<br/>Nginx :8080"]
    BACKEND["backend / backend-gpu<br/>FastAPI :8000"]
    REDIS["Redis :6379"]
    MILVUS["Milvus :19530 / :9091"]
    ETCD["etcd :2379"]
    MINIO["MinIO :9000"]
    MYSQL["MySQL / RDS :3306"]
    MODELS["Host models volume<br/>bge-m3 / bge-reranker-large"]
    DASH["DashScope API"]

    BROWSER --> FRONTEND -->|"Nginx reverse proxy /api"| BACKEND
    BACKEND --> REDIS
    BACKEND --> MILVUS
    BACKEND --> MYSQL
    BACKEND --> MODELS
    BACKEND --> DASH
    MILVUS --> ETCD
    MILVUS --> MINIO
```

## 5. 架构边界

- V2 日常入库只使用 run_v2_file_ingest.py 或 run_v2_corpus_ingest.py。
- frozen_assets 通过 run_v2_ingest.py --precomputed 恢复标准 V2 数据基线。
- /api/knowledge/documents/upload 是 Legacy 入口，不写入 xiaoyi_legal_child_v2。
- V2 检索由 Milvus Child 命中、MySQL Parent 回查、BM25/RRF 融合和 BGE Reranker 组成。
- Live Manifest 只在 MySQL 与 Milvus 写入并核验成功后正式更新。
- Manifest 更新后需要重启后端，使进程缓存的 CanonicalScope 重新加载。
- Milvus 使用 etcd 保存元数据，并使用 MinIO 保存对象数据。
