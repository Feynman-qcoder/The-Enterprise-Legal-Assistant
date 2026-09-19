# Enterprise Legal RAG

Enterprise Legal RAG 是面向企业法务场景的检索增强生成系统，使用 FastAPI、Vue、MySQL、
Redis、Milvus、BGE-M3、BGE-Reranker 和 DashScope LLM，提供混合检索、重排、流式回答、
图片/文件上传与引用核查。

---

## 一、项目简介

**定位**：面向企业法务、合规与合同管理场景的检索增强生成（RAG）系统。回答基于实际检索到的
证据并注明条款出处；检索不到时明确告知，不编造法条。语料为冻结的 53 篇 canonical 公开法律文件
（49 个检索文档 / 2217 个父块 / 4134 个子块）。

**技术栈**

| 层次 | 组件 |
|---|---|
| 后端 | Python 3.11 · FastAPI · SQLAlchemy 2.x Async · SSE 流式输出 |
| 前端 | Vue 3 · Vite（开发 `5173` / Nginx 容器 `8080`） |
| 检索 | Milvus（HNSW）· BGE-M3（1024 维）· BM25 · RRF 融合 · BGE-Reranker-Large 重排 |
| 存储 | MySQL 8（父块 / 映射 / 会话）· Redis（两级缓存 · 滑动窗口限流） |
| 模型服务 | DashScope（意图识别 + LLM 流式生成 + 多模态图片识别） |
| 部署 | 本地 bare-metal · Docker Compose（CPU / GPU profile）· AutoDL + vLLM 私有化 |

**核心能力**

- **混合检索**：稠密向量（BGE-M3）+ BM25 双路召回，RRF 融合，父子两层检索，BGE-Reranker 重排（§2.1）
- **多模态问答**：上传合同照片 / 法条截图（多模态模型识别）或 PDF/DOCX/TXT/MD 文档，
  提取文本作为上下文注入提示词——可直接问「这份合同的违约责任是怎么约定的」，
  回答会引用**该材料本身的具体条款**，而不是只返回通用法条（§2.2）
- **引用核查**：回答中的法条引用（如《劳动合同法》第三十条）逐条与本次检索证据比对，
  输出四级有据状态（grounded / text_mismatch / article_missing / law_not_in_evidence）；
  观察模式——只报告，绝不改写答案（§2.3）
- **数据治理**：冻结资产（frozen_assets）+ SHA256 血缘 + fail-closed 门禁，入库全程可复现（§2.4）
- **私有化部署**：AutoDL + vLLM 自托管 Qwen3.8-27B-FP8，**业务代码零改动**（§4.5）

---

## 二、项目功能

### 2.1 混合检索与父子两层

在线检索主链路：

```text
Query → 意图分流 → BGE-M3 编码 ─┬─ 稠密向量检索（Milvus，Child 子块）
                                └─ BM25 关键词检索
      → RRF 融合 → 父文档回查（MySQL）→ BGE-Reranker 重排
      → 注入来源元数据头 → LLM 流式生成（SSE）
```

| 机制 | 说明 |
|---|---|
| 双路召回 + RRF | 稠密与稀疏两路按**名次**融合（不依赖各路的绝对分数）；候选数与融合参数均可配 |
| 父子两层检索 | 用 child 子块精确命中，回查**父文档全文**送 LLM——兼顾定位精度与上下文完整性 |
| Rerank 候选池截断 | 按 RRF 顺序取前 N 篇父文档进重排（`RERANK_POOL_TOP_N`，默认 `12`） |
| 来源元数据头 | 重排后为每篇父文档注入 manifest 元数据（文档名 / 来源机构 / **生效日期** 等）；lookup 失败保持裸文本，绝不编造字段 |
| FAQ 分支 | 高置信 FAQ 直接作答（不走 LLM）；中等置信作为上下文；不足则进入法律检索 |
| 两级缓存 | exact（问题字面）+ semantic（语义近似，可选开启）；缓存键含语料版本，换版整体失效 |
| 意图路由 | 非专业问题走引导话术，不进入检索 |
| 限流 | Redis 滑动窗口（fail-open） |

### 2.2 多模态功能（上传图片 / 文档提问）

上传一份合同照片、法条截图或 PDF/DOCX/TXT/MD 文档，系统提取其中的文字**作为上下文注入提示词**，
同时用你的问题走**现有 RAG 检索**补充法规依据——因此可以直接问「这份合同的违约责任是怎么约定的」，
回答会引用**该素材本身的具体条款**，而不是只返回通用法条。检索主链（召回→RRF→Rerank）零改动。

#### 使用方式

- 前端：聊天页点「＋上传」选择文件（≤10MB），输入框可留空直接发送（未提问时系统使用缺省
  分析指令，并在回复前展示实际使用的检索 query）。
- API：`POST /api/chat/stream-with-attachment`（multipart，字段 `file` + 可选 `question`）。
  SSE 先推 `transcription` 事件（素材类型 / 提取字符数 / 截断提示 / 实际检索 query），
  再推回答分片 `chunk`，最后 `[DONE]`。
- 支持格式：图片 `jpg/png/webp`（按文件内容 magic bytes 探测真实格式，不信扩展名）+
  文档 `pdf/docx/txt/md`；`.doc` 需本机 LibreOffice，不可用时返回明确错误。

#### 配置（默认整体关闭）

| 键 | 默认值 | 说明 |
|---|---|---|
| `ATTACHMENT_ENABLED` | `false` | 总开关；`false` 时新端点返回 503，现有功能完全不受影响 |
| `VISION_MODEL` | `deepseek-v4.1-flash` | 图片识别用的多模态模型（复用 DASHSCOPE 凭证，零新增配置） |
| `ATTACHMENT_MAX_BYTES` | `10485760` | 单文件上限（10MB），超限返回 413 |
| `ATTACHMENT_TIMEOUT_SECONDS` | `60` | 提取（识别/解析）超时秒数 |
| `ATTACHMENT_MIN_TEXT_CHARS` | `20` | 提取文本低于此字符数判失败（fail-closed，不进入生成） |
| `ATTACHMENT_CONTEXT_MAX_CHARS` | `60000` | 素材文本注入提示词的字符上限；超出截断并在 `transcription` 中告知 |

#### 行为边界（重要）

- **本能力走 DashScope 云端多模态 API**，与本地开发模式的文本生成一致；
  **不属于 §4.5 的私有化部署能力**——私有化叙事（vLLM 自托管）仅覆盖文本问答链路。
- 素材只作本轮上下文：**不写入 Milvus/MySQL，不进入对话历史**（历史仅存提取文本与回答）。
- 提取失败 fail-closed：模糊图片、扫描件 PDF（无文本层）、空文档等返回明确错误提示，
  绝不带垃圾文本进入生成。
- 超长文档（如民法典全文 16.5 万字符）按 `ATTACHMENT_CONTEXT_MAX_CHARS` 截断，
  回答基于前部分内容，`transcription` 事件中明确标注。

### 2.3 引用核查（回答引用的逐条验证）

回答里的法条引用（如《劳动合同法》第三十条）会被逐条提取，与**本次检索到的证据**
（素材路径下还包括上传材料文本）比对，输出每条引用的有据状态，随 SSE 推送
`citation_report` 事件并在前端回答末尾显示核查徽标。

**定位：纵深防御的第二道防线。** 系统已有实测有效的 prompt 层证据规则（模型拒绝编造、
正确排除近似法条）；本能力把“概率性防护”升级为“机制性防护”，把人工抽检
（读答案→对原文→确认命中）自动化、常开化。

#### 四级状态

| 状态 | 含义 |
|---|---|
| `grounded` | 法名在证据中，条号在证据中，引文文本吻合（或相似度达阈值） |
| `text_mismatch` | 法名与条号均在，但引文与证据原文差异超阈值（改写/数字漂移） |
| `article_missing` | 法名在证据中，但该条号不存在（疑似张冠李戴或编造——**最高风险**） |
| `law_not_in_evidence` | 该法名未出现在任何证据中（疑似越出证据引用） |

文本比对为四层漏斗：整段包含 → 句级包含（容忍跳款/跳句的部分引用）→
数字守卫（引文中的金额/比例/期限数字在证据中不存在时直接判 mismatch）→
滑窗相似度（阈值 `CITATION_CHECK_TEXT_THRESHOLD`，默认 0.8）。

#### 配置与事件

| 键 | 默认值 | 说明 |
|---|---|---|
| `CITATION_CHECK_ENABLED` | `false` | 总开关；关闭时 SSE 输出与无此功能时**逐字节一致** |
| `CITATION_CHECK_TEXT_THRESHOLD` | `0.8` | 文本相似度阈值（数字守卫先于此生效） |

事件格式（回答分片之后、`[DONE]` 之前）：

```json
{"type":"citation_report",
 "summary":{"total":5,"grounded":5,"text_mismatch":0,"article_missing":0,"law_not_in_evidence":0},
 "citations":[{"law":"…","article":30,"clause":null,"status":"grounded","evidence_snippet":"第三十条 …"}]}
```

同时输出一行可聚合的审计日志：
`INFO:modules.rag.citation_check:citation_report total=2 grounded=2 mismatch=0 missing=0 no_evidence=0 ms=1`

#### 已知限制（v1 观察模式）

- **只报告，不改答案**：核查器异常时 fail-open（跳过事件，不影响回答流）；
  拦截/重试是 v2，且应与置信度门禁统一设计触发条件。
- 只覆盖《法名》+第X条族引用形态；“该法第五条”类跨句指代不计（漏提不产生错误状态）。
- **验证对象仅为本次证据**：缓存命中的回答没有本次检索证据，不发核查报告
  （对该类答案发报告会系统性误导）；这是 grounding 语义的固有权衡。
- 改写/句序重组的真实引文可能被判 `text_mismatch`（保守面）——观察期攒误报率后再定分层阈值。
- 提取与比对全部为正则/中文数字解析/文本算法，**零 LLM 调用**（确定性即本能力的价值）。

### 2.4 数据治理（冻结语料与可复现入库）

- **冻结资产**（`frozen_assets/`）：项目检索基线的快照——4134 个子块、2217 个父块、
  映射关系与 BGE-M3 预计算向量。第三方 clone 后无需重新解析原始语料即可复现一致数据（§4.2）。
- **SHA256 血缘**：每个 canonical 文件带 `normalized_sha256`；冻结资产有
  `SHA256_MANIFEST.md` 校验值，可离线验证完整性。
- **canonical 白名单**：在线检索经 `CanonicalScope` 按冻结清单过滤，范围外内容不进入检索。
- **fail-closed 门禁**：清单缺失/不一致时直接报错而非静默放开；解析到扫描件页插入占位符
  而不伪造文本（`no text fabricated`）；入库写入后有 Count/Lookup/Orphan/Dimension 全套核验。
- **链路隔离**：V2 日常入库只走 `run_v2_file_ingest.py` / `run_v2_corpus_ingest.py`；
  `/api/knowledge/documents/upload` 是 Legacy 入口，不写入 V2 集合。

---

## 三、项目架构

### 3.1 分层架构

| 层 | 组成 |
|---|---|
| **访问与交付层** | Vue 3 前端（Vite 开发 / Nginx 部署）· FastAPI（REST + SSE 流式响应） |
| **应用与 RAG 编排层** | `RagPipeline`（缓存 → 意图 → 检索 → 重排 → 生成）· 认证与会话 · Redis 滑动窗口限流 · Legacy 上传入口 |
| **模型层** | BGE-M3（Query / Child 1024 维嵌入）· BGE-Reranker-Large（父块 Cross-Encoder 重排）· DashScope（意图识别 + LLM 流式生成 + 多模态识别） |
| **数据与基础设施层** | Redis（答案 / 语义缓存 · 限流）· MySQL（用户 / 会话 / FAQ / V2 父子块与映射 / corpus_version）· Milvus（`xiaoyi_faq_highfreq`、`xiaoyi_legal_child_v2`）· etcd + MinIO（Milvus 元数据与对象存储）· Canonical Manifest（冻结 / 运行态清单） |
| **离线 V2 入库层** | `run_v2_corpus_ingest.py`（批量）· `run_v2_file_ingest.py`（单文件）· Parser → Cleaner V2 → Strategy B 切块 → Chunk Metadata Contract → 父子包 → 写入与核验 → Live Manifest |

### 3.2 在线问答链路

两个入口共用同一条 RAG 主干：

- `POST /api/chat/stream` —— 纯文字问答
- `POST /api/chat/stream-with-attachment` —— 素材问答：图片 / 文档 → 提取文本作为上下文注入提示词；
  

```text
企业用户 / 管理员
    │
    ▼
Vue 3 前端 ──HTTP/SSE──▶ FastAPI ──▶ 认证 / 限流 ──▶ RagPipeline
                                                      │
      ┌───────────────────────────────────────────────┤
      ▼                                               ▼
  exact 缓存 ──miss──▶ 语义缓存 ──miss──▶ 意图分流 ──▶ 检索
                                                      │
                          ┌───────────────────────────┴──────────────┐
                          ▼                                          ▼
                   FAQ 分支（两级阈值）                        法律检索分支（V2）
                   高置信直达 / 中置信作上下文                  dense + BM25 → RRF
                                                               → 父文档回查（MySQL）
                                                               → BGE-Reranker 重排
                                                               → 注入来源元数据头
                          └──────────────┬───────────────────────────┘
                                         ▼
                                  LLM 流式生成（SSE）
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                        写回两级缓存           写会话历史（MySQL）
                                         （可选）引用核查 → citation_report 事件
```

### 3.3 目录与模块

```text
Legal_System/
├── backend/app/            # HTTP API 层（FastAPI），与领域逻辑解耦
│   ├── main.py             # 应用装配：CORS、/health、挂载 /api
│   ├── lifespan.py         # 启动：异步建表、Milvus ensure、可选后台同步
│   ├── deps.py             # RagPipeline 进程内单例注入、滑动窗口限流
│   ├── schemas.py          # 请求 / 响应模型
│   └── api/                # chat · attachment_chat（素材问答）· auth · knowledge
├── modules/                # 领域模块（backend 与 offline 脚本共用）
│   ├── core/config.py      # 全局 Settings（.env 驱动）
│   ├── auth/               # 口令哈希与鉴权
│   ├── rag/                # pipeline · prompts · intent · hybrid_rrf · retrieval_contract
│   │                       # · retrieval_v2_adapter · corpus_scope · dashscope_http
│   │                       # · vision · attachment · query_refine · citation_check
│   ├── embeddings/         # BGE-M3 本地句向量
│   ├── rerank/             # BGE-Reranker 本地重排
│   ├── milvus_store/       # 连接与集合 schema
│   ├── cache/              # exact + 语义两级缓存、滑动窗口限流
│   ├── database/           # ORM 模型、异步会话、V2 父块仓储
│   ├── ingestion/          # 离线管道：Parser(PDF/DOCX/HTML) · Cleaner V2
│   │                       # · 切块 · 增量入库
│   └── memory/             # 对话历史
├── frontend/               # Vue 3 + Vite（App.vue 单页；Dockerfile + nginx.conf）
├── offline/                # 离线流水线与评测
│   ├── scripts/            # 入库 / 冻结资产恢复 / 同步入口
│   ├── chunking_strategy_v2/ · chunk_metadata_contract_v1/
│   ├── retrieval_eval_v1/ · benchmarks/ · audit/ · tests/
├── frozen_assets/          # 冻结检索基线（子块 / 父块 / 映射 / 预计算向量 / SHA256）
├── data_corpus/            # 公开法律语料（法规 / 司法解释 / 合同示范文本）
├── deploy/                 # Dockerfile · docker-compose.yml · autodl/（私有化部署）
├── scripts/                # 运维与验证脚本（建库 SQL、图片识别 / 引用核查探针）
├── docs/                   # 设计与复盘文档
├── tests/                  # 单测与集成测试（素材问答 20 项 · 引用核查 31 项）
└── （根目录另有 README / ARCHITECTURE / PROJECT_ARCHITECTURE / 启动和部署 / 软件工具及版本）
```

**三张完整架构图**（总体架构 / V2 入库链路 / Docker 拓扑）见
[`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)；
逐目录逐文件注释见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 四、启动与部署

### 4.1 本地安装

#### 4.1.1 环境要求

- Python 3.11
- Node.js 18 或更高版本
- MySQL 8
- Docker Desktop 或 Docker Engine（包含 Docker Compose v2）
- 至少 8 GiB 可用磁盘空间
- NVIDIA GPU 可选；CPU 环境也可运行

#### 4.1.2 克隆项目并安装 Python 依赖

```bash
git clone https://github.com/Feynman-qcoder/The-Enterprise-Legal-Assistant.git
cd The-Enterprise-Legal-Assistant
conda create -n xiaoyi_rag python=3.11 -y
conda activate xiaoyi_rag
pip install -r requirements.txt
```

#### 4.1.3 启动 Redis 与 Milvus

在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```

默认开放：

- Redis：`6379`
- Milvus：`19530`
- Milvus 健康检查：`9091`

#### 4.1.4 准备 MySQL

创建数据库和项目账号，密码请替换为实际值：

```sql
CREATE DATABASE IF NOT EXISTS xiaoyi_rag
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'xiaoyi'@'%' IDENTIFIED BY 'YOUR_MYSQL_PASSWORD';
GRANT ALL PRIVILEGES ON xiaoyi_rag.* TO 'xiaoyi'@'%';
FLUSH PRIVILEGES;
```

#### 4.1.5 配置环境变量

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Linux：

```bash
cp .env.example .env
```

编辑 `.env`，至少确认以下配置：

```ini
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=xiaoyi
MYSQL_PASSWORD=YOUR_MYSQL_PASSWORD
MYSQL_DATABASE=xiaoyi_rag

REDIS_URL=redis://127.0.0.1:6379/0
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

RETRIEVAL_CONTRACT=v2
RETRIEVAL_V2_COLLECTION=xiaoyi_legal_child_v2
CANONICAL_MANIFEST_PATH=frozen_assets/ingest_manifest_v1.jsonl
HOT_UPDATE_ENABLED=false

EMBEDDING_MODEL_PATH=models/bge-m3
RERANK_MODEL_PATH=models/bge-reranker-large
EMBEDDING_DEVICE=cpu
RERANK_DEVICE=cpu
```

使用 NVIDIA GPU 时，可把模型设备调整为 `cuda` 或 `cuda:0`。

> **可选：用 Milvus Lite 替代 Milvus standalone。**
> 默认按上方的 `MILVUS_HOST` / `MILVUS_PORT` 连接 Milvus 服务，本地与 Docker 部署都无需改动。
> 只有在无法运行 Docker 的环境（如 AutoDL 等嵌套容器）才需要切换：设置
> `MILVUS_LITE_URI=/绝对路径/milvus.db` 即可改用 Milvus Lite 本地文件。
> 该变量**留空（默认）时行为与不设置完全一致**，不影响本节的本地启动流程。
> ⚠️ 变量名必须是 `MILVUS_LITE_URI`，**不要**写成 `MILVUS_URI` —— 后者是 pymilvus
> 自身的环境变量，pymilvus 会在 import 阶段从当前工作目录的 `.env` 读取并校验它，
> 只接受 `http(s)://` 形式，传本地路径会导致连 `import pymilvus` 都失败。

> 素材问答与引用核查的配置项见 §2.2 / §2.3，两者**默认关闭**，不影响本节的启动流程。

#### 4.1.6 下载模型

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-large --local-dir models/bge-reranker-large
```

首次启动前，请先按 §4.2 完成冻结资产复现。

### 4.2 冻结资产复现（首次启动前必做）

`frozen_assets/` 保存了项目的 V2 检索基线。第三方 clone 项目后，可直接使用这些资产恢复
与项目一致的父子块、映射关系和 BGE-M3 预计算向量，不需要重新解析原始语料。

#### 4.2.1 冻结资产组成

| 文件 | 内容 |
|---|---|
| `strategy_b_chunks.json` | 4134 个 Strategy B child chunks |
| `parent_sections.json` | 2217 个 parent sections，覆盖 49 个检索文档 |
| `child_parent_mapping.json` | 4134 条 child-parent 映射 |
| `embeddings.npy` | 4134×1024 的 BGE-M3 预计算向量 |
| `embedding_manifest.json` | chunk 与向量位置的映射 |
| `ingest_manifest_v1.jsonl` | 53 条 canonical source metadata |
| `SHA256_MANIFEST.md` | 冻结资产完整性校验值 |

#### 4.2.2 校验冻结资产

Windows PowerShell：

```powershell
Get-ChildItem frozen_assets -File | ForEach-Object {
  "{0}  {1}" -f `
    (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
}
```

Linux：

```bash
sha256sum frozen_assets/strategy_b_chunks.json \
  frozen_assets/parent_sections.json \
  frozen_assets/child_parent_mapping.json \
  frozen_assets/embeddings.npy \
  frozen_assets/embedding_manifest.json \
  frozen_assets/ingest_manifest_v1.jsonl
```

校验结果以 `frozen_assets/SHA256_MANIFEST.md` 为准。

#### 4.2.3 写入 MySQL 与 Milvus

确认 MySQL 和 Milvus 已启动、`.env` 已配置，然后在项目根目录执行：

```bash
conda activate xiaoyi_rag
python offline/scripts/run_v2_ingest.py --precomputed
```

`--precomputed` 直接使用冻结的 `embeddings.npy`，可以快速恢复与项目一致的向量数据。
脚本会创建或确认 V2 表和 `xiaoyi_legal_child_v2` collection，并通过既有 V2 写入流程完成灌入。

复现完成后的校验结果为：

| 校验项 | 预期值 |
|---|---:|
| MySQL `legal_parent_contract_v1` | 2217 |
| MySQL distinct `document_identity` | 49 |
| MySQL `legal_child_parent_map_v1` | 4134 |
| Milvus `xiaoyi_legal_child_v2` | 4134 |
| embedding dimension | 1024 |
| `corpus_meta.corpus_version` | ≥ 1 |

控制台最终应显示：

```text
FINAL_CHECK=PASS
```

仅在明确需要重新灌入已经包含数据的 V2 collection 时使用：

```bash
python offline/scripts/run_v2_ingest.py --precomputed --confirm-prod
```

### 4.3 启动后端与前端

#### 4.3.1 启动后端

```bash
conda activate xiaoyi_rag
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

后端健康检查：

```text
http://127.0.0.1:8000/health
```

#### 4.3.2 启动前端

新开一个终端，在项目根目录执行：

```bash
cd frontend
npm ci
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

### 4.4 Docker 部署

Docker 部署复用根目录的 `.env`、宿主机 MySQL、`models/` 模型目录，以及已经写入
MySQL/Milvus 的冻结语料。

#### 4.4.1 部署准备

1. 按 §4.1.4～§4.1.6 准备 MySQL、`.env` 和模型权重。
2. 执行 §4.2 的冻结资产复现命令。
3. 确保容器可以访问 MySQL。Docker Desktop 可使用 `host.docker.internal`；云服务器可使用
   MySQL 或 RDS 的内网地址，并同步设置 `deploy/docker-compose.yml` 中应用服务的
   `MYSQL_HOST`。

#### 4.4.2 CPU 部署

在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.yml --profile app up -d --build
```

服务地址：

- 前端：`http://服务器地址:8080`
- 后端：`http://服务器地址:8000`
- 后端健康检查：`http://服务器地址:8000/health`

#### 4.4.3 GPU 部署

宿主机安装好 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit 后执行：

```bash
docker compose -f deploy/docker-compose.yml --profile app-gpu up -d --build
```

GPU 部署使用 `backend-gpu` 服务，并通过宿主机的 `models/` 目录加载模型权重。

#### 4.4.4 常用管理命令

查看服务状态：

```bash
docker compose -f deploy/docker-compose.yml ps
```

查看应用日志：

```bash
docker compose -f deploy/docker-compose.yml logs -f backend frontend
```

GPU 部署查看后端日志：

```bash
docker compose -f deploy/docker-compose.yml logs -f backend-gpu frontend
```

停止服务：

```bash
docker compose -f deploy/docker-compose.yml --profile app --profile app-gpu down
```

### 4.5 私有化部署（AutoDL）

把线上调用的 DashScope 云端千问，换成**本机自托管的 Qwen3.8-27B-FP8**，实现数据不出内网的
私有化部署。**业务代码零改动** —— `ChatOpenAI(base_url=..., model=...)` 本就是配置驱动，
换模型只需要改 `.env`。

适用场景：需要私有化 / 信创合规、或想验证「同一套 RAG 链路在本地开源模型上的表现」。

| 项 | 说明 |
|---|---|
| GPU | 显存 **≥48GB**（Ada/Hopper 才有 FP8 张量核；A10/A100 等 Ampere 卡不建议走 FP8） |
| 服务栈 | vLLM + Qwen3.8-27B-FP8 + Milvus Lite + 原生 MySQL/Redis + FastAPI |
| 入口脚本 | `deploy/autodl/`（分阶段、幂等、可断点） |

**文档与脚本**：

- 操作手册：[`deploy/autodl/README.md`](deploy/autodl/README.md) —— 选卡、显存预算、
  分阶段部署命令、排错、回滚
- 部署实录：[`docs/autodl-private-deploy.md`](docs/autodl-private-deploy.md) ——
  三层校验：`/health` → vLLM `/v1/models` → 端到端 SSE 问答。

---

完整文档索引见 [`docs/README.md`](docs/README.md)。

有何问题欢迎留言指正。
