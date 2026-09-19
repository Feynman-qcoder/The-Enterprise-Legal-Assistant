# Legal RAG Live Ingestion Smoke V1

> 最新结论（2026-08-16 Rollback Gate Correction）：authoritative Milvus logical count 与 exact-ID visibility 证明 rollback PASS；Cleaner V1.1 retry 后 3-doc ingestion、clean text、MySQL/Milvus、retrieval 3/3、RAG/SSE 3/3 全部 PASS。本文保留此前两轮阻塞记录作为审计历史，末尾“Final PASS Addendum”取代早期 Final Gate。

> 2026-08-16 V1.1 follow-up：本文原始部分记录第一次 Cleaner Gate 阻塞。Cleaner V1.1 修复、53-doc dry-run、metadata consistency 和 exact rollback 后，Milvus target IDs 已不可见且 visible count=44，但 `Collection.num_entities` 仍为 55。依据最新任务的强制 count Gate，本轮最终状态已更新为 `ROLLBACK_INTEGRITY_FAILED`；下方新增 addendum 为最新结论。

- 执行日期：2026-08-16（Asia/Shanghai）
- 执行范围：Legal RAG Full Environment Startup + LIVE CLEAN INGESTION PHASE 1
- 原计划：仅 3 个 Frozen Canonical Documents
- 实际终止点：第 1 个 Web-derived MD 入库后触发 `CLEANING_INTEGRATION_BLOCKED`
- 数据策略：`SMOKE DATA KEPT`
- 生产代码修改：ZERO
- Corpus Cleaning 的原始实现/测试归属：WorkBuddy；本次 Codex 仅验证当前磁盘代码及其 LIVE integration

## A. Production Pipeline Confirmation

快速复核通过，未发现 `STATE_CHANGED_SINCE_PREFLIGHT`：

- `modules/ingestion/document_cleaning.py` 存在。
- `modules/ingestion/incremental.py::ingest_legal_document` 当前调用顺序：
  `parse_document`（L118）→ `clean_parsed_document`（L122）→ `chunk_pdf_pages_to_parents` / `split_children_from_parent`（L125-L126）→ BGE-M3（L132-L135）→ MySQL（L138-L172）→ Milvus incremental insert（L175-L176）。
- 本轮真实调用 `ingest_legal_document(filename, canonical_file.read_bytes())`，没有直接执行 MySQL INSERT、没有直接构造/插入 Milvus vector、没有使用 `normalized_preview_v1`。
- Cleaner 计时使用运行时 wrapper/monkeypatch；没有修改生产文件。

## B. Frozen Corpus

- Manifest：`D:\xiaoyi\data_source\_meta\ingest_manifest_v1.jsonl`
- Rows：53
- Unique `logical_document_id`：53
- `LAW_003` count：1
- `LAW_003` version：`2025_amended`
- `LAW_003 normalized_sha256`：`f571d7567e108b9068370aa8185bfb794b42219460c9fb923722a31f58d01f27`
- 旧 2017 网络安全法没有被选择或入库。

## C. Environment Startup

- Docker Desktop：4.86.0
- Docker Engine：29.7.2，READY
- Compose services：`etcd`, `minio`, `redis`, `standalone`
- 使用既有镜像；本轮没有 pull、没有改 image tag、没有改 Compose。
- Containers：`milvus-etcd`, `milvus-minio`, `xiaoyi-redis`, `milvus-standalone` 均为 Up。
- MySQL：项目此前使用的小皮 MySQL 8.0.12，`127.0.0.1:3306` READY。
- Milvus：`127.0.0.1:19530` READY。
- Redis：`127.0.0.1:6379` READY。
- Backend：尚未启动；按照失败策略，在 Retrieval 前的 Cleaning Gate 失败后停止。
- Local BGE-M3：`D:\xiaoyi\Legal_System\models\bge-m3` AVAILABLE。

说明：首次通过 PATH 发现并启动的 `D:\AI\MYSQL\...\MySQL 5.7` 不是项目先前数据实例，项目账号鉴权失败。该刚启动的实例已停止，随后识别并启动既有小皮 MySQL 8.0.12；没有修改账号、密码、schema 或数据目录。

## D. Resource Usage

环境 READY 后采样：

| Container | CPU | Memory |
|---|---:|---:|
| milvus-standalone | 5.93% | 120.4 MiB |
| xiaoyi-redis | 0.40% | 5.703 MiB |
| milvus-etcd | 0.50% | 22.85 MiB |
| milvus-minio | 0.00% | 87.91 MiB |

- Windows RAM：15.86 GB total / 12.12 GB used / 3.74 GB free / 76.4% used。
- 未达到 90%，未触发 `RESOURCE_PRESSURE_BLOCKER`。

## E. Database Before State

MySQL：

- Database：`xiaoyi_rag`
- `legal_tab`：exists
- `faq_tab`：exists
- Legal Parent：12
- Legal Child：44
- FAQ：30

Milvus：

- Collections：`xiaoyi_legal_child`, `xiaoyi_faq_highfreq`
- Legal entities：44
- FAQ entities：30
- Legal/FAQ embedding dimension：1024

本轮没有 DROP、DELETE、Collection recreate 或 `recreate=True`。

## F. Three Smoke Documents

三者均来自 53-row Frozen Manifest：

| Role | Logical Document ID | Canonical file | Selection evidence |
|---|---|---|---|
| Web-derived MD | `ENT_COMPLIANCE_002` | `ENT_POLICY_002_中央企业法律纠纷案件管理办法.md` | 原始文件真实含 JiaThis、二维码、访问量、打印/关闭窗口及 Footer 组件噪声 |
| Contract DOCX | `CONTRACT_021` | `ENT_CONTRACT_037_建设工程施工合同（住房城乡建设部、国家工商总局2017版）.docx` | Static precheck：`placeholder_normalized=3`，clean text 含 `[待填写]` |
| Legal PDF | `LAW_010` | `LEGAL_LAW_004_中华人民共和国电子签名法.pdf` | 正式法律 PDF；8 pages，static precheck 预计 3 Parents / 14 Children |

`CONTRACT_021` 和 `LAW_010` 因第一个文件触发 Cleaning Gate 阻塞而没有继续入库。

## G. Live Ingestion

### ENT_COMPLIANCE_002

- Production call：PASS
- Parser：PASS
- Cleaner invocation：PASS（但输出质量 Gate FAIL，见 K）
- Chunk：PASS
- Embedding：PASS
- MySQL incremental insert：PASS
- Milvus incremental insert：PASS
- Parents：2
- Children/Vectors：11
- MySQL Parent IDs：185, 186
- MySQL Child IDs / Milvus PKs：187–197

### CONTRACT_021

- Status：NOT INGESTED — stopped before MySQL write after upstream Cleaning Gate failure
- MySQL rows：0

### LAW_010

- Status：NOT INGESTED — stopped after upstream Cleaning Gate failure
- MySQL rows：0

因此 3/3 ingestion Gate 未满足。

## H. Stage Timing

| Logical ID | parse_ms | clean_ms | chunk_ms | embedding_ms | mysql_ms | milvus_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENT_COMPLIANCE_002 | 4.94 | 1.60 | 0.40 | 9618.68 | 39.55 | 3107.08 | 43696.13 |
| CONTRACT_021 | NOT COMPLETED | NOT COMPLETED | NOT COMPLETED | INTERRUPTED | NOT WRITTEN | NOT WRITTEN | NOT RECORDED |
| LAW_010 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

第一个文件的 total 包含 BGE-M3 首次模型加载等进程冷启动成本；这不是正式性能 Benchmark。

## I. MySQL Evidence

`ENT_POLICY_002_中央企业法律纠纷案件管理办法.md`：

- Parent rows：185, 186
- Child rows：187–197
- Child 187–196 → Parent 185
- Child 197 → Parent 186
- 所有 Parent/Child `source_file` 正确。
- 全库 orphan child count：0。

结构与引用映射正确，但 Parent 186 / Child 197 仍含清洗遗漏噪声，故 MySQL Clean Text Gate FAIL。

## J. Milvus Evidence

- BEFORE Legal entities：44
- AFTER Legal entities：55
- Delta：+11
- 实际新增 MySQL Child：11
- Delta 与新增 Child 数一致。
- 新增 Milvus PK：187–197，逐项映射同 ID MySQL Child。
- `parent_id`、`source_file`、stored text 与 MySQL Child 逐项完全一致。
- Vector dimension：1024。
- 即时 query 可见 11/11；没有 `CONSISTENCY_VISIBILITY_ISSUE`。
- 全库 orphan vector IDs：[]。
- 全库 missing vector IDs：[]。

Milvus 本身通过映射/可见性验证；但向量化文本忠实复制了 Cleaner 遗漏的垃圾，因此 Clean Vector Gate FAIL。

## K. Cleaning Evidence

Cleaner metadata：

- Profile：`generic`
- Raw chars：4190
- Clean chars：3898
- Removed chars：292
- Removed ratio：0.0697
- `boilerplate_hits`：11
- `placeholder_normalized`：0
- `quality_status`：PASS
- `normalized_sha256`：`6201d3cdcf3a867042050a65d04ca8595a941caeb101d1b8eea330832bc76a4d`

但数据库实际 clean output 仍出现：

| Residual | Count in all inserted Parent/Child rows |
|---|---:|
| `jiathis` URL | 2 |
| `打印` | 2 |
| `关闭窗口` | 2 |
| `end component 文档组件` | 2 |
| `baidu` decoration | present |

实际残留片段：

```text
[](http://www.jiathis.com/share)

| 打印 | | 关闭窗口 |
|----|--|------|

-------------------baidu------------------------
--------------------baidu------------------------------
end component 文档组件(文章正文)
```

结论：Cleaner 确实位于 Parser 与 Chunk 之间并真实执行，但规则覆盖不完整；这不是调用顺序或旁路问题。

## L. Retrieval Smoke

- Status：NOT RUN
- Reason：第一个文档已违反强制 Clean Text Gate；按 Failure Policy 先报告并等待确认，不继续扩大验证。
- 不声明 Recall@K、MRR 或 NDCG。

## M. RAG/SSE Smoke

- Status：NOT RUN
- Backend 未启动。
- 未进入 GLM / DashScope / SSE 阶段。
- 这不是 `RAG_GENERATION_BLOCKED_EXTERNAL`；阻塞发生在本地 Cleaner 输出质量。

## N. Integrity

当前全库：

- MySQL orphan children：0
- Milvus orphan vectors：0
- MySQL children missing Milvus vectors：0
- 新增 Parent/Child mapping：PASS
- 新增 Milvus/MySQL ID mapping：PASS
- 新增 Milvus/MySQL text equality：PASS
- Clean text quality：FAIL

## O. Database After State

MySQL：

- Legal Parent：14（delta +2）
- Legal Child：55（delta +11）
- FAQ：30（delta 0）

Milvus：

- Legal entities：55（delta +11）
- FAQ entities：30（delta 0）

## P. Smoke IDs Kept

按照要求未 cleanup：

- Logical ID：`ENT_COMPLIANCE_002`
- File：`ENT_POLICY_002_中央企业法律纠纷案件管理办法.md`
- MySQL Parent IDs：185, 186
- MySQL Child IDs：187–197
- Milvus IDs：187–197

没有执行 DELETE、Milvus delete、DROP 或 Collection recreate。

## Q. Blockers / Technical Debt

### CLEANING_INTEGRATION_BLOCKED

Failure category：`CLEANER`

Root Cause：

1. `document_cleaning.py` L100 的 Markdown 导航表格正则只允许一个 junk token，无法匹配同时含两个 token 的 `| 打印 | | 关闭窗口 |`。
2. L76-L81 的 component prefix 只覆盖 `HTML组件`，未覆盖真实 Corpus 的 `end component 文档组件(...)`。
3. 当前规则未覆盖 `http://www.jiathis.com/share` 形式的分享 URL。
4. `PURE_SEPARATOR_RE` 只接受纯分隔字符，包含 `baidu` 字母的装饰分隔行不会被识别。
5. `quality_status=PASS` 只基于移除比例/最小长度，没有 residual boilerplate negative check，因此统计 PASS 与实际 Gate 结果不一致。

Evidence：

- Cleaner 运行 metadata 与 normalized hash 已生成。
- 残留文本真实存在于 MySQL Parent 186 / Child 197。
- 相同残留文本真实存在于 Milvus PK 197。
- Pipeline 顺序确认正确，所以问题定位为 Cleaner rule coverage，而非 Cleaner 未接入。

Minimal Fix Proposal（仅提案，本轮未实施）：

1. 扩展 junk-table 判断：拆分 Markdown cells，确认所有非空 cell 均属于 `JUNK_NAV_TOKENS` 后删除整行。
2. 添加明确的 `start/end component 文档组件` whole-line rule。
3. 添加仅针对独立 Markdown link/URL 行的 `jiathis.com/share` rule，避免误伤正文。
4. 添加明确的 `^-+baidu-+$` decoration rule。
5. 加入该 Frozen canonical MD 的回归 fixture，并对 Cleaner output 做 residual negative assertions。
6. 修复并得到用户确认后，再决定保留 Smoke IDs 185–197 的处理策略及重跑余下两个文档。

其他技术债：PyMilvus ORM API 已打印 3.1 deprecation warning；本轮不迁移、不优化。

## R. Final Gate

- 3/3 ingestion：NOT MET（1/3 completed）
- Parser：PASS（completed document）
- Cleaner invocation：PASS
- Cleaner output quality：FAIL
- Chunk：PASS（completed document）
- Embedding：PASS（completed document）
- MySQL mapping：PASS（completed document）
- Milvus mapping/visibility：PASS（completed document）
- Retrieval 3/3：NOT RUN
- No orphans：PASS

**LIVE CLEAN INGESTION GATE = BLOCKED — CLEANING_INTEGRATION_BLOCKED**

**RAG LIVE SMOKE = NOT RUN**

已停止；禁止的 53-doc Full Ingestion、Eval Dataset、Recall/MRR/NDCG、正式 Benchmark 与优化均未执行。

---

# Cleaner V1.1 Follow-up Addendum（最新状态）

## A. Environment

Docker、MySQL、Milvus、Redis 与 BGE-M3 均 READY；未启动 Backend，因为 Phase 11 已阻塞。

## B. Baseline

- Failed Smoke before rollback：MySQL 14/55/30；Milvus 55/30
- Exact rollback after：MySQL 12/44/30
- Milvus：target 187–197 query=0；visible query count=44；`num_entities=55`；FAQ=30

## C. Three Canonical Docs

- Web MD：ENT_COMPLIANCE_002
- Contract DOCX：CONTRACT_021
- Legal PDF：LAW_010

三者保持 Frozen Manifest canonical；由于 rollback count Gate 阻塞，本轮未重新写入。

## D. Live Ingestion

Cleaner V1.1 retry：NOT RUN。未进行第二次 3-doc 写入。

## E. Cleaning Evidence

- Cleaner tests：PASS
- ENT_COMPLIANCE_002 offline Parser→Cleaner V1.1：PASS
- 53/53 canonical Parser→Cleaner dry-run：PASS
- residual high-confidence junk：0/53
- Changed：3；全部 diff 安全，仅删除已知技术噪声
- Metadata/hash consistency：PASS

## F. MySQL

精准删除 child 187–197、parent 185/186 后恢复 12 Parent / 44 Child / 30 FAQ，target remaining=0。

## G. Milvus

精准删除 PK 187–197 后：

- Target query：0
- Visible unique IDs：44
- FAQ：30
- `Collection.num_entities`：55

强制期望为 44，因此 `ROLLBACK_INTEGRITY_FAILED`。

## H. Retrieval

NOT RUN — stopped before re-ingestion。

## I. RAG

NOT RUN — Backend/GLM/SSE 未进入。

## J. Final Integrity

- MySQL exact rollback：PASS
- Milvus deleted IDs visibility：PASS
- Milvus required entity statistic：FAIL
- 3/3 retry：NOT RUN
- Retrieval 3/3：NOT RUN

**LIVE CLEAN INGESTION GATE = BLOCKED — ROLLBACK_INTEGRITY_FAILED**

**RAG LIVE SMOKE = NOT RUN**

Smoke retry data不存在；旧 failed-smoke IDs 已按最新任务授权精准删除。未执行 53-doc full ingestion、recreate、compaction、cleanup 扩展、正式 Eval 或 commit。

---

# Final PASS Addendum（最新、权威状态）

## A. Environment

- Docker Engine / etcd / MinIO / Redis / Milvus：READY
- MySQL 8.0.12：READY
- BGE-M3 / BGE reranker：AVAILABLE
- DashScope：3 次 intent/generation 请求正常
- 未执行 restart、manual compaction、drop/recreate 或优化

## B. Baseline and Rollback Correction

Milvus 2.4 authoritative logical rollback evidence：

- MySQL：12 Parent / 44 Child / 30 FAQ
- Legal `query count(*)`：44
- Visible / unique IDs：44 / 44
- Exact failed IDs 187–197：0
- FAQ `query count(*)`：30
- 剩余 44 条 MySQL/Milvus ID、parent、source、text mapping：PASS

结论：`FAILED SMOKE LOGICAL ROLLBACK = PASS`。

历史 `Collection.num_entities=55` 是 delete tombstone/statistics lag；本次复核时已由后台自然更新为 44，未做 manual compaction。

## C. Three Canonical Documents

| Role | Logical ID | File |
|---|---|---|
| Web-derived MD | ENT_COMPLIANCE_002 | `ENT_POLICY_002_中央企业法律纠纷案件管理办法.md` |
| Contract DOCX | CONTRACT_021 | `ENT_CONTRACT_037_建设工程施工合同（住房城乡建设部、国家工商总局2017版）.docx` |
| Legal PDF | LAW_010 | `LEGAL_LAW_004_中华人民共和国电子签名法.pdf` |

三者均来自当前 53-row Frozen Manifest。

## D. Live Ingestion

真实调用 `ingest_legal_document`：

| Logical ID | Parent IDs | Child/Milvus IDs | Parents | Children | Result |
|---|---|---|---:|---:|---|
| ENT_COMPLIANCE_002 | 198–199 | 200–210 | 2 | 11 | PASS |
| CONTRACT_021 | 211–229 | 230–417 | 19 | 188 | PASS |
| LAW_010 | 418–420 | 421–434 | 3 | 14 | PASS |

完整生产链：Original Canonical → Parser → Cleaner V1.1 → Quality Gate → Parent/Child Chunk → BGE-M3 → MySQL → Milvus。

Stage timing：

| Logical ID | parse_ms | clean_ms | chunk_ms | embedding_ms | mysql_ms | milvus_ms | total_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENT_COMPLIANCE_002 | 4.82 | 2.30 | 0.58 | 9596.25 | 38.60 | 3107.74 | 39766.83 |
| CONTRACT_021 | 531.97 | 30.70 | 6.71 | 176205.86 | 325.31 | 3119.35 | 180226.16 |
| LAW_010 | 143.37 | 3.37 | 0.50 | 11227.14 | 26.01 | 3074.20 | 14480.77 |

## E. Cleaning Evidence

Web MD 在 MySQL Parent/Child 与 Milvus stored text 中均满足：

- jiathis / JiaThis：0
- 二维码生成专用：0
- Produced By CMS：0
- 访问量统计：0
- end component：0
- 关闭窗口：0
- baidu decorative junk：0
- Cleaner residual hits：0
- quality status：PASS

Contract：

- Cleaner `placeholder_normalized=3`
- 入库 Parent/Child 合并文本含 `[待填写]` 7 处
- 长下划线：0
- 标题、发包人、承包人、专用合同条款均保留
- Markdown table lines：203，表格结构存在

PDF：标题、第一条、数据电文证据条款、可靠电子签名同等法律效力条款、第五章附则均保留。

## F. MySQL Integrity

- BEFORE：12 Parent / 44 Child / 30 FAQ
- AFTER：36 Parent / 257 Child / 30 FAQ
- Delta：+24 Parent / +213 Child
- 213/213 new child.parent_id 指向对应新 Parent
- Orphan children：0

## G. Milvus Integrity

- BEFORE Legal logical `count(*)`：44
- AFTER Legal logical `count(*)`：257
- Delta：+213，与新增 Child 完全一致
- Visible / unique：257 / 257
- FAQ logical `count(*)`：30
- New PK ↔ MySQL Child ID：213/213 PASS
- parent_id / source_file / clean text：213/213 PASS
- Vector dimension：1024
- Orphan vectors：0
- Missing vectors：0
- Informational `num_entities`：257

## H. Live Retrieval Smoke

真实链路：BGE-M3 Dense + BM25 + RRF + Parent Retrieval + CrossEncoder。

| Query | Expected file | Final rank | Dense | BM25 | RRF | Reranker |
|---|---|---:|---:|---:|---:|---:|
| 5000万元重大案件应在多少工作日内备案？ | ENT_POLICY_002...md | 1 | 0.620675 | 12.813741 | 0.024201 | 0.360744 |
| GF-2017-0201 示范文本由哪三部分组成？ | ENT_CONTRACT_037...docx | 1 | 0.802892 | 36.467412 | 0.032787 | 0.999834 |
| 电子签名法第八条审查数据电文真实性因素？ | LEGAL_LAW_004...pdf | 1 | 0.698290 | 34.222963 | 0.032522 | 0.904557 |

**LIVE RETRIEVAL SMOKE = PASS（3/3）**。本阶段不是 Recall/MRR/NDCG benchmark。

## I. RAG/SSE Smoke

真实 `/api/chat/stream`：

| Logical ID | HTTP | SSE DONE | Error | Answer chars | Observed first event / total ms |
|---|---:|---|---|---:|---:|
| ENT_COMPLIANCE_002 | 200 | YES | None | 167 | 53051.93 / 53054.47 |
| CONTRACT_021 | 200 | YES | None | 32 | 22651.08 / 22651.20 |
| LAW_010 | 200 | YES | None | 127 | 27724.92 / 27725.21 |

回答分别正确给出 10 个工作日、合同协议书/通用合同条款/专用合同条款、电子签名法第八条四项因素。ASGI TestClient transport 可能缓冲 first-event，因此该数字不是正式 TTFT benchmark。

**RAG LIVE SMOKE = PASS（3/3）**。

## J. Final Integrity and Gate

- Cleaner V1.1 Regression：PASS
- 53-doc Cleaner Dry-run：PASS
- Rollback Logical Integrity：PASS
- 3/3 Live Ingestion：PASS
- Clean Text：PASS
- MySQL：PASS
- Milvus Logical Data：PASS
- Retrieval 3/3：PASS
- No Orphans：PASS
- RAG/SSE 3/3：PASS

**CLEANER V1.1 = PASS**

**LIVE CLEAN INGESTION GATE = PASS**

**RAG LIVE SMOKE = PASS**

Smoke data 198–434 按要求保留。未执行 53-doc Full Ingestion、manual compaction、正式 Eval/Benchmark、Milvus 优化或 commit。
