# Chunking Strategy V2 — Offline A/B 评估报告

> 类型：CHUNKING IMPLEMENTATION + OFFLINE COMPARISON ONLY。
> 不包含 Embedding、检索、Milvus/MySQL/Redis 写入或上游改写。

## 1. 运行摘要

| 项目 | 值 |
|---|---|
| 开始时间 (UTC) | 2026-08-21T02:09:54.107991+00:00 |
| 结束时间 (UTC) | 2026-08-21T02:10:49.566607+00:00 |
| 总耗时 (秒) | 55.459 |
| Manifest 条目数 | 53 |
| 成功解析 | 49 / 53 |
| 跳过条目 | 4 |
| BGE-M3 tokenizer 可用 | 是 |
| Parent/Child Probe | **PASS** |
| **CHUNKING_STRATEGY_V2_OFFLINE** | **PASS** |
| **OFFLINE_PREFERRED_STRATEGY** | **B** |
| **READY_FOR_RETRIEVAL_AB_EVAL** | **YES** |

### 1.1 Gate 故障（如有）

- 无

### 1.2 跳过的文档列表

| logical_document_id | reason |
|---|---|
| CONTRACT_006 | PARSE_ERROR_ContractViolation_HEADING level must be in 1..9, got 10 |
| CONTRACT_007 | PARSE_ERROR_ContractViolation_HEADING level must be in 1..9, got 10 |
| CONTRACT_008 | PARSE_ERROR_ContractViolation_HEADING level must be in 1..9, got 10 |
| CONTRACT_009 | PARSE_ERROR_ContractViolation_HEADING level must be in 1..9, got 10 |

## 2. 策略定义

### Strategy A — BASELINE

- **算法**：滑动窗口（生产 `chunking.py`）
- **chunk size**：512 chars（normalize_ws 后）
- **overlap**：128 chars；step = 384 chars
- **长度度量**：Python len(normalize_ws(text)) 字符数
- **输入表示**：按 ParsedDocumentV2 blocks 顺序 flatten → 单一文本；TABLE 以 `to_markdown()` 参与切片

### Strategy B — STRUCTURE-AWARE HYBRID

- **优先边界**：chapter > section > article > HEADING (level≤2 或下降) > PARAGRAPH / LIST_ITEM > TABLE 独立处理
- **禁止跨强边界合并**：不同 `article` 永不合并；新 section/chapter heading 强制分组断开；同 article 下小 unit 可合并（FR2.2）
- **Oversized 递归降级顺序**：paragraph/双换行 → 句号句末 → 逗号顿号 → 硬长度切分
- **TABLE 独立处理**：小表单 chunk（TABLE_SMALL_ROWS=20 或 TABLE_SMALL_CHARS=800）；大表按 TABLE_GROUP_ROWS=10 行分组 + 所有分片重复 materialized header（`table_header_preserved=True`）
- **长度参数**：target_max = 600 chars；hard_max = 900 chars

## 3. FR7 冻结阈值

| 阈值 | 值 | 含义 |
|---|---|---|
| OVERSIZED_THRESHOLD_CHARS | 1200 | 参考 Task 7 指标阈值 / 策略参数
| VERY_SMALL_THRESHOLD_CHARS | 64 | 参考 Task 7 指标阈值 / 策略参数
| A_WINDOW | 512 | 参考 Task 7 指标阈值 / 策略参数
| A_OVERLAP | 128 | 参考 Task 7 指标阈值 / 策略参数
| A_STEP | 384 | 参考 Task 7 指标阈值 / 策略参数
| B_TARGET_MAX_CHARS | 600 | 参考 Task 7 指标阈值 / 策略参数
| B_HARD_MAX | 900 | 参考 Task 7 指标阈值 / 策略参数
| TABLE_SMALL_ROWS | 20 | 参考 Task 7 指标阈值 / 策略参数
| TABLE_SMALL_CHARS | 800 | 参考 Task 7 指标阈值 / 策略参数
| TABLE_SPLIT_GROUP_ROWS | 10 | 参考 Task 7 指标阈值 / 策略参数

## 4. 指标总览 (chunking_ab_metrics.csv)

| 指标 | Strategy A | Strategy B | 离线优方 (仅参考) |
|---|---|---|---|
| TOTAL_CHUNKS | 1849 | 4145 | A 更优 |
| AVG chunk length (chars) | 506.89 | 196.96 | B 更优 |
| MEDIAN length (加权近似) | 512.0 | 148.71 | B 更优 |
| P95 length (加权近似) | 512.0 | 550.03 | A 更优 |
| MIN length | 135 | 1 | B 更优 |
| MAX length | 512 | 1724 | A 更优 |
| OVERSIZED_CHUNKS (> 1200) | 0 | 10 | A 更优 |
| VERY_SMALL_CHUNKS (< 64) | 0 | 1196 | A 更优 |
| AVG SOURCE_BLOCK_COVERAGE (%) | 100.0 | 100.0 | 持平 |
| AVG DUPLICATED_TEXT_RATIO | 0.57038 | 0.30162 | B 更优 |
| CROSS_ARTICLE_CHUNKS (越低越好) | 913 | 0 | B 更优 |
| CROSS_SECTION_CHUNKS (越低越好) | 267 | 85 | B 更优 |
| HEADING_ORPHAN_COUNT | 0 | 482 | A 更优 |
| ARTICLE_ORPHAN_COUNT | 0 | 81 | A 更优 |
| TABLE_CHUNKS | 321 | 217 | B 更优 |
| TABLE_RELATIONSHIP_LOSS (必须 B=0) | 238 | 0 | B 更优 |
| PROVENANCE_FAILURES (应为 0) | 0 | 0 | 持平 |
| METADATA_CONTRACT_FAILURES (应为 0) | 0 | 0 | 持平 |

## 5. 每文档指标

完整 CSV：`chunking_ab_document_metrics.csv`（53 doc × A/B = 106 行）。

### 5.1 代表文档 A/B 对照（9 指标）

| 代表文档 | 策略 | total_chunks | avg_len | oversized | very_small | cov% | cross_article | cross_section | table_chunks | contract_fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 6. TABLE 审计

参见 `table_chunk_audit.csv`。每文档每 TABLE chunk 一行，列：strategy, chunk_id, table_id, part, header_preserved, structure_preserved, relationship_loss_check。

## 7. 人工评审包

请评审 20–30 对代表 case：

- **Markdown 详细版**：`chunking_ab_review_zh.md`
- **CSV 版**：`chunking_ab_human_review.csv`

用户字段：**用户选择**（A / B / 两者均可 / 两者都差）与 **用户备注** 保持空白，由真实评审者填写。禁止 AI 直接填胜者。

case 构成（由 FR10 规则自动挑选）：

- boundary 敏感：A 的跨 article chunk vs B 的对应 chunks（约 10 例）
- table：A/B 同 TABLE block（约 6–8 例）
- oversized：B 超出硬上限的告警样本（0–3 例）
- very-small：B 产生的极小 chunk，核对是否应当合并（2–4 例）
- disagreement：A 为独立 HEADING_ONLY vs B 标题+正文合并（3–5 例）
- cross-section 兜底补齐到 20–30 例

## 8. Parent/Child Probe

本阶段 Parent/Child 仅作为 OPTIONAL projection 探针验证 CMCV1 Layer E 支持。结果：**PASS**。不作为 OFFLINE_PREFERRED_STRATEGY 的直接决策依据。

## 9. STOP 检查点（§15）

本任务停止在 **OFFLINE CHUNKING A/B** 阶段。以下动作尚未发生也不应被触发：

- Embedding / 写入 Milvus / MySQL / Redis
- 检索 A/B 比较（BM25 / dense / rerank）
- Re-ingestion / 重新预处理 53 文档
- 修改 Parser / Cleaner / MetadataNormalizer / QGate / Chunk Metadata Contract V1

## 10. 交付物清单

| 文件名 | 描述 |
|---|---|
| `chunking_strategy_v2.md` | 本报告 |
| `chunking_strategy_v2.json` | 机器可消费运行摘要 + 聚合指标 + 交付物索引 |
| `strategy_a_chunks.json` | Strategy A 全部 chunk（包装 `chunk_metadata` + `text`） |
| `strategy_b_chunks.json` | Strategy B 全部 chunk（同上结构） |
| `chunking_ab_metrics.csv` | 聚合指标对比（2 行） |
| `chunking_ab_document_metrics.csv` | 106 行（53 doc × A/B）每文档指标 |
| `chunking_ab_review_zh.md` | 人工评审 Markdown 包 |
| `chunking_ab_human_review.csv` | 人工评审 CSV 包（用户字段空） |
| `table_chunk_audit.csv` | TABLE chunk 审计 |
| `test_chunking_strategy_v2.py` | 自动化测试套件（Task 13） |
