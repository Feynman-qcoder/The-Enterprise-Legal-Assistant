# XIAOYI ENTERPRISE LEGAL RAG: Chunk Metadata Contract V1

> Contract version: `CMCV1-202601`  |  Status: **FROZEN candidate**  |  Scope: Metadata only (no chunking algo / no embedding / no DB writes)
>
> Companion files in this directory:
>
> - `chunk_metadata_contract_v1.py` — typed contract (dataclass) with validation
> - `test_chunk_metadata_contract_v1.py` — 47 unit tests, 16+ invariants
> - `chunk_metadata_field_matrix.csv` — ownership / filter / embedding matrix
> - `chunk_metadata_real_probe.csv` — 5-format real-document probes (MD/PDF/DOCX/TABLE/层级)
> - `chunk_metadata_contract_v1.json` — machine-readable contract surface

---

## 0. STOP GATE

| Stage | Status |
|---|---|
| **CHUNK_METADATA_CONTRACT_V1** | ✅ DELIVERED |
| Chunking Strategy V2 | ❌ STOPPED — NO production chunker change in this task |
| Chunk Generation | ❌ STOPPED |
| Embedding / Milvus / MySQL write | ❌ STOPPED |
| Re-ingestion | ❌ STOPPED |
| Upstream mutation (Parser / Cleaner / MetadataContractV1 / QGate / Manifest) | ❌ **FORBIDDEN** |

---

## 1. Design Principles

### 1.1 Five-Layer Architecture

Chunk metadata is split into **5 independent layers**. Each layer owns a distinct lifetime and a distinct **source_owner** (PARSER / METADATA_NORMALIZER / CHUNKER / QGATE / DERIVED_DETERMINISTIC).

```
┌────────────────────────────────────────────────────────────────────────┐
│ CHUNK METADATA CONTRACT V1 (CMCV1-202601)                              │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER A — IDENTITY                 (source_owner: DERIVED_DETERMINISTIC)│
│   chunk_id, document_id, logical_document_id, source_file,             │
│   source_sha256, source_format, chunk_index, contract_version          │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER B — DOCUMENT METADATA        (source_owner: METADATA_NORMALIZER) │
│   title, source_org, effective_date, expiry_date, document_type,       │
│   authority_level, legal_status, jurisdiction, theme_category, …       │
│   → INHERITANCE-ONLY: never recompute; NULL stays NULL.                │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER C — STRUCTURAL METADATA      (source_owner: CHUNKER)             │
│   chapter, section, article, heading_path[], heading_level,            │
│   block_type, content_type                                              │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER D — PROVENANCE               (source_owner: PARSER + CHUNKER)    │
│   source_block_orders[], source_block_ids[],                           │
│   page_start / page_end / page_numbers[],                              │
│   original_block_order_start / original_block_order_end,               │
│   parser_name, parser_version                                          │
│   → MANDATORY. source_block_orders non-empty + monotonic + in doc.     │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER E — RETRIEVAL / RUNTIME      (source_owner: CHUNKER + QGATE)     │
│   TableContext (contains_table / table_ids / table_header_preserved /  │
│       table_structure_preserved / table_part_index / table_parts_total │
│   ParentChild (parent_chunk_id, chunk_level ∈ {PARENT, CHILD, ATOMIC}) │
│   QualityFlags (quality_verdict ∈ {PASS, WARNING, POLICY_REVIEW,       │
│       REVIEW, FAIL}, quality_warnings[], has_review_content,           │
│       review_rule_ids[])                                                │
│   extra — forbidden (strict schema; sent through extras={}).           │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Non-Goals (FROZEN boundary)

The contract does **not** cover or prescribe any of the following:

1. A specific chunking algorithm — sliding-window, structure-aware, recursive-fallback, parent/child, table-aware strategies are free to change as long as they **emit** this contract.
2. Embedding text composition (§10 only classifies which metadata must never be embedded; contextual-heading embedding decision is deferred to Retrieval Eval).
3. Milvus / MySQL / Redis collection schema (§9 only separates FILTER_SCALAR_CANDIDATE vs JSON_AUXILIARY).
4. Re-ingestion / re-chunking of the 53-document canonical corpus.

---

## 2. Layer A — Identity

### 2.1 Deterministic chunk_id generation contract

```
chunk_id = hex( sha256(
    "CMCV1"           "|"
    source_sha256     "|"
    document_id       "|"
    str(chunk_index)  "|"
    str(original_block_order_start) "|"
    str(original_block_order_end)
) )[0:32]    # lowercase hex, length 32
```

Guarantees:
- **Same chunk inputs → same chunk_id** (replayable re-ingestion).
- **Different provenance range → different chunk_id** even for same chunk_index.
- **Document identity survives unchanged** — it is an input, so ID changes if the upstream parser/Cleaner output SHA changes.

### 2.2 Non-negotiable IDENTITY fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `chunk_id` | `str` length-32 hex lowercase | ✅ | Generated exactly per contract above. |
| `document_id` | `str` non-empty | ✅ | Copied 1:1 from `ParsedDocument.document_id`. |
| `logical_document_id` | `str \| None` | ⭕ | Projected from MetadataContractV1 if present. |
| `source_file` | `str` non-empty | ✅ | Canonical corpus path (for recovery, always writeable). |
| `source_sha256` | `str` 64-hex lowercase | ✅ | SHA256 of original raw source bytes. |
| `source_format` | `str` ∈ `{md,txt,pdf,docx,html}` lowercase | ✅ | Parser key used to ingest the doc. |
| `chunk_index` | `int` ≥ 0 | ✅ | Logical chunk number per document (chunker-specific, but must be ≥0). |
| `contract_version` | `str` | ✅ | Hard-coded `"CMCV1-202601"`. |

---

## 3. Layer B — Document Metadata Projection (from frozen MetadataContractV1)

### 3.1 Semantic contract — **NO-LWW**, **NO-FABRICATION**

Every Layer-B field is inherited from the **frozen** `Metadata Contract V1` (see
`F:\DataBase\trae_work\RAG\metadata_contract_v1\`). Rules:

1. **Inherit verbatim.** The chunker MUST call `project_document_metadata_to_chunk(metadata_v1_dict)` once. The helper copies values unchanged.
2. **Null stays null.** No `"Unknown"` string fabrication, no `"-"`, no `"—"`.
3. **No Last-Write-Wins.** Chunker code cannot “recompute” `effective_date` from a regex on chunk text. Provenance never reconstructs document-level fields.
4. **Internal sentinels never leak.** Any value in
   `FORBIDDEN_SENTINEL_STRINGS = { "__UNSET__", "__UNKNOWN__", "__METADATA_BLOCK_VALUE__",
   "__REVIEW_BLOCK__", "__REVIEW__" }` is rejected at `validate()` time.

### 3.2 DOCUMENT_METADATA_FIELD → CHUNK_METADATA_FIELD mapping

Every row also tags whether the field is a candidate for Milvus scalar indexing (`FILTERABLE`) and whether the value supports legal citation. See `chunk_metadata_field_matrix.csv` for the full 60-field matrix.

| DOCUMENT V1 field | CHUNK field | Required | FILTERABLE | CITATION_RELEVANT |
|---|---|---|---|---|
| `title` | `title` | ⭕ optional | Yes | Yes |
| `logical_document_id` | `logical_document_id` | ⭕ optional | **FILTER_SCALAR_CANDIDATE** | Yes |
| `source_org` | `source_org` | ⭕ optional | **FILTER_SCALAR_CANDIDATE** | Yes |
| `publish_date` | `publish_date` | ⭕ optional | JSON_AUXILIARY | Yes |
| `creation_date` | `creation_date` | ⭕ optional | JSON_AUXILIARY | No |
| `effective_date` | `effective_date` | ⭕ optional | **FILTER_SCALAR_CANDIDATE** | Yes |
| `expiry_date` | `expiry_date` | ⭕ optional | JSON_AUXILIARY | Yes |
| `document_number` | `document_number` | ⭕ optional | JSON_AUXILIARY | Yes |
| `document_index_number` | `document_index_number` | ⭕ optional | JSON_AUXILIARY | Yes |
| `document_type` | `document_type` | ⭕ optional | **FILTER_SCALAR_CANDIDATE** | Yes |
| `authority_level` | `authority_level` | ⭕ optional | JSON_AUXILIARY | Yes |
| `status` → V1 status | `legal_status` | ⭕ optional | JSON_AUXILIARY | Yes |
| `jurisdiction` | `jurisdiction` | ⭕ optional | JSON_AUXILIARY | Yes |
| `region` | `region` | ⭕ optional | JSON_AUXILIARY | No |
| `theme_category` | `theme_category` | ⭕ optional | JSON_AUXILIARY | No |
| `source_page_url` | `source_url` | ⭕ optional | NOT_FILTERED | No |

---

## 4. Layer C — Structural Metadata

### 4.1 Semantics

| Field | Example | When to use null |
|---|---|---|
| `chapter` | `"第三编 合同"` / `"第一编 总则"` | Source has no 编/卷/篇 hierarchy. |
| `section` | `"第二章 合同的订立"` / `"第三章 合同的效力"` | Source has no 章/节. |
| `article` | `"第五百零二条"` / `"第一条"` | Source not legal-code. |
| `heading_path` | `["民法典","第三编 合同","第二章 合同的订立"]` | No parser-available heading stack. |
| `heading_level` | `1` (h1), `3` (h3) | Plain stream (no MD/PDF heading emitted). |
| `block_type` | `PARAGRAPH / HEADING / TABLE / FOOTER / …` | Vocabulary-enforced; never `None` for valid chunks. |
| `content_type` | `LEGAL_TEXT / HEADING_ONLY / TABLE_ONLY / …` | Vocabulary-enforced; never `None`. |

### 4.2 Hard rules

- **Do not hallucinate hierarchy from chunk text.** If Parser V2 did not emit heading / structure blocks, set structural fields to `None`.
- **Whitespace-only chapter/section/article** is equivalent to `None` — `__post_init__` raises `ChunkContractViolation`.
- `heading_path` must be `list[str]` of non-empty strings; length 0 allowed.

---

## 5. Layer D — Provenance (MANDATORY)

### 5.1 Fields

| Field | Type | Required |
|---|---|---|
| `source_block_orders: list[int]` | Strictly increasing; no duplicates; length ≥1 | ✅ |
| `source_block_ids: list[str] \| None` | Same length as `source_block_orders` if present | ⭕ |
| `page_start: int \| None` | First page number the chunk’s provenance covers | ⭕ — but if `page_start` set, `page_end` must be ≥ |
| `page_end: int \| None` | Last page number the chunk’s provenance covers | ⭕ |
| `page_numbers: list[int]` | All pages actually spanned; each `∈ [page_start, page_end]` when both set | ✅ list (empty OK for MD/TXT) |
| `original_block_order_start` | `int`, auto-computed := `min(source_block_orders)` | ✅ |
| `original_block_order_end` | `int`, auto-computed := `max(source_block_orders)` | ✅ |
| `parser_name` | `str` non-empty | ✅ |
| `parser_version` | `str` non-empty | ✅ |

### 5.2 Invariants enforced in `ChunkProvenance.__post_init__`

1. INV4 — `source_block_orders` non-empty.
2. INV5 — Strictly monotonic (strict < ).
3. INV6 — No duplicates.
4. INV7 — `original_block_order_start == min(source_block_orders)` and `original_block_order_end == max(source_block_orders)`. If caller overrides these, mismatch raises immediately.
5. INV8 — If both `page_start` and `page_end` exist then `page_start <= page_end`.
6. INV9 — Every page number in `page_numbers` must lie in `[page_start, page_end]` when both endpoints are set.

---

## 6. Layer E1 — TABLE lineage

Tables never flatten into paragraph-only metadata.

### 6.1 Fields (`ChunkTableContext`)

- `contains_table: bool` — when True the chunk references a table block in provenance.
- `table_ids: list[str]` — stable ids for each referenced table.
- `source_table_block_orders: list[int]` — block orders carrying the TABLE; each **must** be in `provenance.source_block_orders` (INV13 enforced).
- `table_header_preserved: bool` — whether the canonical TableData header rows survive inside the chunk (important for retrieval of split tables).
- `table_structure_preserved: bool` — whether the chunk retains the **whole** `TableData` (True if unsplit, False if split across multiple chunks).
- `table_part_index: int | None` — 0-based part of a single TableData chunked across parts.
- `table_parts_total: int | None` — total number of parts for this TableData.

### 6.2 Contract rule (INV13)

`contains_table == True` ⇔ `len(table_ids) > 0 AND len(source_table_block_orders) > 0`.

### 6.3 Schema support for future table splitting

Parent/Child chunking over tables:
- Table-level parent: `chunk_level = PARENT`, `table_structure_preserved = True` (whole `TableData` JSON stored for re-ranking / citation)
- Row-range children: `chunk_level = CHILD`, `table_part_index ∈ [0, table_parts_total-1]`, `parent_chunk_id` points back to the whole-table parent.

Canonical truth: `ParsedDocument.Block.table_data: TableData`. Markdown/HTML rendering of tables is **projection only**.

---

## 7. Layer E2 — Parent / Child compatibility

### 7.1 `ChunkParentChildInfo`

| Field | Semantics |
|---|---|
| `parent_chunk_id: str \| None` | When present: references the parent’s `chunk_id` (same format) |
| `chunk_level: str` | Enumerated, extensible: `PARENT`, `CHILD`, `ATOMIC` (default) |
| `parent_document_id: str \| None` | Reserved for cross-document aggregation (e.g. multi-doc “definition parent”). |

### 7.2 Stability promise

The schema will not break for future “hybrid” levels such as `GRAND_PARENT` or multi-level parent/child; new values can be added to the `ChunkLevel` enum as long as `parent_chunk_id` references remain valid.

---

## 8. Layer E3 — Review / Quality flags

`ChunkQualityFlags` preserves QGate policy results separately from `chunk.text`:

- `quality_verdict ∈ { "PASS", "WARNING", "POLICY_REVIEW", "REVIEW", "FAIL" }`.
- `quality_warnings: list[str]` — human-readable warnings (if any).
- `has_review_content: bool` — whether the chunk body contains an inserted QGate review block.
- `review_rule_ids: list[str]` — rule ids from QGate that triggered WARNING/POLICY_REVIEW/REVIEW.

Policy rule: these values **must never** be embedded as legal text. They are metadata only.

---

## 9. Retrieval-facing field classification

Fields are tagged in `chunk_metadata_field_matrix.csv` for future Milvus schema planning.

Classification:

- **FILTER_SCALAR_CANDIDATE** → high-frequency filters; prefer scalar / index-backed before ANN.
- **JSON_AUXILIARY** → occasional ad-hoc filter; store in Milvus JSON metadata.
- **NOT_FILTERED** → never used for filtering; keep only for audit / citation.

Current HIGH-FREQUENCY FILTER set (§9 spec):
```
logical_document_id, source_file, source_format, source_org,
effective_date, document_type, chapter, section, article
```

---

## 10. Embedding text vs. metadata classification

`chunk.text != chunk.metadata`.  Embedding-text composition is the job of a later
Chunking Strategy + Retrieval Eval stage.  CMCV1-202601 only imposes the
following policy classification so that the Chunker / Embedder pipeline has a
stable contract to follow:

| Category | Meaning | Fields |
|---|---|---|
| **RETRIEVAL_FILTER_ONLY** | Pure filter. Never put into embedding text. | `chunk_id`, `document_id`, `source_sha256`, `source_format`, `contract_version`, `chunk_index`, `extras`, table_ids, review_rule_ids. |
| **CITATION_ONLY** | May appear in the **citation footer** only. Never in the embedding-query-sensitive body. | `source_file`, `source_org`, `publish_date`, `creation_date`, `expiry_date`, `document_number`, `document_index_number`, `authority_level`, `legal_status`, `jurisdiction`, `source_url`, `parser_name`, `parser_version`. |
| **OPTIONAL_CONTEXT_PREFIX** | MAY be prepended as context prefix **only if a later Retrieval/Eval decides.** Default: DO NOT embed. | `title`, `logical_document_id` (as id), `effective_date`, `document_type`, `region`, `theme_category`, `chapter`, `section`, `article`, `heading_path`, `heading_level`, `block_type`, `content_type`. |
| **NEVER_EMBED** | Implementation / quality / runtime noise. Must never leak into legal text embeddings. | `quality_verdict`, `quality_warnings`, `has_review_content`, `parent_chunk_id`, `chunk_level`, `table_part_index`, `table_parts_total`, `page_start`, `page_end`, `page_numbers`. |

Decision: metadata is **not** blindly concatenated into `chunk.text`.

---

## 11. Validation Invariants (17 frozen checks)

Running the test suite: `pytest test_chunk_metadata_contract_v1.py -v`.

| ID | Invariant | Category | Test class |
|---|---|---|---|
| INV1 | `chunk_id` length=32 lowercase hex | Identity | `TestInvariant1ChunkId` |
| INV1.2 | `chunk_id` generated correctly per contract | Identity | `TestInvariant17ChunkIdDeterminismDeep` |
| INV2 | `document_id` non-empty string | Identity | `TestInvariant2DocumentIdentity` |
| INV3 | `source_file != ""`, `source_sha256` is 64-hex, `chunk_index ≥ 0` | Identity | `TestInvariant3SourceIdentity` |
| INV4 | `source_block_orders` non-empty | Provenance | `TestInvariant4_5_6_7BlockOrders` |
| INV5 | `source_block_orders` strictly monotonic | Provenance | `TestInvariant4_5_6_7BlockOrders` |
| INV6 | `source_block_orders` no duplicates | Provenance | `TestInvariant4_5_6_7BlockOrders` |
| INV7 | `original_block_order_start / end` consistent with min/max | Provenance | `TestInvariant4_5_6_7BlockOrders` |
| INV8 | `page_start ≤ page_end` when both set | Provenance | `TestInvariant8_9Pages` |
| INV9 | `page_numbers` all inside [page_start, page_end] | Provenance | `TestInvariant8_9Pages` |
| INV10 | chapter/section/article: nullable, non-empty if set, no whitespace-only pseudo-nulls | Structural | `TestInvariant10StructuralNulls` |
| INV11 | Sentinels forbidden everywhere | Contract integrity | `TestInvariant11SentinelsForbidden` |
| INV12 | Strict schema: `extras={}` | Contract integrity | `TestInvariant12StrictSchema` |
| INV13 | TABLE: contains_table ⇔ refs present; table orders ∊ provenance | Table lineage | `TestInvariant13TableConsistency` |
| INV14 | Parent/Child optional; chunk_level vocabulary valid | P/C compatibility | `TestInvariant14ParentChildOptional` |
| INV15 | Projected doc metadata: inherit-only, no overwrites, no fabrication, no sentinels | Doc metadata | `TestInvariant15DocMetadataNoOverwriteNoFabrication` |
| INV16 | to_dict → from_dict round-trips with stable JSON serialization | Contract integrity | `TestInvariant16RoundTripStable` |

Additional quality/test surface tests (not invariants, but checked):
- Quality verdict vocabulary (`PASS`, `WARNING`, `POLICY_REVIEW`, `REVIEW`, `FAIL`).
- Metadata projection table completeness (§3 fields all covered by projection helper).
- Enumeration surface stability (metadata layers count, chunk_level values, source_format vocabulary).
- Invariants count and prefix registry (defensive: count 17; prefixes INV1…INV17 present).

---

## 12. Real-document probe results

| Scenario | Source | Format | Structural demo | Tables | P/C demo | Validation |
|---|---|---|---|---|---|---|
| S1_MD_LEGAL_HIERARCHY_ARTICLE_1 | 个人信息保护法 | MD | chapter=第一章 总则, article=第一条 | no | — | PASS |
| S2_PDF_CIVIL_CODE_CHAPTER_SECTION_ARTICLE | 民法典 | PDF | 第三编 合同 / 第二章 合同的订立 / 第四百七十一条 | no | — | PASS |
| S3_DOCX_CONTRACT_PREAMBLE | 合同节水管理项目服务合同 | DOCX | heading_path+level demo | no | — | PASS |
| S4_DOCX_TABLE_PARTY_SIGNATURE_INFO | 同 DOCX 合同 | DOCX | TABLE block_type + heading_path | ✅ party/signature table; header+structure preserved | — | PASS |
| S5_PDF_PARENT_CHILD_DEMO_PARENT | 民法典 | PDF | chapter+section range | no | ✅ PARENT | PASS |
| S5_PDF_PARENT_CHILD_DEMO_CHILD | 民法典 | PDF | article=第四百七十二条 | no | ✅ CHILD, parent_chunk_id linked | PASS |

Full per-field probe rows → `chunk_metadata_real_probe.csv`.

---

## 13. Source Ownership Matrix (summary)

| Owner | Fields it may write |
|---|---|
| **PARSER** | `source_sha256`, `source_format`, `parser_name`, `parser_version`, block order/page provenance. |
| **METADATA_NORMALIZER** (frozen Contract V1) | `title`, `logical_document_id`, `source_org`, dates, document_type, authority_level, legal_status, jurisdiction, region, theme_category, `source_url`, `canonical_file` → `source_file`. |
| **CHUNKER** (future) | `chunk_index`, `chunk_id` (per contract rule), `structural.*`, `provenance.source_block_orders/source_block_ids`, `table_context.*`, `parent_child.*`. |
| **QGATE** | `quality.quality_verdict`, `quality.quality_warnings`, `quality.has_review_content`, `quality.review_rule_ids`. |
| **DERIVED_DETERMINISTIC** | `contract_version`, all `to_dict` round-trip invariants, chunk_id format checks. |

---

## 14. 冻结 / FROZEN promise

After this contract is ratified by the FINAL GATE below, only **additive** changes
(new optional fields, new enum values, new helper functions) will be permitted in
minor revisions (CMCV1.1, CMCV1.2, …). Breaking changes (removing/renaming
fields, narrowing validation) require CMCV2 and a parallel ratification cycle.

---

## 15. FINAL GATE (outputs per §17)

```
CHUNK_METADATA_CONTRACT_V1        = PASS
READY_FOR_CHUNKING_STRATEGY_V2    = YES
READY_FOR_REINGESTION             = NO
READY_FOR_EMBEDDING               = NO
```

Evidence:
- schema implemented: ✅ `ChunkMetadata` dataclass with 5 typed sub-structures, +1 helper for projection, +1 chunk_id function, +1 to_dict/from_dict.
- validation invariants pass: ✅ 47/47 unit tests (INV1…INV17 all covered and passing).
- document metadata semantics unchanged: ✅ inherit-only, null-preserving, sentinel-rejecting projection helper.
- provenance mandatory and validated: ✅ source_block_orders non-empty + monotonic + 2× pages invariants.
- table lineage supported: ✅ `ChunkTableContext` + INV13 bidirectional rule.
- Parent/Child compatible: ✅ optional parent_chunk_id, enum vocabulary (PARENT/CHILD/ATOMIC).
- real probe representable: ✅ 6 probes × MD/PDF/DOCX + TABLE + P/C, all PASS.
- no upstream production mutation: ✅ parser/cleaner/normalizer/QGate/gold/manifest **not touched** (constraint: forbidden).
- tests clean: ✅ `pytest` exit code 0, 47 passed in 0.20s.
