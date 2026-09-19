# Frozen Assets SHA256 Manifest

V2 语料冻结资产清单。第三方复现时以此文件为自验锚点：clone 后逐文件校验 SHA256，
任何不匹配说明资产损坏或被篡改，请停止灌入。

复现链路：`strategy_b_chunks.json` + `parent_sections.json` + `child_parent_mapping.json`
→ `offline/scripts/run_v2_ingest.py` → MySQL 三表 + Milvus `xiaoyi_legal_child_v2`。
`embeddings.npy` 为预计算 BGE-M3 向量（可选 `--precomputed` 加速灌入，且与生产向量逐位一致）。
`ingest_manifest_v1.jsonl` 为在线元数据头（[Source Metadata]）数据源。

| 文件 | 字节数 | SHA256 | 说明 |
|---|---:|---|---|
| strategy_b_chunks.json | 18041271 | `331d46df3d31ba249e9fccc87890f1d3ddf7db09ea101fd0426b6a8069b38293` | Strategy B Production V2 冻结切块（4134 children） |
| parent_sections.json | 12002273 | `e71eb5da6f06827e12c520ffae44e84324a1c598a5d460269635c29b1667b943` | Parent Contract V1.1 父段定义（2217 parents / 49 documents） |
| child_parent_mapping.json | 5819686 | `44f55e54885b7030527c6d2bdad637d621b86e4e0c42a0e7b794193729f2f57e` | Child-Parent Link V1.0 映射（4134 links） |
| ingest_manifest_v1.jsonl | 45681 | `882ab4ed3bba3169c57964a0f09da1a44bad8c695f6fcc3566709dae64dc7d8a` | Canonical 53 文件清单（在线元数据头数据源） |
| embeddings.npy | 16932992 | `0a5716ba2285f4f74b9c15b4a4adbe95330edee61c41429862bc6202748680ad` | 预计算嵌入（4134×1024 float32，两遍生成 bitwise 一致） |
| embedding_manifest.json | 1030147 | `e63bc8e4d6afb71d950c314188e986ff345423037e4340bafefa5acfe263c575` | 嵌入清单（chunk_id → vector_index 血缘） |

来源血缘（上游冻结记录，见 `parent_mapping_v2_retry/mapping_report.md`）：

- `strategy_b_chunks.json` ← Strategy B Production V2（输入 gate SHA256 PASS）
- `parent_sections.json` / `child_parent_mapping.json` ← Parent Mapping V2 Retry（READY_FOR_PARENT_PERSISTENCE=YES）
- `embeddings.npy` ← Embedding V1（source_chunks_sha256 = 上表 chunks SHA256，血缘闭合）

自验命令（PowerShell）：

```powershell
Get-ChildItem frozen_assets -File | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
}
```

输出与上表逐一比对。冻结资产只读：任何脚本不得写 `frozen_assets/` 目录。
