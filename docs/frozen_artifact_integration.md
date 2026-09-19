# Frozen Artifact Integration

This document records immutable references to offline-validated artifacts used by the next ingestion phase. The artifacts remain external to the production repository and are not copied, imported, or executed by this integration step.

## Artifact references

| Artifact | External path | SHA256 | Version | Stage |
|---|---|---|---|---|
| Chunk Metadata Contract | `F:\DataBase\trae_work\RAG_小号\chunk_metadata_contract_v1\chunk_metadata_contract_v1.py` | `415baa294167e7cbde2e11c3ef2784ff0b98ef1b761d85a5920a01db737e0cb6` | `CHUNK_METADATA_CONTRACT_V1.0` | Metadata Contract V1 |
| Strategy B chunks | `F:\DataBase\trae_work\RAG_小号\chunking_strategy_v2\strategy_b_chunks.json` | `f7877ef1ff246aedb12f657a778a6b57d81a7e046480558496103b5aba2b5045` | `Chunking Strategy V2 / Strategy B` | Chunking Strategy V2 offline output |

The Strategy B artifact declares `count = 4145`. Its chunk metadata identifies the frozen contract as `CHUNK_METADATA_CONTRACT_V1.0`.

## Integration rules

- Treat both referenced files as immutable frozen artifacts.
- Verify the recorded SHA256 values before every ingestion run.
- A hash mismatch must fail closed and require a new reviewed reference record.
- Do not edit, normalize, regenerate, or overwrite either artifact from the production repository.
- Do not copy Parser, Cleaner, QGate, Chunking, or other offline production code into this repository.
- This reference does not authorize Milvus, embedding, retrieval, API, or deployment work.

## Allowed next-phase use

Milvus Ingestion may read these exact external files only after revalidating their paths and SHA256 values. This document is an artifact reference, not an ingestion implementation.
