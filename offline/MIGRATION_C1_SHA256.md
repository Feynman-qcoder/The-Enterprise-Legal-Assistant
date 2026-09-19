# Task 22 C1 迁移 SHA256 清单（源 vs 副本）

- 源：f:\DataBase\trae_work\RAG_小号\{chunking_strategy_v2, chunk_metadata_contract_v1, retrieval_eval_v1}
- 目标：D:\xiaoyi\Legal_System\offline\ 同名目录
- 纪律：逐字节复制；BEYTES-EQUAL 行双端哈希一致
- ADAPTED 行：迁移前做了最小路径适配（绝对路径→仓库相对/环境变量），双端哈希预期不同，适配内容见 commit message
- 排除：__pycache__ / .pytest_cache / _run_pytest_result.xml（测试运行残留）

| 文件 | 状态 | 源 SHA256 | 副本 SHA256 |
|---|---|---|---|
| offline/chunk_metadata_contract_v1/_build_real_probe.py | BYTE-EQUAL | 8b3ba23325a9bb96… | 8b3ba23325a9bb96… |
| offline/chunk_metadata_contract_v1/chunk_metadata_contract_v1.json | BYTE-EQUAL | 071504a6a8d7403d… | 071504a6a8d7403d… |
| offline/chunk_metadata_contract_v1/chunk_metadata_contract_v1.md | BYTE-EQUAL | 7088e6913c8b0965… | 7088e6913c8b0965… |
| offline/chunk_metadata_contract_v1/chunk_metadata_contract_v1.py | BYTE-EQUAL | 415baa294167e7cb… | 415baa294167e7cb… |
| offline/chunk_metadata_contract_v1/chunk_metadata_field_matrix.csv | BYTE-EQUAL | a22c13fc209f76e1… | a22c13fc209f76e1… |
| offline/chunk_metadata_contract_v1/chunk_metadata_real_probe.csv | BYTE-EQUAL | b14fb93bdc82448b… | b14fb93bdc82448b… |
| offline/chunk_metadata_contract_v1/test_chunk_metadata_contract_v1.py | BYTE-EQUAL | e73e56de3624cb94… | e73e56de3624cb94… |
| offline/chunking_strategy_v2/_v2_pipeline.py | BYTE-EQUAL | 2280fcf9b3aab18b… | 2280fcf9b3aab18b… |
| offline/chunking_strategy_v2/_v2_strategies.py | BYTE-EQUAL | b33f9be81c3353ed… | b33f9be81c3353ed… |
| offline/chunking_strategy_v2/_v2_utils.py | ADAPTED | f4dfb921f09d8eff… | c00e4d6f0ec0ac86… |
| offline/chunking_strategy_v2/chunking_ab_document_metrics.csv | BYTE-EQUAL | cb6057c259f40f20… | cb6057c259f40f20… |
| offline/chunking_strategy_v2/chunking_ab_human_review.csv | BYTE-EQUAL | 5352dd0708702c3c… | 5352dd0708702c3c… |
| offline/chunking_strategy_v2/chunking_ab_metrics.csv | BYTE-EQUAL | ea71d5e82dc38261… | ea71d5e82dc38261… |
| offline/chunking_strategy_v2/chunking_ab_review_zh.md | BYTE-EQUAL | 49bf77e84227d9c3… | 49bf77e84227d9c3… |
| offline/chunking_strategy_v2/chunking_strategy_v2.json | BYTE-EQUAL | 25dddf4c9a3d8402… | 25dddf4c9a3d8402… |
| offline/chunking_strategy_v2/chunking_strategy_v2.md | BYTE-EQUAL | 2a82bfb67a8f9807… | 2a82bfb67a8f9807… |
| offline/chunking_strategy_v2/strategy_a_chunks.json | BYTE-EQUAL | bd2010cbc0ebb5f0… | bd2010cbc0ebb5f0… |
| offline/chunking_strategy_v2/strategy_b_chunks.json | BYTE-EQUAL | f7877ef1ff246aed… | f7877ef1ff246aed… |
| offline/chunking_strategy_v2/table_chunk_audit.csv | BYTE-EQUAL | 700c3d15caed9481… | 700c3d15caed9481… |
| offline/chunking_strategy_v2/test_chunking_strategy_v2.py | ADAPTED | 9c77cc8cdd84e238… | bd0f1841121af6e5… |
| offline/retrieval_eval_v1/_00_probe_runtime.py | BYTE-EQUAL | e3d5a05a36ab2e26… | e3d5a05a36ab2e26… |
| offline/retrieval_eval_v1/_01_build_questions.py | BYTE-EQUAL | 8d6cdbe60fbe24ab… | 8d6cdbe60fbe24ab… |
| offline/retrieval_eval_v1/_02_build_corpora.py | BYTE-EQUAL | 649e6efdc1f629c7… | 649e6efdc1f629c7… |
| offline/retrieval_eval_v1/_03_build_embeddings.py | BYTE-EQUAL | 67a3be0ba9de2317… | 67a3be0ba9de2317… |
| offline/retrieval_eval_v1/_04_run_retrieval_and_metrics.py | BYTE-EQUAL | 940353b6d62f0f0f… | 940353b6d62f0f0f… |
| offline/retrieval_eval_v1/_corpora_stats.json | BYTE-EQUAL | b880b24ca7ea013e… | b880b24ca7ea013e… |
| offline/retrieval_eval_v1/_embedding_build_summary.json | BYTE-EQUAL | a86b110095674e78… | a86b110095674e78… |
| offline/retrieval_eval_v1/_probe_runtime.json | BYTE-EQUAL | 45dd9ef4daa8d1c8… | 45dd9ef4daa8d1c8… |
| offline/retrieval_eval_v1/_process_modules_snapshot.json | BYTE-EQUAL | a6f959fe45386bba… | a6f959fe45386bba… |
| offline/retrieval_eval_v1/_questions_summary.json | BYTE-EQUAL | eafc942046e1edc1… | eafc942046e1edc1… |
| offline/retrieval_eval_v1/_retrieval_stage_summary.json | BYTE-EQUAL | 5babb65b8906a52e… | 5babb65b8906a52e… |
| offline/retrieval_eval_v1/_reval_utils.py | ADAPTED | 1fbece150a59abac… | ea08863708f00cbb… |
| offline/retrieval_eval_v1/_run_orchestrator_summary.json | BYTE-EQUAL | f23fe7406c0784dd… | f23fe7406c0784dd… |
| offline/retrieval_eval_v1/_run_pytest_result.json | BYTE-EQUAL | 5ef2d07163e9ff31… | 5ef2d07163e9ff31… |
| offline/retrieval_eval_v1/failure_samples.json | BYTE-EQUAL | 5b82aec10d540813… | 5b82aec10d540813… |
| offline/retrieval_eval_v1/offline_embedding_index/SYN_citation_block_map.json | BYTE-EQUAL | 0e8e49044804b427… | 0e8e49044804b427… |
| offline/retrieval_eval_v1/offline_embedding_index/corpus_strategy_a.json | BYTE-EQUAL | 5173fb895044a2d0… | 5173fb895044a2d0… |
| offline/retrieval_eval_v1/offline_embedding_index/corpus_strategy_b.json | BYTE-EQUAL | 747f9869abab0a4f… | 747f9869abab0a4f… |
| offline/retrieval_eval_v1/offline_embedding_index/index_manifest.json | BYTE-EQUAL | ff05d5b78d332d00… | ff05d5b78d332d00… |
| offline/retrieval_eval_v1/offline_embedding_index/query_embeddings.npz | BYTE-EQUAL | e978c23db6273837… | e978c23db6273837… |
| offline/retrieval_eval_v1/offline_embedding_index/strategy_a.npz | BYTE-EQUAL | 79ee5ffbcde1e7ae… | 79ee5ffbcde1e7ae… |
| offline/retrieval_eval_v1/offline_embedding_index/strategy_b.npz | BYTE-EQUAL | 55a6c8a5f591a77b… | 55a6c8a5f591a77b… |
| offline/retrieval_eval_v1/retrieval_ab_metrics.csv | BYTE-EQUAL | 9749d1970722b8a5… | 9749d1970722b8a5… |
| offline/retrieval_eval_v1/retrieval_eval_v1.json | BYTE-EQUAL | b576cd2ac6594552… | b576cd2ac6594552… |
| offline/retrieval_eval_v1/retrieval_eval_v1.md | BYTE-EQUAL | cd21a21c4eb174a3… | cd21a21c4eb174a3… |
| offline/retrieval_eval_v1/retrieval_query_results.csv | BYTE-EQUAL | 9e75e65ed4034095… | 9e75e65ed4034095… |
| offline/retrieval_eval_v1/retrieval_questions.jsonl | BYTE-EQUAL | 67bfa3c459aded04… | 67bfa3c459aded04… |
| offline/retrieval_eval_v1/run_retrieval_eval_v1.py | BYTE-EQUAL | 43995ac4f751c715… | 43995ac4f751c715… |
| offline/retrieval_eval_v1/test_retrieval_eval_v1.py | ADAPTED | f9bce28b483d9dcb… | 37a0bc4be19116a8… |
| offline/chunking_strategy_v2/cleaner_v2.py | ADDED | （源 F:\DataBase\trae_work\RAG） | c85259f7ac42e106… |
| offline/chunking_strategy_v2/metadata_normalizer_v1.py | ADDED | （源 F:\DataBase\trae_work\RAG） | 82d0ce54d6312a76… |
| offline/chunking_strategy_v2/pdf_parser_v2_poc.py | ADDED | （源 F:\DataBase\trae_work\RAG） | d4b45bff56eb3a94… |


## ADDED 段说明（迁移补齐，F5 复核新发现）

仓库 modules/ingestion/ 下 cleaner_v2.py / metadata_normalizer_v1.py / docx_parser_v2_poc.py 为 0 字节占位；
Cleaner V2 链路真身仅在开发工作区 F:\DataBase\trae_work\RAG。切块 dispatch 对三者均为动态 import 且
sys.path 中本目录（OUT_DIR）优先于 modules/ingestion，故真身随包迁入 offline/chunking_strategy_v2/ 即被解析。
全仓库零引用（纯离线组件）→ 仓库占位文件保持原样未动（在线零影响）。
