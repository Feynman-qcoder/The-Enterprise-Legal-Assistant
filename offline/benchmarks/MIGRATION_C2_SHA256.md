# Task 22 C2 迁移 SHA256 清单（压测脚本 + gold set）

- 源：f:\DataBase\trae_work\RAG_小号\legal_eval_audit_v1\（bench_* 五脚本 + outputs/aligned_goldset_v2.jsonl）
- 目标：offline/benchmarks/
- PARAMETERIZED：按 F8 清单做 argparse 参数化（--gold/--out）+ 仓库根 sys.path 推导，双端哈希预期不同
- BYTE-EQUAL：逐字节复制
- 排除：outputs/ 其余全部内容（含 model.safetensors 2.16GB；.gitignore 已追加 *.safetensors 兜底）

| 文件 | 状态 | 源 SHA256 | 副本 SHA256 |
|---|---|---|---|
| offline/benchmarks/bench_e2e.py | PARAMETERIZED | 源 cfb04c8f268bc871… | e556b93dd9e8550c… |
| offline/benchmarks/bench_concurrency.py | PARAMETERIZED | 源 df4c6d05e900097a… | cb8fe132075aec4f… |
| offline/benchmarks/bench_quality.py | PARAMETERIZED | 源 2e2215575cd44f77… | 612c8ed26130e722… |
| offline/benchmarks/bench_fastpath.py | PARAMETERIZED | 源 d517cd7d79017cc7… | 67e4c6f651ad0fb0… |
| offline/benchmarks/bench_precheck_redis.py | PARAMETERIZED | 源 ca1399ce15dbdfed… | 3c3c94d4fc4804a8… |
| offline/benchmarks/aligned_goldset_v2.jsonl | BYTE-EQUAL | 2216347ea5d6ba5c… | 2216347ea5d6ba5c… |
