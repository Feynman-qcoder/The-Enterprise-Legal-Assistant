# 文档索引

本目录记录对**冻结资产、部署与评测**的正式存档。

## 部署

| 文档 | 内容 |
|---|---|
| [`autodl-private-deploy.md`](./autodl-private-deploy.md) | **AutoDL 私有化部署实录**：环境事实、显存预算、最终 vLLM 参数、8 个坑、验证结果、提交链 |
| [`thinking_ab_report.md`](./thinking_ab_report.md) | **思考模式 A/B 消融实验报告**：TTFB 6.8 倍恶化的实测数据与决策依据 |
| [`thinking_ab_raw_answers.md`](./thinking_ab_raw_answers.md) | 上述实验的**原始回答全文**（附录，供人工判读质量） |

部署脚本与手册在 [`deploy/autodl/`](../deploy/autodl/)，其中 `README.md` 是操作手册。

## 冻结资产集成

| 文档 | 内容 |
|---|---|
| [`frozen_artifact_integration.md`](./frozen_artifact_integration.md) | 离线验证产物（Chunk Metadata Contract V1、Chunking Strategy V2 / Strategy B）的不可变引用与校验规则 |
