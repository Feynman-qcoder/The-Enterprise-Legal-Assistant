# 思考模式 A/B 消融实验报告

> **场景**：AutoDL 私有化部署 · Qwen3.8-27B-FP8（官方 FP8）
> **方法**：同一批法律问题，走**真实 RAG 链路**（BGE-M3 检索 → RRF → Reranker → 本地千问生成），
> 仅切换 vLLM 的 `--default-chat-template-kwargs` 中 `enable_thinking`。
> 每题使用独立 `user_external_id` 以**绕开 exact 缓存**，保证测得真实推理延迟。
>
> **脚本**：`deploy/autodl/thinking_ab.py`
> **原始回答全文**：见 [`thinking_ab_raw_answers.md`](./thinking_ab_raw_answers.md)

## 1. 实验配置

| | A 组 | B 组 |
|---|---|---|
| `enable_thinking` | `false` | `true` |
| 其余参数 | `max-model-len 65536` / `gpu-memory-utilization 0.80` / `max-num-seqs 64` / `reasoning-parser qwen3` | 同左 |
| 模型 | Qwen3.8-27B-FP8（官方 FP8，Ada sm_89） | 同左 |
| 硬件 | RTX 4090 48GB | 同左 |

> 两轮之间只重启 vLLM，**检索侧、Prompt 模板、采样参数完全不变**，保证变量单一。

## 2. 前置验证：开思考是否会污染用户可见输出

结论先行：**不会**。vLLM 0.29.0 的 `--reasoning-parser qwen3` 会把推理内容分流到独立的
`reasoning` 字段（注意：**不是**旧版的 `reasoning_content`），`delta.content` 只保留最终答案。

实测样本：

```
reasoning_len = 478
content_len   = 1095
THINKING_ACTIVE = True
content 中 " thinking" 残留 = 0
```

→ 这也说明「开思考」的代价**不是**输出被污染，而是纯延迟与 token 预算问题（见 §3）。

## 3. 延迟对比（核心结论）

| 题号 | 类型 | A 首字(s) | B 首字(s) | 倍数 | A 总耗时(s) | B 总耗时(s) | A 字数 | B 字数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 简单·条文 | 5.30 | 18.56 | 3.5× | 29.48 | 37.07 | 1434 | 1118 |
| S2 | 简单·条文 | 4.95 | 30.02 | 6.1× | 32.92 | 47.61 | 1650 | 1055 |
| H1 | 复杂·适用 | 1.95 | 42.69 | 21.9× | 24.93 | 62.26 | 1312 | 1161 |
| H2 | 复杂·多跳 | 5.77 | 300.11 | 52.0× | 41.66 | 300.11 | 2198 | 0 |

**汇总（仅统计成功项）**

| 指标 | A（关思考） | B（开思考） | 变化 |
|---|---:|---:|---|
| 平均首字 TTFB | 4.49 s | 30.42 s | **↑ 6.8 倍** |
| 平均总耗时 | 32.25 s | 48.98 s | ↑ 51.9% |
| 平均可见字数 | 1648 | 1111 | ↓ 32.6% |
| 成功率 | 4/4 | **3/4（H2 超时 300s）** | ↓ |

**延迟倍率随问题复杂度单调上升**：S1 3.5× → S2 6.1× → H1 21.9× → H2 超时。
即**问题越复杂、思考的延迟代价越大**，而复杂问题恰恰是最需要快速反馈的场景。

## 4. 结论与决策

**保持 `enable_thinking=false`。** 依据三条：

1. **可用性不达标**：TTFB 恶化 6.8 倍，且最复杂的多跳题（H2）直接超出 300s 超时；
   结合本项目 `intent.py` 每请求还会多跑一次模型调用，端到端体验会进一步劣化。
2. **产出反而更少**：可见字数下降 32.6% —— 推理 token 挤占了同一 `max_tokens` 预算下的答案篇幅。
3. **未见明显质量增益**：在抽检的 4 题上，A 组答案均 grounded、带条款引用、无编造
   （见附录原文），未观察到开思考带来的实质提升。

> ⚠️ **本结论的边界**：样本仅 4 题、单次运行，属于**方向性判断**而非统计结论。
> 若后续要做严谨评测，建议用项目自带的 `data/法律问答对.xlsx` + `offline/retrieval_eval_v1`
> 跑更大样本，并补上第三组折中方案。

**折中方案（未执行）**：Qwen3.8 原生支持 `reasoning_effort` 三档（low / medium / high）。
若确有需要，可试 `reasoning_effort=low`，在延迟与推理深度之间取中间点。

## 5. 复现方式

```bash
# 在 AutoDL 实例上（vLLM 已就绪的前提下）
cd /root/autodl-tmp/Legal_System

# A 组：vLLM 以 enable_thinking=false 启动
python deploy/autodl/thinking_ab.py --label A_off --out /root/autodl-tmp/ab_A_off.json

# 切到 B 组：用 ENABLE_THINKING=true 重启 vLLM，待就绪后
python deploy/autodl/thinking_ab.py --label B_on  --out /root/autodl-tmp/ab_B_on.json
```

脚本要点：4 题分层（2 简单条文 + 2 复杂适用/多跳），**每题独立 `user_external_id`**
——否则第 2 题起会命中 exact 缓存、返回 0.01s 的假延迟数据。
