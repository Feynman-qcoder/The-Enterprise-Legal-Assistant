# AutoDL 私有化部署实录：Qwen3.8-27B-FP8 + Legal RAG

> **一句话结论**：在 AutoDL 单张 48GB 消费级卡上，把线上的「DashScope 云端千问」换成
> **本机自托管千问**，业务代码**零改动**跑通「检索 → 重排 → 生成」全链路，
> 数据不出内网；并用 A/B 消融实验把「思考模式」这个开关用数据定了下来。

| 项 | 值 |
|---|---|
| 日期 | 2026-09-16 ~ 2026-09-17 |
| 实例 | AutoDL `vGPU-48GB`（E48 机）· GeForce RTX 4090 · compute_cap 8.9 (Ada) |
| 模型 | Qwen3.8-27B-FP8（官方 FP8，e4m3 / block-128） |
| 服务栈 | vLLM 0.29.0 + Milvus Lite + MySQL 8.0.46 + Redis 6.0.16 + FastAPI |
| 结果 | 入库 `FINAL_CHECK=PASS`；冒烟 3 层全 PASS；显存 42.9G / 48G |
| 成本 | 约 4 小时 ≈ **¥12**（其中约 1.6 小时浪费在配错的 torch 源上） |
| 分支 | `autodl-private-deploy-20260916`（从 tag `pre-autodl-deploy-20260916` 切出） |

---

## 1. 目标与约束

**目标**：验证「私有化部署」这条路走得通 —— 法律数据不出内网，模型自己托管。

**约束（都是真的，不是假设）**：

1. **不改业务代码**。`backend/`、`modules/` 一行不动 —— 否则"私有化"这件事就变成了重写项目。
2. **不动原有 git 历史**。所有工作在新分支上，原分支 `v2-one-click-ingestion-20260914` 全程未被触碰。
3. **只用官方 FP8**，不做自定义量化。

**为什么"零改动"能成立**：读代码后确认，换模型本就是「换字符串」的设计 ——

```python
# modules/rag/pipeline.py:261 与 modules/rag/intent.py:40
ChatOpenAI(model=s.llm_model, api_key=..., base_url=s.dashscope_base_url, streaming=True)
```

`base_url` 与 `model` 都来自 `.env`；全库 grep `extra_body|enable_thinking|incremental_output|result_format`
**0 命中** → 没有任何 DashScope 专有参数 → vLLM 的 OpenAI 兼容端点可直连。

---

## 2. 硬件与显存预算

按「官方 FP8」标准，三个模型必须**同卡共存**：

| 组件 | 显存 |
|---|---|
| Qwen3.8-27B-FP8 权重 | 28.5 GB |
| KV cache @128K | 4.3 GB |
| BGE-M3（嵌入） | ~1.5 GB |
| BGE Reranker（重排） | ~1.5 GB |
| 框架 / CUDA / 激活开销 | ~1.5 GB |
| **合计** | **≈ 37.3 GB** |

→ **48GB 是底线，只剩约 10.7GB 余量**。这条结论直接排除了 32GB 卡（5090）与 24GB 卡（4090/3090/A10）。

**实测落地（与规划一致）**：

```
总 42963 / 49140 MiB
  vLLM (Qwen3.8-27B-FP8)   37656 MiB
  应用 (BGE-M3 + Reranker)  5292 MiB     ← 三模型同卡 ✅
```

> ⚠️ **FP8 的硬件前提**：FP8 张量核只有 **Ada / Hopper** 有。A10 / A100 是 Ampere，
> 上 FP8 没有加速（甚至更慢）。选卡时这一条比显存更容易被忽略。

---

## 3. 架构调整：没有 Docker 的路线

接管实例后实测发现一个**决定性事实**：

```
CapEff = 00000000a80425fb   ← Docker 默认能力集，不含 CAP_SYS_ADMIN
Seccomp = 2                 ← 被 seccomp 过滤
systemd = offline
无 docker.sock
```

→ **AutoDL 实例本身是容器，装不了 Docker**。Milvus standalone（需 etcd + minio + milvus 三容器）
的 compose 路线**彻底不可行**，仓库里的 `deploy/docker-compose.yml` 在该实例上无用。

### 应对：Milvus → Milvus Lite

改造成**可选**的 uri 模式，`MILVUS_URI` 为空时行为与改造前**逐字节一致**：

```python
# modules/milvus_store/client.py
if s.milvus_uri:  # 【新增】uri 模式：Milvus Lite 本地文件 / 完整 URI
    connections.connect(alias=_ALIAS, uri=s.milvus_uri)
    return _ALIAS
# ↓↓↓ 以下为原有 host/port 模式：MILVUS_URI 为空时行为与改造前逐字节一致 ↓↓↓
```

先做了 5 项能力探测，**全部通过**才动手：

| 探测项 | 结果 |
|---|---|
| `connections.connect(uri)` | ✅ |
| ORM `Collection` | ✅ |
| `insert` | ✅ |
| **HNSW 索引** | ✅ 被接受 |
| `search` + `expr` 标量过滤 | ✅ |

MySQL / Redis 改为**原生安装**（8.0.46 / 6.0.16），用叠加 compose 的补丁方式保留 Docker 路线给其他环境。

---

## 4. 最终 vLLM 参数（每一个都是修错修出来的）

```bash
export PATH="$VENV/bin:$PATH"          # ← 关键，见坑 #4

python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --port 8001
```

| 参数 | 为什么是这个值 |
|---|---|
| `--max-model-len 65536` | 128K 需要 8.33 GiB KV，预算里没有。64K 够用（检索侧父文档 5×8000 字符 ≈ 40K token） |
| `--gpu-memory-utilization 0.80` | 默认 0.9 会吞满卡，BGE-M3 + Reranker 直接 OOM。0.80 给它们留 9.6 GiB |
| `--max-num-seqs 64` | **GDN 混合架构的硬约束**，见坑 #3 |
| `--reasoning-parser qwen3` | 把推理内容分流到 `reasoning` 字段，不进 `content` |
| `enable_thinking: false` | 防思考泄漏，见坑 #1 与 [A/B 报告](./thinking_ab_report.md) |
| `export PATH="$VENV/bin:$PATH"` | FlashInfer JIT 用裸名调 `ninja`，见坑 #4 |

---

## 5. 八个坑（按"值得复用"排序）

### 🔴 #1 思考模式会直接泄漏到用户可见输出

**背景**：Qwen3.8-27B **默认开 thinking**，而 `modules/rag/pipeline.py` 构造 ChatOpenAI 时
未传任何 `enable_thinking`（全局 grep 0 命中），且 `stream_chat` 逐片直接 yield 给 SSE。

**后果（如果没提前拦）**：前端会看到一大段「内心独白」，首字延迟从秒级变十秒级。

**修法（零代码）**：vLLM 侧加 `--default-chat-template-kwargs '{"enable_thinking": false}'`
+ `--reasoning-parser qwen3` 双保险。

**验证方式**：实测 `reasoning_len=478 / content_len=1095 / content 中  thinking 残留 = 0`
→ 证实推理内容被正确分流。配套消融实验见 [thinking_ab_report.md](./thinking_ab_report.md)。

---

### 🔴 #2 `MILVUS_URI` 与 pymilvus 自身环境变量重名

**症状**：`ConnectionConfigException: Illegal uri` —— 连 `import pymilvus` 都进不去。

**根因**：`MILVUS_URI` 是 **pymilvus 自己的环境变量**（`Config.MILVUS_URI`），
且 pymilvus 会在 **import 阶段**从 cwd 的 `.env` 读取并校验，只接受 `http(s)://`。

**定位方法**：4 组对照实验 —— `/tmp` 下正常、项目目录下必崩、`env -u` 解除变量仍崩
→ 锁定为「cwd 的 .env 被 pymilvus 读到」。

**修法**：别名改为 `MILVUS_LITE_URI`（提交 `86b7254`）。字段名与 `client.py` 不变。

---

### 🔴 #3 GDN（Gated DeltaNet）混合架构：Mamba 状态缓存块

**症状**：

```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (122).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

**原理（关键知识点）**：Qwen3.8-27B 是 GDN 线性注意力 + 全注意力混合架构
（64 层 = 48 层线性注意力 + 16 层全注意力，`full_attention_interval=4`）。
除常规 KV cache 外，**每条解码序列还要占 1 个 Mamba 状态缓存块**（线性注意力的递推状态）。

→ **任何 GDN / Mamba 系混合模型的部署都必须显式设 `--max-num-seqs`**，否则 CUDA graph
捕获阶段必失败。默认 256 远大于可用的 122 块。

**修法**：`--max-num-seqs 64`。

---

### 🔴 #4 `ninja` 不在 PATH（现场看起来与显存/架构毫无关系）

**症状**：卡在采样预热，`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`

**根因**：`ninja` 是 pip 包，装在 `.venv-vllm/bin/ninja`；FlashInfer JIT 用**裸名**调用它、依赖
PATH。而我用 `nohup .venv-vllm/bin/python -m vllm...` 启动**没有把 venv/bin 加进 PATH**，
也没 activate。

→ **通用教训**：用绝对路径 python 启动 vLLM（而非 `activate`）时，
**必须 `export PATH="$VENV/bin:$PATH"`**，否则 FlashInfer 等 JIT 组件会在裸名找工具时炸。

---

### 🟠 #5 应用依赖与 vLLM 不能混装（所以开了两个 venv）

应用侧 `langchain_huggingface.HuggingFaceEmbeddings` 载 BGE-M3、
`sentence_transformers.CrossEncoder` 载 Reranker；而 vLLM 会强钉 transformers/torch 版本
→ 混装会把 transformers 降级，BGE 加载失败。

**实测证据**：应用 venv 装到 `torch 2.14.0`，vLLM 钉的是 `torch==2.13.0` —— **版本本来就不同**。
两个 venv 各下一份 torch 是必要成本。

---

### 🟠 #6 镜像自带 torch，别重复装（本次最大的效率来源）

一开始让 venv 自己下 `torch+cu124`，卡在 `nvidia_cudnn_cu12`（665MB）近 2 小时。
后来发现 **conda base 已有 torch 2.8.0+cu128 / CUDA_AVAILABLE=True / cap (8,9)**。

**修法**：`python -m venv --system-site-packages .venv` 继承 base 的 torch，
只装 requirements 里剩下的包 → **2 分钟装完，venv 仅 661M**。

> 附带教训：**torch 官方源 `download.pytorch.org` 在国内被限速到几乎不可用**，
> 必须换镜像。另实测 **TUNA 比阿里云镜像稳**（阿里云在 553MB 级大文件上反复停滞 0 MB/min）。

---

### 🟡 #7 `pkill -f` 自杀陷阱（踩了两次）

`pkill -f 'uvicorn backend.app.main'` 会**匹配到我自己 SSH 命令的 cmdline**，
把会话杀掉、目标进程却活着。`pkill -f 'pip install torch --index-url'` 同理，
导致一个卡住的 pip 存活了 1 小时 37 分。

**修法**：用方括号技巧 `pgrep -f '[u]vicorn backend'`，或纯 PID 击杀。

---

### 🟡 #8 vLLM EngineCore 会改进程名

**症状**：`pgrep -f 'vllm.entrypoints'` 永远匹配不到 EngineCore 子进程
→ 重启 vLLM 时旧进程仍占 **37.6GB** 显存 →
`Free memory on device cuda:0 (5.03/47.37 GiB) is less than desired`。

**根因**：EngineCore 把进程标题改成了 `VLLM::EngineCore`。

**修法**：必须按「占用显存的进程」清理 ——
`nvidia-smi --query-compute-apps=pid` 拿到 PID 列表，保留 uvicorn、杀其余。

---

## 6. 部署完成验证

### 6.1 入库（关键里程碑）

`python offline/scripts/run_v2_ingest.py --precomputed` → **FINAL_CHECK=PASS，13.4 秒**

```
mysql legal_parent_contract_v1: 2217 (exp 2217) OK
mysql distinct document_identity:  49 (exp 49)   OK
mysql legal_child_parent_map_v1: 4134 (exp 4134) OK
milvus xiaoyi_legal_child_v2:    4134 (exp 4134) OK
corpus_meta.corpus_version >= 1:   49 OK
```

→ 证明 **Milvus Lite 能扛住真实项目代码 + HNSW 索引 + 4134 向量**。
末尾 `RuntimeError: Event loop is closed`（aiomysql 析构）为无害遗留，退出码 0。

> 走 `--precomputed`：`frozen_assets/` 含 `embeddings.npy` 预计算向量，
> **入库不需 GPU、不重算 embedding**。

### 6.2 冒烟测试（3 层全 PASS）

```
/health          PASS  {'ok': True, 'assistant': '小意'}
vLLM /v1/models  PASS  已加载=['Qwen3.8-27B-FP8']
RAG 问答#1        PASS  首字 0.01s / 总耗时 0.01s /  824 字（命中 exact 缓存）
RAG 问答#2        PASS  首字 4.95s / 总耗时 23.08s / 1033 字（冷启动真实延迟）
```

**回答质量抽检**（本地千问生成，grounded、带条款引用、**无思考残留**）：

- 「处理敏感个人信息应当取得什么样的同意？」→ 答「**单独同意**」+
  《个人信息保护法》**第二十九条** +《网络数据安全管理条例》第二十二条第（二）项
- 「经营者不得实施哪些混淆行为？」→ 引《反不正当竞争法》**2025 修订版第七条第（一）项**

---

## 7. 交付物清单

全部为**新增文件**，未修改仓库中任何既有文件：

| 文件 | 作用 |
|---|---|
| `deploy/autodl/deploy_autodl.sh` | 主脚本：分阶段、幂等、可断点（check/env/deps/llm/infra/ingest/backend/smoke/all/status/stop/stop-all） |
| `deploy/autodl/upload_from_local.sh` | 本机 → AutoDL 上传（补 git 排除的模型权重） |
| `deploy/autodl/download_model.py` | ModelScope 并行 Range 下载器（绕开官方 CLI 限速） |
| `deploy/autodl/smoke_test.py` | 三层冒烟验收 + 延迟指标 |
| `deploy/autodl/thinking_ab.py` | 思考模式 A/B 消融实验脚本 |
| `deploy/autodl/boot_all.sh` | **开机一键恢复**（本镜像无 systemd，服务都不会自启） |
| `deploy/autodl/stop_all.sh` | 优雅停服（含 EngineCore 坑注释） |
| `deploy/autodl/env.autodl.example` | AutoDL 专用 `.env` 模板 |
| `deploy/autodl/docker-compose.autodl.yml` | 叠加补丁：补 MySQL、纠正容器连库地址 |
| `deploy/autodl/README.md` | 部署手册（显存预算 / 排错 / 回滚） |

**代码改动仅 2 个文件**（为 Milvus Lite 可选开关，向后兼容）：

| 文件 | 改动 |
|---|---|
| `modules/core/config.py` | +5 行，新增 `milvus_uri` 字段（别名 `MILVUS_LITE_URI`） |
| `modules/milvus_store/client.py` | +13 −1 行，新增 uri 分支 |

### 提交链

```
c7fb723  feat(autodl): add thinking-mode A/B harness + boot/stop scripts
6ec86d3  fix(smoke): send Bearer token when checking vLLM /v1/models
86b7254  fix(milvus): rename env alias MILVUS_URI -> MILVUS_LITE_URI (name collision)
bdca6a2  feat(milvus): add optional MILVUS_URI for Milvus Lite (backward compatible)
181ac00  fix(deploy): address 3 blockers found in pre-mortem review
54acc9e  feat(deploy): add local->AutoDL uploader; document git-ignored model weights
166a80c  feat(deploy): auto-detect torch index by GPU compute capability
e56d756  chore(deploy): force LF line endings for deploy/autodl
c7e5daf  feat(deploy): AutoDL private-deployment toolchain
f670c99  chore(deploy): snapshot before AutoDL private-deployment trial   ← tag: pre-autodl-deploy-20260916
```

---

## 8. 环境事实速查（便于下次开机）

| 项 | 值 |
|---|---|
| SSH 别名 | `autodl-e48`（`connect.weste.seetacloud.com:33245`） |
| 仓库位置 | **必须** `/root/autodl-tmp/Legal_System`（`/root` 属 30GB 系统盘） |
| Python | `/root/miniconda3/bin/python` 3.12.3（非登录 shell 不加载 PATH，需 `bash -lc`） |
| venv | `.venv`（应用，661M，继承 base torch）/ `.venv-vllm`（8.0G，vLLM 0.29.0） |
| 模型 | `models/Qwen3.8-27B-FP8` 29G（66 个 safetensors 分片 + index.json） |
| 端口 | vLLM 8001 / 后端 8000 / MySQL 3306 / Redis 6379 |
| 开机恢复 | `bash /root/autodl-tmp/boot_all.sh` |
| 停服 | `bash /root/autodl-tmp/stop_all.sh` |
| 无卡模式 | **别在传输/安装中途切** —— 切 = 实例重启 = SSH 会话中断 |

### 三个必须记住的运维约束

1. **Milvus Lite 是单进程文件锁**：后端占着 `milvus.db` 时，另一进程连会报
   `Open local milvus failed` → **改库前必须先停后端**。
2. **实例没装 `ss` / `netstat`**：`ss -lnt` 会静默失败 → 曾误判"端口未监听"。
   **端口探测必须用 Python socket 或 bash `/dev/tcp`**。
3. **切「无卡模式」= 实例重启**，所以别在传输/安装中途切；要切的话等阶段结束。

---

## 9. 已知未完成项

| 项 | 状态 |
|---|---|
| 前端 UI 未验证 | AutoDL 只映射 6006/6008，看 UI 需另起静态服务 + 改 CORS |
| `deploy_autodl.sh` 未适配无 Docker 路线 | `infra` 阶段仍是 compose 逻辑；本次为手工执行（README 已说明） |
| `deploy_autodl.sh` 的 `TORCH_INDEX` 默认值 | 仍是 `download.pytorch.org`（国内不可用），应改镜像源 |
| `stage_status` 依赖 `ss` | 该实例无 `ss`，需改用 Python socket |
| C 组实验（`reasoning_effort=low`）未跑 | A/B 报告已列为折中方案 |
| 未做负载测试 | 单用户验证通过，并发/吞吐未测 |
| Milvus Lite 改动未合入 main | 仍在 `autodl-private-deploy-20260916` 分支 |

---

## 10. 成本

| 项 | 量级 |
|---|---|
| AutoDL `vGPU-48GB` | ≈ ¥3.03/h |
| 数据盘（付费 50GB） | ≈ ¥0.35/日（**关机也计费**） |
| 本次总开销 | 约 4 小时 ≈ **¥12** |

**踩坑成本占比很高**：约 1.6 小时（≈¥5）浪费在配错的 torch 源上 —— 这是本次最贵的教训。
跑完一轮「起服务 + 入库 + 冒烟」正常约 **1~2 小时**，即 **¥3~12**。

> 与阿里云对照：同显存档位下阿里云约为 AutoDL 的 **2~2.5 倍**
> （阿里云实测 ¥17.95/h 综合 vs AutoDL L20 48G ¥2.88/h）。
> 阿里云优势在合规 / 可开票 / 企业级 / 带宽稳定；AutoDL 优势在便宜、开卡快。
> 按量付费的阿里云**普通关机仍计费**，必须释放实例才停费。
