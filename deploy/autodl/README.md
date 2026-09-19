# AutoDL 私有化部署（Legal RAG）

把线上的 DashScope 云端千问换成**自托管的 Qwen3.8-27B-FP8**，数据不出内网。
**业务代码零改动** —— 换模型只改 `.env`，起服务靠本目录脚本。

---

## 1. 选型

| 项 | 要求 / 建议 |
|---|---|
| GPU | **≥48GB，Ada/Hopper 架构**：L20 48G / RTX 4090 48G 均可，96G 更宽裕 |
| 避开 | 32GB（5090 / V100）与 24GB 卡；`vGPU-48GB` 能跑但算力共享、延迟会抖动 |
| 系统盘 | 30GB（默认）—— **仓库必须放数据盘** `/root/autodl-tmp/` |
| 数据盘 | **扩到 100GB**（实测约需 50.5GB，默认 50GB 不够）；关机仍按 0.007 元/日/GB 计费 |
| 镜像 | PyTorch 2.8.0 / Python 3.12 (ubuntu22.04) / CUDA 12.8（Blackwell 选它） |
| 成本 | L20 48G ≈¥2.88/h ｜ PRO6000 96G ≈¥5.98/h ｜ 无卡模式 ≈¥0.1/h |

---

## 2. 部署

### 第 0 步（本机）：上传仓库 + 模型权重

```bash
bash deploy/autodl/upload_from_local.sh <实例host> <端口>
# 默认传到 /root/autodl-tmp/Legal_System（勿用 /root，系统盘只有 30GB）
```

> `models/`（BGE-M3 + Reranker ≈6GB）被 `.gitignore` 排除，clone 拿不到，必须单独上传。

### 第 1 步（实例上）：分阶段推进，每步都有 Gate

```bash
cd /root/autodl-tmp/Legal_System
bash deploy/autodl/deploy_autodl.sh check   # ① 体检：不写任何东西
bash deploy/autodl/deploy_autodl.sh env     # ② 生成 .env（改掉两个密码后再继续）
bash deploy/autodl/deploy_autodl.sh all     # ③ deps → llm → infra → ingest → backend → smoke
```

| 阶段 | PASS 标准 |
|---|---|
| check | GPU ≥48GB、Python ≥3.11、资产齐全、端口空闲 |
| deps | `torch.cuda.is_available() == True` |
| llm | `GET /v1/models` 返回 `Qwen3.8-27B-FP8` |
| infra | 3306 / 6379 / 19530 就绪 |
| ingest | 内置硬门禁：parents=2217 / identity=49 / map=4134 / Milvus=4134 |
| backend | `GET /health` 返回 `{"ok": true, ...}` |
| smoke | 三层冒烟全过（`/health` → `/v1/models` → 端到端 SSE），并打印 **TTFB 首字延迟** |

**省钱**：下载权重阶段用 AutoDL 的「无卡模式」（≈¥0.1/h），下完再切 GPU 跑 `llm` 之后的阶段。

---

## 3. `.env` 关键改动（4 处 + 1 条必设）

| 变量 | 线上 | AutoDL |
|---|---|---|
| `DASHSCOPE_BASE_URL` | dashscope 官方地址 | `http://127.0.0.1:8001/v1`（自托管 vLLM） |
| `DASHSCOPE_API_KEY` | 真实 Key | `sk-autodl-local`（vLLM 本地占位值） |
| `LLM_MODEL` / `INTENT_MODEL` | `qwen-max` 等 | `Qwen3.8-27B-FP8`（**须与 vLLM `--served-model-name` 完全一致**） |
| `EMBEDDING_DEVICE` / `RERANK_DEVICE` | `cpu` | `cuda:0` / `cuda`（三模型同卡） |

```ini
NO_PROXY=127.0.0.1,localhost,::1
no_proxy=127.0.0.1,localhost,::1
```

> `modules/rag/dashscope_http.py` 的 httpx 用 `trust_env=True`，会读 `HTTP_PROXY`/`HTTPS_PROXY`。
> 不排除 127.0.0.1，本机 LLM 调用会被代理劫持并失败。

---

## 4. 显存预算与调优（三模型同卡）

| 组件 | 显存 |
|---|---:|
| Qwen3.8-27B FP8 权重 | 28.5 GB |
| KV Cache @128K（1 路） | 4.3 GB |
| BGE-M3 + bge-reranker-large | ≈3.0 GB |
| 框架 / CUDA 上下文 | ≈1.5 GB |
| **合计** | **≈37.3 GB** |

→ 48GB 是底线（余约 10GB）；每多 1 路 128K 并发 +4.3GB。

脚本刻意把 vLLM 的 `--gpu-memory-utilization` 压到 `0.72`（默认 `0.9` 会吞满整卡 → BGE 两模型启动即 OOM）。
显存不够时按序调：① 降 `GPU_MEM_FRAC` ② 降 `MAX_MODEL_LEN`（如 32768）③ 嵌入 / 重排改回 `cpu`。

---

## 5. 常见问题

| 现象 | 根因 | 处理 |
|---|---|---|
| 问答立刻返回兜底文案 | vLLM 未起 / 模型名不匹配 / 被代理劫持 | 查 `logs/vllm.log`；核对 `LLM_MODEL` 与 `NO_PROXY` |
| `Connection refused 127.0.0.1:8001` | vLLM 进程崩了（多因显存） | `status` 看进程与日志 |
| BGE 模型启动即 CUDA OOM | vLLM 吞满显存 | 降 `GPU_MEM_FRAC` |
| 检索为空、答非所问 | 未入库 | 跑 `ingest` 阶段 |
| 系统盘被 Docker 撑爆 | Docker `data-root` 默认在系统盘 | 改到数据盘（见下） |
| 拉镜像慢 / 失败 | 网络 | 配镜像加速（见下） |

**把 Docker 数据目录挪到数据盘**（顺带配镜像加速，列表取自仓库自带 `deploy/docker-daemon-cn.json`）：

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "data-root": "/root/autodl-tmp/docker",
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
EOF
sudo systemctl restart docker
docker info | grep -i "docker root dir"   # 应显示 /root/autodl-tmp/docker
```

**关于思考模式**：Qwen3.8 默认开启思考，用户会在前端看到大段「内心独白」、首字延迟变十秒级。
脚本已在服务端关闭（`--default-chat-template-kwargs '{"enable_thinking": false}'`）；
若要保留思考：`ENABLE_THINKING=true bash deploy/autodl/deploy_autodl.sh llm`。

**另一个隐性开销**：填了 `DASHSCOPE_API_KEY` 时，每个请求会先跑一次大模型做意图二分类，再跑生成 → **TTFB 翻倍**。若不需「闲聊引导」分支，可让 vLLM 不设 `--api-key`、`.env` 该值留空以短路意图识别。

**若 `docker info` 失败**（AutoDL 实例本身是嵌套容器）：Milvus 改连远程实例，MySQL / Redis 用 `apt` 原生安装 —— 只是换连接地址，业务代码同样不用改。

---

## 6. 运维与回滚

```bash
bash deploy/autodl/deploy_autodl.sh status    # 进程 / 端口 / 显存
tail -f deploy/autodl/logs/vllm.log           # 模型日志（backend.log 同理）
bash deploy/autodl/deploy_autodl.sh stop      # 停 vllm + backend
bash deploy/autodl/deploy_autodl.sh stop-all  # 连容器一起停（数据卷保留）

# 回滚：本目录全部是新增文件，删掉日志并清理即可
git clean -fd deploy/autodl/logs deploy/autodl/run
```

---

## 7. 本目录文件清单

| 文件 | 作用 |
|---|---|
| `deploy_autodl.sh` | 主脚本（分阶段、幂等、可断点）**← 在实例上跑** |
| `upload_from_local.sh` | **← 在本机跑**：上传仓库 + `models/` 权重 |
| `env.autodl.example` | AutoDL 专用 `.env` 模板 |
| `docker-compose.autodl.yml` | **叠加补丁**：补 MySQL、纠正容器连库地址（不动基座） |
| `download_model.py` | ModelScope 并行下载器（绕开官方 CLI 限速） |
| `smoke_test.py` | 三层冒烟验收 + TTFB 指标 |
| `logs/`、`run/` | 运行时生成（已被 `.gitignore` 忽略） |
