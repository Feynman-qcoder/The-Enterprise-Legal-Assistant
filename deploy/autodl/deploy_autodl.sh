#!/usr/bin/env bash
# =============================================================================
# 小意 Legal RAG · AutoDL 私有化部署一键脚本
# -----------------------------------------------------------------------------
# 设计目标（为什么这么写，而不是一条长命令）：
#   1) 分阶段：每一步可单独执行、单独重跑；失败时不必从头再来
#   2) 幂等：已在运行/已下载/已入库的步骤自动跳过
#   3) 不动业务代码：本脚本只读代码、只写自己目录下的日志/pid 与仓库根的 .env
#   4) 成本自觉：AutoDL 按小时计费，每阶段开头都会提示当前在烧多少钱
#
# 用法（仓库根目录）：
#   bash deploy/autodl/deploy_autodl.sh check      # 环境体检（先跑这个）
#   bash deploy/autodl/deploy_autodl.sh env        # 生成 .env
#   bash deploy/autodl/deploy_autodl.sh deps       # 建 venv + 装依赖
#   bash deploy/autodl/deploy_autodl.sh llm        # 下载并启动本地千问(vLLM)
#   bash deploy/autodl/deploy_autodl.sh infra      # 起 MySQL/Redis/Milvus
#   bash deploy/autodl/deploy_autodl.sh ingest     # 冻结资产入库
#   bash deploy/autodl/deploy_autodl.sh backend    # 起 FastAPI
#   bash deploy/autodl/deploy_autodl.sh smoke      # 冒烟验收
#   bash deploy/autodl/deploy_autodl.sh all        # 全流程串跑
#   bash deploy/autodl/deploy_autodl.sh status     # 看当前状态
#   bash deploy/autodl/deploy_autodl.sh stop       # 停应用层（llm/backend）
#   bash deploy/autodl/deploy_autodl.sh stop-all   # 连中间件一起停
#
# 可用环境变量覆盖（含默认值）：
#   MODEL_ID=Qwen/Qwen3.8-27B-FP8      MODEL_REV=master
#   MODEL_DIR=/root/autodl-tmp/models/Qwen3.8-27B-FP8
#   SERVED_NAME=Qwen3.8-27B-FP8        VLLM_PORT=8001     BACKEND_PORT=6006
#   GPU_MEM_FRAC=0.72                  MAX_MODEL_LEN=131072
#   TORCH_INDEX=(留空=按 GPU 架构自动选 cu121/cu124/cu128；显式设置则强制覆盖)
#   FORCE=1  # env 阶段覆盖已有 .env（会先自动备份）
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. 基础路径与变量
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_DIR="$SCRIPT_DIR"
LOG_DIR="$AUTODL_DIR/logs"
RUN_DIR="$AUTODL_DIR/run"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B-FP8}"
MODEL_REV="${MODEL_REV:-master}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.8-27B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B-FP8}"
VLLM_PORT="${VLLM_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-6006}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.72}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# torch wheel 源：留空 = 按 GPU 架构自动选择（见 detect_torch_index）：
#   sm_120(Blackwell)→cu128 / sm_90(Hopper)→cu124 / sm_89(Ada)→cu121 / sm_80+→cu121
TORCH_INDEX="${TORCH_INDEX:-}"

# ⚠️ vLLM 独立 venv —— 不要装进应用 venv！
# 应用侧用 sentence-transformers / langchain-huggingface 加载 BGE-M3 与 Reranker，
# 而 vLLM 会强钉 transformers/torch 版本；混装会把 transformers 降级，
# 导致 BGE 加载失败或行为漂移。两者必须隔离。
VLLM_VENV="${VLLM_VENV:-$PROJECT_ROOT/.venv-vllm}"
VLLM_PY="$VLLM_VENV/bin/python"
# Qwen3.8-27B 是 GDN(Gated DeltaNet) 混合架构，需 vLLM ≥ 0.17.0
VLLM_MIN_VERSION="${VLLM_MIN_VERSION:-0.17.0}"
# 国内 pip 源：vLLM 依赖体量很大，默认 PyPI 会非常慢
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# Qwen3.8 默认开启「思考模式」且默认 preserve_thinking。
# 项目侧没有任何 enable_thinking 处理（ChatOpenAI 未传 extra_body），
# 所以必须在服务端关掉，否则思考内容会随 SSE 逐片推给用户。
ENABLE_THINKING="${ENABLE_THINKING:-false}"
VENV="$PROJECT_ROOT/.venv"
PY="$VENV/bin/python"

COMPOSE_BASE="$PROJECT_ROOT/deploy/docker-compose.yml"
COMPOSE_OVERLAY="$AUTODL_DIR/docker-compose.autodl.yml"

# ⚠️ 关键：modules/rag/dashscope_http.py 里 httpx 用了 trust_env=True，
#    必须排除本机地址，否则「本机 LLM 调用」会被 HTTP(S)_PROXY 劫持。
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

# --------------------------------------------------------------------------- #
# 1. 输出与工具函数
# --------------------------------------------------------------------------- #
C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[36m'; C_0=$'\033[0m'
log()  { printf '%s\n' "${C_B}[*]${C_0} $*"; }
ok()   { printf '%s\n' "${C_G}[✓]${C_0} $*"; }
warn() { printf '%s\n' "${C_Y}[!]${C_0} $*"; }
err()  { printf '%s\n' "${C_R}[✗]${C_0} $*" >&2; }
die()  { err "$*"; exit 1; }

banner() {
  echo
  echo "=============================================================="
  echo " $*"
  echo "=============================================================="
}

cost_note() {
  warn "当前实例按小时计费——本阶段结束若不再继续，请及时关机（关机不计费；"
  warn "但注意 AutoDL 数据盘会保留收费，模型目录建议放 /root/autodl-tmp）。"
}

load_env() {
  [ -f "$PROJECT_ROOT/.env" ] || return 0
  local line k v
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    k="$(printf '%s' "$k" | tr -d '[:space:]')"
    [ -n "$k" ] || continue
    case "$k" in *[!A-Za-z0-9_]*) continue ;; esac
    export "$k=$v"
  done < "$PROJECT_ROOT/.env"
}

is_running() {
  local pidfile="$RUN_DIR/$1.pid"
  [ -f "$pidfile" ] || return 1
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_bg() {
  local name="$1"; shift
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  if is_running "$name"; then ok "$name 已在运行（pid $(cat "$RUN_DIR/$name.pid")）"; return 0; fi
  ( cd "$PROJECT_ROOT" && nohup "$@" >"$LOG_DIR/$name.log" 2>&1 & echo $! >"$RUN_DIR/$name.pid" )
  sleep 2
  if is_running "$name"; then
    ok "$name 已启动（pid $(cat "$RUN_DIR/$name.pid")，日志 $LOG_DIR/$name.log）"
  else
    err "$name 启动失败，日志尾部："; tail -n 25 "$LOG_DIR/$name.log" >&2 || true
    return 1
  fi
}

stop_bg() {
  local name="$1"
  if ! is_running "$name"; then rm -f "$RUN_DIR/$name.pid"; ok "$name 未在运行"; return 0; fi
  local pid; pid="$(cat "$RUN_DIR/$name.pid")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$RUN_DIR/$name.pid"
  ok "$name 已停止（pid $pid）"
}

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&- && return 0 || return 1; }

wait_port() {
  local port="$1" name="$2" timeout="${3:-120}" i=0
  log "等待 $name 就绪（127.0.0.1:$port，最多 ${timeout}s）…"
  while [ "$i" -lt "$timeout" ]; do
    port_open "$port" && { ok "$name 已就绪"; return 0; }
    sleep 2; i=$((i + 2))
  done
  err "$name 在 ${timeout}s 内未就绪"; return 1
}

wait_http() {
  local url="$1" name="$2" timeout="${3:-180}" i=0
  log "等待 $name 响应（$url，最多 ${timeout}s）…"
  while [ "$i" -lt "$timeout" ]; do
    if curl -fsS --max-time 5 --noproxy '*' "$url" >/dev/null 2>&1; then ok "$name 已就绪"; return 0; fi
    sleep 3; i=$((i + 3))
  done
  err "$name 在 ${timeout}s 内未响应"; return 1
}

detect_torch_index() {
  # torch wheel 源：留空则按 GPU 计算能力自动选择；显式设置 TORCH_INDEX 则强制覆盖。
  #
  # 为什么必须自动判断：cu121 的 wheel 里没有 sm_120(Blackwell) 的 kernel，
  # 在 5090 / RTX PRO 6000 这类新卡上会报
  #   RuntimeError: CUDA error: no kernel image is available for execution on the device
  # 反过来，新 wheel 在老卡上一般没问题，但仍按架构精确匹配最稳。
  if [ -n "${TORCH_INDEX:-}" ]; then printf '%s' "$TORCH_INDEX"; return 0; fi
  local cap=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  fi
  case "${cap}" in
    12.*)        printf '%s' "https://download.pytorch.org/whl/cu128" ;;  # Blackwell sm_120
    9.*)         printf '%s' "https://download.pytorch.org/whl/cu124" ;;  # Hopper    sm_90
    8.9)         printf '%s' "https://download.pytorch.org/whl/cu121" ;;  # Ada       sm_89（原生支持 FP8）
    8.0|8.6|8.7) printf '%s' "https://download.pytorch.org/whl/cu121" ;;  # Ampere（无 FP8 张量核）
    *)           printf '%s' "https://download.pytorch.org/whl/cu124" ;;  # 未知：取较新保守值
  esac
}

# --------------------------------------------------------------------------- #
# 2. 阶段实现
# --------------------------------------------------------------------------- #
stage_check() {
  banner "阶段 check · 环境体检"
  load_env
  local fail=0

  log "GPU："
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version \
               --format=csv,noheader || true
    local cap; cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    case "${cap}" in
      12.*)        warn "计算能力 sm_${cap}（Blackwell）：torch 将自动选 cu128；用 cu121 会报 no kernel image" ;;
      9.*)         ok   "计算能力 sm_${cap}（Hopper）：torch 将选 cu124" ;;
      8.9)         ok   "计算能力 sm_${cap}（Ada）：原生支持 FP8 张量核，torch 选 cu121" ;;
      8.0|8.6|8.7) warn "计算能力 sm_${cap}（Ampere）：无 FP8 张量核；跑 FP8 权重只省显存不加速，建议改 BF16" ;;
      *)           warn "计算能力 sm_${cap:-未知}：torch 取保守值；若报 no kernel image 请显式指定 TORCH_INDEX" ;;
    esac
    local total_mib; total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ -n "${total_mib:-}" ] && [ "$total_mib" -ge 46000 ]; then
      ok "显存 ${total_mib} MiB：满足三模型同卡的最低要求（≥48GB）"
    else
      warn "显存 ${total_mib:-未知} MiB：低于 48GB 底线，FP8 128K 大概率 OOM（可降 MAX_MODEL_LEN）"
    fi
  else
    err "未找到 nvidia-smi —— 请确认实例已选 GPU 机型并正确开机"; fail=1
  fi

  log "磁盘："
  df -h / /root /root/autodl-tmp 2>/dev/null | sort -u || true

  log "Python（需 ≥3.11）："
  if command -v python3 >/dev/null 2>&1; then
    python3 -V
    python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)' \
      && ok "Python 版本满足要求" || { err "Python 版本低于 3.11"; fail=1; }
  else
    err "未找到 python3"; fail=1
  fi

  log "Docker（中间件依赖）："
  if command -v docker >/dev/null 2>&1; then
    docker --version || true
    if docker info >/dev/null 2>&1; then
      ok "Docker 可用"
      docker info 2>/dev/null | grep -iE 'nvidia|runtimes' || warn "未探测到 nvidia runtime（仅用中间件则无妨）"
    else
      warn "docker 命令存在但 daemon 不可用 —— AutoDL 常需特殊镜像才能跑 Docker"
      warn "若确认无法跑 Docker，请走备用路线：外部/远程 Milvus + 宿主机原生 MySQL"
      fail=1
    fi
  else
    warn "未找到 docker：中间件需另想办法（见 deploy/autodl/README.md「无 Docker 备用路线」）"
    fail=1
  fi

  log "端口占用（应为空闲）：$VLLM_PORT(vLLM) $BACKEND_PORT(backend) 3306 6379 19530"
  local p
  for p in "$VLLM_PORT" "$BACKEND_PORT" 3306 6379 19530; do
    if port_open "$p"; then warn "端口 $p 已被占用"; else ok "端口 $p 空闲"; fi
  done

  log "必需资产："
  local f
  for f in "frozen_assets/embeddings.npy" "frozen_assets/ingest_manifest_v1.jsonl" \
           "models/bge-m3" "models/bge-reranker-large" "requirements.txt" \
           "offline/scripts/run_v2_ingest.py" "scripts/init_mysql.sql"; do
    if [ -e "$PROJECT_ROOT/$f" ]; then ok "存在 $f"; else err "缺失 $f"; fail=1; fi
  done

  echo
  if [ "$fail" -eq 0 ]; then
    ok "体检通过，可以继续：bash deploy/autodl/deploy_autodl.sh env"
  else
    warn "体检存在告警/失败项，请先处理上面的 [✗] 行再继续"
  fi
  cost_note
  return "$fail"
}

stage_env() {
  banner "阶段 env · 生成 .env"
  local src="$AUTODL_DIR/env.autodl.example" dst="$PROJECT_ROOT/.env"
  [ -f "$src" ] || die "缺少模板：$src"
  if [ -f "$dst" ]; then
    local bak="$dst.bak.autodl.$(date +%Y%m%d_%H%M%S)"
    cp -p "$dst" "$bak"
    warn "已存在 .env，已备份到 $(basename "$bak")"
    if [ "${FORCE:-0}" != "1" ]; then
      warn "默认不覆盖。确认要覆盖请用：FORCE=1 bash deploy/autodl/deploy_autodl.sh env"
      return 0
    fi
    warn "FORCE=1：覆盖 .env"
  fi
  cp "$src" "$dst"
  ok "已生成 $dst"
  warn "请务必修改 MYSQL_PASSWORD / MYSQL_ROOT_PASSWORD 再继续"
}

stage_deps() {
  banner "阶段 deps · Python 依赖"
  load_env
  [ -d "$VENV" ] || { log "创建 venv：$VENV"; python3 -m venv "$VENV"; }
  "$PY" -m pip install --upgrade pip -q
  local torch_index; torch_index="$(detect_torch_index)"
  log "安装 torch（$torch_index）——体积大，耐心等…"
  "$PY" -m pip install torch --index-url "$torch_index"
  log "安装项目依赖（requirements.txt，只读不改；pip 源 $PIP_INDEX）…"
  "$PY" -m pip install -i "$PIP_INDEX" -r "$PROJECT_ROOT/requirements.txt"
  # aiomysql/PyMySQL 连 MySQL8 的 caching_sha2 认证需要 cryptography
  "$PY" -m pip install -i "$PIP_INDEX" cryptography
  log "校验 CUDA 可用性："
  "$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda_available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("⚠️ CUDA 不可用：嵌入/重排会退回 CPU（rerank 约 13s/次），请检查 torch 是否装成 CPU 版")
PYEOF
  ok "依赖安装完成（venv: $VENV）"
  cost_note
}

stage_llm() {
  banner "阶段 llm · 下载并启动本地千问（vLLM）"
  load_env
  local api_key="${DASHSCOPE_API_KEY:-sk-autodl-local}"

  # --- 1) 权重 ---
  if [ -f "$MODEL_DIR/config.json" ]; then
    ok "权重已存在：$MODEL_DIR"
  else
    warn "权重缺失，开始下载（约 28.5GB；期间 GPU 空转，建议先切「无卡模式」下载）"
    "$PY" "$AUTODL_DIR/download_model.py" --model "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
  fi

  # --- 2) vLLM（独立 venv：绝不能和应用 venv 混装）---
  if [ ! -x "$VLLM_PY" ]; then
    log "创建 vLLM 独立 venv：$VLLM_VENV"
    python3 -m venv "$VLLM_VENV"
  fi
  "$VLLM_PY" -m pip install --upgrade pip -q -i "$PIP_INDEX"
  if ! "$VLLM_PY" -c "import vllm" >/dev/null 2>&1; then
    log "安装 vLLM（独立 venv；不动 requirements.txt。依赖体量大，耐心等…）"
    "$VLLM_PY" -m pip install -i "$PIP_INDEX" vllm
  fi
  vv="$("$VLLM_PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 0)"
  log "vLLM 版本：$vv（Qwen3.8-27B 的 GDN 混合架构要求 ≥ $VLLM_MIN_VERSION）"
  if [ "$(printf '%s\n%s\n' "$VLLM_MIN_VERSION" "$vv" | sort -V | head -n1)" != "$VLLM_MIN_VERSION" ]; then
    warn "vLLM 版本低于 $VLLM_MIN_VERSION，Qwen3.8-27B(GDN) 可能起不来"
    warn "请升级：$VLLM_PY -m pip install -U -i $PIP_INDEX 'vllm>=$VLLM_MIN_VERSION'"
  fi

  # --- 3) 启动 ---
  if is_running vllm; then ok "vllm 已在运行，跳过启动"; return 0; fi
  log "启动 vLLM：port=$VLLM_PORT gpu_mem=$GPU_MEM_FRAC max_len=$MAX_MODEL_LEN"
  warn "gpu_mem=$GPU_MEM_FRAC 是刻意留白：要给 BGE-M3 + Reranker 留出约 5GB，"
  warn "若让 vLLM 用默认 0.9 会吞满整卡，嵌入/重排启动即 OOM。"

  # 思考模式：默认在服务端关闭（Qwen3.8 默认开思考 + preserve_thinking，
  # 而应用侧 ChatOpenAI 未传任何 enable_thinking，思考内容会随 SSE 推给用户）。
  # 若 vLLM 报 unknown argument，删掉 --default-chat-template-kwargs 这一行即可。
  local extra_args=(--reasoning-parser qwen3)
  if [ "$ENABLE_THINKING" != "true" ]; then
    extra_args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
  else
    warn "ENABLE_THINKING=true：保留思考模式，用户会看到推理过程且 TTFB 明显变长"
  fi

  start_bg vllm "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_DIR" \
      --served-model-name "$SERVED_NAME" \
      --host 127.0.0.1 --port "$VLLM_PORT" \
      --api-key "$api_key" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_FRAC" \
      --enable-prefix-caching \
      --trust-remote-code \
      "${extra_args[@]}"

  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM" 600
  log "已加载模型清单："
  curl -fsS --noproxy '*' -H "Authorization: Bearer $api_key" \
       "http://127.0.0.1:$VLLM_PORT/v1/models" || true
  echo
  ok "本地千问就绪。请确认 .env 的 LLM_MODEL / INTENT_MODEL = $SERVED_NAME"
  cost_note
}

stage_infra() {
  banner "阶段 infra · 起 MySQL / Redis / Milvus"
  load_env
  command -v docker >/dev/null 2>&1 || die "未找到 docker：请先解决（见 README「无 Docker 备用路线」）"
  docker info >/dev/null 2>&1 || die "docker daemon 不可用"

  # Milvus 三件套若本地已有镜像则不重复拉
  if ! docker image inspect docker.m.daocloud.io/milvusdb/milvus:v2.4.6 >/dev/null 2>&1; then
    if [ -f "$PROJECT_ROOT/deploy/milvus-deps.tar" ]; then
      log "从已打包离线包导入 Milvus 依赖镜像…"
      docker load -i "$PROJECT_ROOT/deploy/milvus-deps.tar"
    else
      warn "本地无 Milvus 镜像且无离线包，将尝试联网拉取（国内建议配置 deploy/docker-daemon-cn.json 镜像加速）"
    fi
  fi

  log "docker compose up -d（基座 + AutoDL 补丁：补 MySQL）"
  ( cd "$PROJECT_ROOT/deploy" && docker compose \
      -f docker-compose.yml \
      -f autodl/docker-compose.autodl.yml \
      up -d )

  wait_port 3306 "MySQL" 180
  wait_port 6379 "Redis" 60
  wait_port 19530 "Milvus" 240

  log "容器状态："
  ( cd "$PROJECT_ROOT/deploy" && docker compose \
      -f docker-compose.yml -f autodl/docker-compose.autodl.yml ps ) || true
  ok "中间件就绪"
  cost_note
}

stage_ingest() {
  banner "阶段 ingest · 冻结资产入库（V2）"
  load_env
  [ -x "$PY" ] || die "venv 不存在，请先跑 deps 阶段"
  log "恢复预计算基线：49 documents / 2217 parents / 4134 children（含 embeddings.npy，无需重算向量）"
  local logf="$LOG_DIR/ingest-$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$LOG_DIR"
  if ( cd "$PROJECT_ROOT" && "$PY" offline/scripts/run_v2_ingest.py --precomputed ) 2>&1 | tee "$logf"; then
    ok "入库脚本执行成功（完整日志：$logf）"
  else
    err "入库失败，日志尾部："; tail -n 30 "$logf" >&2 || true
    die "请先确认 MySQL / Milvus 就绪（infra 阶段）后重跑"
  fi
  ok "入库完成。注意：Manifest 更新后需重启后端刷新 CanonicalScope 缓存"
  cost_note
}

stage_backend() {
  banner "阶段 backend · 启动 FastAPI"
  load_env
  [ -x "$PY" ] || die "venv 不存在，请先跑 deps 阶段"
  if is_running backend; then ok "backend 已在运行，跳过启动"; return 0; fi
  log "启动 uvicorn：0.0.0.0:$BACKEND_PORT（AutoDL 对外映射端口为 6006）"
  start_bg backend "$PY" -m uvicorn backend.app.main:app --host 0.0.0.0 --port "$BACKEND_PORT"
  wait_http "http://127.0.0.1:$BACKEND_PORT/health" "backend" 300
  curl -fsS --noproxy '*' "http://127.0.0.1:$BACKEND_PORT/health" || true
  echo
  ok "后端就绪"
  cost_note
}

stage_smoke() {
  banner "阶段 smoke · 冒烟验收"
  load_env
  "$PY" "$AUTODL_DIR/smoke_test.py" --base-url "http://127.0.0.1:$BACKEND_PORT"
}

stage_status() {
  banner "当前状态"
  mkdir -p "$RUN_DIR"
  local n
  for n in vllm backend; do
    if is_running "$n"; then ok "$n 运行中（pid $(cat "$RUN_DIR/$n.pid")）"; else warn "$n 未运行"; fi
  done
  echo
  log "监听端口："
  (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E ":(3306|6379|19530|$VLLM_PORT|$BACKEND_PORT)\b" || echo "  (无相关监听)"
  echo
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo
  log "最近日志尾部："
  for n in vllm backend; do
    [ -f "$LOG_DIR/$n.log" ] && { echo "--- $n ---"; tail -n 5 "$LOG_DIR/$n.log"; }
  done
}

stage_stop() {
  banner "停止应用层"
  stop_bg backend
  stop_bg vllm
  warn "中间件容器仍在运行。全部停止：bash deploy/autodl/deploy_autodl.sh stop-all"
  cost_note
}

stage_stop_all() {
  stage_stop
  banner "停止中间件"
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    ( cd "$PROJECT_ROOT/deploy" && docker compose \
        -f docker-compose.yml -f autodl/docker-compose.autodl.yml down ) || true
    ok "容器已停止（数据卷保留，下次 up -d 即可恢复）"
  else
    warn "docker 不可用，跳过"
  fi
}

stage_all() {
  banner "全流程：check → env → deps → llm → infra → ingest → backend → smoke"
  warn "本流程会持续占用 GPU 计费；中途可 Ctrl+C 中断，之后单阶段重跑即可续上。"
  stage_check
  stage_env
  stage_deps
  stage_llm
  stage_infra
  stage_ingest
  stage_backend
  stage_smoke
  banner "全流程完成 ✅"
}

# --------------------------------------------------------------------------- #
# 3. 入口
# --------------------------------------------------------------------------- #
case "${1:-help}" in
  check)    stage_check ;;
  env)      stage_env ;;
  deps)     stage_deps ;;
  llm)      stage_llm ;;
  infra)    stage_infra ;;
  ingest)   stage_ingest ;;
  backend)  stage_backend ;;
  smoke)    stage_smoke ;;
  all)      stage_all ;;
  status)   stage_status ;;
  stop)     stage_stop ;;
  stop-all) stage_stop_all ;;
  *)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
esac
