#!/usr/bin/env bash
# =============================================================================
# 本地 → AutoDL 上传脚本（在**你的 Windows / Mac 本机**上执行，不是在实例上）
# -----------------------------------------------------------------------------
# 为什么需要它：
#   本仓库的 .gitignore 排除了 `models/`（BGE-M3 + Reranker 约 6GB），
#   所以 git clone 到实例上**不会**带过去；而这两个目录是嵌入/重排的权重，缺了必报错。
#   同时仓库里有一堆不该上传的大目录（node_modules、.venv、软件安装包），
#   直接 `scp -r 整个仓库` 会传几个 GB 的垃圾。
#
# 用法（本机 Git Bash / WSL / macOS）：
#   bash deploy/autodl/upload_from_local.sh <host> <port> [remote_root]
#
# 例：
#   bash deploy/autodl/upload_from_local.sh connect.westd.seetacloud.com 34857
#   # 默认上传到 /root/autodl-tmp/Legal_System
#
# 提示：AutoDL 的**系统盘只有 30GB**，装不下 venv+模型，
#       所以 remote_root 默认指向数据盘 /root/autodl-tmp —— 别改成 /root。
#
# 前置：本机 SSH 能登录实例（推荐把 ~/.ssh/id_rsa.pub 贴到 AutoDL 控制台「SSH 公钥」）
# =============================================================================
set -euo pipefail

HOST="${1:-}"
PORT="${2:-}"
REMOTE_ROOT="${3:-/root/autodl-tmp/Legal_System}"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
fi

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=15)

# 不上传的东西：版本库/虚拟环境/前端依赖/软件安装包/临时产物/大压缩包
EXCLUDES=(
  ".git"
  ".venv"
  "venv"
  "__pycache__"
  "node_modules"
  ".pytest_cache" ".pytest_tmp" ".uv-cache-codex"
  ".codex_tmp_adjudication" ".codex_tmp_audit_connectivity.py"
  ".idea" ".vscode"
  "项目用到的软件及工具"      # 里面有 Anaconda/MySQL/Docker 安装包，好几个 GB
  "adjudication_output"
  "cleaner_v1_1_metadata_update"
  "deploy/milvus-deps.tar"    # 如需离线 Milvus 镜像，单独手动传
  "deploy/autodl/logs" "deploy/autodl/run"
)

echo "=============================================================="
echo "源目录   : $LOCAL_ROOT"
echo "目标     : root@$HOST:$REMOTE_ROOT  (端口 $PORT)"
echo "=============================================================="

echo
echo "[1/3] 预检：必须存在的目录（缺一不可）"
missing=0
for d in "models/bge-m3" "models/bge-reranker-large" "frozen_assets" \
         "backend" "modules" "offline/scripts" "scripts/init_mysql.sql"; do
  if [ -e "$LOCAL_ROOT/$d" ]; then
    printf '  ✅ %s\n' "$d"
  else
    printf '  ❌ %s（缺失）\n' "$d"; missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo; echo "有必需目录缺失，先补齐再上传。"; exit 1; }

echo
echo "[2/3] 预估体积（含/不含被排除项）"
if command -v du >/dev/null 2>&1; then
  echo "  仓库总大小 : $(du -sh "$LOCAL_ROOT" 2>/dev/null | cut -f1)"
  echo "  models/    : $(du -sh "$LOCAL_ROOT/models" 2>/dev/null | cut -f1)"
  echo "  frozen_assets: $(du -sh "$LOCAL_ROOT/frozen_assets" 2>/dev/null | cut -f1)"
  echo "  （被排除的大目录：node_modules / .venv / 项目用到的软件及工具）"
fi

echo
echo "[3/3] 开始传输…"
ssh "${SSH_OPTS[@]}" "root@$HOST" "mkdir -p '$REMOTE_ROOT'"

if command -v rsync >/dev/null 2>&1; then
  # rsync：支持断点续传与增量，6GB 的模型目录推荐用这个
  echo "  使用 rsync（可断点续传）"
  RSYNC_ARGS=(-az --info=progress2 --partial --human-readable
              -e "ssh ${SSH_OPTS[*]}")
  for e in "${EXCLUDES[@]}"; do RSYNC_ARGS+=(--exclude "$e"); done
  rsync "${RSYNC_ARGS[@]}" "$LOCAL_ROOT/" "root@$HOST:$REMOTE_ROOT/"
else
  # 没有 rsync：用 tar 流式压缩传输，排除列表同样生效
  echo "  未找到 rsync，改用 tar over ssh（不可断点续传）"
  TAR_EXCLUDES=()
  for e in "${EXCLUDES[@]}"; do TAR_EXCLUDES+=(--exclude="$e"); done
  ( cd "$LOCAL_ROOT" && tar czf - "${TAR_EXCLUDES[@]}" . ) \
    | ssh "${SSH_OPTS[@]}" "root@$HOST" "mkdir -p '$REMOTE_ROOT' && tar xzf - -C '$REMOTE_ROOT'"
fi

echo
echo "✅ 上传完成。到实例上验证："
echo "     ssh -p $PORT root@$HOST"
echo "     cd $REMOTE_ROOT && ls models/ frozen_assets/ deploy/autodl/"
echo "     bash deploy/autodl/deploy_autodl.sh check"
