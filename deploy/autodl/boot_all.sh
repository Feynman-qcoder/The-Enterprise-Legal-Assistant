#!/usr/bin/env bash
# =============================================================================
# AutoDL 实例「开机后一键恢复」脚本
# -----------------------------------------------------------------------------
# 为什么需要它：本镜像【没有 systemd】，实例重启/开机后 MySQL、Redis、vLLM、
# 后端 FastAPI 都不会自动起来，必须按顺序手工拉起。
#
# 用法（实例上执行，建议放 /root/autodl-tmp/ 下）：
#   bash boot_all.sh      # 一键恢复全部服务
#   bash stop_all.sh      # 停止全部服务（关机前可选）
#
# 前置：模型与 venv 已就绪（位于 /root/autodl-tmp/）
# 本脚本与仓库内 deploy/autodl/ 的其他脚本配套使用：
#   start_vllm.sh / start_backend.sh 由部署流程生成，见 README
# =============================================================================
set -x
P=/root/autodl-tmp/Legal_System
VENV=$P/.venv
VVENV=$P/.venv-vllm
export PATH="$VVENV/bin:$PATH"          # ★ FlashInfer JIT 需要 venv/bin 里的 ninja
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY

echo "########## 1/4 MySQL ##########"
pgrep -x mysqld >/dev/null || (service mysql start || mysqld_safe --user=mysql &)
sleep 8

echo "########## 2/4 Redis ##########"
pgrep -x redis-server >/dev/null || redis-server --daemonize yes
sleep 2

echo "########## 3/4 vLLM（本地千问）##########"
if ! curl -s --noproxy '*' --max-time 3 -H 'Authorization: Bearer sk-autodl-local' \
        http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q Qwen3; then
  nohup bash /root/autodl-tmp/start_vllm.sh > /root/autodl-tmp/vllm_restart.log 2>&1 < /dev/null &
  for i in $(seq 1 30); do
    curl -s --noproxy '*' --max-time 5 -H 'Authorization: Bearer sk-autodl-local' \
      http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q 'Qwen3.8-27B-FP8' && break
    sleep 15
  done
fi
curl -s --noproxy '*' --max-time 5 -H 'Authorization: Bearer sk-autodl-local' \
  http://127.0.0.1:8001/v1/models 2>/dev/null | head -c 120; echo

echo "########## 4/4 后端 FastAPI ##########"
pgrep -f '[u]vicorn backend' >/dev/null || \
  nohup bash /root/autodl-tmp/start_backend.sh > /root/autodl-tmp/backend_start.log 2>&1 < /dev/null &
sleep 30
curl -s --noproxy '*' --max-time 8 http://127.0.0.1:6006/health; echo

echo "########## 状态 ##########"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "BOOT_ALL_DONE"
