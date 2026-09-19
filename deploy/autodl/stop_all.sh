#!/usr/bin/env bash
# =============================================================================
# 停止全部服务（AutoDL 实例关机前可选执行；优雅停 MySQL 可确保数据落盘）
# -----------------------------------------------------------------------------
# ⚠️ 关键坑：vLLM 的 EngineCore 子进程会把进程标题改成 `VLLM::EngineCore`，
#    因此 `pgrep -f 'vllm.entrypoints'` **匹配不到它**，会导致重启 vLLM 时
#    旧进程仍占着 ~37GB 显存，新进程报：
#      ValueError: Free memory on device (5.03/47.37 GiB) on startup is less than
#                  desired GPU memory utilization (0.8, 37.9 GiB)
#    → 必须按「占用显存的进程」来清理（本脚本的做法）。
# =============================================================================
set -x

# 1) vLLM：按占卡进程清理，保留 uvicorn（应用）以便随后单独处理
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  args=$(ps -p "$p" -o args= 2>/dev/null)
  case "$args" in
    *uvicorn*) echo "KEEP $p (app)";;
    *)         echo "KILL $p (vllm) -> $(echo "$args" | cut -c1-40)"; kill -9 "$p" 2>/dev/null;;
  esac
done

# 2) 后端 / 残留下载进程
for p in $(pgrep -f '[u]vicorn backend' 2>/dev/null); do kill -9 "$p"; done
for p in $(pgrep -f '[d]ownload_model.py' 2>/dev/null); do kill -9 "$p"; done

# 3) 中间件（MySQL 用 mysqladmin 优雅停，保证落盘）
redis-cli shutdown nosave 2>/dev/null || pkill -x redis-server
mysqladmin -uroot shutdown 2>/dev/null || pkill -x mysqld

sleep 3
echo "--- 剩余占卡 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "STOP_ALL_DONE"
