#!/bin/bash
# 83 混布启动脚本 (TP=16, enable_disagg_pd=false) - 修复 non-login shell LD_LIBRARY_PATH
# 用法: bash start_colocate.sh [tag]
#   tag: 日志目录后缀，默认 manual

set -e

TAG=${1:-manual}
MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech
HOST=11.87.191.83
NNODES=16
START_PORT=19994
LOG_DIR=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/colocate_tp16_${TAG}
XLLM_BIN=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm

mkdir -p $LOG_DIR

# 清理残留进程
echo "[INFO] cleaning residual xllm processes..."
pkill -9 -f /site-packages/xllm/xllm 2>/dev/null || true
sleep 3
remaining=$(ps -eo pid,args | grep /site-packages/xllm/xllm | grep -v grep | wc -l)
echo "[INFO] remaining xllm procs: $remaining"

# 启动 16 个 rank（用 bash -lc 加载完整环境变量）
for ((rank=0; rank<NNODES; rank++)); do
  port=$((START_PORT + rank))
  log_file=$LOG_DIR/rank_${rank}.log
  nohup bash -lc "
    export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:\${LD_PRELOAD:-}
    export LD_LIBRARY_PATH=/export/home/shifengmin.3/workspace/glm5_2_pd/lib:\${LD_LIBRARY_PATH:-}
    export HCCL_EXEC_TIMEOUT=300
    export HCCL_CONNECT_TIMEOUT=300
    unset HCCL_OP_EXPANSION_MODE
    unset ASCEND_SLOG_PRINT_TO_STDOUT
    $XLLM_BIN \
      --model=$MODEL_PATH --backend=llm \
      --host=$HOST --port=$port --master_node_addr=${HOST}:19888 \
      --nnodes=$NNODES --node_rank=$rank \
      --dp_size=1 --cp_size=1 \
      --npu_kernel_backend=ATB --communication_backend=hccl \
      --enable_disagg_pd=false --instance_role=DEFAULT \
      --enable_prefix_cache=false --enable_chunked_prefill=false \
      --enable_schedule_overlap=true \
      --max_memory_utilization=0.85 \
      --num_speculative_tokens=0
  " >$log_file 2>&1 &
  echo "[INFO] started rank=$rank port=$port"
done

echo "[INFO] ALL_STARTED tag=$TAG log_dir=$LOG_DIR"
echo "[INFO] wait for 'Application startup complete' in rank_0.log, then run: bash test_smoke.sh"
