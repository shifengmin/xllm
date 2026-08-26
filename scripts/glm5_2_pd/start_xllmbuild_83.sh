#!/bin/bash
# 用 xllm_build 启动混布测试（通过 bash -lc 加载完整环境）
set -e
export MODEL_PATH=/export/home/models/GLM-5.2-W8A8-EcoTech
export HOST=11.87.191.83
export NNODES=16
export START_PORT=19994
export LOG_DIR=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/colocate_tp16_xllmbuild
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
unset HCCL_OP_EXPANSION_MODE
unset ASCEND_SLOG_PRINT_TO_STDOUT
mkdir -p $LOG_DIR

pkill -9 -f /site-packages/xllm/xllm 2>/dev/null || true
pkill -9 -f xllm_build 2>/dev/null || true
sleep 3

for ((rank=0; rank<NNODES; rank++)); do
  port=$((START_PORT + rank))
  log_file=$LOG_DIR/rank_${rank}.log
  nohup bash -lc "
    export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:\${LD_PRELOAD:-}
    export LD_LIBRARY_PATH=/export/home/shifengmin.3/workspace/glm5_2_pd/lib:\${LD_LIBRARY_PATH:-}
    /export/home/shifengmin.3/workspace/xllm_build \
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
  echo started rank=$rank
done
echo ALL_STARTED
