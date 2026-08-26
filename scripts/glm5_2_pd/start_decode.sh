#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 PD decode: default DP=2, TP=8, layerwise split=4, MTP default 3.
XLLM_BIN="${XLLM_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/xllm/xllm}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GLM-5.2 checkpoint}"
HOST="${HOST:-127.0.0.1}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-${HOST}:19888}"
ETCD_ADDR="${ETCD_ADDR:-127.0.0.1:2379}"
NNODES="${NNODES:-16}"
START_PORT="${START_PORT:-19994}"
START_TRANSFER_PORT="${START_TRANSFER_PORT:-37100}"
DISAGG_PD_PORT="${DISAGG_PD_PORT:-8878}"
LOG_DIR="${LOG_DIR:-logs/glm5_2_pd/decode}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
LAYERWISE_SPLIT_SIZE="${LAYERWISE_SPLIT_SIZE:-4}"
KV_CACHE_TRANSFER_TYPE="${KV_CACHE_TRANSFER_TYPE:-Mooncake}"
DP_SIZE="${DP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
KV_SPLIT_SIZE="${KV_SPLIT_SIZE:-1}"

# AIV expands AllGather/AllReduce onto Vector cores (`_sub_1`). With
# --rank_tablefile this PD Decode path is stable and lowers TPOT.
# Leave the variable unset to take the default; export it empty to disable.
if [[ ! -v HCCL_OP_EXPANSION_MODE ]]; then
  export HCCL_OP_EXPANSION_MODE=AIV
fi

draft_args=()
if [[ "${NUM_SPECULATIVE_TOKENS}" -gt 0 ]]; then
  DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:?set DRAFT_MODEL_PATH to the GLM-5.2 MTP checkpoint}"
  draft_args=(--draft_model="$DRAFT_MODEL_PATH" --num_speculative_tokens="$NUM_SPECULATIVE_TOKENS")
fi

mkdir -p "$LOG_DIR"
for ((rank = 0; rank < NNODES; rank++)); do
  port=$((START_PORT + rank))
  transfer_port=$((START_TRANSFER_PORT + rank))
  log_file="$LOG_DIR/rank_${rank}.log"
  nohup "$XLLM_BIN" \
    --model="$MODEL_PATH" "${draft_args[@]}" --backend=llm \
    --host="$HOST" --port="$port" --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" --node_rank="$rank" \
    --dp_size="$DP_SIZE" --cp_size="$CP_SIZE" --layerwise_split_size="$LAYERWISE_SPLIT_SIZE" --kv_split_size="$KV_SPLIT_SIZE" \
    --npu_kernel_backend=ATB --communication_backend=hccl \
    --enable_disagg_pd=true --instance_role=DECODE \
    --etcd_addr="$ETCD_ADDR" --disagg_pd_port="$DISAGG_PD_PORT" \
    --transfer_listen_port="$transfer_port" \
    --kv_cache_transfer_type="$KV_CACHE_TRANSFER_TYPE" \
    --enable_prefix_cache=false --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    ${EXTRA_XLLM_ARGS:-} >"$log_file" 2>&1 &
  echo "started decode rank=$rank port=$port log=$log_file"
done
