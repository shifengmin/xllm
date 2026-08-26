#!/usr/bin/env bash
set -euo pipefail

# GLM-5.2 PD prefill: default CP=2, TP=8, KV split=2, MTP default 3.
XLLM_BIN="${XLLM_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/xllm/xllm}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the GLM-5.2 checkpoint}"
HOST="${HOST:-127.0.0.1}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-${HOST}:18888}"
ETCD_ADDR="${ETCD_ADDR:-127.0.0.1:2379}"
NNODES="${NNODES:-16}"
START_PORT="${START_PORT:-18994}"
START_TRANSFER_PORT="${START_TRANSFER_PORT:-36100}"
DISAGG_PD_PORT="${DISAGG_PD_PORT:-8877}"
LOG_DIR="${LOG_DIR:-logs/glm5_2_pd/prefill}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
KV_CACHE_TRANSFER_TYPE="${KV_CACHE_TRANSFER_TYPE:-Mooncake}"
CP_SIZE="${CP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
KV_SPLIT_SIZE="${KV_SPLIT_SIZE:-2}"

# AIV op-expansion creates AllGather/AllReduce `_sub_1` on first decode/prefill
# op. PD Decode has been dying during that lazy link; keep it unset unless
# an experiment explicitly wants AIV.
unset HCCL_OP_EXPANSION_MODE || true

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
    --cp_size="$CP_SIZE" --dp_size="$DP_SIZE" --kv_split_size="$KV_SPLIT_SIZE" \
    --npu_kernel_backend=ATB --communication_backend=hccl \
    --enable_disagg_pd=true --instance_role=PREFILL \
    --etcd_addr="$ETCD_ADDR" --disagg_pd_port="$DISAGG_PD_PORT" \
    --transfer_listen_port="$transfer_port" \
    --kv_cache_transfer_type="$KV_CACHE_TRANSFER_TYPE" \
    --enable_prefix_cache=true --enable_chunked_prefill=true \
    --enable_schedule_overlap=false \
    ${EXTRA_XLLM_ARGS:-} >"$log_file" 2>&1 &
  echo "started prefill rank=$rank port=$port log=$log_file"
done
