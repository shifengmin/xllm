#!/usr/bin/env bash

set -euo pipefail
export PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin:/usr/sbin:/sbin
if [[ -z "${DCP_KV_HOST:-}" ]]; then
  DCP_KV_HOST=11.87.191.98
fi

ACTION="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$WORKTREE/build/lib.linux-aarch64-cpython-311/xllm/xllm"
MODEL="${DCP_KV_MODEL:-/export/home/models/GLM-5-final-w8a8}"
MODEL_ID="${DCP_KV_MODEL_ID:-$(basename "$MODEL")}"
RUN_ROOT="${DCP_KV_RUN_ROOT:-$WORKTREE/build/dcp-kv-validation}"
PID_FILE="$RUN_ROOT/pids"
LOG_ROOT="$RUN_ROOT/logs"
NNODES="${DCP_KV_NNODES:-16}"
DCP_SIZE="${DCP_KV_DCP_SIZE:-2}"
LAYERWISE_KV="${DCP_KV_LAYERWISE_KV:-true}"
DP_SIZE="${DCP_KV_DP_SIZE:-2}"
EP_SIZE="${DCP_KV_EP_SIZE:-16}"
START_PORT="${DCP_KV_START_PORT:-52100}"
MASTER_PORT="${DCP_KV_MASTER_PORT:-22198}"
TRANSFER_START_PORT="${DCP_KV_TRANSFER_START_PORT:-37100}"
HOST="${DCP_KV_HOST:-$(hostname -I | awk '{print $1}')}"
MASTER_ADDR="$HOST:$MASTER_PORT"

usage() {
  echo "usage: $0 preflight|start|status|stop"
}

port_is_listening() {
  local port="$1"
  local hex_port
  hex_port="$(printf '%04X' "$port")"
  awk -v suffix=":$hex_port" \
    '$2 ~ suffix && $4 == "0A" {found=1} END {exit found ? 0 : 1}' \
    /proc/net/tcp /proc/net/tcp6 2>/dev/null
}

collect_owned_pids() {
  OWNED_PIDS=()
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
      local cmd
      cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
      if [[ "$cmd" == *"$BIN"* && "$cmd" == *"--master_node_addr=$MASTER_ADDR"* ]]; then
        OWNED_PIDS+=("$pid")
      fi
    done < "$PID_FILE"
  fi

  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    local pid="${proc##*/}"
    local cmd
    cmd="$(tr '\0' ' ' < "$proc/cmdline")"
    if [[ "$cmd" == *"$BIN"* && "$cmd" == *"--master_node_addr=$MASTER_ADDR"* ]]; then
      if [[ ! " ${OWNED_PIDS[*]} " =~ " $pid " ]]; then
        OWNED_PIDS+=("$pid")
      fi
    fi
  done
}

collect_owned_rank_pids() {
  OWNED_RANK_PIDS=()
  [[ -f "$PID_FILE" ]] || return 0
  while IFS= read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
    local cmd
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$cmd" == *"$BIN"* && "$cmd" == *"--master_node_addr=$MASTER_ADDR"* ]]; then
      OWNED_RANK_PIDS+=("$pid")
    fi
  done < "$PID_FILE"
}

check_ports() {
  local rank port
  for rank in $(seq 0 $((NNODES - 1))); do
    port=$((START_PORT + rank))
    if port_is_listening "$port"; then
      echo "PORT_BUSY=$port"
      return 1
    fi
    port=$((TRANSFER_START_PORT + rank))
    if port_is_listening "$port"; then
      echo "PORT_BUSY=$port"
      return 1
    fi
  done
  if port_is_listening "$MASTER_PORT"; then
    echo "PORT_BUSY=$MASTER_PORT"
    return 1
  fi
}

preflight() {
  [[ -x "$BIN" ]] || { echo "MISSING_BINARY=$BIN"; return 1; }
  [[ -f "$MODEL/config.json" ]] || { echo "MISSING_MODEL_CONFIG=$MODEL/config.json"; return 1; }
  [[ "$NNODES" -gt 0 ]] || { echo "INVALID_NNODES=$NNODES"; return 1; }
  [[ "$DP_SIZE" -gt 0 ]] || { echo "INVALID_DP_SIZE=$DP_SIZE"; return 1; }
  [[ "$DCP_SIZE" -ge 1 && "$DCP_SIZE" -le $((NNODES / DP_SIZE)) ]] || {
    echo "INVALID_DCP_SIZE=$DCP_SIZE"
    return 1
  }
  [[ "$LAYERWISE_KV" == "true" || "$LAYERWISE_KV" == "false" ]] || {
    echo "INVALID_LAYERWISE_KV=$LAYERWISE_KV"
    return 1
  }
  if [[ "$LAYERWISE_KV" == "true" && "$DCP_SIZE" -le 1 ]]; then
    echo "LAYERWISE_KV_REQUIRES_DCP_GT_ONE=$DCP_SIZE"
    return 1
  fi
  [[ $((NNODES % DP_SIZE)) -eq 0 ]] || {
    echo "WORLD_NOT_DIVISIBLE_BY_DP world=$NNODES dp=$DP_SIZE"
    return 1
  }
  [[ $((NNODES / DP_SIZE)) -ge "$DCP_SIZE" ]] || {
    echo "DCP_EXCEEDS_ATTENTION_TP tp=$((NNODES / DP_SIZE)) dcp=$DCP_SIZE"
    return 1
  }
  collect_owned_pids
  if ((${#OWNED_PIDS[@]} > 0)); then
    echo "INSTANCE_ALREADY_RUNNING=${OWNED_PIDS[*]}"
    return 1
  fi
  check_ports
  echo "PREFLIGHT_OK=1"
  echo "MODEL=$MODEL"
  set +u
  echo "MODEL_ID=$MODEL_ID"
  echo "WORLD=$NNODES DP=$DP_SIZE ATTENTION_TP=$((NNODES / DP_SIZE)) EP=$EP_SIZE DCP=$DCP_SIZE LAYERWISE_KV=$LAYERWISE_KV"
  set -u
  echo "INSTANCE_ROLE=DECODE ENABLE_DISAGG_PD=false TRANSFER=LlmDataDist"
  echo "MASTER_ADDR=$MASTER_ADDR"
}

configure_runtime() {
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
  set -u
  export PATH="/usr/local/python3.11.15/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
  export NPU_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export NPU_MEMORY_FRACTION=0.96
  export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
  export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
  export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
  export ATB_CONTEXT_WORKSPACE_SIZE=0
  export OMP_NUM_THREADS=12
  export HCCL_IF_BASE_PORT=54800
  export HCCL_EXEC_TIMEOUT=600
  export HCCL_CONNECT_TIMEOUT=600
  export HCCL_OP_EXPANSION_MODE=AIV
  export LCCL_DETERMINISTIC=1
  export HCCL_DETERMINISTIC=true
  export ASDOPS_LOG_LEVEL=ERROR
  export ASDOPS_LOG_TO_STDOUT=1
  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_xllm_math/op_api/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/aarch64-linux/devlib:/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/device/lib64:${LIBRARY_PATH:-}"
  TORCH_NPU_PRELOAD=/usr/local/python3.11.15/lib/libpython3.11.so.1.0:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so
  export TORCH_NPU_PRELOAD
}

start() {
  preflight
  mkdir -p "$LOG_ROOT"
  : > "$PID_FILE"
  configure_runtime
  {
    echo "MODEL=$MODEL"
    echo "MODEL_ID=$MODEL_ID"
    echo "WORLD=$NNODES DP=$DP_SIZE ATTENTION_TP=$((NNODES / DP_SIZE)) EP=$EP_SIZE DCP=$DCP_SIZE LAYERWISE_KV=$LAYERWISE_KV"
    echo "INSTANCE_ROLE=DECODE ENABLE_DISAGG_PD=false TRANSFER=LlmDataDist"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "BINARY_SHA=$(sha256sum "$BIN" | awk '{print $1}')"
  } | tee "$RUN_ROOT/config.txt"

  local rank port transfer_port cpu_start cpu_end log_file
  for rank in $(seq 0 $((NNODES - 1))); do
    port=$((START_PORT + rank))
    transfer_port=$((TRANSFER_START_PORT + rank))
    cpu_start=$((rank * 40))
    cpu_end=$((cpu_start + 39))
    log_file="$LOG_ROOT/rank_${rank}.log"
    local args=(
      --model="$MODEL"
      --model_id="$MODEL_ID"
      --host="$HOST"
      --port="$port"
      --master_node_addr="$MASTER_ADDR"
      --nnodes="$NNODES"
      --node_rank="$rank"
      --dp_size="$DP_SIZE"
      --ep_size="$EP_SIZE"
      --cp_size=1
      --decode_dcp_size="$DCP_SIZE"
      --enable_decode_dcp_layerwise_kv_cache="$LAYERWISE_KV"
      --max_memory_utilization=0.85
      --host_blocks_factor=0
      --block_size=128
      --communication_backend=hccl
      --npu_kernel_backend=ATB
      --max_tokens_per_batch=4096
      --expert_parallel_degree=1
      --max_seqs_per_batch=32
      --enable_chunked_prefill=false
      --enable_schedule_overlap=false
      --enable_graph=false
      --enable_graph_mode_decode_no_padding=false
      --enable_shm=true
      --enable_prefix_cache=false
      --enable_service_routing=false
      --enable_disagg_pd=false
      --instance_role=DECODE
      --kv_cache_transfer_type=LlmDataDist
      --kv_cache_transfer_mode=PUSH
      --transfer_listen_port="$transfer_port"
    )
    if command -v numactl >/dev/null 2>&1; then
      setsid numactl -C "$cpu_start-$cpu_end" env LD_PRELOAD="$TORCH_NPU_PRELOAD" \
        "$BIN" "${args[@]}" > "$log_file" 2>&1 &
    else
      setsid env LD_PRELOAD="$TORCH_NPU_PRELOAD" \
        "$BIN" "${args[@]}" > "$log_file" 2>&1 &
    fi
    echo "$!" >> "$PID_FILE"
  done
  echo "LAUNCHED=$(wc -l < "$PID_FILE")"
  echo "RUN_ROOT=$RUN_ROOT"
  if ! wait_ready; then
    stop || true
    return 1
  fi
}

wait_ready() {
  local elapsed dead weight_count transfer_count pid
  for elapsed in $(seq 0 10 600); do
    dead=0
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && dead=$((dead + 1))
    done < "$PID_FILE"
    if [[ "$dead" -gt 0 ]]; then
      echo "PROCESS_DIED_BEFORE_READY=$dead"
      tail -120 "$LOG_ROOT/rank_0.log" || true
      return 1
    fi
    weight_count="$(grep -l 'Weight loading completed' "$LOG_ROOT"/rank_*.log 2>/dev/null | wc -l)"
    transfer_count="$(grep -l 'Initialize LlmDataList success' "$LOG_ROOT"/rank_*.log 2>/dev/null | wc -l)"
    echo "WAITING_SECONDS=$elapsed WEIGHT_READY_RANKS=$weight_count TRANSFER_READY_RANKS=$transfer_count"
    if [[ "$weight_count" -eq "$NNODES" && "$transfer_count" -eq "$NNODES" ]] &&
       grep -q 'Application startup complete' "$LOG_ROOT/rank_0.log"; then
      echo "SERVICE_READY_SECONDS=$elapsed"
      grep 'kv cache capacity:' "$LOG_ROOT/rank_0.log" | tail -1
      grep 'Application startup complete' "$LOG_ROOT/rank_0.log" | tail -1
      return 0
    fi
    sleep 10
  done
  echo "STARTUP_TIMEOUT=600"
  tail -160 "$LOG_ROOT/rank_0.log" || true
  return 1
}

status() {
  collect_owned_pids
  collect_owned_rank_pids
  echo "RUN_ROOT=$RUN_ROOT"
  echo "ALIVE_RANKS=${#OWNED_RANK_PIDS[@]}/$NNODES"
  echo "ALIVE_ASSOCIATED_PROCESSES=${#OWNED_PIDS[@]}"
  [[ -f "$RUN_ROOT/config.txt" ]] && cat "$RUN_ROOT/config.txt"
  if [[ -d "$LOG_ROOT" ]]; then
    echo "WEIGHT_READY_RANKS=$(grep -l 'Weight loading completed' "$LOG_ROOT"/rank_*.log 2>/dev/null | wc -l)"
    echo "TRANSFER_READY_RANKS=$(grep -l 'Initialize LlmDataList success' "$LOG_ROOT"/rank_*.log 2>/dev/null | wc -l)"
    grep 'kv cache capacity:' "$LOG_ROOT/rank_0.log" 2>/dev/null | tail -1 || true
    grep 'Application startup complete' "$LOG_ROOT/rank_0.log" 2>/dev/null | tail -1 || true
  fi
}

stop() {
  collect_owned_pids
  if ((${#OWNED_PIDS[@]} == 0)); then
    echo "NO_OWNED_PROCESS"
    return 0
  fi
  local pid
  for pid in "${OWNED_PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 60); do
    collect_owned_pids
    ((${#OWNED_PIDS[@]} == 0)) && { echo "STOPPED=1"; return 0; }
    collect_owned_rank_pids
    ((${#OWNED_RANK_PIDS[@]} == 0)) && break
    sleep 1
  done
  collect_owned_pids
  for pid in "${OWNED_PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    collect_owned_pids
    ((${#OWNED_PIDS[@]} == 0)) && { echo "STOPPED=FORCED"; return 0; }
    sleep 1
  done
  echo "STOP_FAILED_REMAINING=${OWNED_PIDS[*]}"
  return 1
}

case "$ACTION" in
  preflight) preflight ;;
  start) start ;;
  wait) wait_ready ;;
  status) status ;;
  stop) stop ;;
  *) usage; exit 2 ;;
esac
