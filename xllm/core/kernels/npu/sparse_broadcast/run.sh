#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/build"
N_PES="${1:-2}"
IPPORT="${2:-tcp://127.0.0.1:8766}"
N_NPUS="${3:-${N_PES}}"
FIRST_NPU="${4:-0}"

if [[ ! -x "${BUILD}/sparse_broadcast_test" ]]; then
  echo "missing ${BUILD}/sparse_broadcast_test; run cmake --build ${BUILD}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${BUILD}:${SHMEM_ROOT:-}/build/lib:${SHMEM_ROOT:-}/lib:${SHMEM_ROOT:-}/install/shmem/lib:${ASCEND_HOME_PATH:-}/lib64:${LD_LIBRARY_PATH:-}"

pids=()
status=0
for ((pe = 0; pe < N_PES; pe++)); do
  "${BUILD}/sparse_broadcast_test" "${N_PES}" "${pe}" "${IPPORT}" "${N_NPUS}" "${FIRST_NPU}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
