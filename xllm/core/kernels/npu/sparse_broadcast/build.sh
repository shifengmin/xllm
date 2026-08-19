#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/build"

cmake -S "${ROOT}" -B "${BUILD}" ${SHMEM_ROOT:+-DSHMEM_ROOT="${SHMEM_ROOT}"} "$@"
cmake --build "${BUILD}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
"${BUILD}/sparse_broadcast_layout_test"
