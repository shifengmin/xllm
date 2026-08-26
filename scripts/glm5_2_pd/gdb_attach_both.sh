#!/bin/bash
# GDB attach 到 rank0 的主进程和 worker 子进程，捕获 SIGSEGV
# 用法: bash gdb_attach_both.sh <main_pid> <worker_pid>
MAIN_PID=$1
WORKER_PID=$2
XLLM=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
TS=$(date +%H%M%S)
MAIN_OUT=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/gdb_main_${TS}.log
WORKER_OUT=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/gdb_worker_${TS}.log

echo "[INFO] attaching gdb to main=$MAIN_PID -> $MAIN_OUT"
echo "[INFO] attaching gdb to worker=$WORKER_PID -> $WORKER_OUT"

# 后台 attach worker
gdb -batch \
  -ex "set pagination off" \
  -ex "set print thread-events off" \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "attach $WORKER_PID" \
  -ex "continue" \
  -ex "bt 80" \
  -ex "info threads" \
  -ex "thread apply all bt 30" \
  $XLLM > $WORKER_OUT 2>&1 &
WORKER_GDB=$!

# 前台 attach main
gdb -batch \
  -ex "set pagination off" \
  -ex "set print thread-events off" \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "attach $MAIN_PID" \
  -ex "continue" \
  -ex "bt 80" \
  -ex "info threads" \
  -ex "thread apply all bt 30" \
  $XLLM > $MAIN_OUT 2>&1

wait $WORKER_GDB

echo "[INFO] gdb finished"
echo "[INFO] main output: $MAIN_OUT"
echo "[INFO] worker output: $WORKER_OUT"
