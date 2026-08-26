#!/bin/bash
# GDB attach 到 rank0，捕获 SIGSEGV 并打印 backtrace
# 用法: bash gdb_attach_rank0.sh <pid>
PID=$1
XLLM=/usr/local/python3.11.15/lib/python3.11/site-packages/xllm/xllm
OUT=/export/home/shifengmin.3/workspace/glm5_2_pd/logs/gdb_rank0_$(date +%H%M%S).log

echo "[INFO] attaching gdb to pid $PID, output to $OUT"
gdb -batch \
  -ex "set pagination off" \
  -ex "set print thread-events off" \
  -ex "handle SIGSEGV stop print nopass" \
  -ex "attach $PID" \
  -ex "continue" \
  -ex "bt 50" \
  -ex "info threads" \
  -ex "thread apply all bt 20" \
  -ex "info sharedlibrary" \
  $XLLM > $OUT 2>&1

echo "[INFO] gdb finished, output in $OUT"
echo "[INFO] === bt summary ==="
grep -A 50 "^#" $OUT | head -60
