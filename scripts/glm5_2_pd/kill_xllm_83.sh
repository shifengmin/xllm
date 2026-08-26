#!/bin/bash
# 清理 83 残留 xllm 进程
# 用法: bash kill_xllm_83.sh

echo "[INFO] killing all xllm processes..."
pkill -9 -f /site-packages/xllm/xllm 2>/dev/null || true
sleep 3
remaining=$(ps -eo pid,args | grep /site-packages/xllm/xllm | grep -v grep | wc -l)
echo "[INFO] remaining xllm procs: $remaining"
if [ "$remaining" -ne 0 ]; then
  echo "[WARN] still $remaining procs, listing:"
  ps -eo pid,etime,args | grep /site-packages/xllm/xllm | grep -v grep
fi
echo "[INFO] done"
