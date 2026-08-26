#!/bin/bash
# 83 smoke 测试脚本
# 用法: bash test_smoke.sh [port]
#   port: 默认 19994

PORT=${1:-19994}
HOST=11.87.191.83

echo "[INFO] probing port $PORT..."
if ! timeout 2 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
  echo "[ERROR] port $PORT not open, instance not ready"
  exit 1
fi
echo "[INFO] port $PORT open, sending smoke request..."

curl -sS -m 180 -w '\nhttp_code=%{http_code} time=%{time_total}\n' \
  http://$HOST:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.2-W8A8-EcoTech",
    "max_tokens": 16,
    "temperature": 0,
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}]
  }'

echo ""
echo "[INFO] smoke test done. exit code=$?"
