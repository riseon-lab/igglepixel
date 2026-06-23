#!/bin/sh
set -eu

export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"
export QWEN_2512_RUNNER_URL="${QWEN_2512_RUNNER_URL:-http://127.0.0.1:8011}"
export QWEN_EDIT_2511_RUNNER_URL="${QWEN_EDIT_2511_RUNNER_URL:-http://127.0.0.1:8012}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export RUNNER_CPU_OFFLOAD="${RUNNER_CPU_OFFLOAD:-1}"

mkdir -p "$CITIVIA_DATA_DIR/models" "$CITIVIA_DATA_DIR/loras"

RUNNER_MODE=txt2img \
MODEL_ID="${QWEN_2512_MODEL_ID:-Qwen/Qwen-Image}" \
MODEL_CACHE_DIR="$CITIVIA_DATA_DIR/models/qwen-2512" \
OUTPUT_DIR="$CITIVIA_DATA_DIR/outputs/qwen-2512" \
LORA_DIR="$CITIVIA_DATA_DIR/loras" \
PORT=8011 \
python /runner/runner.py &
QWEN_PID=$!

RUNNER_MODE=edit \
MODEL_ID="${QWEN_EDIT_2511_MODEL_ID:-Qwen/Qwen-Image-Edit}" \
MODEL_CACHE_DIR="$CITIVIA_DATA_DIR/models/qwen-edit-2511" \
OUTPUT_DIR="$CITIVIA_DATA_DIR/outputs/qwen-edit-2511" \
LORA_DIR="$CITIVIA_DATA_DIR/loras" \
PORT=8012 \
python /runner/runner.py &
EDIT_PID=$!

cleanup() {
  kill "$QWEN_PID" "$EDIT_PID" 2>/dev/null || true
  wait "$QWEN_PID" "$EDIT_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

./scripts/start.sh &
UI_PID=$!
wait "$UI_PID"
STATUS=$?
cleanup
exit "$STATUS"
