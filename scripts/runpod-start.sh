#!/bin/sh
set -eu

export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"
export QWEN_2512_RUNNER_URL="${QWEN_2512_RUNNER_URL:-http://127.0.0.1:8011}"
export QWEN_EDIT_2511_RUNNER_URL="${QWEN_EDIT_2511_RUNNER_URL:-http://127.0.0.1:8012}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# RUNNER_CPU_OFFLOAD is intentionally left unset: the runner auto-decides based on
# available VRAM (full GPU on 96GB/48GB cards, offload only on small ones). Set it
# to 0 or 1 to override.

mkdir -p "$CITIVIA_DATA_DIR/models" "$CITIVIA_DATA_DIR/loras"

# Prefer the runner from the pulled repo (kept current by start.sh's git pull)
# over the baked copy, so runner updates ship without a full image rebuild.
# Falls back to the baked runner on the very first boot.
RUNNER_PY="/runner/runner.py"
PULLED_RUNNER="${CITIVIA_REPO_DIR:-/workspace/igglepixel}/runners/common/runner.py"
if [ -f "$PULLED_RUNNER" ]; then
  RUNNER_PY="$PULLED_RUNNER"
fi

RUNNER_MODE=txt2img \
MODEL_ID="${QWEN_2512_MODEL_ID:-Qwen/Qwen-Image-2512}" \
MODEL_CACHE_DIR="$CITIVIA_DATA_DIR/models/qwen-2512" \
OUTPUT_DIR="$CITIVIA_DATA_DIR/outputs/qwen-2512" \
LORA_DIR="$CITIVIA_DATA_DIR/loras" \
PORT=8011 \
python "$RUNNER_PY" &
QWEN_PID=$!

RUNNER_MODE=edit \
MODEL_ID="${QWEN_EDIT_2511_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}" \
MODEL_CACHE_DIR="$CITIVIA_DATA_DIR/models/qwen-edit-2511" \
OUTPUT_DIR="$CITIVIA_DATA_DIR/outputs/qwen-edit-2511" \
LORA_DIR="$CITIVIA_DATA_DIR/loras" \
PORT=8012 \
python "$RUNNER_PY" &
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
