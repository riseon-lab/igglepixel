#!/bin/sh
set -eu

export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"
export PORT="${PORT:-3000}"
export QWEN_2512_RUNNER_URL="${QWEN_2512_RUNNER_URL:-http://127.0.0.1:8011}"
export QWEN_EDIT_2511_RUNNER_URL="${QWEN_EDIT_2511_RUNNER_URL:-http://127.0.0.1:8012}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# RUNNER_CPU_OFFLOAD is intentionally left unset: the runner auto-decides based on
# available VRAM (full GPU on 96GB/48GB cards, offload only on small ones). Set it
# to 0 or 1 to override.

mkdir -p "$CITIVIA_DATA_DIR/models" "$CITIVIA_DATA_DIR/loras"

stop_port_listener() {
  python - "$1" <<'PY'
import os
import signal
import sys
import time

port = int(sys.argv[1])
inodes = set()

for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        rows = open(proc_file, encoding="utf-8").read().splitlines()[1:]
    except OSError:
        continue

    for row in rows:
        parts = row.split()
        if len(parts) < 10 or parts[3] != "0A":
            continue

        if int(parts[1].rsplit(":", 1)[1], 16) == port:
            inodes.add(parts[9])

targets = set()
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue

    pid = int(name)
    fd_dir = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        continue

    for fd in fds:
        try:
            link = os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue

        if link.startswith("socket:[") and link[8:-1] in inodes:
            targets.add(pid)
            break

for sig in (signal.SIGTERM, signal.SIGKILL):
    live = []
    for pid in sorted(targets):
        try:
            os.kill(pid, 0)
            live.append(pid)
        except OSError:
            pass

    if not live:
        break

    for pid in live:
        try:
            os.kill(pid, sig)
        except OSError:
            pass

    if sig == signal.SIGTERM:
        time.sleep(2)

for pid in sorted(targets):
    print(f"[runpod-start] stopped stale process pid={pid} on port {port}", flush=True)
PY
}

stop_port_listener 8011
stop_port_listener 8012
stop_port_listener "$PORT"

# The runner comes from the pulled repo — nothing is baked. bootstrap.sh has
# already cloned/synced it before we get here.
RUNNER_PY="${CITIVIA_REPO_DIR:-/workspace/igglepixel}/runners/common/runner.py"
if [ ! -f "$RUNNER_PY" ]; then
  echo "[runpod-start] FATAL: runner not found at $RUNNER_PY (repo not synced?)" >&2
  exit 1
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
UI_PID=""

cleanup() {
  if [ -n "${UI_PID:-}" ]; then
    kill "$UI_PID" 2>/dev/null || true
    wait "$UI_PID" 2>/dev/null || true
  fi

  kill "$QWEN_PID" "$EDIT_PID" 2>/dev/null || true
  wait "$QWEN_PID" "$EDIT_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

sh "${CITIVIA_REPO_DIR:-/workspace/igglepixel}/scripts/start.sh" &
UI_PID=$!
STATUS=0
wait "$UI_PID" || STATUS=$?
cleanup
exit "$STATUS"
