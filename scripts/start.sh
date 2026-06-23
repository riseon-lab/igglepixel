#!/bin/sh
set -eu

APP_DIR="/app"
REPO_URL="${CITIVIA_REPO_URL:-https://github.com/riseon-lab/igglepixel.git}"
REPO_REF="${CITIVIA_REPO_REF:-main}"
REPO_DIR="${CITIVIA_REPO_DIR:-/workspace/igglepixel}"
AUTO_PULL="${CITIVIA_AUTO_PULL:-1}"
GIT_RETRIES="${CITIVIA_GIT_RETRIES:-20}"
GIT_RETRY_SLEEP="${CITIVIA_GIT_RETRY_SLEEP:-5}"

export PORT="${PORT:-3000}"
export HOSTNAME="${HOSTNAME:-0.0.0.0}"
export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"

start_baked() {
  cd "$APP_DIR"
  exec node server.js
}

git_sync() {
  attempt=1
  while [ "$attempt" -le "$GIT_RETRIES" ]; do
    if [ -d "$REPO_DIR/.git" ]; then
      cd "$REPO_DIR"
      git remote set-url origin "$REPO_URL" 2>/dev/null || true
      git fetch origin "$REPO_REF" && git reset --hard FETCH_HEAD && return 0
    else
      git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$REPO_DIR" && return 0
    fi

    echo "Git sync failed ($attempt/$GIT_RETRIES); retrying in ${GIT_RETRY_SLEEP}s." >&2
    attempt=$((attempt + 1))
    sleep "$GIT_RETRY_SLEEP"
  done
  return 1
}

if [ "$AUTO_PULL" = "1" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"

  if ! git_sync; then
    echo "Git sync failed after retries; starting baked image instead." >&2
    start_baked
  fi

  cd "$REPO_DIR"
  if [ -f package-lock.json ]; then
    # Build the pulled repo; on any failure fall back to the baked image instead
    # of crashing the container. Extra heap headroom for the production build.
    if NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}" \
        sh -c 'npm ci --include=dev && npm run build'; then
      exec node .next/standalone/server.js
    fi
    echo "Build of pulled repo failed; starting baked image instead." >&2
  fi
fi

start_baked
