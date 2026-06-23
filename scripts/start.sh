#!/bin/sh
set -eu

APP_DIR="/app"
REPO_URL="${CITIVIA_REPO_URL:-https://github.com/riseon-lab/igglepixel.git}"
REPO_REF="${CITIVIA_REPO_REF:-main}"
REPO_DIR="${CITIVIA_REPO_DIR:-/workspace/igglepixel}"
AUTO_PULL="${CITIVIA_AUTO_PULL:-1}"

export PORT="${PORT:-3000}"
export HOSTNAME="${HOSTNAME:-0.0.0.0}"
export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"

start_baked() {
  cd "$APP_DIR"
  exec node server.js
}

if [ "$AUTO_PULL" = "1" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"

  if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git remote set-url origin "$REPO_URL" 2>/dev/null || true
    if ! git pull --ff-only origin "$REPO_REF"; then
      echo "Git pull failed; starting baked image instead." >&2
      start_baked
    fi
  elif ! git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" "$REPO_DIR"; then
    echo "Git clone failed; starting baked image instead." >&2
    start_baked
  fi

  cd "$REPO_DIR"
  if [ -f package-lock.json ]; then
    npm ci --include=dev
    npm run build
    exec node .next/standalone/server.js
  fi
fi

start_baked
