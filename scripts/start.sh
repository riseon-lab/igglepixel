#!/bin/sh
# Build (if needed) and serve the Next UI from the pulled repo.
#
# bootstrap.sh has already cloned/synced the repo and we run from inside it.
# There is no baked image to fall back to: if the build fails we exit loudly so
# the failure is visible, instead of silently serving stale code.
set -eu

REPO_DIR="${CITIVIA_REPO_DIR:-/workspace/igglepixel}"
export PORT="${PORT:-3000}"
export HOSTNAME="${HOSTNAME:-0.0.0.0}"
export CITIVIA_DATA_DIR="${CITIVIA_DATA_DIR:-/workspace}"

cd "$REPO_DIR"

SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
BUILT_MARKER=".next/.built-sha"

# Skip the (slow) rebuild when the checked-out commit already has a matching
# standalone build on the persistent volume. Keeps restarts fast.
if [ -f "$BUILT_MARKER" ] && [ "$(cat "$BUILT_MARKER" 2>/dev/null)" = "$SHA" ] \
    && [ -f .next/standalone/server.js ]; then
  echo "[start] UI already built for $SHA; skipping build"
else
  echo "[start] building UI for $SHA"
  NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}" npm ci --include=dev
  NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}" npm run build
  echo "$SHA" > "$BUILT_MARKER"
  echo "[start] build complete for $SHA"
fi

exec node .next/standalone/server.js
