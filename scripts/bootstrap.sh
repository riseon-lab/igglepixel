#!/bin/sh
# Minimal, stable launcher — the ONLY code baked into the image.
#
# Its whole job: make sure the repo is present on the persistent volume, pull the
# latest, and hand off to the repo's own start script. Everything that can change
# (UI, runners, start scripts) lives in git, so there is nothing baked to go stale.
set -eu

REPO_URL="${CITIVIA_REPO_URL:-https://github.com/riseon-lab/igglepixel.git}"
REPO_REF="${CITIVIA_REPO_REF:-main}"
REPO_DIR="${CITIVIA_REPO_DIR:-/workspace/igglepixel}"
RETRY_SLEEP="${CITIVIA_GIT_RETRY_SLEEP:-5}"

mkdir -p "$(dirname "$REPO_DIR")"

# Clone if we don't have it yet. Keep retrying — never give up silently, because
# there is no baked app to fall back to (that's the whole point).
while [ ! -d "$REPO_DIR/.git" ]; do
  echo "[bootstrap] cloning $REPO_URL ($REPO_REF) -> $REPO_DIR"
  if git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"; then
    break
  fi
  echo "[bootstrap] clone failed; retrying in ${RETRY_SLEEP}s" >&2
  sleep "$RETRY_SLEEP"
done

cd "$REPO_DIR"
git remote set-url origin "$REPO_URL" 2>/dev/null || true

# Pull latest. If the network is down, fall back to the code already on the
# volume (the last good pull) — NOT to anything baked, because nothing is.
if git fetch origin "$REPO_REF" && git reset --hard FETCH_HEAD; then
  echo "[bootstrap] synced to $(git rev-parse --short HEAD)"
else
  echo "[bootstrap] WARNING: git pull failed; running last-synced code $(git rev-parse --short HEAD)" >&2
fi

exec sh "$REPO_DIR/scripts/runpod-start.sh"
