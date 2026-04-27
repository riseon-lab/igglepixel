#!/usr/bin/env bash
# ── Forge entrypoint ────────────────────────────────────────────────────
# The Docker image only ships the runtime (Python + torch + diffusers).
# The backend/UI/runners live in a public Git repo and are pulled at boot,
# so model code can be iterated without rebuilding the image.
#
# Env vars (all optional):
#   FORGE_REPO       Git URL to clone. Default: riseon-lab/pleoai
#   FORGE_BRANCH     Branch to track.  Default: main
#   FORGE_COMMIT     Pin to a specific commit/tag (overrides BRANCH HEAD).
#   FORGE_CACHE_DIR  Where to cache the clone. Default: /workspace/forge-src
#                    (lives on the persistent volume so a stale-but-working
#                    copy survives a GitHub outage).
# ───────────────────────────────────────────────────────────────────────

set -uo pipefail

REPO="${FORGE_REPO:-https://github.com/riseon-lab/igglepixel.git}"
BRANCH="${FORGE_BRANCH:-main}"
COMMIT="${FORGE_COMMIT:-}"
CACHE_DIR="${FORGE_CACHE_DIR:-/workspace/forge-src}"

log() { echo "[forge] $*"; }

clone_fresh() {
    rm -rf "$CACHE_DIR"
    log "cloning $REPO ($BRANCH) → $CACHE_DIR"
    # depth 50 keeps it small but leaves room to check out a recent tag/commit.
    git clone --depth 50 --branch "$BRANCH" "$REPO" "$CACHE_DIR"
}

update_existing() {
    cd "$CACHE_DIR"
    log "fetching $REPO ($BRANCH)"
    git remote set-url origin "$REPO" 2>/dev/null || true
    git fetch --depth 50 origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
    git clean -fd
}

# ── Acquire source ─────────────────────────────────────────────────────
mkdir -p "$(dirname "$CACHE_DIR")"

if [ -d "$CACHE_DIR/.git" ]; then
    # Re-clone if the configured remote URL has changed since last boot.
    cd "$CACHE_DIR"
    CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || echo '')"
    cd /
    if [ "$CURRENT_REMOTE" != "$REPO" ]; then
        log "remote URL changed ($CURRENT_REMOTE → $REPO); re-cloning"
        clone_fresh || { log "FATAL: clone failed"; exit 1; }
    else
        if ! update_existing; then
            log "WARN: fetch failed — booting from stale cache (HEAD: $(cd "$CACHE_DIR" && git rev-parse --short HEAD))"
        fi
    fi
else
    clone_fresh || { log "FATAL: no cache and clone failed"; exit 1; }
fi

# Pin to a specific commit/tag if requested.
if [ -n "$COMMIT" ]; then
    cd "$CACHE_DIR"
    log "pinning to $COMMIT"
    git fetch --depth 50 origin "$COMMIT" 2>/dev/null || true
    git checkout "$COMMIT" || { log "FATAL: checkout $COMMIT failed"; exit 1; }
fi

cd "$CACHE_DIR"
log "HEAD: $(git log -1 --pretty='%h %s')"

# ── Per-deployment dep installs ────────────────────────────────────────
# A new runner can declare extra Python packages by adding them to
# requirements-runtime.txt at the repo root. Installed on every boot
# (cheap if already present) so we don't have to rebuild the image
# for new model deps.
if [ -f "requirements-runtime.txt" ]; then
    log "installing requirements-runtime.txt"
    pip install --no-cache-dir -r requirements-runtime.txt || \
        log "WARN: requirements-runtime.txt install failed; continuing"
fi

# ── Redirect HuggingFace cache to the persistent volume ───────────────
# By default HF writes to ~/.cache/huggingface which is on the ephemeral
# container disk (typically 10–20 GB). Models are tens of GB, so they must
# land on /workspace (the persistent network volume, 90+ GB).
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub   # older transformers compat
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# ── Workspace skeleton (the volume may be empty on first boot) ─────────
mkdir -p /workspace/models /workspace/loras /workspace/checkpoints \
         /workspace/assets/uploads /workspace/assets/generated /workspace/logs \
         /workspace/.cache/huggingface

# ── Run ────────────────────────────────────────────────────────────────
log "starting Forge (port ${UI_PORT:-3000})"
exec python backend/main.py
