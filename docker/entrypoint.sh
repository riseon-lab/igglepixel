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

# ── Persistent caches on the workspace volume ─────────────────────────
# Both HF and pip caches live on /workspace so subsequent boots can
# reuse them. The pip cache in particular is the foundation for the
# smart-deps strategy: per-model `pip_requirements` (when we add them)
# and any future runtime adds become near-instant on warm pods.
#
# HF_HOME redirect is here too because the default ~/.cache/huggingface
# sits on the 10–20 GB ephemeral container disk, and a single model is
# tens of GB. Pin everything model-shaped to /workspace.
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export PIP_CACHE_DIR=/workspace/.cache/pip

# ── Workspace skeleton (the volume may be empty on first boot) ─────────
mkdir -p /workspace/models /workspace/loras /workspace/checkpoints \
         /workspace/assets/uploads /workspace/assets/generated /workspace/logs \
         /workspace/.cache/huggingface /workspace/.cache/pip \
         /workspace/venvs /workspace/repos

# ── uv (per-runner venv tooling) ───────────────────────────────────────
# uv is a single static binary that handles Python version installs
# (e.g. 3.12 alongside our system 3.11) plus venv creation plus fast pip.
# We persist it under /workspace/.local/bin so first-boot downloads it once
# and every subsequent boot uses the cached binary. backend/venv_manager.py
# uses uv when present, falling back to `python -m venv` / virtualenv otherwise.
export UV_INSTALL_DIR=/workspace/.local/bin
export PATH="$UV_INSTALL_DIR:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv (one-time, persistent under $UV_INSTALL_DIR)"
    mkdir -p "$UV_INSTALL_DIR"
    curl -fsSL https://astral.sh/uv/install.sh | env INSTALL_DIR="$UV_INSTALL_DIR" sh \
        || log "WARN: uv install failed; per-runner venvs will fall back to python -m venv"
fi

# ── Per-deployment dep installs ────────────────────────────────────────
# requirements-runtime.txt at the repo root lists feature-flag deps
# (peft, ftfy, opencv-python-headless, …) the image doesn't carry.
# We install with the persistent pip cache enabled — first cold boot
# downloads, every later boot installs in seconds from the cache, even
# if the image is rebuilt or the pod is recreated on the same volume.
if [ -f "requirements-runtime.txt" ]; then
    log "installing requirements-runtime.txt (pip cache: $PIP_CACHE_DIR)"
    pip install -r requirements-runtime.txt || \
        log "WARN: requirements-runtime.txt install failed; continuing"
fi

# ── Run ────────────────────────────────────────────────────────────────
log "starting Forge (port ${UI_PORT:-3000})"
exec python backend/main.py
