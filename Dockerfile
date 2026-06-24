# Citivia Studio — thin RunPod runtime image.
#
# This image bakes ONLY the runtime: CUDA + PyTorch, a Node binary, git, and the
# heavy Python deps. It bakes NO application code. At boot it pulls the repo into
# the persistent /workspace volume and runs everything from there, so a redeploy
# is just a pod restart — nothing app-level is ever baked, so nothing goes stale.

# Source a Node binary to drop into the CUDA runtime.
FROM node:22-slim AS node

# ---- runtime ----
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS runtime

# Node + npm (to build/serve the pulled Next app) and git (to pull it).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Bake the heavy Python deps so a cold boot is a git pull, not a torch install.
# Also record a hash of what we baked: at boot, runpod-start.sh compares it to the
# pulled requirements and only re-installs if a dep was added/changed since this
# image was built — so "add a dep, push, restart" works without an image rebuild.
COPY runners/common/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && md5sum /tmp/requirements.txt | cut -d' ' -f1 > /opt/deps-baked.md5 \
    && rm /tmp/requirements.txt

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    CITIVIA_DATA_DIR=/workspace \
    CITIVIA_REPO_URL=https://github.com/riseon-lab/igglepixel.git \
    CITIVIA_REPO_REF=main \
    CITIVIA_REPO_DIR=/workspace/igglepixel \
    QWEN_2512_RUNNER_URL=http://127.0.0.1:8011 \
    QWEN_EDIT_2511_RUNNER_URL=http://127.0.0.1:8012 \
    QWEN_2512_MODEL_ID=Qwen/Qwen-Image-2512 \
    QWEN_EDIT_2511_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

# The ONLY baked code: a tiny, stable launcher that pulls the repo and hands off
# to the repo's own scripts. Everything that can change lives in git.
COPY scripts/bootstrap.sh /bootstrap.sh
RUN chmod +x /bootstrap.sh

VOLUME ["/workspace"]
EXPOSE 3000
CMD ["/bootstrap.sh"]
