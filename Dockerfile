# ── Forge RunPod Launcher ──────────────────────────────────────────────────
# Runtime-only image: Python + torch + diffusers + system tools.
# The backend, UI, and runners are pulled from the public Git repo at boot
# by /entrypoint.sh, so model code can be iterated without a rebuild.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Forge Launcher"
LABEL description="Universal open-source model launcher for RunPod"

# ── Base system ──
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace
ENV UI_PORT=3000

# Public-repo defaults; override at pod-create time if you fork.
ENV FORGE_REPO=https://github.com/riseon-lab/igglepixel.git
ENV FORGE_BRANCH=main
ENV FORGE_CACHE_DIR=/workspace/forge-src

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev \
        git curl wget git-lfs jq unzip p7zip-full \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg libavcodec-dev libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 - "pip==26.1.2"

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip && \
    git lfs install

# ── Python base deps (API + runner host) ──
# Everything below is pinned on purpose: this image is rebuilt rarely, so an
# unpinned rebuild silently picks up whatever PyPI serves that day. Bump
# versions deliberately, in lockstep with requirements-runtime.txt.
RUN pip install --no-cache-dir \
        fastapi==0.136.3 'uvicorn[standard]==0.49.0' \
        httpx==0.28.1 \
        'huggingface_hub[cli]==0.36.2' \
        python-multipart==0.0.32 \
        pydantic==2.13.4 \
        cryptography==44.0.0

# ── ML stack (used by backend/runners/*) ──
# Torch is built against CUDA 12.8 (cu128 wheels) so the binary fatbin
# includes sm_120 kernels for Blackwell (RTX PRO 6000, B100, B200) while
# still covering older arches (sm_70 → sm_90 for Ampere/Hopper).
# Older cu121 wheels did NOT have sm_120 and refused to launch on Blackwell.
# New runners can pull extra packages at boot via requirements-runtime.txt
# in the repo root — no rebuild needed.
# Pinned to the exact versions requirements-runtime.txt enforces at boot, so
# the boot-time install is a no-op instead of a multi-GB downgrade that could
# also swap the cu128 build for PyPI's default CUDA variant (losing sm_120).
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
RUN pip install --no-cache-dir \
        diffusers==0.38.0 \
        transformers==4.57.6 \
        accelerate==1.14.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.1 \
        Pillow==12.2.0 \
        bitsandbytes==0.49.2 \
        spandrel==0.4.2

# ── Entrypoint ──
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000
ENTRYPOINT ["/entrypoint.sh"]
