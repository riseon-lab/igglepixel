# ── Forge RunPod Launcher ──────────────────────────────────────────────────
# Runtime-only image: Python + torch + diffusers + system tools.
# The backend, UI, and runners are pulled from the public Git repo at boot
# by /entrypoint.sh, so model code can be iterated without a rebuild.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip && \
    git lfs install

# ── Python base deps (API + runner host) ──
RUN pip install --no-cache-dir \
        fastapi 'uvicorn[standard]' \
        httpx \
        'huggingface_hub[cli]' \
        python-multipart \
        pydantic \
        cryptography==44.0.0

# ── ML stack (used by backend/runners/*) ──
# Torch is pinned to the CUDA-12.1 wheels matching the base image.
# New runners can pull extra packages at boot via requirements-runtime.txt
# in the repo root — no rebuild needed.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1
RUN pip install --no-cache-dir \
        diffusers \
        transformers \
        accelerate==1.2.1 \
        safetensors==0.4.5 \
        sentencepiece==0.2.0 \
        Pillow==11.0.0

# ── Entrypoint ──
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000
ENTRYPOINT ["/entrypoint.sh"]
