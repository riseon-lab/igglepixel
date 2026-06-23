# Citivia Studio — single-container RunPod template image.
# UI and model runners stay separate processes, but ship as one image.

# ---- UI deps ----
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- UI build ----
FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- runtime ----
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime

# Node/npm are needed for the baked Next standalone server and optional startup pull.
COPY --from=deps /usr/local/bin/node /usr/local/bin/node
COPY --from=deps /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    CITIVIA_DATA_DIR=/workspace \
    QWEN_2512_RUNNER_URL=http://127.0.0.1:8011 \
    QWEN_EDIT_2511_RUNNER_URL=http://127.0.0.1:8012 \
    HF_HOME=/workspace/.cache/huggingface \
    RUNNER_CPU_OFFLOAD=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY runners/common/requirements.txt /runner/requirements.txt
RUN pip install --no-cache-dir -r /runner/requirements.txt
COPY runners/common/runner.py /runner/runner.py

COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
COPY --from=build /app/scripts/start.sh ./scripts/start.sh
COPY --from=build /app/scripts/runpod-start.sh ./scripts/runpod-start.sh
RUN chmod +x ./scripts/start.sh ./scripts/runpod-start.sh

VOLUME ["/workspace"]
EXPOSE 3000
CMD ["./scripts/runpod-start.sh"]
