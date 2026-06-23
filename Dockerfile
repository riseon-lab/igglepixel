# Citivia Studio — RunPod-targeted image.
# Multi-stage build producing a slim Next.js standalone server on port 3000.

# ---- deps ----
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- build ----
FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- runtime ----
FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Persist all encrypted assets/account data on the mounted volume.
ENV CITIVIA_DATA_DIR=/workspace

# git so startup can pull latest without a full pod rebuild (plan.md).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Standalone output bundles only the files needed to run.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
COPY --from=build /app/scripts/start.sh ./scripts/start.sh
RUN chmod +x ./scripts/start.sh

VOLUME ["/workspace"]
EXPOSE 3000

CMD ["./scripts/start.sh"]
