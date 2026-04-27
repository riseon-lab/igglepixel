# Forge — RunPod Launcher

One Docker container. Open-source models. Clean monochrome UI.

## What it is

A self-hosted FastAPI + static UI that runs on a RunPod GPU pod and:
- Loads supported models via per-model Python runners (no third-party UI exposed)
- Configures generation params from the drawer (prompt, seed, steps, cfg, dimensions, …)
- Surfaces uploads and generated outputs in an Asset manager
- Pulls its own backend/UI/runner code from this public repo at every pod boot,
  so iteration on models is `git push` + restart, not `docker build`

## Deploy model

The Docker image is a **runtime shell** (Python + torch + diffusers + system tools).
The actual app code lives in this repo and is cloned into `/workspace/forge-src` on
every container start by [`docker/entrypoint.sh`](docker/entrypoint.sh).

```
┌─────────────────────────────┐
│  Docker image (heavy, slow) │  torch, diffusers, transformers, …
│  rebuild rarely             │
└──────────────┬──────────────┘
               │  on container start, entrypoint.sh:
               │   1. git clone / fetch+reset to /workspace/forge-src
               │   2. pip install -r requirements-runtime.txt (if present)
               │   3. exec python backend/main.py
               ▼
┌─────────────────────────────┐
│  Public repo (fast iter)    │  backend/, ui/, runners/, registry
│  push to main → pod restart │
└─────────────────────────────┘
```

## Quick deploy on RunPod

1. **New Pod** → Custom Docker Image
2. Image: `your-dockerhub/forge:latest`
3. Expose port `3000`
4. Optional env:
   - `HF_TOKEN` — gated HuggingFace repos
   - `FORGE_REPO`, `FORGE_BRANCH`, `FORGE_COMMIT` — see below
5. Start → open port 3000

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `FORGE_REPO`      | `https://github.com/riseon-lab/igglepixel.git` | Git URL pulled on boot |
| `FORGE_BRANCH`    | `main` | Branch to track |
| `FORGE_COMMIT`    | _(unset)_ | Pin to a specific commit/tag — overrides BRANCH HEAD |
| `FORGE_CACHE_DIR` | `/workspace/forge-src` | Where the clone is cached (persistent volume) |
| `HF_TOKEN`        | _(unset)_ | For gated HuggingFace repos |
| `WORKSPACE`       | `/workspace` | Root for models, assets, logs |
| `UI_PORT`         | `3000` | FastAPI port |
| `IGGLEPIXEL_MODERATION` | `true` | Run NSFW classifier on every generated image. Set to `false` for private dev pods or fork operators accepting their own responsibility. |
| `HF_HOME`         | `/workspace/.cache/huggingface` | HF cache location (set by entrypoint so weights land on the persistent volume) |

If `git fetch` fails on boot (GitHub outage), the pod boots from the cached clone
in `FORGE_CACHE_DIR` instead of failing.

## Architecture

```
User browser
    │
    ▼
FastAPI (port 3000)                                 (public; only thing exposed)
    │  GET  /api/models       registry + GPU info
    │  POST /api/launch       spawn runner subprocess
    │  POST /api/generate     proxies to runner
    │  POST /api/cancel/:id   asks runner to stop at next step
    │  POST /api/stop/:id     kill runner
    │  GET  /api/status       what's running
    │  GET  /api/assets       generated + uploaded media
    │  POST /api/assets/upload
    │  …
    │
    │  python -m backend.runner_host backend.runners.<model> <port>
    ▼
Runner subprocess  (127.0.0.1:17000+)               (private; no proxy access)
    GET  /healthz   ready / loading / load_error
    POST /generate  inference
    POST /cancel    interrupt at next step boundary
```

## Content moderation

Generated images pass through `Falconsai/nsfw_image_detection` (88M-param ViT,
loaded once on runner spawn, kept resident on GPU) before they're saved.
Flagged outputs are dropped — nothing persists, nothing reaches the user beyond
a neutral toast.

- **Default: on.** This is the safe default for public deployments.
- **Opt out:** set `IGGLEPIXEL_MODERATION=false` in the pod env vars.
- **Fail-closed:** if the moderation model can't load, all outputs are blocked
  until you fix it or disable moderation.
- **Threshold:** `nsfw > 0.85` (tunable in `backend/moderator.py`).
- **Scope:** image-only, post-generation. Does not check prompts. Does not
  catch violence, hate symbols, or other categories beyond NSFW.

If you fork this repo and run your own deployment, you take responsibility for
what your instance generates. The default-on toggle exists so the maintained
public image (`riseonlab/igglepixel`) ships moderated.

## Adding a new model

1. Drop `backend/runners/<model>.py` — subclass `backend.runners.base.Runner`,
   implement `load()` and `generate(params, loras)`.
2. Add an entry to `backend/model_registry.json` with `runner_module`, `param_groups`,
   and any `param_overrides` / `param_keys` whitelist.
3. If the runner needs new pip packages, add them to `requirements-runtime.txt` at
   the repo root — installed on every boot.
4. Mirror the entry in `mockModels()` inside `ui/app.js` so the dev preview shows it.
5. `git push` → restart any pod tracking `main`.

## Local development (no GPU)

```bash
pip install fastapi 'uvicorn[standard]' httpx huggingface_hub pydantic python-multipart
python backend/main.py
# Open http://localhost:3000
# Models won't actually load without a GPU, but the UI works against mock data.
```

## Folder layout

```
PleoAI/
├── backend/
│   ├── main.py             FastAPI app
│   ├── launcher.py         spawns runner subprocesses
│   ├── runner_host.py      generic FastAPI shell wrapping any Runner
│   ├── gpu_detect.py
│   ├── model_registry.json
│   └── runners/
│       ├── base.py         Runner ABC + helpers
│       └── qwen_image.py   …one file per model
├── ui/{index.html, styles.css, app.js}
├── docker/entrypoint.sh    git-pulls the repo on boot
├── Dockerfile              runtime-only image
```

## Workspace layout (on the pod)

```
/workspace/
├── forge-src/              ← this repo, cloned on boot
├── models/                 ← HF weights cache (per-runner)
├── loras/                  ← .safetensors LoRAs
├── checkpoints/            ← manual checkpoints
├── assets/
│   ├── uploads/            ← user-uploaded references
│   └── generated/          ← runner outputs (auto-picked-up by Assets view)
└── logs/                   ← per-runner logs (SSE-streamable)
```
