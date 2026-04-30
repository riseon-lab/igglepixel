# IgglePixel

One self-hosted UI for the open-source generation models people actually want to run: image, edit, video, and text.

IgglePixel gives you a clean RunPod-ready web app over a FastAPI backend, model runners, LoRA management, encrypted assets, and a registry-driven catalogue. Download a model, start it, open the workspace, generate. No ComfyUI graph wrangling and no one-off Python scripts.

IgglePixel is designed for people who want a focused alternative to stitching together separate model UIs. The Docker image provides the runtime; this public repo provides the app code, model registry, and runner implementations.

Forks are welcome. Pull requests are welcome. Please test before opening one.

## Quick Start

The easiest path is a RunPod template using the maintained image:

- RunPod: [runpod.io?ref=cgt80jay](https://runpod.io?ref=cgt80jay)
- Template/image family: `riseonlab/igglepixel`

The tag changes as releases move forward, so use the current `riseonlab/igglepixel:<version>` tag shown on the template.

1. Create a RunPod GPU pod.
2. Use the IgglePixel template or a custom Docker image based on `riseonlab/igglepixel`.
3. Expose port `3000`.
4. Add `HF_TOKEN` if you need gated Hugging Face models.
5. Start the pod and open the HTTP service.

## What It Does

- Launches supported models through isolated Python runner subprocesses.
- Provides workspace UIs for image, image editing, video, and text generation.
- Manages generated and uploaded assets.
- Supports LoRA download, assignment, and per-model strengths.
- Supports BF16 / INT8 / NF4 where runners expose quantized loading.
- Uses a model registry for parameters, VRAM recommendations, context policy, quantization, variants, and model-specific defaults.
- Pulls app code from this repo at pod boot, so iteration can be `git push` plus pod restart instead of rebuilding the runtime image every time.

## Supported Models

The live catalogue is registry-driven, so support will keep expanding without changing the UI shell.

| Category | Models | Notes |
| --- | --- | --- |
| Text-to-image | Qwen-Image, Qwen-Image-2512, FLUX.1-dev | Qwen is Apache 2.0; FLUX.1-dev is non-commercial. |
| Image edit | Qwen-Image-Edit, Qwen-Image-Edit 2511 | Reference-image editing with LoRA support. |
| Image-to-video | Wan 2.2 I2V | 14B Lightning 8-step, 14B, and 5B variants. |
| Image-to-video | LTX-2.3 | Distilled 1.1 / 1.0 and dev variants; LTX Community License, non-commercial. |
| Text / chat | Qwen 2.5 Chat, TinyLlama Chat 1.1B | Qwen 7B / 14B / 32B variants with VRAM-aware context defaults. |

More models are coming. The intended path is simple: add a runner, add registry metadata, expose the right controls, and keep the UX consistent.

## How The Runtime Works

The Docker image is the heavy runtime layer: Python, Torch, Diffusers, Transformers, system packages, and the startup script.

The app code is pulled on boot:

```text
RunPod container starts
  -> docker/entrypoint.sh
  -> clone/fetch this repo into /workspace/forge-src
  -> install requirements-runtime.txt if present
  -> start backend/main.py on port 3000
```

This keeps the Docker image stable while the public repo can move quickly.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `FORGE_REPO` | `https://github.com/riseon-lab/igglepixel.git` | Git repo pulled on boot. |
| `FORGE_BRANCH` | `main` | Branch to track. |
| `FORGE_COMMIT` | unset | Pin to a specific commit or tag. Overrides branch head. |
| `FORGE_CACHE_DIR` | `/workspace/forge-src` | Cached clone location on the persistent volume. |
| `WORKSPACE` | `/workspace` | Root for models, assets, LoRAs, logs, and cache. |
| `UI_PORT` | `3000` | FastAPI/UI port. |
| `HF_TOKEN` | unset | Hugging Face token for gated repos. You can also set this in the UI. |
| `HF_HOME` | `/workspace/.cache/huggingface` | Hugging Face cache location. |
| `IGGLEPIXEL_MODERATION` | `true` | Enables image moderation before outputs are saved. |
| `FORGE_QUANT` | unset | Set by launcher per runner from the UI. Usually do not set manually. |

If GitHub is unavailable during boot, the entrypoint falls back to the cached clone when possible.

## Architecture

```text
Browser
  |
  |  port 3000
  v
FastAPI backend
  |  /api/models
  |  /api/launch
  |  /api/generate
  |  /api/assets
  |  /api/download/hf
  |
  |  python -m backend.runner_host backend.runners.<model> <port>
  v
Runner subprocess on 127.0.0.1:17000+
  |  load weights
  |  run inference
  |  write assets/logs
  v
/workspace
```

Only the main FastAPI app is exposed. Runner subprocesses bind to localhost and are reached only by the backend.

## Encryption & Asset Privacy

IgglePixel encrypts uploaded and generated assets at rest on the workspace volume. The goal is to protect persistent storage, backups, and accidentally exposed asset URLs while keeping the normal image/video workflow usable in the browser.

Asset encryption is built around one data key:

- The data key is derived from the user's password with PBKDF2-SHA256.
- The derived key is 32 bytes and is used for AES-256-GCM.
- Encrypted files are written as `<visible-filename>.enc`.
- The encrypted file format is `[12-byte nonce][ciphertext + 16-byte GCM tag]`.
- The user-facing asset path stays as the original filename, so the UI works with `photo.png` while disk stores `photo.png.enc`.

During setup/login, the backend stores the username, password hash, session token, signing key, PBKDF2 salt, and an encrypted canary in the auth file. It does not store the raw data key. On login or unlock, the password derives the key again and verifies it against the canary. The data key then lives in process memory until the backend restarts or the user logs out.

Uploads can be encrypted in the browser before they reach the backend. The UI derives a non-extractable Web Crypto `CryptoKey`, encrypts the file, and sends it with `X-Forge-Encrypted: 1`. In that path, the backend stores the ciphertext directly and does not need to see the plaintext upload. If browser-side encryption is unavailable but the backend is unlocked, the backend falls back to server-side encryption before writing the file. A plaintext fallback exists only for local/dev situations where no key is available.

Generated outputs are encrypted by the runner helpers. When the backend launches a model runner, it passes the unlocked data key to the subprocess through `FORGE_DATA_KEY`. Runners use that key to read encrypted references and write generated images/videos back as encrypted assets.

Viewing assets works through signed URLs and the service worker:

- Asset URLs are served from `/api/assets/file/...` with a short-lived signature.
- The backend returns the encrypted bytes and marks encrypted responses with `X-Forge-Encrypted: 1`.
- `ui/sw.js` intercepts those asset requests, decrypts the blob in the browser, and returns normal media bytes to `<img>` and `<video>`.
- Encrypted videos are fetched as complete blobs rather than byte ranges because AES-GCM cannot decrypt an arbitrary partial range safely.

Important trust boundary: this is encryption at rest, not a zero-knowledge hosted system. While a pod is unlocked, the backend and runner processes must be able to decrypt assets to generate, edit, preview, and serve media. Plaintext can also exist briefly in GPU memory, process memory, browser memory, and temporary encode buffers. If a running pod is compromised while unlocked, the attacker may be able to access decrypted data.

Operational notes:

- If the password is lost, encrypted assets cannot be recovered.
- After a pod/backend restart, users may need to unlock again before encrypted assets can be viewed or used as references.
- Do not commit `/workspace/assets`, `.forge_auth.json`, tokens, generated outputs, model weights, or local workspace data.
- PRs that touch auth, encryption, downloads, uploads, service-worker asset handling, or runner asset I/O should test upload, preview, download, generated output, locked state, and post-restart unlock.

## Model Registry

Models live in `backend/model_registry.json`. The registry describes:

- model id, name, category, and Hugging Face repo
- runner module
- VRAM minimum and recommended VRAM
- quantization options
- parameter groups and overrides
- aspect presets or video dimensions
- LoRA compatibility
- context limits for text models
- variant-specific settings for models with multiple sizes

For text models, context can be configured per model and per VRAM tier:

```json
{
  "context_window": 2048,
  "context_by_vram": [
    { "min_gb": 12, "tokens": 4096, "label": "extended" },
    { "min_gb": 0, "tokens": 2048, "label": "safe default" }
  ],
  "context_policy": {
    "mode": "workspace",
    "auto_compact_at": 0.82,
    "keep_recent_messages": 8
  }
}
```

The UI resolves the actual working context from model capability, detected VRAM, selected quantization, and model policy.

## Adding A Model

1. Add a runner in `backend/runners/<model>.py`.
2. Subclass `backend.runners.base.Runner`.
3. Implement `load()` and `generate(params, loras)`.
4. Add an entry to `backend/model_registry.json`.
5. Mirror the model in `mockModels()` in `ui/app.js` so `?preview` works without a backend.
6. Add any extra Python dependencies to `requirements-runtime.txt`.
7. Test locally where possible, then test on a GPU pod before opening a PR.

## Local Development

UI-only preview works without a GPU:

```bash
python3 -m http.server 4175 -d ui
open "http://127.0.0.1:4175/?preview"
```

Backend development:

```bash
pip install fastapi "uvicorn[standard]" httpx huggingface_hub pydantic python-multipart
python backend/main.py
```

Models will not load properly without suitable GPU/runtime support, but the UI and API shell can still be exercised.

## Testing Checklist

Before opening a pull request, run the checks that match your change:

```bash
node --check ui/app.js
python3 -m json.tool backend/model_registry.json >/dev/null
python3 -m py_compile backend/runners/*.py backend/*.py
```

For UI changes:

- Test `http://127.0.0.1:4175/?preview`.
- Check desktop and mobile widths.
- Make sure text does not overflow buttons, chips, cards, or controls.
- Verify any new controls remain usable with long labels.

For model or runner changes:

- Validate registry JSON.
- Confirm runner import/compile succeeds.
- Test launch and generation on a suitable GPU pod.
- Include VRAM assumptions in the registry.
- Do not claim support for models or hardware you have not tested.

For download/auth changes:

- Test with and without `HF_TOKEN`.
- Check gated Hugging Face repos.
- Confirm typed tokens are not lost on UI re-render.

## Contributing

This repository is public. Forks, experiments, and PRs are welcome.

Please keep contributions focused:

- Prefer small PRs over broad rewrites.
- Follow existing UI and backend patterns.
- Add registry data instead of hardcoding model-specific behavior where possible.
- Keep model claims honest: include VRAM requirements, license notes, and known limitations.
- Do not commit secrets, tokens, generated assets, model weights, or local workspace data.
- Run the relevant tests before submitting.
- Describe what you tested in the PR body.

PRs that touch safety, downloads, auth, encryption, or model execution should explain the risk and verification path clearly.

## Content Moderation

Generated images and sampled video frames pass through `Falconsai/nsfw_image_detection` before they are saved. Flagged outputs are dropped and never reach the asset library.

- Default: on.
- Disable with `IGGLEPIXEL_MODERATION=false`.
- Scope: image outputs and sampled video frames.
- Video sample count: `IGGLEPIXEL_VIDEO_MODERATION_FRAMES` (default `7`).
- Threshold and behavior live in `backend/moderator.py`.

Fork operators are responsible for their own deployments and outputs.

## Workspace Layout

```text
/workspace/
├── forge-src/              repo clone pulled on boot
├── .cache/huggingface/     Hugging Face model cache
├── models/                 model files and runner-managed downloads
├── loras/                  LoRA files
├── checkpoints/            manual checkpoints
├── assets/
│   ├── uploads/            uploaded references
│   └── generated/          generated outputs
└── logs/                   runner logs
```

## Repository Layout

```text
backend/
  main.py                   FastAPI app
  launcher.py               starts and stops runner subprocesses
  runner_host.py            localhost wrapper for one runner
  model_registry.json       model and UI registry
  runners/                  one runner module per model

ui/
  index.html
  styles.css
  app.js

docker/
  entrypoint.sh             pulls repo and starts the app

Dockerfile                  runtime image
```

## License

Check the repository license before redistributing. Individual models, LoRAs, datasets, and upstream weights have their own licenses and restrictions.
