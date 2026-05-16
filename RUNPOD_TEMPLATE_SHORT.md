# Igglepixel

**One UI, every open generation model — image, edit, video, chat, voice, LoRA.**

A self-hosted launcher with a clean UI over the open-source generation stack. Pick a model, click Download → Start → Open. No ComfyUI graphs, no Python.

## What you get

- **Catalogue UI** — browse, download, run; per-model workspaces with VRAM-aware defaults.
- **Quantisation** — BF16 / INT8 / NF4 auto-picked for your GPU.
- **Live previews** — step-by-step previews where the runner supports them.
- **LoRA support** — browse CivitAI in-app; per-LoRA strength sliders on supported models.
- **Encrypted storage** — AES-GCM at rest; key derived from your password.
- **Auth gate** — username + password set on first boot, never on a server.
- **Hot reload** — UI/backend pulled from the public Git repo on every boot.

## Pick the right GPU

The UI now groups families into model cards with variants. Quants are auto-picked where available; tiers below show the smallest sensible GPU for each current variant path.

**Grouped variants in the app:** Wan 2.2 I2V/T2V each expose 5B, 14B, and 14B Lightning 4/8-step; LTX-2.3 exposes distilled 8-step I2V; Qwen 2.5 Chat exposes 7B/14B/32B; HunyuanVideo exposes T2V/I2V; HiDream-I1 exposes Fast/Dev/Full; SenseNova-U1 exposes 8B-MoT, preview, and SFT.

### 16 GB
FLUX.2 [klein] 4B · Z-Image Turbo · FLUX.1-dev INT8/NF4 · LongCat Image/Edit INT8/NF4 · Qwen-Image / Edit / 2511 / 2512 NF4 · Qwen ControlNet Union/Inpaint NF4 · Qwen 2.5 7B BF16 · Qwen 2.5 14B INT8/NF4 · Qwen 2.5 32B NF4 · Wan 2.2 5B INT8/NF4

### 24 GB
HunyuanVideo NF4 T2V/I2V · HiDream-I1 Fast · Qwen-Image / Edit / 2511 / 2512 INT8 · Qwen ControlNet INT8 · Wan 2.2 14B NF4 · FLUX.2 [klein] 9B INT8 · LongCat Image/Edit BF16 · FLUX.1-dev BF16

### 36 – 48 GB
Wan 2.2 5B BF16 · Wan 2.2 14B INT8 · LTX-2.3 distilled minimum · FLUX.2 [klein] 9B BF16 · Qwen 2.5 14B BF16 · Qwen 2.5 32B INT8 · SenseNova-U1 BF16 minimum · HiDream-I1 Dev

### 80 GB+
Wan 2.2 14B BF16 · Wan 2.2 14B Lightning 4/8-step I2V/T2V · LTX-2.3 distilled recommended · HunyuanVideo BF16 T2V/I2V · HiDream-I1 Full · Qwen-Image / Edit / 2511 / 2512 BF16 · Qwen ControlNet BF16 · Qwen 2.5 32B BF16 · SenseNova-U1 all variants comfortably

### CPU only
Kyutai Pocket TTS. 🛣️ Planned API-only video: **Kling**, **Seedance**, **Veo 3**, **Sora 2**, **Runway**, **Pika**. Bring your own provider key.

> **Running locally:** this image targets RunPod's `/workspace` volume layout. To run on your own Docker host, mount any persistent dir at `/workspace` and override `FORGE_REPO` to point at your fork.

## Quick start

1. Deploy → pick a GPU from the tier above.
2. Open the proxy URL on port `3000`.
3. Set username + password (**no recovery — bound to your encryption key**).
4. Pick a model → Download → Start → generate.

First load pulls weights to the persistent volume (one-time). Subsequent boots load in seconds from cache.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Gated HuggingFace repos (or paste in Settings) |
| `IGGLEPIXEL_MODERATION_DISABLE_ACK` | — | Fork-operator opt-out token (see Moderation) |

The container pulls latest UI/backend from [github.com/riseon-lab/igglepixel](https://github.com/riseon-lab/igglepixel) on every boot — fixes ship without rebuilds.

## Content moderation

On by default; the shared moderation policy gates four stages:

- **Prompt** — `KoalaAI/Text-Moderation` (CPU) catches sexual / hate / violence / harassment / self-harm before any model spins up.
- **Reference image** — `Falconsai/nsfw_image_detection` scans uploaded refs at first use, post-decrypt inside the runner.
- **Output** — same classifier scans generated stills and sampled video frames before anything lands on disk.
- **CivitAI browse** — NSFW-tagged LoRAs filtered server-side; tampered clients can't bypass.

HF downloads aren't gated — checks happen at *use*, not *download*. Disabling is friction-by-design: paste the acknowledgement token from `backend/moderator.py` either as `IGGLEPIXEL_MODERATION_DISABLE_ACK` (fork operator, visible in pod config) or in Settings → Content moderation (runtime override, requires login). **The token is the operator's written declaration of liability.**

## Privacy & terms

- Credentials and outputs live encrypted on `/workspace`. No telemetry, no external auth.
- HF / CivitAI requests come from your pod's IP; tokens browser-stored only.
- No model weights are bundled — they're pulled from their publishers under each publisher's licence. You are responsible for compliance and for what you generate.
- Provided as-is, without warranty. **Forks own their own moderation policy and liability.**

[Source · github.com/riseon-lab/igglepixel](https://github.com/riseon-lab/igglepixel) — MIT.

---
_Built on RunPod. Powered by open-source models._
