# Igglepixel

**One UI for every open-source generation model you actually want to run.**

A self-contained launcher with a clean, commercial-grade interface over the entire open-source generation stack — image, edit, and (soon) video, text, and voice. Deploy, sign in, generate. No ComfyUI node graphs, no Python scripts.

## What you get

- **One UI, every model** — pick from the catalogue, click Download, click Start, generate.
- **Auth gate** — pod boots locked, you set username/password on first run. Credentials live in the pod, not on a server.
- **Live previews** — ComfyUI-style step-by-step image previews while inference runs.
- **Multi-image carousel** — reference cells hold multiple images per slot (◀ ▶).
- **LoRA support** — browse CivitAI in-app, download to pod, assign to models, per-LoRA strength sliders.
- **Quantisation** — BF16 / INT8 / NF4 with auto-pick by GPU. INT8 lets larger generations fit smaller cards.
- **VRAM-aware defaults** — output size, steps, and quant default to what your card can handle.
- **Encrypted storage** — uploads and outputs are AES-GCM encrypted at rest, key derived from your password.
- **HF / CivitAI integration** — gated repos and LoRA browsing inside the app.

## What it currently runs

| Category | Models | Status |
|---|---|---|
| Text-to-image | Qwen-Image | ✅ Live |
| Image-to-image edit | Qwen-Image-Edit | ✅ Live |
| Text-to-image | FLUX.1-dev, SDXL | 🛣️ Roadmap |
| Image edit | FLUX.1-Kontext | 🛣️ Roadmap |
| Pose / Style ControlNet | InstantX Qwen ControlNet | 🛣️ Roadmap |
| Video | Wan, HunyuanVideo, LTX-Video | 🛣️ Roadmap |
| Voice / Audio | Whisper, Bark, MusicGen | 🛣️ Roadmap |
| Text / LLM | Llama 3, Qwen2.5, DeepSeek | 🛣️ Roadmap |

## Quick start

1. **Deploy** this template, pick a GPU (24 GB+ for INT8, 48 GB+ for BF16).
2. Open the proxy URL on port `3000`.
3. Set username + password. **No password recovery** — it's bound to your encryption key.
4. Pick a model → Download → Start → Open workspace → generate.

First model load pulls ~40 GB to the persistent volume (one-time). Subsequent boots load from cache in seconds.

## Configuration

All optional. Defaults work out of the box.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Gated HuggingFace repos (or paste in Settings tab) |
| `IGGLEPIXEL_MODERATION` | `true` | NSFW classifier on every output (see below) |

The container pulls latest UI/backend from [github.com/riseon-lab/igglepixel](https://github.com/riseon-lab/igglepixel) on every boot — fixes ship without rebuilds.

## Content moderation

Ships with NSFW image moderation **enabled by default**. Outputs are classified before they hit disk; flagged ones are dropped silently with a neutral toast. Classifier is `Falconsai/nsfw_image_detection` (~350 MB VRAM, on-pod, no external calls).

Set `IGGLEPIXEL_MODERATION=false` to disable. **If you do, you accept full responsibility for your deployment's outputs.** Appropriate for private dev pods, not for any deployment where untrusted users can generate.

## Source

[github.com/riseon-lab/igglepixel](https://github.com/riseon-lab/igglepixel) — MIT licensed. Issues, PRs, and new model contributions welcome (one-file drop-in under `backend/runners/`).

## Terms & responsibility

This template provides infrastructure to run open-source generation models. **No model weights are bundled.** Models are downloaded directly from their original publishers under whatever licence those publishers set. You are responsible for complying with each model's licence.

Generated outputs are your responsibility:

- Moderation is a defensive measure, not a guarantee. It catches NSFW; it does not catch every category of harmful content.
- You are responsible for what you generate, what you publish, and what you allow others to generate via your deployment.
- Outputs are statistical — may be unexpected, biased, or reflect training data biases.

**Forks operate under their own moderation policy and own their own responsibility.** The maintainer is responsible only for the official `riseonlab/igglepixel` image with its default-on moderation. Forks may disable moderation; that is the fork operator's choice and liability.

Provided as-is, without warranty of any kind, express or implied. In no event shall the maintainer be liable for any claim or damages arising from use of this software.

## Privacy

- Credentials live in the pod's `/workspace` volume. No external auth.
- Generated content is encrypted on the pod's volume. No telemetry.
- Moderation runs entirely on-pod. No external moderation calls.
- HF / CivitAI requests come from your pod's IP; tokens are browser-stored only.

---

_Built on RunPod. Powered by open-source models._
