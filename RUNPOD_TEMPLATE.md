# Igglepixel

**One UI for every open-source generation model you actually want to run.**

A self-contained launcher that gives you a clean, commercial-grade interface over the entire open-source generation stack — image, edit, and (soon) video, text, and voice. Deploy it on this RunPod template, sign in once, and start generating. No ComfyUI node graphs, no Python scripts, no fighting with virtualenvs.

---

## What you get

- **One UI, every model.** Pick a model from the catalogue, click Download, click Start, generate. The launcher handles weight downloads, runtime loading, GPU placement, and inference — you just pick what you want to make.
- **Auth gate.** Every pod boots locked. You set a username and password on first run; credentials are bound to the pod (not stored on a server). Subsequent visits remember you on the same browser.
- **Live previews.** ComfyUI-style step-by-step image previews while generation is in progress. See it forming in real time.
- **Carousel inputs.** Reference image cells hold multiple images per slot — flip through with ◀ ▶ to compare options.
- **Full LoRA support.** Browse CivitAI in-app, download LoRAs to your pod, assign them to models, set per-LoRA strength, all from the UI.
- **Quantisation built in.** Pick BF16, INT8, or NF4 per model, or let it auto-pick based on your GPU's VRAM. INT8 lets you run larger generations on smaller cards without the bf16 OOM crashes.
- **Auto-detect sensible defaults.** Output dimensions, step count, and quant default to what your GPU can comfortably handle. Override anything manually.
- **Encrypted storage.** Uploaded references and generated outputs are AES-GCM encrypted at rest with a key derived from your password. Even with disk access, the files are unreadable without your password.
- **Asset manager.** Every generated and uploaded image is one click away. 3-dot menu for re-use, download, or delete.
- **HuggingFace and CivitAI integration.** HF tokens for gated repos, CivitAI API for browsing and downloading LoRAs from inside the app.

---

## What it currently runs

| Category | Models | Status |
|---|---|---|
| Text-to-image | Qwen-Image | ✅ Live |
| Image-to-image (edit) | Qwen-Image-Edit | ✅ Live |
| Text-to-image | FLUX.1-dev, SDXL | 🛣️ Roadmap |
| Image edit (alt) | FLUX.1-Kontext | 🛣️ Roadmap |
| Pose / Style ControlNet | InstantX Qwen ControlNet | 🛣️ Roadmap |
| Video | Wan 2.2 I2V, HunyuanVideo | ✅ Live |
| Voice / Audio | Whisper, Bark, MusicGen | 🛣️ Roadmap |
| Text / LLM | Llama 3, Qwen2.5, DeepSeek | 🛣️ Roadmap |

The architecture supports any model with a Python runner — adding a new one is a single-file drop-in.

---

## Quick start

1. Click **Deploy** on this template.
2. Pick a GPU. Recommended: 24 GB+ for INT8, 48 GB+ for BF16.
3. Wait for the pod to come up, open the proxy URL on port `3000`.
4. Set a username and password. Remember the password — there is no recovery; it's bound to the encryption key for your assets.
5. Pick a model, hit Download (one-time pull from HuggingFace), then Start.
6. Open the workspace and generate.

The first model load downloads ~40 GB of weights to the persistent volume. This only happens once — subsequent pod starts boot in seconds from the cached weights.

---

## Configuration

All optional. Defaults work out of the box.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | _(unset)_ | For gated HuggingFace repos. You can also paste it in the Settings tab. |
| `IGGLEPIXEL_MODERATION` | `true` | NSFW image classifier on every output. See below. |
| `FORGE_BRANCH` | `main` | Track a different branch of the source repo |
| `FORGE_COMMIT` | _(unset)_ | Pin to a specific commit/tag for reproducibility |

The container is a runtime shell — it pulls the latest UI/backend code from [github.com/riseon-lab/igglepixel](https://github.com/riseon-lab/igglepixel) on every boot. You always get the latest fixes without rebuilding the image.

---

## Content moderation

This template ships with NSFW image moderation **enabled by default**. Every generated image is classified before it lands on disk — flagged outputs are dropped silently with a neutral toast to the user. The classifier (Falconsai/nsfw_image_detection, ~350 MB VRAM) loads once per runner and stays resident.

You can disable it by setting `IGGLEPIXEL_MODERATION=false` in the pod environment variables. **If you do, you take full responsibility for what your instance generates.** This is the right call for private dev pods or research instances where you control all access. It's not appropriate for any deployment where untrusted users can generate.

---

## Source, contributions, licensing

- **Source:** https://github.com/riseon-lab/igglepixel — MIT licensed.
- **Issues / feature requests:** the repo issue tracker.
- **Pull requests welcome.** Adding a new model is a one-file change in `backend/runners/`. See the README for the full architecture.

---

## Terms of use & responsibility

This template provides infrastructure to run open-source generation models. **It does not bundle any model weights.** Models are downloaded directly from their original publishers (HuggingFace, etc.) on first use, under whatever licence those publishers set. You are responsible for understanding and complying with the licence of any model you download and run.

Generated outputs are your responsibility:

- The moderation gate is a defensive measure, not a guarantee. It catches NSFW imagery; it does not catch every category of harmful content (violence, hate symbols, copyrighted likenesses, etc.).
- You are responsible for what you generate, what you publish, and what you allow others to generate via your deployment.
- Generation models are statistical — outputs may be unexpected, biased, or reflect the training data of the underlying model. The maintainer makes no warranty as to fitness for any purpose.

**If you fork this project and run your own deployment, you take full responsibility for your fork.** The maintainer of this RunPod template (`riseonlab/igglepixel`) is responsible only for the official, moderated image. Forks operate under their own moderation policy, including the choice to disable moderation entirely.

The official image is provided as-is, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the maintainer be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

---

## Privacy

- **Credentials:** stored in the pod's `/workspace` volume only. No external auth server. Resetting the pod's volume wipes them.
- **Generated content:** stored encrypted on the pod's volume. No telemetry, no upload to any external service.
- **Model downloads:** HuggingFace and CivitAI requests come from your pod's IP. Your tokens are stored in your browser only (not on any server we control).
- **Moderation:** runs entirely on-pod. No images are sent to any external moderation service.

---

_Built on RunPod. Powered by open-source models. Designed for people who want to make things, not configure tools._
