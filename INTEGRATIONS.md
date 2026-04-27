# Future Integrations & Notes

## Pose Conditioning (Qwen-Image-Edit)

**Status:** Not implemented. Pose/Style image slots hidden in UI.

**What's needed:**
- Switch from `QwenImageEditPipeline` to `QwenImageControlNetPipeline` (diffusers)
- Download `InstantX/Qwen-Image-ControlNet-Union` (~5 GB)
- Download a DWPose keypoint extractor (e.g. `yzd-v/DWPose` or `Kea-W/FaceChain-SuDe-Bridge`) to preprocess the pose image into a keypoint map before passing to the pipeline
- New runner: `backend/runners/qwen_image_edit_controlnet.py`
- Registry: re-add `{ "key": "pose", "label": "Pose", "required": false }` to `qwen-image-edit` image_inputs once the controlnet runner exists
- Key params: `controlnet_conditioning_scale` (0.8–1.0), `true_cfg_scale` (4.0)

**Pipeline call shape:**
```python
from diffusers import QwenImageControlNetPipeline
pipe = QwenImageControlNetPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
)
result = pipe(
    image=ref_img,
    control_image=pose_keypoint_map,   # output of DWPose preprocessor
    prompt=prompt,
    controlnet_conditioning_scale=0.9,
)
```

---

## Style Conditioning (Qwen-Image-Edit)

**Status:** Not implemented. Style slot hidden.

**What's needed (two options):**

**Option A — VLM description (no extra models):**
When a style image is attached, run it through Qwen2.5-VL (already used as text encoder inside this pipeline) to generate a short style description, then append it to the prompt. No additional weights to download.

**Option B — IP-Adapter:**
Check if a Qwen-compatible IP-Adapter exists. At time of writing, none confirmed. Generic IP-Adapters (e.g. for SDXL) won't work with QwenImageEditPipeline.

Recommendation: implement Option A first since it reuses the existing model.

---

## Additional Models (Future)

### FLUX.1-dev (text-to-image)
- Repo: `black-forest-labs/FLUX.1-dev`
- Pipeline: `FluxPipeline` (diffusers)
- Runner: `backend/runners/flux1_dev.py`
- ~24 GB, good quality baseline alongside Qwen-Image

### FLUX.1-Kontext-dev (image editing)
- Repo: `black-forest-labs/FLUX.1-Kontext-dev`
- Pipeline: `FluxKontextPipeline` (diffusers)
- Runner: `backend/runners/flux1_kontext.py`
- High-quality reference-guided editing; alternative to Qwen-Image-Edit

### SDXL (text-to-image)
- Repo: `stabilityai/stable-diffusion-xl-base-1.0`
- Pipeline: `StableDiffusionXLPipeline`
- Smaller footprint (~6 GB), fast, broad LoRA ecosystem

---

## Pre-quantised Weights (FP8, GGUF)

**Status:** Not implemented. Current quant story is runtime-only (BitsAndBytes
INT8 / NF4 applied at `from_pretrained` time on the same official HF repo).

**What's needed for FP8:**
- Comfy-Org publishes `qwen_image_edit_2511_fp8mixed.safetensors` etc — these
  are ComfyUI-format split files, not directly loadable by diffusers'
  `from_pretrained`. Either:
  - Find a diffusers-format FP8 repo, or
  - Hand-load the safetensors into pipeline components (transformer + VAE +
    text encoder loaded individually, then assembled via `Pipeline(...)` ctor)
- Add `fp8` to the model's `quants` array in `model_registry.json` with a
  separate `hf_repo` / `weight_file` per quant
- Runner's `load()` branches on the quant id and uses the right loader path

**What's needed for GGUF:**
- `gguf` Python library + diffusers' `GGUFQuantizationConfig` (recent feature,
  works for FluxPipeline; QwenImage support unverified at time of writing)
- Repos like `city96/Qwen-Image-gguf` provide Q4_K_M / Q5_K_M / Q6_K / Q8_0
- Pre-load probe: the GGUF loader is finicky — should be tested against a
  specific QwenImage repo before exposing in the UI

**Why not yet:** runtime BnB INT8 / NF4 covers 80 % of the use case with one
HF repo per model. Pre-quantised weights add per-model curation overhead.
Defer until users hit a real limitation BnB doesn't solve.

---

## Hot-swap Pipeline Components

**Status:** Not implemented. ComfyUI does this; diffusers makes it harder.

**What's needed:**
- Load each component separately (`AutoencoderKLQwenImage`, transformer,
  Qwen2.5-VL text encoder) instead of via `from_pretrained` on the full repo
- Construct the pipeline manually: `QwenImagePipeline(text_encoder=..., transformer=..., vae=...)`
- Registry entry per component variant
- UI: separate dropdowns in the configure drawer for each slot

**Why not yet:** quant covers the main use case (smaller resident model).
Component-swap is mostly useful for the rare case of mixing a fine-tuned
transformer with a stock text encoder — niche.

---

## VRAM Hygiene on Model Switch

**Status:** Not implemented.

**Problem:** `launcher.py` only checks if the model the user is launching is
already running. It doesn't stop other models. So if a user loads Qwen-Image
(uses ~47 GB), then opens Qwen-Image-Edit and clicks Start, a second runner
process spawns and tries to load another ~47 GB model — instant OOM on a
48 GB card. Same problem when adding more models (Flux, SDXL, etc).

**Fix:**
- Before spawning a new runner, call `launcher.stop()` on every other live
  runner so its VRAM is freed
- Optionally: track approximate VRAM per runner and only auto-stop if the
  combined footprint would exceed the available card. On a 96 GB card you
  could keep two ~47 GB models loaded at once.
- Frontend: when user clicks Start on model B while model A is running,
  show a "Switching from A → B" status briefly so the eviction is visible

**Where:** `backend/launcher.py`, `launch()` method. Add a sweep over
`self._procs` calling `_terminate()` on anything that isn't `mid` before
spawning the new process.

---

## Queue Persistence

**Status:** Queue is in-memory per browser tab. Closing the tab loses pending jobs.

**To implement:**
- Persist `state.queue` to `localStorage` on every change
- On boot, restore pending jobs and re-submit any that were `running` (mark as pending again since the pod can't know the outcome)
- Low priority for RunPod single-user use — pods stay up, tabs rarely close mid-batch
