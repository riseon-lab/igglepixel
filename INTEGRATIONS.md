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

## Queue Persistence

**Status:** Queue is in-memory per browser tab. Closing the tab loses pending jobs.

**To implement:**
- Persist `state.queue` to `localStorage` on every change
- On boot, restore pending jobs and re-submit any that were `running` (mark as pending again since the pod can't know the outcome)
- Low priority for RunPod single-user use — pods stay up, tabs rarely close mid-batch
