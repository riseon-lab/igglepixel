"""Image upscaling via spandrel.

Spandrel auto-detects the architecture of any common upscaler weight file
(RealESRGAN, ESRGAN, SwinIR, HAT, DAT, …) and returns a uniform descriptor
we can call. We use it so adding a new upscaler is a registry entry plus
a weight file on HuggingFace — no per-model loader code.

Lazy loading: each upscaler is downloaded + initialised on first use,
then cached in process for the runner's lifetime. Typical upscaler is
50–100 MB on disk and stays resident — negligible vs the diffusion model.

Weight files are cached under HF_HOME like everything else, so they live
on the persistent volume and survive pod restarts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Loaded model descriptors keyed by upscaler id. Stays alive for the
# runner subprocess's lifetime.
_loaded: dict = {}
_registry_cache = None


def _registry() -> list:
    """Read the upscalers section of model_registry.json. Cached."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "backend" / "model_registry.json") as f:
        reg = json.load(f)
    _registry_cache = reg.get("upscalers", [])
    return _registry_cache


def _entry(upscaler_id: str):
    for u in _registry():
        if u["id"] == upscaler_id:
            return u
    return None


def _load(upscaler_id: str):
    """Resolve, download, and load the upscaler. Cached."""
    if upscaler_id in _loaded:
        return _loaded[upscaler_id]
    entry = _entry(upscaler_id)
    if not entry:
        raise ValueError(f"Unknown upscaler: {upscaler_id}")

    from huggingface_hub import hf_hub_download
    from spandrel import ModelLoader
    import torch

    print(f"[upscaler] loading {entry['label']}…", flush=True)
    weight_path = hf_hub_download(
        repo_id=entry["hf_repo"],
        filename=entry["weight_file"],
        token=os.environ.get("HF_TOKEN"),
    )

    descriptor = ModelLoader().load_from_file(weight_path)
    if torch.cuda.is_available():
        descriptor.cuda()
    descriptor.eval()
    _loaded[upscaler_id] = descriptor
    print(f"[upscaler] {entry['label']} ready (×{descriptor.scale})", flush=True)
    return descriptor


def upscale(image, upscaler_id: str):
    """Run an image through the chosen upscaler. Returns a new PIL Image.

    Returns the original image unchanged if upscaler_id is empty/None or
    if anything goes wrong — better to ship the un-upscaled output than
    to fail the whole generation.
    """
    if not upscaler_id:
        return image
    try:
        import numpy as np
        import torch
        from PIL import Image

        descriptor = _load(upscaler_id)
        device = next(descriptor.model.parameters()).device

        # PIL → tensor (1, 3, H, W) in [0, 1]
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        # Match dtype to model — mixed precision can blow up half-precision models.
        tensor = tensor.to(next(descriptor.model.parameters()).dtype)

        with torch.no_grad():
            out = descriptor(tensor)

        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((out * 255).astype(np.uint8))
    except Exception as e:
        print(f"[upscaler] failed, returning original: {e}", flush=True)
        return image
