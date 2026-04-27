"""Qwen-Image-Edit runner — image-conditioning variant of Qwen-Image.

Repo:        Qwen/Qwen-Image-Edit
Pipeline:    diffusers.QwenImageEditPipeline

Accepts a `ref_image` param (path or URL). Pose/Style conditioning today maps
to a single edit pipeline call; future iterations may chain multiple
ControlNet adapters for true multi-conditioning. For now extra inputs are
collected from the UI but only the primary ref drives generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Runner(RunnerBase):
    model_id            = "qwen-image-edit"
    model_name          = "Qwen-Image-Edit"
    category            = "image"
    supports_lora       = True
    min_vram_gb         = 24
    recommended_vram_gb = 48

    HF_REPO = "Qwen/Qwen-Image-Edit"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    def load(self) -> None:
        import torch
        from diffusers import QwenImageEditPipeline

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()

        kwargs = {"torch_dtype": torch.bfloat16, "token": token}
        if quant in ("int8", "nf4"):
            from diffusers import BitsAndBytesConfig
            if quant == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            print(f"[runner] loading QwenImageEditPipeline with {quant} quantisation…", flush=True)
        else:
            print("[runner] loading QwenImageEditPipeline (bf16)…", flush=True)

        pipe = QwenImageEditPipeline.from_pretrained(self.HF_REPO, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= 80:
                    pipe.to("cuda")
                else:
                    # Sequential offload streams individual layers to GPU one at
                    # a time — lower peak VRAM than model_cpu_offload.
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()
        # int8 / nf4: bnb already placed weights on GPU during from_pretrained
        print("[runner] ready", flush=True)
        self._pipe = pipe

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import secrets
        import torch

        if self._pipe is None:
            raise RuntimeError("Runner not loaded")
        self._cancel = False
        loras_loaded = self._load_loras(loras or [])

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        ref_path = params.get("ref_image") or params.get("ref")
        if not ref_path:
            raise ValueError("`ref_image` is required for Qwen-Image-Edit")

        rp = Path(ref_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        try:
            ref_img = self.load_image(rp).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 30))
        cfg    = float(params.get("cfg", 4.0))
        width  = int(params.get("width",  ref_img.width))
        height = int(params.get("height", ref_img.height))
        if seed < 0:
            seed = secrets.randbits(31)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)
        preview_path = WORKSPACE / "assets" / f".preview_{self.model_id}.jpg"

        runner = self
        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            print(f"[gen] step {step + 1}/{steps}", flush=True)
            if step % 5 == 0 and "latents" in callback_kwargs:
                try:
                    lat = callback_kwargs["latents"]
                    with torch.no_grad():
                        img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
                    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
                    from PIL import Image as _PIL
                    _PIL.fromarray((img * 255).astype("uint8")).save(preview_path, "JPEG", quality=80)
                except Exception:
                    pass
            return callback_kwargs

        try:
            result = self._pipe(
                image=ref_img,
                prompt=prompt,
                negative_prompt=(params.get("negative_prompt") or "").strip() or None,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                width=width,
                height=height,
                generator=gen,
                callback_on_step_end=_on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            if loras_loaded:
                self._clear_loras()

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        out_image = result.images[0]
        from backend import moderator
        if moderator.is_flagged(out_image):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(prefix=f"{self.model_id}_{seed}")
        on_disk = self.save_image(out_image, out_path, format="PNG")
        return self.asset_response([on_disk], meta={
            "model":  self.model_id,
            "prompt": prompt,
            "ref":    ref_path,
            "seed":   seed,            # ← actual seed used (always concrete)
            "steps":  steps,
            "cfg":    cfg,
        })

    def cancel(self) -> None:
        self._cancel = True
