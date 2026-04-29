"""Qwen-Image-Edit 2511 runner via the official Diffusers pipeline.

Repo:     Qwen/Qwen-Image-Edit-2511
Pipeline: diffusers.QwenImageEditPlusPipeline

This is the stable path for 2511. It loads the matching transformer, VAE,
text encoder, tokenizer, and scheduler from the official repo instead of
mixing a Comfy split transformer with the older Qwen-Image-Edit stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Runner(RunnerBase):
    model_id            = "qwen-image-edit-2511"
    model_name          = "Qwen-Image-Edit 2511"
    category            = "image"
    supports_lora       = True
    min_vram_gb         = 14
    recommended_vram_gb = 47

    HF_REPO = "Qwen/Qwen-Image-Edit-2511"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    def load(self) -> None:
        import torch

        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as e:
            raise RuntimeError(
                "Qwen-Image-Edit 2511 requires diffusers with "
                "QwenImageEditPlusPipeline support. Update the runtime to "
                "diffusers>=0.37.1 or the latest Hugging Face diffusers build."
            ) from e

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
            print(f"[runner] loading QwenImageEditPlusPipeline 2511 with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading QwenImageEditPlusPipeline 2511 (bf16)...", flush=True)

        pipe = QwenImageEditPlusPipeline.from_pretrained(self.HF_REPO, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= 80:
                    pipe.to("cuda")
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()

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
            raise ValueError("`ref_image` is required for Qwen-Image-Edit 2511")

        ref_images = [self._load_ref_image(ref_path)]
        extra_refs = params.get("ref_images") or []
        if isinstance(extra_refs, str):
            extra_refs = [extra_refs]
        for extra in extra_refs:
            if extra and extra != ref_path:
                ref_images.append(self._load_ref_image(extra))
        pipe_image = ref_images if len(ref_images) > 1 else ref_images[0]

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 40))
        cfg    = float(params.get("cfg", 4.0))
        width  = int(params.get("width",  ref_images[0].width))
        height = int(params.get("height", ref_images[0].height))
        negative = (params.get("negative_prompt") or "").strip()
        if not negative and cfg > 1:
            negative = " "
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
                image=pipe_image,
                prompt=prompt,
                negative_prompt=negative or None,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                guidance_scale=1.0,
                width=width,
                height=height,
                generator=gen,
                num_images_per_prompt=1,
                callback_on_step_end=_on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            if loras_loaded:
                self._clear_loras()

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        out_image = result.images[0]
        out_image = self._upscale_if_requested(out_image, params)
        from backend import moderator
        if moderator.is_flagged(out_image):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(prefix=f"{self.model_id}_{seed}")
        on_disk = self.save_image(out_image, out_path, format="PNG")
        return self.asset_response([on_disk], meta={
            "model":  self.model_id,
            "prompt": prompt,
            "ref":    ref_path,
            "seed":   seed,
            "steps":  steps,
            "cfg":    cfg,
            "width":  width,
            "height": height,
        })

    def _load_ref_image(self, ref_path: str):
        rp = Path(ref_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        try:
            return self.load_image(rp).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

    def cancel(self) -> None:
        self._cancel = True
