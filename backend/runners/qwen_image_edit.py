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
    min_vram_gb         = 14
    recommended_vram_gb = 47

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

        # Optional split-file component swaps. The launcher passes absolute
        # paths via FORGE_COMPONENT_<TARGET> when the user picked a curated
        # component (e.g. the Comfy-Org 2511 transformer). When set, we load
        # via from_single_file and inject the resulting object into
        # from_pretrained — this lets us run Comfy's exact 2511 weights while
        # keeping diffusers' tokenizer + scheduler config from the base repo.
        # bnb quantisation can't be combined with custom-loaded sub-modules
        # because the BitsAndBytesConfig only applies to weights pulled by
        # from_pretrained, so we ignore swaps when quant != bf16 and warn.
        component_overrides: dict = {}
        if quant == "bf16":
            try:
                from diffusers import (
                    QwenImageTransformer2DModel,
                    AutoencoderKLQwenImage,
                )
                # Some diffusers builds expose AutoencoderKLQwenImage; older ones
                # only have AutoencoderKL. Fall back if the specific class is
                # missing so this never blocks the whole launch.
            except ImportError:
                from diffusers import QwenImageTransformer2DModel  # type: ignore
                from diffusers import AutoencoderKL as AutoencoderKLQwenImage  # type: ignore

            tx_path = os.environ.get("FORGE_COMPONENT_TRANSFORMER")
            if tx_path:
                try:
                    print(f"[runner] loading transformer from split file: {tx_path}", flush=True)
                    component_overrides["transformer"] = QwenImageTransformer2DModel.from_single_file(
                        tx_path, torch_dtype=torch.bfloat16
                    )
                except Exception as e:
                    print(f"[runner] WARN: transformer split-file load failed ({e}); falling back to base repo", flush=True)

            vae_path = os.environ.get("FORGE_COMPONENT_VAE")
            if vae_path:
                try:
                    print(f"[runner] loading VAE from split file: {vae_path}", flush=True)
                    component_overrides["vae"] = AutoencoderKLQwenImage.from_single_file(
                        vae_path, torch_dtype=torch.bfloat16
                    )
                except Exception as e:
                    print(f"[runner] WARN: VAE split-file load failed ({e}); falling back to base repo", flush=True)

            # Text encoder swap is left for a follow-up — it requires picking
            # the right Qwen2.5-VL class + tokenizer pairing, which differs
            # from the standard transformers loader path. Falls through silently.
        elif os.environ.get("FORGE_COMPONENT_TRANSFORMER") or os.environ.get("FORGE_COMPONENT_VAE"):
            print(f"[runner] WARN: component swaps ignored (FORGE_QUANT={quant}; only bf16 supported with custom components)", flush=True)

        kwargs.update(component_overrides)
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
            "seed":   seed,            # ← actual seed used (always concrete)
            "steps":  steps,
            "cfg":    cfg,
        })

    def cancel(self) -> None:
        self._cancel = True
