"""Z-Image Turbo runner.

Repo:     Tongyi-MAI/Z-Image-Turbo
Pipeline: diffusers.ZImagePipeline, with optional ZImageImg2ImgPipeline when a
          reference image is supplied.
Defaults: Turbo wants 9 scheduler steps, which corresponds to 8 DiT forwards,
          and guidance_scale=0.0.
Licence: Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview


class Runner(RunnerBase):
    model_id = "z-image-turbo"
    model_name = "Z-Image Turbo"
    category = "image"
    supports_lora = True
    min_vram_gb = 16
    recommended_vram_gb = 24

    HF_REPO = "Tongyi-MAI/Z-Image-Turbo"
    DEFAULT_STEPS = 9
    DEFAULT_CFG = 0.0
    GPU_DIRECT_THRESHOLD_GB = 20

    def __init__(self) -> None:
        self._pipe = None
        self._img2img_pipe = None
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import ZImagePipeline
        except ImportError as e:
            raise RuntimeError(
                "ZImagePipeline is missing. Install a current diffusers release with Z-Image support."
            ) from e

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "token": token,
            "low_cpu_mem_usage": False,
        }
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
            print(f"[runner] loading ZImagePipeline with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading ZImagePipeline (bf16)...", flush=True)

        pipe = ZImagePipeline.from_pretrained(self.HF_REPO, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= self.GPU_DIRECT_THRESHOLD_GB:
                    pipe.to("cuda")
                elif hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_sequential_cpu_offload()

        print("[runner] ready", flush=True)
        self._pipe = pipe

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    def _reference_image(self, params: dict, width: int, height: int):
        ref_path = params.get("ref_image")
        if not ref_path:
            return None
        from PIL import Image

        rp = Path(ref_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        image = self.load_image(rp).convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        return image.resize((width, height), resampling)

    def _img2img(self):
        if self._img2img_pipe is not None:
            return self._img2img_pipe
        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as e:
            raise RuntimeError(
                "ZImageImg2ImgPipeline is missing. Update diffusers before using a Z-Image reference."
            ) from e
        self._img2img_pipe = ZImageImg2ImgPipeline(**self._pipe.components)
        return self._img2img_pipe

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

        seed = int(params.get("seed", -1))
        steps = int(params.get("steps", self.DEFAULT_STEPS))
        cfg = float(params.get("cfg", self.DEFAULT_CFG))
        width = self._round_to_multiple(int(params.get("width", 1024)))
        height = self._round_to_multiple(int(params.get("height", 1024)))
        ref_strength = float(params.get("ref_strength", 0.6))
        ref_strength = max(0.0, min(1.0, ref_strength))
        ref_image = self._reference_image(params, width, height)

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
            if step % 2 == 0 and "latents" in callback_kwargs:
                save_latent_preview(pipe, callback_kwargs["latents"], height, width, preview_path)
            return callback_kwargs

        pipe = self._img2img() if ref_image is not None else self._pipe
        call_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if ref_image is not None:
            call_kwargs["image"] = ref_image
            call_kwargs["strength"] = ref_strength

        try:
            result = pipe(**call_kwargs)
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
            "model": self.model_id,
            "prompt": prompt,
            "reference": bool(ref_image),
            "ref_strength": ref_strength if ref_image is not None else None,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
        })

    def cancel(self) -> None:
        self._cancel = True
