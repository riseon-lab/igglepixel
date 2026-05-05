"""LongCat-Image-Edit-Turbo runner.

Repo:     meituan-longcat/LongCat-Image-Edit-Turbo
Pipeline: diffusers.LongCatImageEditPipeline
Defaults: 8 steps, guidance_scale=1.0.
Licence: Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview


class Runner(RunnerBase):
    model_id = "longcat-image-edit-turbo"
    model_name = "LongCat-Image-Edit Turbo"
    category = "image"
    supports_lora = True
    min_vram_gb = 18
    recommended_vram_gb = 32

    HF_REPO = "meituan-longcat/LongCat-Image-Edit-Turbo"
    DEFAULT_STEPS = 8
    DEFAULT_CFG = 1.0
    GPU_DIRECT_THRESHOLD_GB = 36

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import LongCatImageEditPipeline
        except ImportError as e:
            raise RuntimeError(
                "LongCatImageEditPipeline is missing. Install diffusers from source or a release with LongCat support."
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
            print(f"[runner] loading LongCatImageEditPipeline with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading LongCatImageEditPipeline (bf16)...", flush=True)

        pipe = LongCatImageEditPipeline.from_pretrained(self.HF_REPO, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= self.GPU_DIRECT_THRESHOLD_GB:
                    pipe.to("cuda", torch.bfloat16)
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

    def _load_reference(self, ref_path: str):
        rp = Path(ref_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        try:
            return self.load_image(rp).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

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
            raise ValueError("`ref_image` is required for LongCat-Image-Edit Turbo")

        ref_img = self._load_reference(ref_path)
        seed = int(params.get("seed", -1))
        steps = int(params.get("steps", self.DEFAULT_STEPS))
        cfg = float(params.get("cfg", self.DEFAULT_CFG))
        negative = (params.get("negative_prompt") or "").strip()

        if seed < 0:
            seed = secrets.randbits(31)

        gen = torch.Generator("cpu").manual_seed(seed)
        width = int(params.get("width", ref_img.width))
        height = int(params.get("height", ref_img.height))
        preview_path = WORKSPACE / "assets" / f".preview_{self.model_id}.jpg"
        runner = self

        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            print(f"[gen] step {step + 1}/{steps}", flush=True)
            if step % 2 == 0 and "latents" in callback_kwargs:
                save_latent_preview(pipe, callback_kwargs["latents"], height, width, preview_path)
            return callback_kwargs

        call_kwargs = dict(
            negative_prompt=negative,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        try:
            attempts = [
                call_kwargs,
                {k: v for k, v in call_kwargs.items() if not k.startswith("callback_on_step_end")},
            ]
            last_error = None
            for attempt in attempts:
                try:
                    result = self._pipe(ref_img, prompt, **attempt)
                    break
                except TypeError as e:
                    last_error = e
            else:
                raise last_error
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
            "ref": ref_path,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
        })

    def cancel(self) -> None:
        self._cancel = True
