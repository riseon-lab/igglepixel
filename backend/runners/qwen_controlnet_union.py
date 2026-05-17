"""InstantX Qwen-Image ControlNet Union runner.

Repo:     InstantX/Qwen-Image-ControlNet-Union
Base:     Qwen/Qwen-Image
Pipeline: diffusers.QwenImageControlNetPipeline
Controls: uploaded/preprocessed control image, plus optional canny extraction.
Licence: Apache 2.0 for the ControlNet weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview
from .diffusers_quant import pipeline_bnb_quantization_config


class Runner(RunnerBase):
    model_id = "qwen-controlnet-union"
    model_name = "Qwen ControlNet Union"
    category = "image"
    supports_lora = True
    min_vram_gb = 16
    recommended_vram_gb = 47

    BASE_REPO = "Qwen/Qwen-Image"
    CONTROLNET_REPO = "InstantX/Qwen-Image-ControlNet-Union"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import QwenImageControlNetModel, QwenImageControlNetPipeline
        except ImportError as e:
            raise RuntimeError(
                "QwenImageControlNetPipeline is missing. Install diffusers from source or a release with InstantX Qwen ControlNet support."
            ) from e

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()

        print("[runner] loading Qwen ControlNet Union...", flush=True)
        controlnet = QwenImageControlNetModel.from_pretrained(
            self.CONTROLNET_REPO,
            torch_dtype=torch.bfloat16,
            token=token,
        )

        kwargs = {"controlnet": controlnet, "torch_dtype": torch.bfloat16, "token": token}
        if quant in ("int8", "nf4"):
            kwargs["quantization_config"] = pipeline_bnb_quantization_config(quant, torch)
            print(f"[runner] loading Qwen base with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading Qwen base (bf16)...", flush=True)

        pipe = QwenImageControlNetPipeline.from_pretrained(self.BASE_REPO, **kwargs)

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

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    def _load_control_image(self, params: dict):
        control_path = params.get("control_image") or params.get("control") or params.get("ref_image")
        if not control_path:
            raise ValueError("A control image is required")
        rp = Path(control_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        try:
            return self.load_image(rp).convert("RGB"), control_path
        except FileNotFoundError:
            raise FileNotFoundError(f"Control image not found: {control_path}")

    @staticmethod
    def _resize(image, width: int, height: int):
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        return image.resize((width, height), resampling)

    @staticmethod
    def _canny(image):
        try:
            import cv2
            import numpy as np
            from PIL import Image

            arr = np.asarray(image.convert("RGB"))
            edges = cv2.Canny(arr, 100, 200)
            edges = np.stack([edges, edges, edges], axis=-1)
            return Image.fromarray(edges)
        except Exception:
            from PIL import ImageFilter, ImageOps

            # Fallback is less precise than cv2.Canny but still produces a
            # usable edge-like conditioning map without adding a hard runtime dep.
            return ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES).convert("RGB")

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

        control_image, control_path = self._load_control_image(params)
        width = self._round_to_multiple(int(params.get("width", control_image.width)))
        height = self._round_to_multiple(int(params.get("height", control_image.height)))
        if control_image.size != (width, height):
            control_image = self._resize(control_image, width, height)

        control_mode = str(params.get("control_mode") or "preprocessed").lower()
        if control_mode == "canny":
            control_image = self._canny(control_image)

        seed = int(params.get("seed", -1))
        steps = int(params.get("steps", 30))
        cfg = float(params.get("cfg", 4.0))
        control_strength = float(params.get("control_strength", 1.0))
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
                save_latent_preview(pipe, callback_kwargs["latents"], height, width, preview_path)
            return callback_kwargs

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative or None,
            control_image=control_image,
            controlnet_conditioning_scale=control_strength,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        try:
            try:
                result = self._pipe(**call_kwargs)
            except TypeError:
                call_kwargs.pop("callback_on_step_end", None)
                call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
                result = self._pipe(**call_kwargs)
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
            "control": control_path,
            "control_mode": control_mode,
            "control_strength": control_strength,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
        })

    def cancel(self) -> None:
        self._cancel = True
