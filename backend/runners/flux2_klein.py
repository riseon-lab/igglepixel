"""Shared FLUX.2 [klein] runner implementation.

The concrete runner modules (`flux2_klein_4b.py`, `flux2_klein_9b.py`) only
set metadata and the Hugging Face repo. The generation path is identical:
Diffusers' Flux2KleinPipeline supports text-to-image and optional reference
images in one call.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Flux2KleinRunner(RunnerBase):
    model_id = "flux2-klein"
    model_name = "FLUX.2 [klein]"
    category = "image"
    supports_lora = True
    min_vram_gb = 8
    recommended_vram_gb = 16

    HF_REPO = ""
    DEFAULT_STEPS = 4
    DEFAULT_CFG = 1.0
    GPU_DIRECT_THRESHOLD_GB = 24

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "missing"

    @staticmethod
    def _projection_shape(pipe) -> str:
        transformer = getattr(pipe, "transformer", None)
        for name in (
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.0.attn.to_q",
        ):
            module = transformer
            for part in name.split("."):
                module = getattr(module, part, None) if module is not None else None
                if module is None:
                    break
            weight = getattr(module, "weight", None)
            if weight is not None:
                return f"{name}.weight={tuple(weight.shape)}"
        return "unknown"

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as e:
            raise RuntimeError(
                "Flux2KleinPipeline is missing. Install diffusers>=0.38.0 in the runtime."
            ) from e

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()
        print(
            "[runner] versions "
            f"torch={torch.__version__} "
            f"diffusers={self._package_version('diffusers')} "
            f"peft={self._package_version('peft')} "
            f"torchao={self._package_version('torchao')}",
            flush=True,
        )

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
            print(f"[runner] loading {self.model_name} with {quant} quantisation...", flush=True)
        else:
            print(f"[runner] loading {self.model_name} (bf16)...", flush=True)

        pipe = Flux2KleinPipeline.from_pretrained(self.HF_REPO, **kwargs)
        print(f"[runner] loaded repo={self.HF_REPO} projection={self._projection_shape(pipe)}", flush=True)

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

    def _reference_images(self, params: dict) -> list:
        refs = []
        for key in ("ref_image", "ref2", "ref3", "ref4"):
            ref_path = params.get(key)
            if not ref_path:
                continue
            rp = Path(ref_path)
            if not rp.is_absolute():
                rp = WORKSPACE / rp
            refs.append(self.load_image(rp).convert("RGB"))
        return refs

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
        refs = self._reference_images(params)

        if seed < 0:
            seed = secrets.randbits(31)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)

        runner = self

        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            print(f"[gen] step {step + 1}/{steps}", flush=True)
            return callback_kwargs

        pipe_kwargs = dict(
            prompt=prompt,
            image=refs if refs else None,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        try:
            result = self._pipe(**pipe_kwargs)
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
            "references": len(refs),
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
        })

    def cancel(self) -> None:
        self._cancel = True
