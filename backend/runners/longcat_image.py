"""LongCat-Image text-to-image runner.

Repo:     meituan-longcat/LongCat-Image
Pipeline: diffusers.LongCatImagePipeline
Defaults: 50 steps, guidance_scale=4.0, cfg renorm + prompt rewrite enabled.
Licence: Apache 2.0.
"""

from __future__ import annotations

from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview


class Runner(RunnerBase):
    model_id = "longcat-image"
    model_name = "LongCat-Image"
    category = "image"
    supports_lora = True
    min_vram_gb = 17
    recommended_vram_gb = 32

    HF_REPO = "meituan-longcat/LongCat-Image"
    DEFAULT_STEPS = 50
    DEFAULT_CFG = 4.0
    GPU_DIRECT_THRESHOLD_GB = 36

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import LongCatImagePipeline
        except ImportError as e:
            raise RuntimeError(
                "LongCatImagePipeline is missing. Install diffusers from source or a release with LongCat support."
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
            print(f"[runner] loading LongCatImagePipeline with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading LongCatImagePipeline (bf16)...", flush=True)

        pipe = LongCatImagePipeline.from_pretrained(self.HF_REPO, **kwargs)

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

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

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

        if seed < 0:
            seed = secrets.randbits(31)

        gen = torch.Generator("cpu").manual_seed(seed)
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
            height=height,
            width=width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=gen,
            enable_cfg_renorm=True,
            enable_prompt_rewrite=True,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        try:
            attempts = [
                call_kwargs,
                {k: v for k, v in call_kwargs.items() if not k.startswith("callback_on_step_end")},
                {
                    k: v for k, v in call_kwargs.items()
                    if not k.startswith("callback_on_step_end")
                    and k not in {"enable_cfg_renorm", "enable_prompt_rewrite"}
                },
            ]
            last_error = None
            for attempt in attempts:
                try:
                    result = self._pipe(**attempt)
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
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
        })

    def cancel(self) -> None:
        self._cancel = True
