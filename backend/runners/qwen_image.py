"""Qwen-Image runner — Alibaba's Qwen-Image text-to-image via diffusers.

Repo:        Qwen/Qwen-Image     (~40 GB on first download)
Pipeline:    diffusers.QwenImagePipeline
CFG knob:    true_cfg_scale  (Qwen's actual classifier-free guidance scale —
             `guidance_scale` exists too but is the *distilled* one and is
             usually left at the default).

Imports of torch/diffusers happen inside `load()` so the host can boot and
report `loading=true` on /healthz before paying the cold-start cost.
"""

from __future__ import annotations

from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Runner(RunnerBase):
    model_id            = "qwen-image"
    model_name          = "Qwen-Image"
    category            = "image"
    supports_lora       = False
    min_vram_gb         = 24
    recommended_vram_gb = 48

    HF_REPO = "Qwen/Qwen-Image"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False   # toggled by cancel(); checked in callback

    # ── Lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        from diffusers import QwenImagePipeline

        import os
        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()

        # Build quantisation config when requested. bnb auto-places the
        # quantised model on GPU during from_pretrained, so we skip the
        # post-load .to("cuda") / offload dance below for int8 and nf4.
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
            print(f"[runner] loading QwenImagePipeline with {quant} quantisation…", flush=True)
        else:
            print("[runner] loading QwenImagePipeline (bf16)…", flush=True)

        pipe = QwenImagePipeline.from_pretrained(self.HF_REPO, **kwargs)

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
        # int8 / nf4: bnb already placed weights on GPU during from_pretrained
        print("[runner] ready", flush=True)

        self._pipe = pipe

    # ── Inference ────────────────────────────────────────────────────
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
        negative = (params.get("negative_prompt") or "").strip() or None

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 30))
        cfg    = float(params.get("cfg", 4.0))
        width  = int(params.get("width",  1024))
        height = int(params.get("height", 1024))

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
                prompt=prompt,
                negative_prompt=negative,
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
        # save_image() encrypts when FORGE_DATA_KEY is set in the env (the
        # backend injects it on spawn). Returned path may be `<out>.enc`.
        on_disk = self.save_image(out_image, out_path, format="PNG")
        return self.asset_response([on_disk], meta={
            "model":  self.model_id,
            "prompt": prompt,
            "seed":   seed,            # ← actual seed used (always concrete, never -1)
            "steps":  steps,
            "cfg":    cfg,
            "width":  width,
            "height": height,
        })

    # ── Cancellation ─────────────────────────────────────────────────
    def cancel(self) -> None:
        """Set a flag the inference callback reads at the next step boundary."""
        self._cancel = True
