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

from .base import Runner as RunnerBase


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

        pipe = QwenImagePipeline.from_pretrained(
            self.HF_REPO,
            torch_dtype=torch.bfloat16,
            token=token,
        )

        if torch.cuda.is_available():
            pipe.to("cuda")
            # Auto-offload on smaller cards so loads still succeed.
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb < 36:
                    pipe.enable_model_cpu_offload()
            except Exception:
                pass

        self._pipe = pipe

    # ── Inference ────────────────────────────────────────────────────
    def generate(self, params: dict, loras: Optional[list[str]] = None) -> dict:
        import secrets
        import torch

        if self._pipe is None:
            raise RuntimeError("Runner not loaded")
        self._cancel = False

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        negative = (params.get("negative_prompt") or "").strip() or None

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 50))
        cfg    = float(params.get("cfg", 4.0))
        width  = int(params.get("width",  1328))
        height = int(params.get("height", 1328))

        # Pre-pick a concrete seed so we can report it back to the UI even when
        # the user asked for "random". 31-bit so it fits in a JS number safely.
        if seed < 0:
            seed = secrets.randbits(31)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)

        runner = self
        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            return callback_kwargs

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            width=width,
            height=height,
            generator=gen,
            callback_on_step_end=_on_step,
        )

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        out_path = self.new_output_path(prefix=f"{self.model_id}_{seed}")
        # save_image() encrypts when FORGE_DATA_KEY is set in the env (the
        # backend injects it on spawn). Returned path may be `<out>.enc`.
        on_disk = self.save_image(result.images[0], out_path, format="PNG")
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
