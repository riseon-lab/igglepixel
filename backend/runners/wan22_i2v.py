"""Wan 2.2 image-to-video runner.

Two size variants from the same registry entry:
  - 14B (Wan-AI/Wan2.2-I2V-A14B-Diffusers)  — best quality, needs ~80 GB BF16
  - 5B  (Wan-AI/Wan2.2-TI2V-5B-Diffusers)   — consumer, ~27 GB BF16

The launcher passes FORGE_VARIANT (14b | 5b) and FORGE_QUANT (bf16 | int8 | nf4)
as env vars; this runner reads them and picks the right HF repo + quant config.

Outputs an mp4 written to assets/generated/. Uses diffusers.export_to_video
(ffmpeg-backed) so the only requirement is that ffmpeg lives on the image,
which it already does.

Lightning LoRA support: the registry's default_loras list includes the
Wan2.2 4-step Lightning LoRA. Selecting it lets the model finish in 4 steps
instead of ~50, with the trade-off being slightly less motion detail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


# Maps variant id → diffusers HF repo. Mirrored from model_registry.json so
# the runner doesn't have to read the registry; the launcher passes
# FORGE_VARIANT and we look it up here.
VARIANTS = {
    "14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "5b":  "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}


class Runner(RunnerBase):
    model_id            = "wan22-i2v"
    model_name          = "Wan 2.2 — Image to Video"
    category            = "video"
    supports_lora       = True
    min_vram_gb         = 16
    recommended_vram_gb = 80

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False
        self._variant = None

    # ── Lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        from diffusers import WanImageToVideoPipeline

        token   = os.environ.get("HF_TOKEN")
        variant = os.environ.get("FORGE_VARIANT", "5b").lower()
        quant   = os.environ.get("FORGE_QUANT",   "bf16").lower()

        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to 5b", flush=True)
            variant = "5b"
        self._variant = variant
        repo = VARIANTS[variant]

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
            print(f"[runner] loading Wan 2.2 {variant} with {quant} quantisation…", flush=True)
        else:
            print(f"[runner] loading Wan 2.2 {variant} (bf16)…", flush=True)

        pipe = WanImageToVideoPipeline.from_pretrained(repo, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                # 14B at bf16 needs ~80 GB; 5B at bf16 needs ~27 GB. Anything
                # below comfortable headroom gets sequential offload.
                threshold = 80 if variant == "14b" else 36
                if total_gb >= threshold:
                    pipe.to("cuda")
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()
        # int8/nf4: bnb auto-placed on GPU during from_pretrained
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
        ref_path = params.get("ref_image") or params.get("ref")
        if not ref_path:
            raise ValueError("`ref_image` is required for Wan i2v")

        rp = Path(ref_path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        try:
            ref_img = self.load_image(rp).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        seed    = int(params.get("seed", -1))
        steps   = int(params.get("steps", 30))
        cfg     = float(params.get("cfg", 5.0))
        # Dual-expert (MoE) guidance: high-noise expert uses cfg, low-noise
        # uses cfg_low. If unset, diffusers reuses cfg for both.
        cfg_low = params.get("cfg_low")
        cfg_low = float(cfg_low) if cfg_low is not None else None
        width  = int(params.get("width",  832))
        height = int(params.get("height", 480))
        fps    = int(params.get("fps", 24))
        duration = params.get("duration")
        if duration is not None:
            # Wan works best with frame counts of 4n + 1. Let the UI expose
            # seconds, then resolve to the nearest valid frame count here.
            seconds = max(0.1, float(duration))
            raw_frames = max(16, min(121, int(round(seconds * fps))))
            frames = max(17, min(121, ((raw_frames - 1) // 4) * 4 + 1))
        else:
            frames = int(params.get("num_frames", 81))
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
            image=ref_img,
            prompt=prompt,
            negative_prompt=(params.get("negative_prompt") or "").strip() or None,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_frames=frames,
            width=width,
            height=height,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if cfg_low is not None:
            pipe_kwargs["guidance_scale_2"] = cfg_low
        try:
            result = self._pipe(**pipe_kwargs)
        finally:
            if loras_loaded:
                self._clear_loras()

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        # Wan/diffusers can return either [batch][frame] PIL lists or array-like
        # tensors. Normalize before moderation/export so array truth checks do
        # not explode and ffmpeg always receives RGB PIL frames.
        raw_frames = result.frames if hasattr(result, "frames") else result.images
        frames_out = self._normalize_video_frames(raw_frames)
        from backend import moderator
        if moderator.is_video_flagged(frames_out):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(ext="mp4", prefix=f"{self.model_id}_{seed}")
        on_disk = self.save_video(frames_out, out_path, fps=fps)
        return self.asset_response([on_disk], meta={
            "model":    self.model_id,
            "variant":  self._variant,
            "prompt":   prompt,
            "ref":      ref_path,
            "seed":     seed,
            "steps":    steps,
            "cfg":      cfg,
            "frames":   frames,
            "fps":      fps,
            "width":    width,
            "height":   height,
            "duration": round(frames / fps, 2),
        })

    # ── Cancellation ─────────────────────────────────────────────────
    def cancel(self) -> None:
        self._cancel = True
