"""Wan 2.2 image-to-video runner.

Three size variants from the same registry entry:
  - 14B Lightning (base 14B + fused Lightx2v LoRA) — fast 8-step path
  - 14B (Wan-AI/Wan2.2-I2V-A14B-Diffusers)       — best quality, needs ~80 GB BF16
  - 5B  (Wan-AI/Wan2.2-TI2V-5B-Diffusers)        — consumer, ~27 GB BF16

The launcher passes FORGE_VARIANT (14b-lightning | 14b | 5b) and FORGE_QUANT
(bf16 | int8 | nf4) as env vars; this runner reads them and picks the right
HF repo + quant config.

Outputs an mp4 written to assets/generated/. Uses diffusers.export_to_video
(ffmpeg-backed) so the only requirement is that ffmpeg lives on the image,
which it already does.

Lightning support is exposed as a baked variant: the runner loads the
Lightx2v 480p LoRA into both Wan experts, fuses it at load, then unloads the
adapter bookkeeping so jobs behave like normal Wan generations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


# Maps variant id → diffusers HF repo/config. Mirrored from model_registry.json
# so the runner doesn't have to read the registry; the launcher passes
# FORGE_VARIANT and we look it up here.
VARIANTS = {
    "14b-lightning": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "lightning": True,
        "default_steps": 8,
    },
    "14b": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    },
    "5b": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    },
}

LIGHTNING_REPO = "Kijai/WanVideo_comfy"
LIGHTNING_WEIGHT = "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"


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
        self._lightning_baked = False

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
        variant_cfg = VARIANTS[variant]
        repo = variant_cfg["repo"]
        lightning = bool(variant_cfg.get("lightning"))
        if lightning and quant != "bf16":
            raise RuntimeError("Wan Lightning baked variant currently requires BF16. Select BF16 or the standard 14B/5B variant.")

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
        if lightning:
            self._bake_lightning_lora(pipe)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                # 14B at bf16 needs ~80 GB; 5B at bf16 needs ~27 GB. Anything
                # below comfortable headroom gets sequential offload.
                threshold = 80 if variant.startswith("14b") else 36
                if total_gb >= threshold:
                    pipe.to("cuda")
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()
        # int8/nf4: bnb auto-placed on GPU during from_pretrained
        print("[runner] ready", flush=True)
        self._pipe = pipe

    def _bake_lightning_lora(self, pipe) -> None:
        """Fuse Lightx2v's Wan 2.2 I2V 14B Lightning LoRA once at load time.

        The LoRA file contains weights for both experts. Diffusers needs it
        loaded once into the high-noise transformer and once into transformer_2,
        then fused with different scales. After fusion, runtime jobs behave like
        a normal model and do not need to carry a LoRA payload.
        """
        print("[runner] baking Wan Lightning LoRA into 14B pipeline…", flush=True)
        high_adapter = "lightx2v_high"
        low_adapter = "lightx2v_low"
        try:
            pipe.load_lora_weights(
                LIGHTNING_REPO,
                weight_name=LIGHTNING_WEIGHT,
                adapter_name=high_adapter,
            )
            pipe.load_lora_weights(
                LIGHTNING_REPO,
                weight_name=LIGHTNING_WEIGHT,
                adapter_name=low_adapter,
                load_into_transformer_2=True,
            )
            pipe.set_adapters([high_adapter, low_adapter], adapter_weights=[1.0, 1.0])
            pipe.fuse_lora(adapter_names=[high_adapter], lora_scale=3.0, components=["transformer"])
            pipe.fuse_lora(adapter_names=[low_adapter], lora_scale=1.0, components=["transformer_2"])
            pipe.unload_lora_weights()
            self._lightning_baked = True
            print("[runner] Wan Lightning LoRA fused (high=3.0, low=1.0)", flush=True)
        except Exception as e:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            raise RuntimeError(f"Failed to bake Wan Lightning LoRA: {type(e).__name__}: {e}") from e

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
        default_steps = int(VARIANTS.get(self._variant, {}).get("default_steps", 8)) if self._lightning_baked else 30
        default_cfg = 1.0 if self._lightning_baked else 5.0
        steps   = int(params.get("steps", default_steps))
        cfg     = float(params.get("cfg", default_cfg))
        # Dual-expert (MoE) guidance: high-noise expert uses cfg, low-noise
        # uses cfg_low. If unset, diffusers reuses cfg for both.
        cfg_low = params.get("cfg_low")
        cfg_low = float(cfg_low) if cfg_low is not None else (default_cfg if self._lightning_baked else None)
        width  = int(params.get("width",  832))
        height = int(params.get("height", 480))
        fps    = int(params.get("fps", 16 if self._lightning_baked else 24))
        duration = params.get("duration")
        if duration is not None:
            # Wan works best with frame counts of 4n + 1. Let the UI expose
            # seconds, then resolve to the nearest valid frame count here.
            seconds = max(0.1, float(duration))
            if self._lightning_baked:
                # Lightx2v's 480p distilled recipe clamps the model frames to
                # 8-80 and then adds the initial frame, so 3.5s at 16fps = 57.
                model_frames = max(8, min(80, int(round(seconds * fps))))
                frames = 1 + model_frames
            else:
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
            "lightning": self._lightning_baked,
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
