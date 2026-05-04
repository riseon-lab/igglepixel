"""Wan 2.2 text-to-video runner.

Mirrors the Wan I2V runner, but uses Diffusers' WanPipeline and does not
require a source image. Variants:
  - 14B Lightning 8 step (T2V-A14B + fused Lightx2v T2V LoRA)
  - 14B Lightning 4 step
  - 14B standard (T2V-A14B)
  - 5B standard (TI2V-5B in text-only mode)
"""

from __future__ import annotations

import os
from typing import Optional

from .base import Runner as RunnerBase


VARIANTS = {
    "14b-lightning": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "lightning": True,
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_cfg_low": 1.0,
    },
    "14b-lightning-4": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "lightning": True,
        "default_steps": 4,
        "default_cfg": 1.0,
        "default_cfg_low": 1.0,
    },
    "14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "default_steps": 40,
        "default_cfg": 4.0,
        "default_cfg_low": 3.0,
    },
    "5b": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "default_steps": 50,
        "default_cfg": 5.0,
        "default_cfg_low": None,
    },
}

LIGHTNING_REPO = "Kijai/WanVideo_comfy"
LIGHTNING_WEIGHT = "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors"
WAN_DIM_MULTIPLE = 16
WAN_MAX_DURATION_SECONDS = 15.0
WAN_LIGHTNING_MAX_FPS = 24
WAN_STANDARD_MAX_FPS = 30


class Runner(RunnerBase):
    model_id = "wan22-t2v"
    model_name = "Wan 2.2 — Text to Video"
    category = "video"
    supports_lora = True
    min_vram_gb = 16
    recommended_vram_gb = 80

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False
        self._variant = None
        self._lightning_baked = False
        self._has_low_expert = False

    def load(self) -> None:
        import torch
        from diffusers import WanPipeline

        token = os.environ.get("HF_TOKEN")
        variant = os.environ.get("FORGE_VARIANT", "5b").lower()
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()

        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to 5b", flush=True)
            variant = "5b"
        self._variant = variant
        variant_cfg = VARIANTS[variant]
        repo = variant_cfg["repo"]
        lightning = bool(variant_cfg.get("lightning"))
        if lightning and quant != "bf16":
            raise RuntimeError("Wan T2V Lightning baked variant currently requires BF16. Select BF16 or the standard 14B/5B variant.")

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
            print(f"[runner] loading Wan 2.2 T2V {variant} with {quant} quantisation...", flush=True)
        else:
            print(f"[runner] loading Wan 2.2 T2V {variant} (bf16)...", flush=True)

        pipe = WanPipeline.from_pretrained(repo, **kwargs)
        self._has_low_expert = getattr(pipe, "transformer_2", None) is not None
        if lightning:
            self._bake_lightning_lora(pipe)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                threshold = 80 if variant.startswith("14b") else 36
                if total_gb >= threshold:
                    pipe.to("cuda")
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()

        print("[runner] ready", flush=True)
        self._pipe = pipe

    def _bake_lightning_lora(self, pipe) -> None:
        print("[runner] baking Wan T2V Lightning LoRA into 14B pipeline...", flush=True)
        high_adapter = "lightx2v_t2v_high"
        low_adapter = "lightx2v_t2v_low"
        try:
            pipe.load_lora_weights(
                LIGHTNING_REPO,
                weight_name=LIGHTNING_WEIGHT,
                adapter_name=high_adapter,
            )
            if getattr(pipe, "transformer_2", None) is not None:
                pipe.load_lora_weights(
                    LIGHTNING_REPO,
                    weight_name=LIGHTNING_WEIGHT,
                    adapter_name=low_adapter,
                    load_into_transformer_2=True,
                )
                pipe.set_adapters([high_adapter, low_adapter], adapter_weights=[1.0, 1.0])
                pipe.fuse_lora(adapter_names=[high_adapter], lora_scale=1.0, components=["transformer"])
                pipe.fuse_lora(adapter_names=[low_adapter], lora_scale=1.0, components=["transformer_2"])
            else:
                pipe.set_adapters([high_adapter], adapter_weights=[1.0])
                pipe.fuse_lora(adapter_names=[high_adapter], lora_scale=1.0)
            pipe.unload_lora_weights()
            self._lightning_baked = True
            print("[runner] Wan T2V Lightning LoRA fused (high=1.0, low=1.0)", flush=True)
        except Exception as e:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            raise RuntimeError(f"Failed to bake Wan T2V Lightning LoRA: {type(e).__name__}: {e}") from e

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = WAN_DIM_MULTIPLE) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    @staticmethod
    def _wan_frame_count(seconds: float, fps: int) -> int:
        raw_frames = max(8, int(round(seconds * fps)))
        return max(9, ((raw_frames - 1 + 3) // 4) * 4 + 1)

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

        variant_cfg = VARIANTS.get(self._variant, VARIANTS["5b"])
        seed = int(params.get("seed", -1))
        steps = int(params.get("steps", variant_cfg["default_steps"]))
        cfg = float(params.get("cfg", variant_cfg["default_cfg"]))
        cfg_low_default = variant_cfg.get("default_cfg_low")
        cfg_low = params.get("cfg_low", cfg_low_default)
        cfg_low = float(cfg_low) if cfg_low is not None else None
        width = self._round_to_multiple(int(params.get("width", 832)))
        height = self._round_to_multiple(int(params.get("height", 480)))
        max_fps = WAN_LIGHTNING_MAX_FPS if self._lightning_baked else WAN_STANDARD_MAX_FPS
        fps = max(1, min(max_fps, int(params.get("fps", 18))))
        duration = params.get("duration")
        requested_seconds = None
        if duration is not None:
            seconds = max(0.1, min(WAN_MAX_DURATION_SECONDS, float(duration)))
            requested_seconds = seconds
            frames = self._wan_frame_count(seconds, fps)
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
        if self._has_low_expert and cfg_low is not None:
            pipe_kwargs["guidance_scale_2"] = cfg_low

        try:
            result = self._pipe(**pipe_kwargs)
        finally:
            if loras_loaded:
                self._clear_loras()

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        raw_frames = result.frames if hasattr(result, "frames") else result.images
        frames_out = self._normalize_video_frames(raw_frames)
        actual_frames = len(frames_out)

        from backend import moderator
        if moderator.is_video_flagged(frames_out):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(ext="mp4", prefix=f"{self.model_id}_{seed}")
        on_disk = self.save_video(frames_out, out_path, fps=fps)
        return self.asset_response([on_disk], meta={
            "model": self.model_id,
            "variant": self._variant,
            "lightning": self._lightning_baked,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "cfg_low": cfg_low,
            "frames": frames,
            "actual_frames": actual_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "requested_duration": round(requested_seconds, 2) if requested_seconds is not None else None,
            "duration": round(actual_frames / fps, 2),
        })

    def cancel(self) -> None:
        self._cancel = True
