"""HunyuanVideo runner — Tencent's flagship open video DiT.

Two pipeline paths from the same registry entry:

  - t2v:  HunyuanVideoPipeline                      (text-to-video)
  - i2v:  HunyuanVideoImageToVideoPipeline          (image-to-video)

The launcher passes FORGE_VARIANT (t2v | i2v) and FORGE_QUANT (bf16 | nf4)
as env vars; this runner reads them and picks the right HF repo + quant
config. NF4 is the default for consumer cards (~24 GB with offload + VAE
tiling); BF16 is the workstation path (~60-80 GB resident).

Diffusers ships HunyuanVideo mainline, so unlike LTX-2.3 this runs on the
shared interpreter — no per-runner venv. Frame counts must follow the
4*k+1 rule per Tencent's reference inference recipe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


VARIANTS = {
    "t2v": {
        "repo":          "hunyuanvideo-community/HunyuanVideo",
        "pipeline":      "t2v",
        "needs_ref":     False,
    },
    "i2v": {
        "repo":          "hunyuanvideo-community/HunyuanVideo-I2V",
        "pipeline":      "i2v",
        "needs_ref":     True,
    },
}

# Hunyuan's reference recipe: num_frames must satisfy `4*k + 1`. Same
# constraint Wan 2.2 uses, so the math helper below mirrors that runner.
HUNYUAN_FRAME_MULTIPLE = 4
HUNYUAN_DIM_MULTIPLE   = 16
HUNYUAN_MAX_FPS        = 30


class Runner(RunnerBase):
    model_id            = "hunyuan-video"
    model_name          = "HunyuanVideo"
    category            = "video"
    supports_lora       = True
    min_vram_gb         = 16   # NF4 + offload with small frames; pad expectations on the registry
    recommended_vram_gb = 80

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False
        self._variant: Optional[str] = None
        self._needs_ref: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        from diffusers import HunyuanVideoPipeline

        token   = os.environ.get("HF_TOKEN")
        variant = os.environ.get("FORGE_VARIANT", "t2v").lower()
        quant   = os.environ.get("FORGE_QUANT",   "nf4").lower()

        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to t2v", flush=True)
            variant = "t2v"
        self._variant   = variant
        variant_cfg     = VARIANTS[variant]
        self._needs_ref = bool(variant_cfg["needs_ref"])
        repo            = variant_cfg["repo"]

        # Pick the right pipeline class based on the variant. The i2v class
        # has a different from_pretrained signature only insofar as it loads
        # the I2V-specific transformer; both expose the same offload/tiling
        # methods we rely on below.
        if variant_cfg["pipeline"] == "i2v":
            from diffusers import HunyuanVideoImageToVideoPipeline as PipelineCls
        else:
            PipelineCls = HunyuanVideoPipeline

        kwargs = {"torch_dtype": torch.bfloat16, "token": token}
        if quant == "nf4":
            # NF4 quantization on the transformer only — text encoder and VAE
            # stay in bf16. Per the diffusers docs, this is the canonical low-
            # VRAM path; pairs naturally with model_cpu_offload + VAE tiling.
            from diffusers.quantizers import PipelineQuantizationConfig
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit":             True,
                    "bnb_4bit_quant_type":      "nf4",
                    "bnb_4bit_compute_dtype":   torch.bfloat16,
                },
                components_to_quantize="transformer",
            )
            print(f"[runner] loading HunyuanVideo {variant} (nf4 transformer + bf16)…", flush=True)
        elif quant != "bf16":
            print(f"[runner] unknown quant '{quant}', falling back to bf16", flush=True)
            quant = "bf16"
            print(f"[runner] loading HunyuanVideo {variant} (bf16)…", flush=True)
        else:
            print(f"[runner] loading HunyuanVideo {variant} (bf16)…", flush=True)

        pipe = PipelineCls.from_pretrained(repo, **kwargs)

        # The transformer is the bulky part (~13B); for BF16 the whole pipe
        # is ~60+ GB resident, so anything below H100 territory needs offload.
        # NF4 always pairs with offload — it's the whole point of the path.
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                # 60 GB headroom rule: above ~70 GB we keep BF16 fully on
                # device for max throughput; below, we offload. NF4 always
                # offloads since the consumer-card story depends on it.
                if quant == "bf16" and total_gb >= 70:
                    pipe.to("cuda")
                else:
                    pipe.enable_model_cpu_offload()
            except Exception:
                pipe.enable_model_cpu_offload()
        # No CUDA → leave on CPU (smoke-test path only; generation will be glacial).

        print("[runner] ready", flush=True)
        self._pipe = pipe

    # ── Inference ────────────────────────────────────────────────────
    @staticmethod
    def _hunyuan_frame_count(seconds: float, fps: int) -> int:
        """Resolve seconds × fps → a Hunyuan-legal frame count (4*k + 1)."""
        raw_frames = max(5, int(round(seconds * fps)))
        return ((raw_frames - 1 + HUNYUAN_FRAME_MULTIPLE - 1) // HUNYUAN_FRAME_MULTIPLE) * HUNYUAN_FRAME_MULTIPLE + 1

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = HUNYUAN_DIM_MULTIPLE) -> int:
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

        # The model exposes a single optional `ref` slot in the UI. i2v
        # requires it; t2v silently ignores anything in there so users can
        # flip variants without clearing the slot first.
        ref_path = params.get("ref_image") or params.get("ref")
        ref_img  = None
        if self._needs_ref:
            if not ref_path:
                raise ValueError("`ref_image` is required for HunyuanVideo i2v")
            rp = Path(ref_path)
            if not rp.is_absolute():
                rp = WORKSPACE / rp
            try:
                ref_img = self.load_image(rp).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Reference image not found: {ref_path}")
        elif ref_path:
            print(f"[runner] t2v variant: ignoring ref_image '{ref_path}'", flush=True)

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 30))
        cfg    = float(params.get("cfg", 6.0))
        width  = self._round_to_multiple(int(params.get("width",  1280)))
        height = self._round_to_multiple(int(params.get("height",  720)))
        fps    = max(1, min(HUNYUAN_MAX_FPS, int(params.get("fps", 15))))
        duration = params.get("duration")
        requested_seconds = None
        if duration is not None:
            seconds = max(0.1, min(15.0, float(duration)))
            requested_seconds = seconds
            frames = self._hunyuan_frame_count(seconds, fps)
        else:
            frames = int(params.get("num_frames", 61))

        # If ref came in at a different aspect, resize to the requested target
        # while keeping multiple-of-16 dims. Hunyuan I2V has no special bounds
        # like Wan Lightning; we just match the requested width/height.
        original_width = original_height = None
        if ref_img is not None:
            from PIL import Image
            original_width, original_height = ref_img.size
            if (original_width, original_height) != (width, height):
                ref_img = ref_img.resize((width, height), Image.LANCZOS)

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
        if ref_img is not None:
            pipe_kwargs["image"] = ref_img

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
            "model":    self.model_id,
            "variant":  self._variant,
            "prompt":   prompt,
            "ref":      ref_path,
            "seed":     seed,
            "steps":    steps,
            "cfg":      cfg,
            "frames":   frames,
            "actual_frames": actual_frames,
            "fps":      fps,
            "width":    width,
            "height":   height,
            "source_width":  original_width,
            "source_height": original_height,
            "requested_duration": round(requested_seconds, 2) if requested_seconds is not None else None,
            "duration": round(actual_frames / fps, 2),
        })

    # ── Cancellation ─────────────────────────────────────────────────
    def cancel(self) -> None:
        self._cancel = True
