"""FLUX.1-dev runner — Black Forest Labs' open-weight rectified flow transformer.

Repo:        black-forest-labs/FLUX.1-dev   (~24 GB at bf16; gated, needs HF_TOKEN)
Pipeline:    diffusers.FluxPipeline
CFG knob:    guidance_scale (FLUX-specific guidance, default ~3.5;
             this is NOT classifier-free guidance — true CFG is folded
             into training. ~3.5 is the recommended value.)
Native:      1024×1024 sweet spot. Higher is supported but quality varies.
Licence:     FLUX.1 [dev] Non-Commercial — outputs are free to use, but
             redistributing the model itself for commercial use is restricted.

Imports of torch/diffusers happen inside `load()` so the host can boot and
report `loading=true` on /healthz before paying the cold-start cost.
"""

from __future__ import annotations

from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Runner(RunnerBase):
    model_id            = "flux-dev"
    model_name          = "FLUX.1-dev"
    category            = "image"
    supports_lora       = True
    min_vram_gb         = 12
    recommended_vram_gb = 24

    HF_REPO = "black-forest-labs/FLUX.1-dev"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False

    # ── Lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        from diffusers import FluxPipeline

        import os
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
            print(f"[runner] loading FluxPipeline with {quant} quantisation…", flush=True)
        else:
            print("[runner] loading FluxPipeline (bf16)…", flush=True)

        pipe = FluxPipeline.from_pretrained(self.HF_REPO, **kwargs)

        if torch.cuda.is_available() and quant == "bf16":
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= 36:
                    pipe.to("cuda")
                else:
                    # Below ~36 GB the activations + UNet won't both fit
                    # comfortably; offload layer-by-layer.
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()
        # int8 / nf4: bnb auto-placed on GPU during from_pretrained
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
        # FLUX-dev doesn't use a negative prompt — guidance is single-sided.

        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 28))
        cfg    = float(params.get("cfg", 3.5))     # mapped to guidance_scale
        width  = int(params.get("width",  1024))
        height = int(params.get("height", 1024))

        # Pre-pick a concrete seed so the UI can report what was actually used.
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
            # Step preview — FLUX latents are packed and need unpacking
            # before VAE decode. Wrapped in try/except so any incompatibility
            # falls back to no preview without breaking generation.
            if step % 5 == 0 and "latents" in callback_kwargs:
                try:
                    lat = callback_kwargs["latents"]
                    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
                    lat = pipe._unpack_latents(lat, height, width, vae_scale)
                    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                    with torch.no_grad():
                        img = pipe.vae.decode(lat).sample
                    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
                    from PIL import Image as _PIL
                    _PIL.fromarray((img * 255).astype("uint8")).save(preview_path, "JPEG", quality=80)
                except Exception:
                    pass
            return callback_kwargs

        try:
            result = self._pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
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
        on_disk = self.save_image(out_image, out_path, format="PNG")
        return self.asset_response([on_disk], meta={
            "model":  self.model_id,
            "prompt": prompt,
            "seed":   seed,
            "steps":  steps,
            "cfg":    cfg,
            "width":  width,
            "height": height,
        })

    # ── Cancellation ─────────────────────────────────────────────────
    def cancel(self) -> None:
        """Set a flag the inference callback reads at the next step boundary."""
        self._cancel = True
