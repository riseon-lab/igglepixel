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

import os
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview
from .diffusers_quant import (
    pipeline_bnb_quantization_config,
    pipeline_torchao_fp8_quantization_config,
    pipeline_torchao_int8_quantization_config,
    seed_torch_for_pipeline,
    torchao_fp8_component_quantization_config,
    torchao_int8_component_quantization_config,
)


class Runner(RunnerBase):
    model_id            = "qwen-image"
    model_name          = "Qwen-Image"
    category            = "image"
    supports_lora       = True
    min_vram_gb         = 14
    recommended_vram_gb = 47

    HF_REPO = "Qwen/Qwen-Image"

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False   # toggled by cancel(); checked in callback

    # ── Lifecycle ────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        from diffusers import QwenImagePipeline

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()
        if quant in ("fp8wo", "fp8-weight-only", "fp8_weight_only"):
            quant = "fp8"
        elif quant in ("fp8dyn", "fp8-dynamic", "fp8_dynamic"):
            quant = "fp8"
            os.environ["FORGE_QWEN_FP8_DYNAMIC"] = "1"

        # Build quantisation config when requested. By default Qwen quantizes
        # the DiT transformer; the 2512 runtime profile can opt into the
        # Qwen2.5-VL text encoder too. Quantized paths can apply VAE
        # tiling/slicing because that pressure is a decode-time memory spike
        # more than a weight-footprint issue.
        kwargs = {"torch_dtype": torch.bfloat16, "token": token}
        quant_backend = ""
        quant_is_torchao = False
        if quant == "int8":
            backend = os.environ.get("FORGE_QWEN_INT8_BACKEND", "torchao").strip().lower()
            if backend in ("torchao", "ao"):
                components = self._torchao_quant_components("int8")
                kwargs["quantization_config"] = pipeline_torchao_int8_quantization_config(components_to_quantize=components)
                quant_backend = f"torchao-int8 ({'+'.join(components)})"
                quant_is_torchao = True
                print("[runner] loading QwenImagePipeline with int8 TorchAO quantisation…", flush=True)
            else:
                kwargs["quantization_config"] = pipeline_bnb_quantization_config(quant, torch)
                quant_backend = "bitsandbytes-int8"
                print("[runner] loading QwenImagePipeline with int8 bitsandbytes quantisation…", flush=True)
        elif quant == "fp8":
            dynamic_fp8 = os.environ.get("FORGE_QWEN_FP8_DYNAMIC", "0").strip().lower() in ("1", "true", "yes", "on")
            components = self._torchao_quant_components("fp8")
            try:
                kwargs["quantization_config"] = pipeline_torchao_fp8_quantization_config(torch, components_to_quantize=components, dynamic=dynamic_fp8)
                quant_backend = f"{'torchao-fp8-dynamic' if dynamic_fp8 else 'torchao-fp8-weight-only'} ({'+'.join(components)})"
            except RuntimeError as e:
                fallback = os.environ.get("FORGE_QWEN_FP8_FALLBACK", "int8").strip().lower()
                if fallback not in ("int8", "torchao-int8"):
                    raise
                kwargs["quantization_config"] = pipeline_torchao_int8_quantization_config(components_to_quantize=components)
                quant = "int8"
                quant_backend = f"torchao-int8 ({'+'.join(components)})"
                print(f"[runner] WARN: FP8 unavailable ({e}); falling back to TorchAO INT8", flush=True)
            quant_is_torchao = True
            print(f"[runner] loading QwenImagePipeline with {quant_backend} component quantisation…", flush=True)
        elif quant == "nf4":
            kwargs["quantization_config"] = pipeline_bnb_quantization_config(quant, torch)
            quant_backend = "bitsandbytes-nf4"
            print("[runner] loading QwenImagePipeline with nf4 bitsandbytes quantisation…", flush=True)
        else:
            quant = "bf16"
            print("[runner] loading QwenImagePipeline (bf16)…", flush=True)

        try:
            pipe = QwenImagePipeline.from_pretrained(self.HF_REPO, **kwargs)
        except Exception as e:
            if not quant_is_torchao:
                raise
            print(
                f"[runner] WARN: pipeline-level {quant_backend} quantization failed "
                f"({type(e).__name__}: {e}); retrying with explicit transformer load",
                flush=True,
            )
            pipe = self._load_with_quantized_transformer(QwenImagePipeline, torch, token, quant, kwargs)

        if quant_is_torchao:
            self._apply_vae_memory_policy(pipe)

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
        elif torch.cuda.is_available() and quant_is_torchao:
            try:
                pipe.to("cuda")
            except Exception as e:
                print(f"[runner] WARN: could not move quantized Qwen pipeline to CUDA ({type(e).__name__}: {e}); enabling CPU offload", flush=True)
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe.enable_sequential_cpu_offload()
        # bnb int8 / nf4: diffusers/bnb handles quantized module placement.
        print("[runner] ready", flush=True)

        self._pipe = pipe

    @staticmethod
    def _torchao_quant_components(quant: str) -> list[str]:
        env_key = "FORGE_QWEN_FP8_COMPONENTS" if quant == "fp8" else "FORGE_QWEN_INT8_COMPONENTS"
        raw = os.environ.get(env_key) or os.environ.get("FORGE_QWEN_QUANT_COMPONENTS") or "transformer"
        requested = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
        allowed = {"transformer", "text_encoder"}
        components = []
        ignored = []
        for name in requested:
            if name in allowed:
                if name not in components:
                    components.append(name)
            else:
                ignored.append(name)
        if ignored:
            print(f"[runner] WARN: ignoring unsupported Qwen quant component(s): {', '.join(ignored)}", flush=True)
        return components or ["transformer"]

    @staticmethod
    def _apply_vae_memory_policy(pipe) -> None:
        policy = os.environ.get("FORGE_QWEN_VAE_MEMORY", "tiling").strip().lower()
        if policy in ("", "0", "off", "none", "false", "no"):
            return
        want_slicing = policy in ("slice", "slicing", "sliced", "both")
        want_tiling = policy in ("1", "true", "yes", "on", "auto", "tile", "tiling", "tiled", "both")
        if want_slicing and hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
                print("[runner] enabled Qwen VAE slicing", flush=True)
            except Exception as e:
                print(f"[runner] WARN: Qwen VAE slicing unavailable ({type(e).__name__}: {e})", flush=True)
        if want_tiling and hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
                print("[runner] enabled Qwen VAE tiling", flush=True)
            except Exception as e:
                print(f"[runner] WARN: Qwen VAE tiling unavailable ({type(e).__name__}: {e})", flush=True)

    def _load_with_quantized_transformer(self, pipeline_cls, torch, token, quant: str, base_kwargs: dict):
        from diffusers import QwenImageTransformer2DModel

        dynamic_fp8 = os.environ.get("FORGE_QWEN_FP8_DYNAMIC", "0").strip().lower() in ("1", "true", "yes", "on")
        if quant == "fp8":
            qcfg = torchao_fp8_component_quantization_config(torch, dynamic=dynamic_fp8)
        else:
            qcfg = torchao_int8_component_quantization_config()

        transformer = QwenImageTransformer2DModel.from_pretrained(
            self.HF_REPO,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            token=token,
            quantization_config=qcfg,
        )
        kwargs = {k: v for k, v in base_kwargs.items() if k != "quantization_config"}
        kwargs["transformer"] = transformer
        return pipeline_cls.from_pretrained(self.HF_REPO, **kwargs)

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
        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", 25))
        cfg    = float(params.get("cfg", 4.0))
        width  = int(params.get("width",  1024))
        height = int(params.get("height", 1024))
        negative = (params.get("negative_prompt") or "").strip()
        if not negative and cfg > 1:
            negative = " "

        if seed < 0:
            seed = secrets.randbits(31)

        seed_torch_for_pipeline(torch, seed)
        preview_path = WORKSPACE / "assets" / f".preview_{self.model_id}.jpg"

        runner = self
        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            print(f"[gen] step {step + 1}/{steps}", flush=True)
            if step % 5 == 0 and "latents" in callback_kwargs:
                save_latent_preview(pipe, callback_kwargs["latents"], height, width, preview_path)
            return callback_kwargs

        try:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                width=width,
                height=height,
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
