"""LTX-2.3 image-to-video runner.

Lightricks' LTX-2.3 22B checkpoint is not a normal diffusers runner in this
app. It uses the official LTX-2 pipeline packages in an isolated runtime
profile so their Torch and package pins do not disturb the shared runners.
"""

from __future__ import annotations

import gc
import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, _data_key


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


VARIANTS = {
    "distilled-fp8": {
        "repo": "Lightricks/LTX-2.3-fp8",
        "weight": "ltx-2.3-22b-distilled-fp8.safetensors",
        "default_steps": 8,
        "default_cfg": 1.0,
        "pipeline": "distilled",
        "prequantized_fp8": True,
    },
    "custom-fp8": {
        "repo": "Lightricks/LTX-2.3-fp8",
        "weight": "ltx-2.3-22b-distilled-fp8.safetensors",
        "default_steps": 8,
        "default_cfg": 1.0,
        "pipeline": "distilled",
        "prequantized_fp8": True,
        "custom_weight": True,
    },
    "distilled-1.1": {
        "weight": "ltx-2.3-22b-distilled-1.1.safetensors",
        "default_steps": 8,
        "default_cfg": 1.0,
        "pipeline": "distilled",
    },
    "distilled": {
        "weight": "ltx-2.3-22b-distilled.safetensors",
        "default_steps": 8,
        "default_cfg": 1.0,
        "pipeline": "distilled",
    },
    "dev": {
        "weight": "ltx-2.3-22b-dev.safetensors",
        "default_steps": 30,
        "default_cfg": 3.0,
        "pipeline": "two_stage",
    },
}

HF_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DEFAULT_IMAGE_CRF = 33
LTX_TMP_DIR = WORKSPACE / "tmp" / "ltx23"
PRELOAD_COMPONENTS = os.environ.get("FORGE_LTX_PRELOAD_COMPONENTS", "0").lower() in ("1", "true", "yes")
LTX_OFFLOAD_MODE = os.environ.get("FORGE_LTX_OFFLOAD_MODE", "auto").strip().lower()
LTX_QUANTIZATION = os.environ.get("FORGE_LTX_QUANTIZATION", "auto").strip().lower()
# FP8 runtime cast is cheap and effective even on small cards — the whole
# point of FP8 is "make the model fit". Default minimum was 80 GB which
# meant the cast never fired below an H100. Drop to 24 GB so consumer
# cards with the BF16 variant can opt in.
LTX_FP8_MIN_GB = float(os.environ.get("FORGE_LTX_FP8_MIN_GB", "24"))
# CPU offload was forced on for anything under 141 GB (H200/B100 only).
# Lightricks recommends 48 GB minimum; 40 GB is a safe floor that keeps
# everything 48-96 GB resident in VRAM. xformers attention is the real
# memory unlock — see _patch_xformers_attention.
LTX_CPU_OFFLOAD_BELOW_GB = float(os.environ.get("FORGE_LTX_CPU_OFFLOAD_BELOW_GB", "40"))
# Force CPU offload on the prequantized-FP8 variants regardless of VRAM.
# The ltx-pipelines docs are explicit: `--offload` and `--quantization` are
# mutually exclusive, and "prequantized" already means the weights are FP8
# on disk so the runtime quantization policy is None anyway. So offloading
# Gemma + idle components for FP8 variants is pure win — Gemma BF16 (~24 GB)
# was sitting resident the whole session. Default on; env-overridable.
LTX_FORCE_CPU_OFFLOAD_FP8 = os.environ.get("FORGE_LTX_FORCE_CPU_OFFLOAD_FP8", "1").lower() not in ("0", "false", "no")
# Quantize the Gemma text encoder via bitsandbytes when loading. Comfy uses
# a Q4 GGUF (~6 GB); the closest we can get inside ltx-pipelines without
# a GGUF loader is bitsandbytes 8-bit (~12 GB) or NF4 (~7 GB).
#
# DEFAULT OFF — experimental. The ltx-pipelines text-encoder factory
# returns a wrapper with a custom `.encode()` method; replacing it with
# a raw transformers model crashes with `AttributeError: 'Gemma3...'
# object has no attribute 'encode'` at the first forward pass. Until we
# either build a matching wrapper or reach into the wrapper's internal
# model attribute, leave this off. 'int8'/'nf4' for opt-in testing.
LTX_GEMMA_QUANT = os.environ.get("FORGE_LTX_GEMMA_QUANT", "off").strip().lower()
# VAE tile size for the video VAE decode. ltx-pipelines' default is
# unverified-but-large; smaller tiles cap activation spikes during decode
# at the cost of marginal extra compute. 256 is conservative; set to 0
# (or 'default') to fall back to TilingConfig.default().
LTX_VAE_TILE_SIZE = int(os.environ.get("FORGE_LTX_VAE_TILE_SIZE", "256") or "0")
# Skip the spatial upscaler entirely for the distilled-only path. The
# upscaler is a separate ~1-2 GB sub-pipeline that's only used when the
# pipeline opts into 2× upsampling. Default-off keeps it loaded for
# back-compat; set to 1 to drop it on distilled variants.
LTX_SKIP_UPSCALER = os.environ.get("FORGE_LTX_SKIP_UPSCALER", "0").lower() in ("1", "true", "yes")


def _normalise_lora_set(loras) -> tuple:
    if not loras:
        return ()
    out = []
    for entry in loras:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("filename") or entry.get("file") or entry.get("path")
        if not fn:
            continue
        out.append((Path(fn).name, float(entry.get("strength", 1.0))))
    return tuple(sorted(out))


class Runner(RunnerBase):
    model_id = "ltx23"
    model_name = "LTX-2.3"
    category = "video"
    supports_lora = True
    min_vram_gb = 80
    recommended_vram_gb = 141
    requires_ref = True

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False
        self._variant: Optional[str] = None
        self._loaded_lora_key: tuple = ()
        self._loaded_weight_key: Optional[tuple] = None
        self._runtime_variant_cfg: Optional[dict] = None

    def _build_pipeline(self, lora_set: tuple):
        self._disable_torch_compile()
        from huggingface_hub import hf_hub_download
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        self._patch_xformers_attention()

        token = os.environ.get("HF_TOKEN")
        variant_cfg = self._variant_config()
        weight_repo = variant_cfg.get("repo", HF_REPO)
        weight_name = variant_cfg["weight"]
        pipeline_name = variant_cfg.get("pipeline", "two_stage")

        quantization, offload_mode, policy_label = self._resolve_memory_policy(variant_cfg)
        if variant_cfg.get("prequantized_fp8"):
            quantization = None
            policy_label = f"{policy_label}, prequantized_fp8"

        print(f"[runner] resolving LTX-2.3 weights ({weight_repo}/{weight_name})...", flush=True)
        weight_path = hf_hub_download(repo_id=weight_repo, filename=weight_name, token=token)
        # Spatial upscaler: skip when explicitly disabled on distilled-only
        # runs. The upscaler is a separate ~1-2 GB sub-pipeline only used
        # for 2× output upsampling; the distilled-fp8 path can run without
        # it. Two-stage variants always need it.
        upscaler_path = None
        if not (LTX_SKIP_UPSCALER and pipeline_name == "distilled"):
            upscaler_path = hf_hub_download(repo_id=HF_REPO, filename=SPATIAL_UPSCALER, token=token)
        else:
            print("[runner] LTX spatial upscaler skipped (FORGE_LTX_SKIP_UPSCALER=1)", flush=True)
        gemma_root = self._resolve_gemma_root(token)

        loras_dir = Path(os.environ.get("LORAS_DIR", str(WORKSPACE / "loras")))
        ltx_loras = []
        for filename, strength in lora_set:
            p = loras_dir / filename
            if not p.exists():
                matches = list(loras_dir.rglob(filename))
                if not matches:
                    print(f"[runner] WARN: LoRA not found, skipping: {filename}", flush=True)
                    continue
                p = matches[0]
            ltx_loras.append(LoraPathStrengthAndSDOps(str(p), float(strength), LTXV_LORA_COMFY_RENAMING_MAP))

        print(f"[runner] building LTX pipeline (mode={pipeline_name}, variant={self._variant}, loras={len(ltx_loras)}, memory={policy_label})", flush=True)

        def construct_pipeline(q_policy, o_mode):
            common_extra = self._pipeline_memory_kwargs(pipeline_cls, q_policy, o_mode)
            # Pass spatial_upsampler_path only if we have one. Distilled
            # variants tolerate omission when LTX_SKIP_UPSCALER is on;
            # two-stage variants always have a path.
            upscaler_kwargs = {"spatial_upsampler_path": str(upscaler_path)} if upscaler_path else {}
            if pipeline_name == "distilled":
                return pipeline_cls(
                    distilled_checkpoint_path=str(weight_path),
                    gemma_root=str(gemma_root),
                    loras=ltx_loras,
                    **upscaler_kwargs,
                    **common_extra,
                )
            return pipeline_cls(
                checkpoint_path=str(weight_path),
                distilled_lora=distilled_lora,
                gemma_root=str(gemma_root),
                loras=ltx_loras,
                **upscaler_kwargs,
                **common_extra,
            )

        distilled_lora = None
        if pipeline_name == "distilled":
            from ltx_pipelines.distilled import DistilledPipeline
            pipeline_cls = DistilledPipeline
        else:
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

            distilled_lora_path = hf_hub_download(repo_id=HF_REPO, filename=DISTILLED_LORA, token=token)
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    str(distilled_lora_path),
                    0.6,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
            pipeline_cls = TI2VidTwoStagesPipeline

        self._log_cuda_memory("before LTX pipeline build")
        try:
            pipe = construct_pipeline(quantization, offload_mode)
        except Exception as e:
            if LTX_QUANTIZATION == "auto" and quantization is not None:
                print(
                    f"[runner] WARN: LTX FP8 policy failed during load ({type(e).__name__}: {e}); retrying with CPU offload and BF16",
                    flush=True,
                )
                self._cleanup_memory()
                fallback_offload = self._offload_mode("cpu")
                pipe = construct_pipeline(None, fallback_offload)
                policy_label = "offload=cpu, quantization=none (fallback)"
            elif self._is_cuda_oom(e):
                self._cleanup_memory()
                self._log_cuda_memory("after LTX load OOM cleanup")
                raise RuntimeError(
                    "LTX ran out of VRAM while loading the pipeline. This can look like zero VRAM was used "
                    "afterward because CUDA releases memory when the runner fails. Leave FORGE_LTX_OFFLOAD_MODE=auto "
                    "or set FORGE_LTX_OFFLOAD_MODE=cpu for 80-96GB cards; use a shorter/lower-resolution test first."
                ) from e
            else:
                raise

        # Try to swap Gemma for a bitsandbytes-quantized version before any
        # other component is loaded. The default `gemma-3-12b-it-qat-q4_0-
        # unquantized` is 24 GB BF16 despite the misleading "q4_0" in its
        # repo name — replacing it with int8/nf4 frees 12-18 GB of VRAM
        # that previously sat resident the whole session. Failure here is
        # non-fatal: we keep the default Gemma and just log the reason.
        self._maybe_inject_quantized_gemma(pipe, gemma_root)

        if PRELOAD_COMPONENTS:
            self._preload_pipeline_components(pipe)
        else:
            print("[runner] LTX component preload disabled; lazy loading leaves VRAM headroom for generation", flush=True)
        self._pipe = pipe
        self._loaded_lora_key = lora_set
        self._loaded_weight_key = self._weight_key(variant_cfg)
        print("[runner] ready", flush=True)

    @staticmethod
    def _maybe_inject_quantized_gemma(pipe, gemma_root: Path) -> None:
        """Replace the pipeline's Gemma factory with a bitsandbytes-quantized
        instance so the text encoder costs ~6-12 GB instead of ~24 GB.

        ComfyUI sidesteps this by running the text encoder once and unloading
        it before the transformer loads. ltx-pipelines doesn't do that out
        of the box — the text encoder factory under `pipe.model_ledger` is
        invoked lazily, but the model it returns stays resident afterwards.
        We pre-build the quantized model and swap the factory so the
        pipeline picks it up on first use.

        Fails closed: any error here logs a warning and leaves the default
        Gemma path untouched.
        """
        if LTX_GEMMA_QUANT in ("", "off", "none", "0", "false"):
            print(f"[runner] LTX Gemma quantization disabled (FORGE_LTX_GEMMA_QUANT={LTX_GEMMA_QUANT!r})", flush=True)
            return

        ledger = getattr(pipe, "model_ledger", None)
        if ledger is None:
            print("[runner] LTX pipe has no model_ledger; cannot inject quantized Gemma", flush=True)
            return
        factory_name = None
        for candidate in ("text_encoder", "gemma_embeddings_processor", "gemma"):
            if callable(getattr(ledger, candidate, None)):
                factory_name = candidate
                break
        if factory_name is None:
            print("[runner] LTX model_ledger exposes no text-encoder factory; cannot inject quantized Gemma", flush=True)
            return

        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
        except Exception as e:
            print(f"[runner] WARN: quantized Gemma skipped (import failed: {type(e).__name__}: {e})", flush=True)
            return

        choice = LTX_GEMMA_QUANT
        if choice == "nf4":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            label = "nf4 (~7 GB)"
        elif choice in ("int8", "8bit", "fp8"):
            # fp8 here means "8-bit via bitsandbytes" — distinct from the
            # transformer's own torch FP8 path.
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
            label = "int8 (~12 GB)"
        else:
            print(f"[runner] WARN: unknown FORGE_LTX_GEMMA_QUANT={choice!r}; falling back to default Gemma", flush=True)
            return

        try:
            # Pick the right model class. Gemma 3 is multimodal; the text-only
            # path uses `Gemma3ForConditionalGeneration` or its language head.
            # AutoModelForCausalLM resolves to the correct class via config.
            print(f"[runner] loading Gemma {label} from {gemma_root}...", flush=True)
            cfg = AutoConfig.from_pretrained(str(gemma_root))
            quantized = AutoModelForCausalLM.from_pretrained(
                str(gemma_root),
                config=cfg,
                quantization_config=qcfg,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            quantized.eval()
        except Exception as e:
            print(f"[runner] WARN: quantized Gemma load failed ({type(e).__name__}: {e}); keeping default Gemma", flush=True)
            return

        try:
            setattr(ledger, factory_name, lambda obj=quantized: obj)
            print(f"[runner] LTX Gemma swapped → {label} (replaced ledger.{factory_name})", flush=True)
        except Exception as e:
            print(f"[runner] WARN: ledger factory replacement failed ({type(e).__name__}: {e}); default Gemma still in use", flush=True)

    def _variant_config(self) -> dict:
        if self._runtime_variant_cfg:
            return self._runtime_variant_cfg
        if self._variant and self._variant in VARIANTS:
            return VARIANTS[self._variant]
        return VARIANTS["distilled-fp8"]

    def _apply_runtime_variant_params(self, params: dict) -> tuple:
        variant_cfg = dict(VARIANTS[self._variant])
        if variant_cfg.get("custom_weight"):
            repo = str(params.get("ltx_weight_repo") or variant_cfg.get("repo") or HF_REPO).strip()
            weight = str(params.get("ltx_weight_name") or variant_cfg["weight"]).strip()
            pipeline = str(params.get("ltx_weight_pipeline") or variant_cfg.get("pipeline", "distilled")).strip().lower()
            prequantized_fp8 = self._bool_param(
                params.get("ltx_weight_prequantized_fp8"),
                bool(variant_cfg.get("prequantized_fp8")),
            )
            if not repo:
                raise ValueError("`ltx_weight_repo` is required for the custom LTX weight variant")
            if not weight:
                raise ValueError("`ltx_weight_name` is required for the custom LTX weight variant")
            if pipeline not in ("distilled", "two_stage"):
                raise ValueError("`ltx_weight_pipeline` must be `distilled` or `two_stage`")
            variant_cfg.update(
                repo=repo,
                weight=weight,
                pipeline=pipeline,
                prequantized_fp8=prequantized_fp8,
            )
        self._runtime_variant_cfg = variant_cfg
        return self._weight_key(variant_cfg)

    @staticmethod
    def _weight_key(variant_cfg: dict) -> tuple:
        return (
            variant_cfg.get("repo", HF_REPO),
            variant_cfg["weight"],
            variant_cfg.get("pipeline", "two_stage"),
            bool(variant_cfg.get("prequantized_fp8")),
        )

    @staticmethod
    def _bool_param(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _pipeline_memory_kwargs(pipeline_cls, quantization, offload_mode) -> dict:
        params = inspect.signature(pipeline_cls.__init__).parameters
        out = {}
        if quantization is not None and "quantization" in params:
            out["quantization"] = quantization
        if offload_mode is not None and "offload_mode" in params:
            out["offload_mode"] = offload_mode
        return out

    @classmethod
    def _resolve_memory_policy(cls, variant_cfg: Optional[dict] = None):
        vram = cls._gpu_vram_gb()
        offload_choice = LTX_OFFLOAD_MODE
        quant_choice = LTX_QUANTIZATION
        is_prequantized_fp8 = bool((variant_cfg or {}).get("prequantized_fp8"))

        if offload_choice == "auto":
            # For prequantized-FP8 variants, ltx-pipelines docs are explicit
            # that offload + runtime FP8 cast are mutually exclusive. The
            # weights are *already* FP8 on disk so the runtime cast is moot —
            # we get pure win from offloading Gemma + idle components to RAM.
            # Default-on; FORGE_LTX_FORCE_CPU_OFFLOAD_FP8=0 disables.
            if is_prequantized_fp8 and LTX_FORCE_CPU_OFFLOAD_FP8:
                offload_choice = "cpu"
            else:
                offload_choice = "cpu" if vram and vram < LTX_CPU_OFFLOAD_BELOW_GB else "none"
        if quant_choice == "auto":
            quant_choice = "fp8-cast" if offload_choice == "none" and (not vram or vram >= LTX_FP8_MIN_GB) else "none"

        offload_mode = cls._offload_mode(offload_choice)
        quantization = cls._quantization_policy(quant_choice)
        return quantization, offload_mode, f"offload={offload_choice}, quantization={quant_choice}, vram={vram or 'unknown'}GB"

    @staticmethod
    def _offload_mode(choice: str):
        if choice in ("", "0", "false", "no"):
            choice = "none"
        if choice not in ("none", "cpu", "disk"):
            print(f"[runner] WARN: unknown FORGE_LTX_OFFLOAD_MODE={choice!r}; using none", flush=True)
            choice = "none"
        try:
            from ltx_pipelines.utils.types import OffloadMode
        except Exception as e:
            if choice != "none":
                print(f"[runner] WARN: LTX offload mode unavailable ({type(e).__name__}: {e}); continuing without offload", flush=True)
            return None
        for candidate in (choice, choice.upper(), choice.replace("-", "_").upper()):
            try:
                return OffloadMode(candidate)
            except Exception:
                pass
            if hasattr(OffloadMode, candidate):
                return getattr(OffloadMode, candidate)
        for member in OffloadMode:
            if choice in (member.name.lower(), str(member.value).lower()):
                return member
        if choice != "none":
            print(f"[runner] WARN: LTX offload mode {choice!r} not recognised by installed ltx-pipelines; continuing without offload", flush=True)
        return None

    @staticmethod
    def _quantization_policy(choice: str):
        if choice in ("", "0", "false", "no", "none"):
            return None
        try:
            from ltx_core.quantization import QuantizationPolicy
        except Exception as e:
            print(f"[runner] WARN: LTX quantization unavailable ({type(e).__name__}: {e}); continuing without quantization", flush=True)
            return None
        if choice in ("fp8", "fp8-cast", "fp8_cast"):
            return QuantizationPolicy.fp8_cast()
        if choice in ("fp8-scaled-mm", "fp8_scaled_mm"):
            return QuantizationPolicy.fp8_scaled_mm()
        print(f"[runner] WARN: unknown FORGE_LTX_QUANTIZATION={choice!r}; continuing without quantization", flush=True)
        return None

    @staticmethod
    def _gpu_vram_gb() -> Optional[float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / 1024**3, 1)
        except Exception:
            return None

    @staticmethod
    def _preload_pipeline_components(pipe) -> None:
        ledger = getattr(pipe, "model_ledger", None)
        if ledger is None:
            return
        cached = {}
        component_names = (
            "transformer",
            "video_encoder",
            "video_decoder",
            "spatial_upsampler",
            "text_encoder",
            "gemma_embeddings_processor",
        )
        for name in component_names:
            factory = getattr(ledger, name, None)
            if not callable(factory):
                continue
            print(f"[runner] preloading LTX component: {name}", flush=True)
            cached[name] = factory()
        for name, instance in cached.items():
            setattr(ledger, name, lambda obj=instance: obj)
        if cached:
            print(f"[runner] preloaded LTX components: {', '.join(cached)}", flush=True)

    def load(self) -> None:
        variant = os.environ.get("FORGE_VARIANT", "distilled-fp8").lower()
        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to distilled-fp8", flush=True)
            variant = "distilled-fp8"
        self._variant = variant
        self._runtime_variant_cfg = dict(VARIANTS[variant])
        if self._runtime_variant_cfg.get("custom_weight"):
            print("[runner] custom LTX weight selected; delaying pipeline load until generation params arrive", flush=True)
            return
        self._build_pipeline(())

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import secrets

        self._cancel = False

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        ref_path = params.get("ref_image") or params.get("ref")
        if self.requires_ref and not ref_path:
            raise ValueError("`ref_image` is required for LTX-2.3 i2v")

        wanted = _normalise_lora_set(loras)
        weight_key = self._apply_runtime_variant_params(params)
        if self._pipe is None or wanted != self._loaded_lora_key or weight_key != self._loaded_weight_key:
            if self._pipe is None:
                print("[runner] building LTX pipeline for current request...", flush=True)
            if wanted != self._loaded_lora_key:
                print(f"[runner] LoRA set changed (was {len(self._loaded_lora_key)}, now {len(wanted)}); rebuilding pipeline...", flush=True)
            if weight_key != self._loaded_weight_key:
                print("[runner] LTX weight selection changed; rebuilding pipeline...", flush=True)
            try:
                self._cleanup_memory()
                self._pipe = None
                self._build_pipeline(wanted)
            except Exception:
                self._loaded_lora_key = ()
                self._loaded_weight_key = None
                raise

        variant_cfg = self._variant_config()
        seed = int(params.get("seed", -1))
        steps = int(params.get("steps", variant_cfg["default_steps"]))
        cfg = float(params.get("cfg", variant_cfg["default_cfg"]))
        width = self._align_dimension(int(params.get("width", 1024)))
        height = self._align_dimension(int(params.get("height", 576)))
        fps = max(1, int(params.get("fps", 18)))
        duration = float(params.get("duration", 3.0))
        frames = self._frames_from_duration(duration, fps)
        if seed < 0:
            seed = secrets.randbits(31)

        ref_tmp = None
        if ref_path:
            ref_visible = Path(ref_path)
            if not ref_visible.is_absolute():
                ref_visible = WORKSPACE / ref_visible
            ref_tmp = self._decrypt_ref_to_temp(ref_visible)

        out_tmp = self._temp_file(".mp4")
        stripped = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number  # noqa: F401 (TilingConfig used via _make_tiling_config)
            from ltx_pipelines.utils.args import ImageConditioningInput
            from ltx_pipelines.utils.media_io import encode_video

            print(f"[runner] LTX-2.3 generate (variant={self._variant}, steps={steps}, cfg={cfg}, {width}x{height}@{fps}fps, frames={frames}, seed={seed})", flush=True)
            self._log_cuda_memory("before LTX pipeline")
            self._cleanup_memory()
            self._progress(1, 10, "prepared inputs")
            tiling_config = self._make_tiling_config()
            common_kwargs = dict(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=float(fps),
                images=([ImageConditioningInput(str(ref_tmp), 0, 1.0, DEFAULT_IMAGE_CRF)] if ref_tmp else []),
                tiling_config=tiling_config,
            )
            self._progress(2, 10, "running pipeline")
            try:
                if variant_cfg.get("pipeline") == "distilled":
                    video, audio = self._pipe(**common_kwargs)
                else:
                    video, audio = self._pipe(
                        **common_kwargs,
                        negative_prompt=(params.get("negative_prompt") or "").strip() or "",
                        num_inference_steps=steps,
                        video_guider_params=MultiModalGuiderParams(
                            cfg_scale=cfg,
                            stg_scale=float(params.get("stg", 1.0)),
                            rescale_scale=float(params.get("rescale", 0.7)),
                            modality_scale=float(params.get("modality_scale", 3.0)),
                            skip_step=0,
                            stg_blocks=[28],
                        ),
                        audio_guider_params=MultiModalGuiderParams(
                            cfg_scale=float(params.get("audio_cfg", 7.0)),
                            stg_scale=1.0,
                            rescale_scale=0.7,
                            modality_scale=3.0,
                            skip_step=0,
                            stg_blocks=[28],
                        ),
                    )
            except Exception as e:
                if self._is_cuda_oom(e):
                    self._cleanup_memory()
                    self._log_cuda_memory("after LTX OOM cleanup")
                    raise RuntimeError(
                        "LTX ran out of VRAM. Try 1024x576, a shorter duration, or lower FPS. "
                        "The higher-resolution presets can exceed an 80GB card because the 22B model, "
                        "Gemma text stack, VAE/upscaler, and video activations overlap during generation."
                    ) from e
                raise
            self._log_cuda_memory("after LTX pipeline")
            self._progress(8, 10, "encoding video")
            encode_video(
                video=video,
                fps=float(fps),
                audio=audio,
                output_path=str(out_tmp),
                video_chunks_number=get_video_chunks_number(frames, tiling_config),
            )
            self._progress(9, 10, "saving output")

            if self._cancel:
                return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

            from backend import moderator

            try:
                import imageio.v3 as iio
                from PIL import Image

                meta = iio.immeta(str(out_tmp), plugin="pyav")
                total_frames = int(meta.get("nframes") or frames)
                middle_idx = max(0, total_frames // 2)
                frame_arr = iio.imread(str(out_tmp), index=middle_idx, plugin="pyav")
                if moderator.is_flagged(Image.fromarray(frame_arr)):
                    return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})
            except Exception as e:
                print(f"[runner] WARN: moderation frame extract failed ({type(e).__name__}: {e}); skipping mod", flush=True)

            stripped = self._strip_audio(out_tmp)
            out_path = self.new_output_path(ext="mp4", prefix=f"{self.model_id}_{seed}")
            on_disk = self._encrypt_video_to_assets(stripped, out_path)
            self._progress(10, 10, "done")
        finally:
            for p in (ref_tmp, out_tmp, stripped):
                try:
                    if p and p.exists():
                        p.unlink()
                except OSError:
                    pass

        return self.asset_response([on_disk], meta={
            "model": self.model_id,
            "variant": self._variant,
            "prompt": prompt,
            "ref": ref_path,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "frames": frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": round(frames / fps, 2),
            "weight_repo": variant_cfg.get("repo", HF_REPO),
            "weight_name": variant_cfg["weight"],
            "loras": [{"filename": fn, "strength": s} for fn, s in self._loaded_lora_key],
        })

    def cancel(self) -> None:
        self._cancel = True

    @staticmethod
    def _progress(step: int, total: int, label: str) -> None:
        print(f"[gen] step {step} / {total} {label}", flush=True)

    @staticmethod
    def _make_tiling_config():
        """Build the VAE TilingConfig used at decode time.

        The default `TilingConfig.default()` is unverified-but-large; that
        contributes to high VRAM spikes during VAE decode of long video.
        When FORGE_LTX_VAE_TILE_SIZE is set, we try to construct a tighter
        config by inspecting the dataclass's accepted params and setting
        any tile_size-shaped knob we can find. Failure is non-fatal —
        we always fall back to TilingConfig.default().
        """
        from ltx_core.model.video_vae import TilingConfig
        if LTX_VAE_TILE_SIZE <= 0:
            return TilingConfig.default()
        try:
            sig = inspect.signature(TilingConfig.__init__)
            params = sig.parameters
            kwargs = {}
            for tile_arg in ("tile_size", "spatial_tile_size", "tile_height", "tile_width"):
                if tile_arg in params:
                    kwargs[tile_arg] = LTX_VAE_TILE_SIZE
            if not kwargs:
                # Some versions expose it as a class-level setter rather than a
                # constructor arg — try setting attributes on the default cfg.
                cfg = TilingConfig.default()
                applied = False
                for attr in ("tile_size", "spatial_tile_size", "tile_height", "tile_width"):
                    if hasattr(cfg, attr):
                        try:
                            setattr(cfg, attr, LTX_VAE_TILE_SIZE)
                            applied = True
                        except Exception:
                            pass
                if applied:
                    print(f"[runner] LTX VAE tiling: patched default tile_size={LTX_VAE_TILE_SIZE}", flush=True)
                    return cfg
                print("[runner] WARN: TilingConfig accepts no tile_size kwarg; falling back to default", flush=True)
                return TilingConfig.default()
            print(f"[runner] LTX VAE tiling: {kwargs}", flush=True)
            return TilingConfig(**kwargs)
        except Exception as e:
            print(f"[runner] WARN: custom TilingConfig failed ({type(e).__name__}: {e}); using default", flush=True)
            return TilingConfig.default()

    @staticmethod
    def _disable_torch_compile() -> None:
        try:
            import torch

            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
        except Exception:
            pass

    @staticmethod
    def _patch_xformers_attention() -> None:
        try:
            from ltx_core.model.transformer import attention as attn_mod
            from xformers.ops import memory_efficient_attention

            attn_mod.memory_efficient_attention = memory_efficient_attention
            print("[runner] patched LTX attention with xformers", flush=True)
        except Exception as e:
            print(f"[runner] xformers attention patch skipped: {type(e).__name__}: {e}", flush=True)

    @staticmethod
    def _cleanup_memory() -> None:
        try:
            from ltx_pipelines.utils.helpers import cleanup_memory

            cleanup_memory()
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _is_cuda_oom(e: Exception) -> bool:
        if e.__class__.__name__ == "OutOfMemoryError":
            return True
        return "cuda out of memory" in str(e).lower()

    @staticmethod
    def _log_cuda_memory(tag: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free, total = torch.cuda.mem_get_info()
            print(
                f"[runner] VRAM {tag}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB free={free / 1024**3:.2f}GiB total={total / 1024**3:.2f}GiB",
                flush=True,
            )
        except Exception:
            pass

    @staticmethod
    def _resolve_gemma_root(token: Optional[str]) -> Path:
        from huggingface_hub import snapshot_download

        override = os.environ.get("FORGE_LTX_GEMMA_ROOT") or os.environ.get("LTX_GEMMA_ROOT")
        if override:
            p = Path(override).expanduser()
            if p.exists():
                return p
            raise FileNotFoundError(f"LTX Gemma root does not exist: {p}")
        return Path(snapshot_download(repo_id=GEMMA_REPO, token=token))

    @staticmethod
    def _temp_file(suffix: str) -> Path:
        LTX_TMP_DIR.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkstemp(suffix=suffix, dir=str(LTX_TMP_DIR))[1])

    @staticmethod
    def _frames_from_duration(duration: float, fps: int) -> int:
        raw = max(9, int(round(max(0.1, duration) * fps)))
        return ((raw - 1 + 7) // 8) * 8 + 1

    @staticmethod
    def _align_dimension(value: int) -> int:
        return max(64, int(round(value / 64)) * 64)

    @staticmethod
    def _decrypt_ref_to_temp(visible: Path) -> Path:
        key = _data_key()
        if key:
            import backend.crypto as fcrypto

            data = fcrypto.read_decrypted(key, visible)
        else:
            if not visible.exists():
                raise FileNotFoundError(visible)
            data = visible.read_bytes()
        suffix = visible.suffix or ".png"
        LTX_TMP_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(LTX_TMP_DIR))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return Path(tmp)

    @staticmethod
    def _strip_audio(src: Path) -> Path:
        if not shutil.which("ffmpeg"):
            return src
        out = src.with_suffix(".silent.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-c:v", "copy", "-an", str(out)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return out
        except subprocess.CalledProcessError as e:
            print(f"[runner] WARN: ffmpeg -an failed ({e.stderr.decode('utf-8', errors='replace')[-200:]}); using original mp4", flush=True)
            return src

    @staticmethod
    def _encrypt_video_to_assets(src: Path, out_visible: Path) -> Path:
        plaintext = src.read_bytes()
        out_visible.parent.mkdir(parents=True, exist_ok=True)
        key = _data_key()
        if key:
            import backend.crypto as fcrypto

            return fcrypto.write_encrypted(key, out_visible, plaintext)
        out_visible.write_bytes(plaintext)
        return out_visible
