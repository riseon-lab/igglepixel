"""LTX-2.3 image-to-video runner.

Lightricks' LTX-2.3 (22B params) — best-in-class video quality but workstation
class hardware (~46 GB BF16 weights → ~60-80 GB VRAM realistic). Diffusers
doesn't support 2.3 yet (model card says "coming soon"), so this runner uses
Lightricks' own `ltx-pipelines` package directly.

That package brings its own Torch ~2.7-era stack, which conflicts with our
shared runtime. We therefore run this runner inside a per-runner venv managed
by backend/venv_manager.py — see the `runtime` block on the ltx23 entry in
backend/model_registry.json.

Differences from our diffusers runners worth flagging:

  - Their pipeline takes IMAGE FILE PATHS, not PIL Images. We decrypt the
    user's ref to a tempfile, hand the path over, then unlink in finally.

  - LoRAs are passed at pipeline CONSTRUCTION, not loaded dynamically.
    The runner caches the currently-built LoRA set; if a generate() call
    asks for a different set, we rebuild the pipeline. Acceptable since
    LoRA toggles are rare in a session.

  - Their pipeline returns video/audio tensors rather than writing a file.
    We use their `encode_video()` helper to produce an mp4, strip the audio
    track losslessly, then encrypt the result via crypto.write_encrypted.

License: LTX Community License (non-commercial). The UI surfaces this via
the registry's license pill — operators who run a fork commercially are
on their own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import gc
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, _data_key

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Maps registry-side variant ids → the safetensors filename and an inference
# preset hint (default steps when the user hasn't overridden). The full
# Lightricks/LTX-2.3 repo is ~46 GB per variant; users pick exactly one.
VARIANTS = {
    "distilled-1.1": {
        "weight": "ltx-2.3-22b-distilled-1.1.safetensors",
        "default_steps": 8,
        "default_cfg":   1.0,
        "pipeline": "distilled",
    },
    "distilled": {
        "weight": "ltx-2.3-22b-distilled.safetensors",
        "default_steps": 8,
        "default_cfg":   1.0,
        "pipeline": "distilled",
    },
    "dev": {
        "weight": "ltx-2.3-22b-dev.safetensors",
        "default_steps": 30,
        "default_cfg":   3.0,
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


def _normalise_lora_set(loras) -> tuple:
    """Hashable tuple representation of a LoRA list — used to detect set
    changes between generate() calls so we know whether to rebuild the pipe.
    Sorted so order doesn't matter."""
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
    model_id            = "ltx23"
    model_name          = "LTX-2.3"
    category            = "video"
    supports_lora       = True
    # Honest minimums — no FP8 weights exist on the official repo, so the
    # smallest config still needs ~46 GB on disk and similar VRAM.
    min_vram_gb         = 48
    recommended_vram_gb = 80

    def __init__(self) -> None:
        self._pipe = None
        self._cancel = False
        self._variant: Optional[str] = None
        self._loaded_lora_key: tuple = ()   # tuple form of currently-built LoRA set

    # ── Pipeline construction ────────────────────────────────────────
    def _build_pipeline(self, lora_set: tuple):
        """Construct a fresh LTX pipeline for the chosen variant + LoRA set.
        Lazy imports keep cold start before /healthz cheap and ensure the
        package only resolves once we're inside the venv subprocess.
        """
        self._disable_torch_compile()
        from huggingface_hub import hf_hub_download
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        self._patch_xformers_attention()

        token = os.environ.get("HF_TOKEN")
        variant_cfg = VARIANTS[self._variant]
        weight_name = variant_cfg["weight"]

        print(f"[runner] resolving LTX-2.3 weights ({weight_name})…", flush=True)
        weight_path = hf_hub_download(repo_id=HF_REPO, filename=weight_name, token=token)
        upscaler_path = hf_hub_download(repo_id=HF_REPO, filename=SPATIAL_UPSCALER, token=token)
        gemma_root = self._resolve_gemma_root(token)

        # Resolve LoRA paths from $LORAS_DIR (the launcher exports it).
        loras_dir = Path(os.environ.get("LORAS_DIR", str(WORKSPACE / "loras")))
        ltx_loras = []
        for filename, strength in lora_set:
            p = loras_dir / filename
            if not p.exists():
                # Try the recursive fallback the other runners use.
                matches = list(loras_dir.rglob(filename))
                if not matches:
                    print(f"[runner] WARN: LoRA not found, skipping: {filename}", flush=True)
                    continue
                p = matches[0]
            ltx_loras.append(LoraPathStrengthAndSDOps(str(p), float(strength), LTXV_LORA_COMFY_RENAMING_MAP))

        pipeline_name = variant_cfg.get("pipeline", "two_stage")
        print(f"[runner] building LTX pipeline (mode={pipeline_name}, variant={self._variant}, loras={len(ltx_loras)})", flush=True)
        if pipeline_name == "distilled":
            from ltx_pipelines.distilled import DistilledPipeline
            pipe = DistilledPipeline(
                distilled_checkpoint_path=str(weight_path),
                spatial_upsampler_path=str(upscaler_path),
                gemma_root=str(gemma_root),
                loras=ltx_loras,
            )
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
            pipe = TI2VidTwoStagesPipeline(
                checkpoint_path=str(weight_path),
                distilled_lora=distilled_lora,
                spatial_upsampler_path=str(upscaler_path),
                gemma_root=str(gemma_root),
                loras=ltx_loras,
            )
        if PRELOAD_COMPONENTS:
            self._preload_pipeline_components(pipe)
        else:
            print("[runner] LTX component preload disabled; lazy loading leaves VRAM headroom for generation", flush=True)
        self._pipe = pipe
        self._loaded_lora_key = lora_set
        print("[runner] ready", flush=True)

    @staticmethod
    def _preload_pipeline_components(pipe) -> None:
        """Force LTX's lazy ledger to instantiate its heavy components now.

        Without this, the runner can report "ready" while the first generate
        still has to load the transformer/text/VAE stack. That looks like a
        stuck generation in the UI. Preloading shifts that cost into Start
        Runner, where users expect model loading to happen.
        """
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
        variant = os.environ.get("FORGE_VARIANT", "distilled-1.1").lower()
        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to distilled-1.1", flush=True)
            variant = "distilled-1.1"
        self._variant = variant
        # Build with no LoRAs. First generate() call that asks for any will
        # trigger a rebuild — same code path, just with the LoRA list.
        self._build_pipeline(())

    # ── Inference ────────────────────────────────────────────────────
    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import secrets

        if self._pipe is None:
            raise RuntimeError("Runner not loaded")
        self._cancel = False

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        ref_path = params.get("ref_image") or params.get("ref")
        if not ref_path:
            raise ValueError("`ref_image` is required for LTX-2.3 i2v")

        # If the requested LoRA set differs from what's loaded, rebuild the
        # pipeline. Their package locks LoRAs at construction; this is the
        # only correct way to swap them. Slow (10–30s) but rare.
        wanted = _normalise_lora_set(loras)
        if wanted != self._loaded_lora_key:
            print(f"[runner] LoRA set changed (was {len(self._loaded_lora_key)}, now {len(wanted)}); rebuilding pipeline…", flush=True)
            try:
                self._pipe = None
                self._build_pipeline(wanted)
            except Exception:
                # Failed rebuild — try to reset to no-LoRAs so the next
                # call doesn't re-attempt the same broken set.
                self._loaded_lora_key = ()
                raise

        variant_cfg = VARIANTS[self._variant]
        seed   = int(params.get("seed", -1))
        steps  = int(params.get("steps", variant_cfg["default_steps"]))
        cfg    = float(params.get("cfg",   variant_cfg["default_cfg"]))
        width  = self._align_dimension(int(params.get("width",  1024)))
        height = self._align_dimension(int(params.get("height",  576)))
        fps    = max(1, int(params.get("fps", 18)))
        duration = float(params.get("duration", 3.0))
        # LTX accepts an absolute frame count — keep this consistent with how
        # we expose seconds in the UI for Wan and others.
        frames = self._frames_from_duration(duration, fps)
        if seed < 0:
            seed = secrets.randbits(31)

        # Decrypt the ref to a tempfile so we can hand a path to LTX. Their
        # API does not accept PIL objects.
        ref_visible = Path(ref_path)
        if not ref_visible.is_absolute():
            ref_visible = WORKSPACE / ref_visible
        ref_tmp = self._decrypt_ref_to_temp(ref_visible)

        out_tmp = self._temp_file(".mp4")
        stripped = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            from ltx_pipelines.utils.args import ImageConditioningInput
            from ltx_pipelines.utils.media_io import encode_video

            print(f"[runner] LTX-2.3 generate (variant={self._variant}, steps={steps}, cfg={cfg}, {width}x{height}@{fps}fps, frames={frames}, seed={seed})", flush=True)
            self._log_cuda_memory("before LTX pipeline")
            self._cleanup_memory()
            self._progress(1, 10, "prepared inputs")
            tiling_config = TilingConfig.default()
            common_kwargs = dict(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=float(fps),
                images=[ImageConditioningInput(str(ref_tmp), 0, 1.0, DEFAULT_IMAGE_CRF)],
                tiling_config=tiling_config,
            )
            self._progress(2, 10, "running pipeline")
            try:
                if VARIANTS[self._variant].get("pipeline") == "distilled":
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
                        "The high-resolution presets can exceed a 94GB card in BF16 because the 22B model, "
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

            # Middle-frame moderation: read just one frame, no full re-decode.
            # The decoded mp4 from LTX is small (a few hundred MB at most).
            from backend import moderator
            try:
                import imageio.v3 as iio
                meta = iio.immeta(str(out_tmp), plugin="pyav")
                total_frames = int(meta.get("nframes") or frames)
                middle_idx = max(0, total_frames // 2)
                frame_arr = iio.imread(str(out_tmp), index=middle_idx, plugin="pyav")
                from PIL import Image
                if moderator.is_flagged(Image.fromarray(frame_arr)):
                    return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})
            except Exception as e:
                # If moderation read fails, log and continue rather than block
                # generation — the asset is still encrypted at rest either way.
                print(f"[runner] WARN: moderation frame extract failed ({type(e).__name__}: {e}); skipping mod", flush=True)

            # Strip the audio track losslessly so our /api/assets/file/ path
            # consistently serves silent mp4 (matches our other video runners).
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
            "loras":    [{"filename": fn, "strength": s} for fn, s in self._loaded_lora_key],
        })

    # ── Cancellation ─────────────────────────────────────────────────
    def cancel(self) -> None:
        # Their pipeline doesn't expose a clean per-step cancel today, so we
        # mark the flag and let generate() return an empty asset response
        # after the in-flight call finishes. Users can also kill the runner
        # subprocess from the drawer if they need it stopped immediately.
        self._cancel = True

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _progress(step: int, total: int, label: str) -> None:
        print(f"[gen] step {step} / {total} {label}", flush=True)

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
        # LTX two-stage pipelines require num_frames = (8 * k) + 1.
        raw = max(9, int(round(max(0.1, duration) * fps)))
        return ((raw - 1 + 7) // 8) * 8 + 1

    @staticmethod
    def _align_dimension(value: int) -> int:
        # Two-stage LTX requires dimensions divisible by 64. Round to the
        # closest legal value so typed UI values like 1080 become 1088.
        return max(64, int(round(value / 64)) * 64)

    @staticmethod
    def _decrypt_ref_to_temp(visible: Path) -> Path:
        """Write the plaintext bytes of an encrypted ref to a tempfile so we
        can hand a path to LTX (which only accepts file paths, not PIL).
        Caller is responsible for unlinking — we don't keep a handle around.
        """
        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            data = fcrypto.read_decrypted(key, visible)
        else:
            if not visible.exists():
                raise FileNotFoundError(visible)
            data = visible.read_bytes()
        # Preserve the visible suffix so LTX's image loader can decode the
        # right format (PNG/JPEG/etc.) without sniffing.
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
        """Drop the audio track in place (or near-in-place). `-c:v copy -an`
        is lossless and fast. If ffmpeg is missing, we just return src — the
        downstream encrypt step still works; the user gets audio along for
        the ride. ffmpeg is in our docker image so this should always work.
        """
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
        """Encrypt + write a finished mp4 into assets/. Mirrors the encryption
        path inside Runner.save_video() but accepts existing bytes-on-disk
        rather than a frames list (since LTX gives us a finished mp4)."""
        plaintext = src.read_bytes()
        out_visible.parent.mkdir(parents=True, exist_ok=True)
        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            return fcrypto.write_encrypted(key, out_visible, plaintext)
        out_visible.write_bytes(plaintext)
        return out_visible
