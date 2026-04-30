"""LTX-2.3 image-to-video runner.

Lightricks' LTX-2.3 (22B params) — best-in-class video quality but workstation
class hardware (~46 GB BF16 weights → ~60-80 GB VRAM realistic). Diffusers
doesn't support 2.3 yet (model card says "coming soon"), so this runner uses
Lightricks' own `ltx-pipelines` package directly.

That package pins Python ≥3.12 and Torch ~2.7, which conflicts with our
3.11 base image. We therefore run this runner inside a per-runner venv
managed by backend/venv_manager.py — see the `runtime` block on the
ltx23 entry in backend/model_registry.json.

Differences from our diffusers runners worth flagging:

  - Their pipeline takes IMAGE FILE PATHS, not PIL Images. We decrypt the
    user's ref to a tempfile, hand the path over, then unlink in finally.

  - LoRAs are passed at pipeline CONSTRUCTION, not loaded dynamically.
    The runner caches the currently-built LoRA set; if a generate() call
    asks for a different set, we rebuild the pipeline. Acceptable since
    LoRA toggles are rare in a session.

  - Output is an mp4 (theirs), already on disk. We don't re-encode through
    imageio — `ffmpeg -c:v copy -an` strips the audio track losslessly,
    then we encrypt the result via crypto.write_encrypted.

License: LTX Community License (non-commercial). The UI surfaces this via
the registry's license pill — operators who run a fork commercially are
on their own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, _data_key


# Maps registry-side variant ids → the safetensors filename and an inference
# preset hint (default steps when the user hasn't overridden). The full
# Lightricks/LTX-2.3 repo is ~46 GB per variant; users pick exactly one.
VARIANTS = {
    "distilled-1.1": {
        "weight": "ltx-2.3-22b-distilled-1.1.safetensors",
        "default_steps": 8,
        "default_cfg":   1.0,
    },
    "distilled": {
        "weight": "ltx-2.3-22b-distilled.safetensors",
        "default_steps": 8,
        "default_cfg":   1.0,
    },
    "dev": {
        "weight": "ltx-2.3-22b-dev.safetensors",
        "default_steps": 30,
        "default_cfg":   3.0,
    },
}

HF_REPO = "Lightricks/LTX-2.3"


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
        """Construct a fresh TI2VidTwoStagesPipeline with the chosen variant
        + LoRA set. Lazy imports keep cold start before /healthz cheap and
        ensure the package only resolves once we're inside the venv subprocess.
        """
        from huggingface_hub import hf_hub_download
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps

        token = os.environ.get("HF_TOKEN")
        variant_cfg = VARIANTS[self._variant]
        weight_name = variant_cfg["weight"]

        print(f"[runner] resolving LTX-2.3 weights ({weight_name})…", flush=True)
        weight_path = hf_hub_download(repo_id=HF_REPO, filename=weight_name, token=token)

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
            ltx_loras.append(LoraPathStrengthAndSDOps(str(p), float(strength), None))

        print(f"[runner] building TI2VidTwoStagesPipeline (variant={self._variant}, loras={len(ltx_loras)})", flush=True)
        # Their constructor takes the safetensors path + LoRA list. Other
        # ctor kwargs (precision, scheduler, etc.) are left as defaults —
        # the upstream config files inside the package set sensible BF16
        # defaults for the dev/distilled checkpoints.
        pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=str(weight_path),
            loras=ltx_loras,
        )
        self._pipe = pipe
        self._loaded_lora_key = lora_set
        print("[runner] ready", flush=True)

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
        width  = int(params.get("width",  1280))
        height = int(params.get("height",  720))
        fps    = int(params.get("fps", 24))
        duration = float(params.get("duration", 3.0))
        # LTX accepts an absolute frame count — keep this consistent with how
        # we expose seconds in the UI for Wan and others.
        frames = max(8, int(round(duration * fps)))
        if seed < 0:
            seed = secrets.randbits(31)

        # Decrypt the ref to a tempfile so we can hand a path to LTX. Their
        # API does not accept PIL objects.
        ref_visible = Path(ref_path)
        if not ref_visible.is_absolute():
            ref_visible = WORKSPACE / ref_visible
        ref_tmp = self._decrypt_ref_to_temp(ref_visible)

        out_tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
        try:
            from ltx_pipelines import ImageConditioningInput

            print(f"[runner] LTX-2.3 generate (variant={self._variant}, steps={steps}, cfg={cfg}, {width}x{height}@{fps}fps, frames={frames}, seed={seed})", flush=True)
            self._pipe(
                prompt=prompt,
                negative_prompt=(params.get("negative_prompt") or "").strip() or None,
                output_path=str(out_tmp),
                images=[ImageConditioningInput(str(ref_tmp), 0, 1.0, frames)],
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                num_frames=frames,
                seed=seed,
            )

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
        finally:
            for p in (ref_tmp, out_tmp):
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
        fd, tmp = tempfile.mkstemp(suffix=suffix)
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
