"""Base class + helpers for model runners.

Each runner is a Python module under `backend/runners/` that subclasses
`Runner` and implements `load()` and `generate()`. The generic shell at
`backend/runner_host.py` instantiates exactly one Runner per subprocess
and exposes it on a localhost-only FastAPI port.

Conventions:
  - Weights are downloaded into  $WORKSPACE/models/<hf_repo_path>/
  - Generated outputs are written to  $WORKSPACE/assets/generated/
  - Runner.generate() returns a dict shaped like:
        { "assets": [{"path": "...", "kind": "image", "url": "..."}],
          "meta":   { ... arbitrary debug ... } }
    so the frontend can drop the items straight into the Assets view.

Encryption: when the backend launches us, it injects FORGE_DATA_KEY (hex)
in the environment. `save_image_encrypted()` and `load_image_encrypted()`
use it to write outputs as <name>.enc and read encrypted refs back to PIL
images. Runners should always go through these helpers — never PIL.save
straight to disk for user-visible assets.
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

WORKSPACE       = Path(os.environ.get("WORKSPACE", "/workspace"))
MODELS_DIR      = WORKSPACE / "models"
GENERATED_DIR   = WORKSPACE / "assets" / "generated"

# Make `backend.crypto` importable from runner subprocesses. Runners are
# spawned with the repo root on PYTHONPATH (see launcher.py + runner_host),
# so `import crypto` works the same way `from main import auth` would.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _data_key() -> Optional[bytes]:
    """Read the at-rest data key from the env. None if encryption is off."""
    hex_key = os.environ.get("FORGE_DATA_KEY")
    return bytes.fromhex(hex_key) if hex_key else None


class Runner(ABC):
    """Subclass per model. Keep loading lazy in __init__; do the heavy work in load()."""

    # Mandatory metadata
    model_id:    str = "unknown"
    model_name:  str = ""
    category:    str = "image"        # image | video | llm | audio
    supports_lora: bool = False

    # Hardware hints (informational)
    min_vram_gb:         int = 8
    recommended_vram_gb: int = 16

    # ── Lifecycle ────────────────────────────────────────────────────
    @abstractmethod
    def load(self) -> None:
        """Pull weights, build the pipeline, move it to GPU. Called once."""

    @abstractmethod
    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        """Run inference. Return the dict shape described in this module's docstring."""

    # ── LoRA helpers ─────────────────────────────────────────────────
    def _load_loras(self, loras: list) -> bool:
        """Load LoRA adapters onto self._pipe. Returns True if any loaded.

        Each entry is one of:

          {filename, strength}
              Single file, applied uniformly to whatever components the
              pipeline has LoRA weights for.

          {filename, strength_high, strength_low}
              Single file containing both high- and low-noise expert weights.
              Applied per-component for MoE pipelines (Wan etc.) via the
              diffusers per-component weight dict.

          {files: [{filename, target}, ...], strength_high, strength_low}
              Multi-file logical LoRA — Wan Lightning ships as paired
              `high_noise_model.safetensors` + `low_noise_model.safetensors`.
              Each file is loaded as its own adapter and set_adapters
              applies it only to its target component. Strengths come from
              the parent entry, not the per-file dict.

        Files are resolved under LORAS_DIR (default $WORKSPACE/loras).
        Missing files are logged + skipped — generation continues without
        them rather than failing.
        """
        pipe = getattr(self, "_pipe", None)
        if pipe is None:
            return False
        if not loras:
            self._clear_loras()
            return False
        # Diffusers/PEFT keeps adapter names attached to the live pipeline.
        # If a prior generation failed halfway through LoRA loading, `finally`
        # may not have known anything was active yet. Clear first so stacked
        # runs never collide with stale adapter_0/adapter_1 names.
        self._clear_loras()
        loras_dir = Path(os.environ.get("LORAS_DIR", str(WORKSPACE / "loras")))
        run_prefix = f"forge_{uuid.uuid4().hex[:8]}"

        def resolve(filename: str) -> Optional[Path]:
            """Try the filename as relative path, then as basename anywhere under loras_dir."""
            direct = loras_dir / filename
            if direct.exists():
                return direct
            base = Path(filename).name
            matches = list(loras_dir.rglob(base))
            return matches[0] if matches else None

        # Some pipelines validate the LoRA's target_modules against EVERY
        # component when loading at the pipe level — if the LoRA only
        # targets the transformer, the text encoder gets validated, no
        # match, PEFT raises "No modules were targeted for adaptation".
        # Bypass that by loading directly onto the matching submodule:
        #   - For multi-file paired LoRAs (Wan Lightning): high → transformer,
        #     low → transformer_2.
        #   - For single-file LoRAs (Qwen Lightning, FLUX LoRAs, …): target
        #     pipe.transformer if it exposes load_lora_adapter; fall back to
        #     pipe-level load otherwise.
        # When a LoRA loads directly onto a submodule, activate it on that
        # submodule too. Some pipelines (Qwen) do not mirror direct component
        # loads into the pipe-level adapter registry, so pipe.set_adapters()
        # would see an empty adapter list and fail.
        def _registered_adapters(module) -> set[str]:
            names = set()
            peft_config = getattr(module, "peft_config", None)
            if isinstance(peft_config, dict):
                names.update(str(k) for k in peft_config.keys())
            for attr in ("get_list_adapters", "get_adapter_names"):
                fn = getattr(module, attr, None)
                if not callable(fn):
                    continue
                try:
                    value = fn()
                    if isinstance(value, dict):
                        for v in value.values():
                            if isinstance(v, (list, tuple, set)):
                                names.update(str(x) for x in v)
                    elif isinstance(value, (list, tuple, set)):
                        names.update(str(x) for x in value)
                except Exception:
                    pass
            return names

        def _load_to_submodule(path: Path, adapter: str, target: str) -> bool:
            sub = getattr(pipe, "transformer_2", None) if target == "low" else getattr(pipe, "transformer", None)
            if sub is None or not hasattr(sub, "load_lora_adapter"):
                return False
            attempts = []
            if target in ("high", "low"):
                # Most Qwen/Wan LoRAs are saved with transformer-prefixed
                # keys when trained against a full pipeline. Loading straight
                # onto the transformer needs that prefix stripped. Older
                # diffusers builds may not accept `prefix`, so we fall back to
                # the no-prefix call below.
                attempts.append({"prefix": "transformer_2" if target == "low" else "transformer"})
            attempts.append({})
            errors = []
            for kwargs in attempts:
                try:
                    sub.load_lora_adapter(str(path), adapter_name=adapter, **kwargs)
                    registered = _registered_adapters(sub)
                    if registered and adapter not in registered:
                        errors.append(f"adapter {adapter} not registered; present={sorted(registered)}")
                        continue
                    return True
                except TypeError as e:
                    # Likely an older signature without `prefix`. Try the
                    # no-prefix path next.
                    errors.append(f"{type(e).__name__}: {e}")
                    continue
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
                    continue
            print(f"[runner] direct submodule load failed ({'; '.join(errors)}); falling back to pipe-level", flush=True)
            return False

        def _set_adapters(module, names, weights) -> bool:
            try:
                module.set_adapters(names, adapter_weights=weights)
                return True
            except TypeError as e:
                first_error = e
                try:
                    module.set_adapters(names, weights)
                    return True
                except Exception as e:
                    print(
                        "[runner] set_adapters failed "
                        f"({type(first_error).__name__}: {first_error}; "
                        f"{type(e).__name__}: {e})",
                        flush=True,
                    )
                    return False
            except Exception as e:
                print(f"[runner] set_adapters failed ({type(e).__name__}: {e})", flush=True)
                return False

        adapters = []
        for i, entry in enumerate(loras):
            if not isinstance(entry, dict):
                # Bare filename string — treat as single file at strength 1.0.
                entry = {"filename": entry, "strength": 1.0}

            files = entry.get("files")
            if isinstance(files, list) and files:
                # Multi-file logical LoRA (e.g. Wan Lightning high+low pair).
                sh = float(entry.get("strength_high", 1.0))
                sl = float(entry.get("strength_low",  sh))
                for j, f in enumerate(files):
                    fname  = f.get("filename")
                    target = (f.get("target") or "high").lower()
                    path   = resolve(fname) if fname else None
                    if not path:
                        print(f"[runner] LoRA file missing, skipping: {fname}", flush=True)
                        continue
                    adapter  = f"{run_prefix}_{i}_{j}"
                    strength = sl if target == "low" else sh
                    print(f"[runner] loading LoRA {fname} ({target}, strength={strength:.2f})", flush=True)
                    if _load_to_submodule(path, adapter, target):
                        adapters.append({"name": adapter, "weight": strength, "target": target})
                    else:
                        weight = ({"transformer": 0.0, "transformer_2": strength}
                                  if target == "low"
                                  else {"transformer": strength, "transformer_2": 0.0})
                        pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
                        adapters.append({"name": adapter, "weight": weight, "target": "pipe"})
                continue

            # Single-file entry.
            filename = entry.get("filename")
            if not filename:
                continue
            path = resolve(filename)
            if not path:
                print(f"[runner] LoRA not found, skipping: {filename}", flush=True)
                continue
            adapter = f"{run_prefix}_{i}"
            # Explicit target on the entry (e.g. CivitAI Wan LoRA the user
            # tagged as high-noise) — load straight onto that submodule.
            # Defaults to "high" since single-transformer pipes (Qwen, Flux)
            # only have `transformer`, and "high" maps there.
            entry_target = (entry.get("target") or "high").lower()
            if entry_target not in ("high", "low"):
                entry_target = "high"
            if "strength_high" in entry or "strength_low" in entry:
                sh = float(entry.get("strength_high", 1.0))
                sl = float(entry.get("strength_low",  sh))
                print(f"[runner] loading LoRA {filename} (target={entry_target}, high={sh:.2f}, low={sl:.2f})", flush=True)
                # Pick the strength matching the explicit target.
                strength = sl if entry_target == "low" else sh
                if _load_to_submodule(path, adapter, entry_target):
                    adapters.append({"name": adapter, "weight": strength, "target": entry_target})
                else:
                    weight = ({"transformer": 0.0, "transformer_2": sl}
                              if entry_target == "low"
                              else {"transformer": sh, "transformer_2": 0.0})
                    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
                    adapters.append({"name": adapter, "weight": weight, "target": "pipe"})
            else:
                strength = float(entry.get("strength", 1.0))
                print(f"[runner] loading LoRA {filename} (target={entry_target}, strength={strength:.2f})", flush=True)
                if _load_to_submodule(path, adapter, entry_target):
                    adapters.append({"name": adapter, "weight": strength, "target": entry_target})
                else:
                    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
                    adapters.append({"name": adapter, "weight": strength, "target": "pipe"})

        if adapters:
            active = []
            for target in ("pipe", "high", "low"):
                group = [a for a in adapters if a["target"] == target]
                if not group:
                    continue
                module = pipe
                if target == "high":
                    module = getattr(pipe, "transformer", None)
                elif target == "low":
                    module = getattr(pipe, "transformer_2", None)
                if module is None or not hasattr(module, "set_adapters"):
                    print(f"[runner] cannot activate LoRA adapter(s) on {target}: set_adapters unavailable", flush=True)
                    continue
                names = [a["name"] for a in group]
                weights = [a["weight"] for a in group]
                present = _registered_adapters(module)
                if present:
                    keep = [(name, weight) for name, weight in zip(names, weights) if name in present]
                    missing = [name for name in names if name not in present]
                    if missing:
                        print(f"[runner] skipping missing LoRA adapter(s) on {target}: {missing}", flush=True)
                    names = [name for name, _ in keep]
                    weights = [weight for _, weight in keep]
                if not names:
                    continue
                if _set_adapters(module, names, weights):
                    active.extend(names)
            self._active_lora_adapters = active
            if active:
                print(f"[runner] {len(active)} LoRA adapter(s) active", flush=True)
                return True
            print("[runner] LoRA adapter(s) loaded but none could be activated", flush=True)
            self._clear_loras()
        return False

    def _clear_loras(self) -> None:
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:
            names = getattr(self, "_active_lora_adapters", None)
            if names:
                for module in (pipe, getattr(pipe, "transformer", None), getattr(pipe, "transformer_2", None)):
                    if module is not None and hasattr(module, "delete_adapters"):
                        try:
                            module.delete_adapters(names)
                        except Exception:
                            pass
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        self._active_lora_adapters = []

    # ── Upscaling ────────────────────────────────────────────────────
    @staticmethod
    def _upscale_if_requested(image, params: dict):
        """Apply the chosen upscaler if params['upscale']['id'] is set.

        params shape: { "upscale": { "id": "realesrgan-x4-plus" } }
        Lazy-loads and caches the upscaler in the runner subprocess.
        """
        cfg = params.get("upscale") or {}
        upscaler_id = cfg.get("id") if isinstance(cfg, dict) else cfg
        if not upscaler_id:
            return image
        from backend import upscaler as _up
        return _up.upscale(image, upscaler_id)

    # ── Helpers shared by subclasses ─────────────────────────────────
    @staticmethod
    def fetch_weight(hf_repo: str, filename: Optional[str] = None,
                     subfolder: Optional[str] = None,
                     hf_token: Optional[str] = None) -> Path:
        """Download a single file (or whole repo if filename is None) into MODELS_DIR.

        Returns the local path. Skips if already on disk.
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        local_dir = MODELS_DIR / hf_repo.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        token = hf_token or os.environ.get("HF_TOKEN")

        if filename:
            return Path(hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                subfolder=subfolder,
                local_dir=str(local_dir),
                token=token,
            ))
        return Path(snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_dir),
            token=token,
        ))

    @staticmethod
    def new_output_path(ext: str = "png", prefix: str = "") -> Path:
        """Pick a fresh path under assets/generated/."""
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        slug = uuid.uuid4().hex[:8]
        name = f"{prefix + '_' if prefix else ''}{ts}_{slug}.{ext}"
        return GENERATED_DIR / name

    @staticmethod
    def asset_response(paths: list[Path], meta: Optional[dict] = None) -> dict:
        """Wrap a list of saved files into the canonical generate() response.

        URLs are filled in by the backend in `/api/generate` (see _sign_url
        in main.py) — runners only emit the path. We deliberately leave
        `url` blank here because the runner has no signing key.

        Note: paths are reported as VISIBLE names (no .enc suffix) regardless
        of whether the file on disk is encrypted. The auth-gated streaming
        endpoint resolves both forms.
        """
        items = []
        for p in paths:
            visible_name = p.name[:-4] if p.name.endswith(".enc") else p.name
            visible_path = p.with_suffix("") if p.suffix == ".enc" else p
            kind = "video" if visible_path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v", ".mkv"} else "image"
            items.append({
                "path":   str(visible_path),
                "url":    "",
                "name":   visible_name,
                "kind":   kind,
                "source": "generated",
            })
        return {"assets": items, "meta": meta or {}}

    # ── Encrypted asset I/O ─────────────────────────────────────────────
    # When FORGE_DATA_KEY is set in the env (the normal case — backend
    # injects it on spawn), images are encrypted at rest. Runners save
    # outputs through `save_image()` and read refs via `load_image()` so the
    # crypto layer is transparent.

    @staticmethod
    def save_image(image, dest: Path, format: str = "PNG") -> Path:
        """Save a PIL image — encrypted if FORGE_DATA_KEY is set.

        Returns the on-disk path (with `.enc` suffix when encrypted, or the
        bare path in the legacy plaintext fallback).
        """
        buf = io.BytesIO()
        image.save(buf, format=format)
        plaintext = buf.getvalue()

        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            return fcrypto.write_encrypted(key, dest, plaintext)
        # Legacy fallback (encryption disabled): plaintext on disk.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(plaintext)
        return dest

    @staticmethod
    def save_video(frames, dest: Path, fps: int = 24) -> Path:
        """Encode a list of PIL frames to mp4 — encrypted if FORGE_DATA_KEY is set.

        Uses imageio + imageio-ffmpeg (the path diffusers' export_to_video
        prefers when imageio is available). cv2's VideoWriter has a long
        history of silent failures — non-zero but invalid mp4s with bad
        moov atoms, missing faststart, fps mismatches — that produce videos
        which "exist" but show 0 seconds. imageio shells out to the system
        ffmpeg with libx264 + yuv420p + +faststart and never produces those
        broken-but-non-empty outputs.

        Final fallback to a direct ffmpeg subprocess remains as belt-and-
        braces in case imageio-ffmpeg is unavailable on a pod that hasn't
        finished its boot-time pip install yet.
        """
        import subprocess
        import tempfile

        frames = Runner._normalize_video_frames(frames)
        if len(frames) == 0:
            raise ValueError("save_video: empty frames list")

        # H.264 + yuv420p requires even dimensions. Crop one pixel rather
        # than padding to keep the framing tight; same approach Comfy uses.
        w0, h0 = frames[0].size
        ew = w0 if w0 % 2 == 0 else w0 - 1
        eh = h0 if h0 % 2 == 0 else h0 - 1
        if (ew, eh) != (w0, h0):
            frames = [f.crop((0, 0, ew, eh)) for f in frames]

        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            try:
                import imageio.v2 as imageio  # type: ignore
                import numpy as np

                # imageio writes frame-by-frame so memory stays bounded by
                # the encoder's buffer rather than holding the whole movie
                # in RAM. macro_block_size=1 turns off imageio-ffmpeg's
                # default 16-pixel alignment requirement (we already cropped
                # to even dimensions, that's all H.264 needs).
                writer = imageio.get_writer(
                    str(tmp_path),
                    format="FFMPEG",
                    fps=int(fps),
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=1,
                    ffmpeg_params=["-movflags", "+faststart"],
                )
                try:
                    for f in frames:
                        writer.append_data(np.asarray(f.convert("RGB")))
                finally:
                    writer.close()
            except Exception as e:
                print(f"[save_video] imageio path failed ({type(e).__name__}: {e}); falling back to ffmpeg subprocess", flush=True)
                # Wipe any partial output before retrying so the size check
                # below isn't fooled by a half-written file.
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-vcodec", "rawvideo",
                        "-pix_fmt", "rgb24",
                        "-s", f"{ew}x{eh}",
                        "-r", str(fps),
                        "-i", "-",
                        "-an",
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(tmp_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                for frame in frames:
                    proc.stdin.write(frame.convert("RGB").tobytes())
                proc.stdin.close()
                proc.stdin = None
                _, err = proc.communicate(timeout=120)
                if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"save_video: ffmpeg fallback failed "
                        f"(exit={proc.returncode}). stderr tail: "
                        f"{err.decode('utf-8', errors='replace')[-500:]}"
                    )

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("save_video: encoded mp4 is empty")
            plaintext = tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            return fcrypto.write_encrypted(key, dest, plaintext)
        dest.write_bytes(plaintext)
        return dest

    @staticmethod
    def _normalize_video_frames(frames) -> list:
        """Return video frames as a plain list of RGB PIL images.

        Diffusers video pipelines are not perfectly consistent here: depending
        on pipeline/version they can return `[batch][frame]` PIL lists, a
        NumPy array shaped like `(frames, h, w, c)`, or tensor-like arrays. We
        normalize once so truth checks, moderation, and ffmpeg export do not
        trip over array semantics.
        """
        from PIL import Image

        def as_array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            if hasattr(value, "shape"):
                import numpy as np
                return np.asarray(value)
            return None

        def image_from_array(value):
            import numpy as np
            arr = as_array(value)
            if arr is None:
                raise TypeError(f"Unsupported video frame type: {type(value).__name__}")

            # Drop leading batch dimension when present.
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.dtype.kind == "f":
                # Diffusers tensors/arrays are often 0..1 floats; if they are
                # already 0..255, clipping below keeps them sensible.
                if float(np.nanmax(arr)) <= 1.0:
                    arr = arr * 255.0
                arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype("uint8")

            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim != 3:
                raise ValueError(f"Unsupported video frame shape: {arr.shape}")
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
                return Image.fromarray(arr, mode="L").convert("RGB")
            if arr.shape[-1] >= 3:
                return Image.fromarray(arr[..., :3], mode="RGB")
            raise ValueError(f"Unsupported video frame shape: {arr.shape}")

        def frame_to_pil(frame):
            if isinstance(frame, Image.Image):
                return frame.convert("RGB")
            return image_from_array(frame)

        if frames is None:
            return []
        if isinstance(frames, Image.Image):
            return [frames.convert("RGB")]

        arr = as_array(frames)
        if arr is not None:
            if arr.ndim == 5:
                arr = arr[0] if arr.shape[0] == 1 else arr.reshape((-1,) + arr.shape[-3:])
            if arr.ndim == 4:
                return [frame_to_pil(frame) for frame in arr]
            return [frame_to_pil(arr)]

        if not isinstance(frames, (list, tuple)):
            frames = list(frames)

        normalized = []
        for frame in frames:
            if isinstance(frame, (list, tuple)):
                normalized.extend(Runner._normalize_video_frames(frame))
            else:
                normalized.append(frame_to_pil(frame))
        return normalized

    @staticmethod
    def load_image(visible_path: Path):
        """Open whichever of <name>.enc or <name> exists; return a PIL.Image.

        `visible_path` is the user-facing path (no .enc). We resolve the
        encrypted form transparently when the data key is present.
        """
        from PIL import Image
        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            data = fcrypto.read_decrypted(key, visible_path)
            return Image.open(io.BytesIO(data))
        # Legacy plaintext path.
        on_disk = visible_path
        if not on_disk.exists():
            raise FileNotFoundError(visible_path)
        return Image.open(on_disk)
