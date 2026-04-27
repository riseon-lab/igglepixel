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
        def _load_to_submodule(path: Path, adapter: str, target: str) -> bool:
            sub = getattr(pipe, "transformer_2", None) if target == "low" else getattr(pipe, "transformer", None)
            if sub is None or not hasattr(sub, "load_lora_adapter"):
                return False
            try:
                sub.load_lora_adapter(str(path), adapter_name=adapter)
                return True
            except Exception as e:
                print(f"[runner] direct submodule load failed ({type(e).__name__}: {e}); falling back to pipe-level", flush=True)
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
            if "strength_high" in entry or "strength_low" in entry:
                sh = float(entry.get("strength_high", 1.0))
                sl = float(entry.get("strength_low",  sh))
                print(f"[runner] loading LoRA {filename} (high={sh:.2f}, low={sl:.2f})", flush=True)
                # Try high-targeted submodule first; if it works, the same
                # adapter on the same submodule gets weight sh. The low side
                # wouldn't match anyway with a single file.
                if _load_to_submodule(path, adapter, "high"):
                    adapters.append({"name": adapter, "weight": sh, "target": "high"})
                else:
                    weight = {"transformer": sh, "transformer_2": sl}
                    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
                    adapters.append({"name": adapter, "weight": weight, "target": "pipe"})
            else:
                strength = float(entry.get("strength", 1.0))
                print(f"[runner] loading LoRA {filename} @ {strength:.2f}", flush=True)
                if _load_to_submodule(path, adapter, "high"):
                    adapters.append({"name": adapter, "weight": strength, "target": "high"})
                else:
                    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
                    adapters.append({"name": adapter, "weight": strength, "target": "pipe"})

        if adapters:
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
                    continue
                module.set_adapters([a["name"] for a in group], [a["weight"] for a in group])
            self._active_lora_adapters = [a["name"] for a in adapters]
            print(f"[runner] {len(adapters)} LoRA adapter(s) active", flush=True)
            return True
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

        Uses diffusers' export_to_video helper (ffmpeg under the hood). Writes
        to a tmp mp4 first, then either encrypts in place or moves to dest.
        Returns the on-disk path with `.enc` suffix when encrypted.
        """
        import tempfile
        from diffusers.utils import export_to_video

        dest.parent.mkdir(parents=True, exist_ok=True)
        # diffusers.export_to_video expects a real filesystem path; tmpfs in
        # /workspace keeps the plaintext mp4 out of the persistent volume
        # except as ciphertext.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            export_to_video(frames, tmp_path, fps=fps)
            with open(tmp_path, "rb") as f:
                plaintext = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        key = _data_key()
        if key:
            import backend.crypto as fcrypto
            return fcrypto.write_encrypted(key, dest, plaintext)
        dest.write_bytes(plaintext)
        return dest

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
