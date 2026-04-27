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
        """Load LoRA adapters onto self._pipe. Returns True if any were loaded.

        Each entry in `loras` is {filename: str, strength: float}.
        Resolves files against LORAS_DIR env var (default /workspace/loras).
        """
        pipe = getattr(self, "_pipe", None)
        if not loras or pipe is None:
            return False
        loras_dir = Path(os.environ.get("LORAS_DIR", str(WORKSPACE / "loras")))
        names, weights = [], []
        for i, entry in enumerate(loras):
            filename = entry.get("filename") if isinstance(entry, dict) else entry
            strength = float(entry.get("strength", 1.0) if isinstance(entry, dict) else 1.0)
            path = loras_dir / filename
            if not path.exists():
                print(f"[runner] LoRA not found, skipping: {filename}", flush=True)
                continue
            name = f"adapter_{i}"
            print(f"[runner] loading LoRA {filename} @ {strength:.2f}", flush=True)
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
            names.append(name)
            weights.append(strength)
        if names:
            pipe.set_adapters(names, weights)
            print(f"[runner] {len(names)} LoRA(s) active", flush=True)
            return True
        return False

    def _clear_loras(self) -> None:
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

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
