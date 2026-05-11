"""Qwen-Image LoRA trainer wrapper around kohya-ss/musubi-tuner.

Backend launches this script via subprocess with the following env vars
(set in backend/main.py:_run_train_job):

    DATASET_DIR           Pod path to a directory containing image + .txt
                          caption pairs (one .txt per image, same basename).
    OUTPUT_DIR            Where the trained LoRA + checkpoints go.
    OUTPUT_NAME           Filename stem (no extension).
    OUTPUT_PATH           Final .safetensors path the backend expects.
    BASE_MODEL            HF repo id, one of:
                            Qwen/Qwen-Image, Qwen/Qwen-Image-2512,
                            Qwen/Qwen-Image-Edit, Qwen/Qwen-Image-Edit-2511
    TRIGGER_PHRASE        Optional. Tokens like "A woman named Kerry"
                          prefixed/appended to captions if requested.
    TRAIN_STEPS           Total optimizer steps.
    TRAIN_RANK            LoRA network_dim. Alpha = rank (paired).
    TRAIN_LR              Learning rate (float).
    TRAIN_RESOLUTION      Square training resolution (e.g. 1024).
    TRAIN_BATCH_SIZE      Per-GPU batch size.
    TRAIN_REPEATS         Repeats per image per epoch (musubi dataset config).
    TRAIN_MANIFEST        Manifest path written by the backend.
    HF_TOKEN              For gated repos / private uploads.

Optional overrides for ops:
    MUSUBI_PATH           Local musubi-tuner clone (default: /workspace/musubi-tuner).
    MUSUBI_REF            Git ref to pin (default: main).
    MUSUBI_VENV           Venv path (default: /workspace/venvs/musubi-tuner).
    WORKSPACE             /workspace root (default).

Output convention: anything printed to stdout/stderr streams straight to
the backend's job log_tail. The backend's regex parser picks up:
    "base dim (rank): N, alpha: M"      → observed_rank
    "<n>/<total>"                       → step progress

Musubi-tuner emits both naturally; no extra formatting needed.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Required env ────────────────────────────────────────────────────────
def _req(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"[trainer] FATAL: missing required env var {name}", flush=True)
        sys.exit(2)
    return v


DATASET_DIR    = Path(_req("DATASET_DIR"))
OUTPUT_DIR     = Path(_req("OUTPUT_DIR"))
OUTPUT_NAME    = _req("OUTPUT_NAME")
BASE_MODEL     = _req("BASE_MODEL")

# Optional / typed env with defaults
TRIGGER        = os.environ.get("TRIGGER_PHRASE", "").strip()
RANK           = int(os.environ.get("TRAIN_RANK", "64"))
STEPS          = int(os.environ.get("TRAIN_STEPS", "2000"))
LR             = float(os.environ.get("TRAIN_LR", "1e-4"))
RESOLUTION     = int(os.environ.get("TRAIN_RESOLUTION", "1024"))
BATCH_SIZE     = int(os.environ.get("TRAIN_BATCH_SIZE", "1"))
REPEATS        = int(os.environ.get("TRAIN_REPEATS", "10"))
HF_TOKEN       = os.environ.get("HF_TOKEN")

WORKSPACE      = Path(os.environ.get("WORKSPACE", "/workspace"))
MUSUBI_PATH    = Path(os.environ.get("MUSUBI_PATH", str(WORKSPACE / "musubi-tuner")))
MUSUBI_REF     = os.environ.get("MUSUBI_REF", "main")
MUSUBI_VENV    = Path(os.environ.get("MUSUBI_VENV", str(WORKSPACE / "venvs" / "musubi-tuner")))

MUSUBI_REPO    = "https://github.com/kohya-ss/musubi-tuner.git"

# Map BASE_MODEL → (musubi entry script relative path, training type).
# The non-2512/2511 entries use the same scripts the newer variants do;
# musubi-tuner dispatches the model family internally via --dit.
TRAINER_BY_MODEL = {
    "Qwen/Qwen-Image":             ("qwen_image_train_network.py",      "t2i"),
    "Qwen/Qwen-Image-2512":        ("qwen_image_train_network.py",      "t2i"),
    "Qwen/Qwen-Image-Edit":        ("qwen_image_edit_train_network.py", "edit"),
    "Qwen/Qwen-Image-Edit-2511":   ("qwen_image_edit_train_network.py", "edit"),
}


def log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


# ── musubi-tuner provisioning ───────────────────────────────────────────
def _ensure_musubi_clone() -> None:
    if (MUSUBI_PATH / ".git").exists():
        log(f"musubi-tuner present at {MUSUBI_PATH}")
        return
    MUSUBI_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"cloning {MUSUBI_REPO} → {MUSUBI_PATH} (ref {MUSUBI_REF})")
    rc = subprocess.call([
        "git", "clone", "--depth", "1", "--branch", MUSUBI_REF,
        MUSUBI_REPO, str(MUSUBI_PATH),
    ])
    if rc != 0:
        log(f"FATAL: git clone failed (exit {rc})")
        sys.exit(3)


def _venv_python() -> Path:
    return MUSUBI_VENV / "bin" / "python"


def _ensure_musubi_venv() -> Path:
    """Build a persistent venv with musubi-tuner's deps installed.

    Prefers uv (already on the pod via entrypoint.sh); falls back to the
    stdlib venv module if uv is unavailable. The venv lives on /workspace
    so cold pods on the same volume skip the heavy install.
    """
    py = _venv_python()
    marker = MUSUBI_VENV / ".forge_installed"
    if py.exists() and marker.exists():
        log(f"musubi-tuner venv ready at {MUSUBI_VENV}")
        return py

    MUSUBI_VENV.parent.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv:
        log(f"creating venv via uv at {MUSUBI_VENV}")
        rc = subprocess.call([uv, "venv", "--python", "3.11", str(MUSUBI_VENV)])
        if rc != 0:
            log(f"uv venv failed (exit {rc}); falling back to python -m venv")
            uv = None
    if not uv:
        log(f"creating venv via stdlib at {MUSUBI_VENV}")
        rc = subprocess.call([sys.executable, "-m", "venv", str(MUSUBI_VENV)])
        if rc != 0:
            log(f"FATAL: python -m venv failed (exit {rc})")
            sys.exit(4)

    if not py.exists():
        log(f"FATAL: venv python not found at {py}")
        sys.exit(4)

    pip_install = [str(py), "-m", "pip", "install", "-U"]

    log("installing musubi-tuner + deps (first run; may take 5–10 min)")
    # Use the musubi-tuner requirements.txt directly. Torch is heavy but
    # already cached on /workspace/.cache/pip from prior installs.
    requirements_files = [
        MUSUBI_PATH / "requirements.txt",
        MUSUBI_PATH / "requirements-uv.txt",
    ]
    installed = False
    for req in requirements_files:
        if req.exists():
            log(f"  pip install -r {req}")
            rc = subprocess.call(pip_install + ["-r", str(req)])
            if rc == 0:
                installed = True
                break
            log(f"  requirements file failed (exit {rc}); trying next")
    if not installed:
        log("FATAL: no usable requirements file inside musubi-tuner. "
            "The repo layout may have changed; pin MUSUBI_REF to a known good tag.")
        sys.exit(5)

    # Install musubi-tuner itself as editable so `python -m musubi_tuner...`
    # works if the repo uses the modern src/ layout. Tolerate failure on
    # older layouts where there's no setup.py / pyproject.toml.
    if (MUSUBI_PATH / "pyproject.toml").exists() or (MUSUBI_PATH / "setup.py").exists():
        log("  pip install -e musubi-tuner")
        subprocess.call(pip_install + ["-e", str(MUSUBI_PATH)])

    marker.touch()
    log("musubi-tuner venv installed")
    return py


# ── Dataset preparation ─────────────────────────────────────────────────
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _scan_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(image, caption_txt), ...] pairs where both files exist
    and share the same stem (1.png + 1.txt, etc.)."""
    if not dataset_dir.exists():
        log(f"FATAL: DATASET_DIR does not exist: {dataset_dir}")
        sys.exit(6)
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(dataset_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        txt = img.with_suffix(".txt")
        if not txt.exists():
            log(f"  WARN: {img.name} has no matching .txt; skipping")
            continue
        pairs.append((img, txt))
    return pairs


def _write_dataset_toml(pairs: list[tuple[Path, Path]], path: Path) -> None:
    """Write a musubi-tuner dataset config.

    Single-bucket, single-resolution dataset that points at DATASET_DIR
    with caption_extension=".txt". Musubi auto-discovers all image/.txt
    pairs at that path.
    """
    contents = f"""# Auto-generated by backend/trainers/qwen_lora_train.py
[general]
resolution = [{RESOLUTION}, {RESOLUTION}]
caption_extension = ".txt"
batch_size = {BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{DATASET_DIR}"
num_repeats = {REPEATS}
"""
    path.write_text(contents, encoding="utf-8")
    log(f"wrote dataset config → {path}")


# ── Training command ────────────────────────────────────────────────────
def _build_train_cmd(py: Path, entry_script: Path, dataset_toml: Path) -> list[str]:
    """Assemble the musubi-tuner training subprocess argv.

    Uses `accelerate launch` when accelerate is installed in the venv —
    even single-GPU benefits from its mixed-precision plumbing. Falls
    back to direct `python` invocation otherwise.
    """
    accelerate = MUSUBI_VENV / "bin" / "accelerate"
    if accelerate.exists():
        base = [str(accelerate), "launch", "--num_cpu_threads_per_process", "8"]
    else:
        base = [str(py)]

    cmd = base + [
        str(entry_script),
        # Model
        "--dit", BASE_MODEL,
        "--dit_dtype", "bfloat16",
        # Dataset
        "--dataset_config", str(dataset_toml),
        # Output
        "--output_dir", str(OUTPUT_DIR),
        "--output_name", OUTPUT_NAME,
        # LoRA network
        "--network_module", "networks.lora_qwen_image",
        "--network_dim", str(RANK),
        "--network_alpha", str(RANK),
        # Optimizer
        "--learning_rate", str(LR),
        "--max_train_steps", str(STEPS),
        "--optimizer_type", "adamw8bit",
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(max(1, STEPS // 50)),
        # Memory
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--persistent_data_loader_workers",
        "--sdpa",
        # Saving
        "--save_every_n_steps", str(max(250, STEPS // 4)),
        "--save_state",
        "--save_model_as", "safetensors",
        "--seed", "42",
    ]

    # Edit-variant models train against paired (input, target) data and
    # need a different sample column expectation. We surface the dataset
    # mode here so musubi-tuner picks the right collate.
    if BASE_MODEL.startswith("Qwen/Qwen-Image-Edit"):
        cmd.extend(["--training_comment", "qwen-image-edit lora"])

    if HF_TOKEN:
        cmd.extend(["--huggingface_token", HF_TOKEN])

    return cmd


def _find_entry_script() -> Path:
    """Locate the musubi-tuner training entry script. Layout changed in
    early-2026 to src/musubi_tuner/<entry>.py; older clones kept them
    at root. Try both."""
    entry_name, _ = TRAINER_BY_MODEL.get(BASE_MODEL, (None, None))
    if not entry_name:
        log(f"FATAL: unsupported base model '{BASE_MODEL}'. Supported: {list(TRAINER_BY_MODEL)}")
        sys.exit(7)
    candidates = [
        MUSUBI_PATH / "src" / "musubi_tuner" / entry_name,
        MUSUBI_PATH / entry_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    log(f"FATAL: could not locate training entry '{entry_name}' under {MUSUBI_PATH}. "
        "musubi-tuner may have renamed the script — pin MUSUBI_REF to a known good tag.")
    sys.exit(8)


# ── Main ────────────────────────────────────────────────────────────────
def main() -> int:
    started = time.time()
    log(f"BASE_MODEL={BASE_MODEL}  rank={RANK} steps={STEPS} lr={LR} res={RESOLUTION} batch={BATCH_SIZE} repeats={REPEATS}")
    log(f"DATASET_DIR={DATASET_DIR}")
    log(f"OUTPUT_DIR={OUTPUT_DIR}  OUTPUT_NAME={OUTPUT_NAME}")

    if BASE_MODEL not in TRAINER_BY_MODEL:
        log(f"FATAL: unsupported BASE_MODEL '{BASE_MODEL}'. "
            f"Supported: {list(TRAINER_BY_MODEL)}")
        return 7

    pairs = _scan_pairs(DATASET_DIR)
    log(f"dataset: {len(pairs)} image+caption pairs")
    if len(pairs) == 0:
        log("FATAL: no image/.txt pairs found in DATASET_DIR. "
            "Each image must have a .txt sidecar with the same stem.")
        return 9

    _ensure_musubi_clone()
    py = _ensure_musubi_venv()
    entry = _find_entry_script()
    log(f"entry script: {entry}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_toml = OUTPUT_DIR / "dataset.toml"
    _write_dataset_toml(pairs, dataset_toml)

    cmd = _build_train_cmd(py, entry, dataset_toml)
    log(f"launching: {' '.join(shlex.quote(c) for c in cmd)}")

    env = os.environ.copy()
    # Push the venv bin onto PATH so musubi's internal `accelerate`/`bitsandbytes`
    # subprocesses resolve to the right interpreter.
    env["PATH"] = f"{MUSUBI_VENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    # Repo-root on PYTHONPATH so `--network_module networks.lora_qwen_image`
    # resolves to <musubi>/networks (or src/musubi_tuner/networks) without
    # the user having to set this themselves.
    extra_paths = [
        str(MUSUBI_PATH),
        str(MUSUBI_PATH / "src"),
    ]
    env["PYTHONPATH"] = ":".join(extra_paths + [env.get("PYTHONPATH", "")]).rstrip(":")

    # cwd=MUSUBI_PATH so relative imports inside the trainer succeed.
    rc = subprocess.call(cmd, env=env, cwd=str(MUSUBI_PATH))
    elapsed = int(time.time() - started)
    log(f"training subprocess exited rc={rc} (elapsed {elapsed}s)")

    # Confirm the expected output exists so the backend doesn't claim
    # success when the trainer crashed mid-save.
    expected = OUTPUT_DIR / f"{OUTPUT_NAME}.safetensors"
    if rc == 0 and not expected.exists():
        log(f"FATAL: subprocess returned 0 but {expected} is missing")
        return 10
    if expected.exists():
        size_mb = expected.stat().st_size / (1024 * 1024)
        log(f"output: {expected} ({size_mb:.1f} MB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
