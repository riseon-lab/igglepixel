#!/usr/bin/env python3
"""Igglepixel Qwen LoRA trainer wrapper.

This script is called by backend/main.py with training settings passed through
environment variables. It bootstraps Ostris AI Toolkit into /workspace, writes
an AI Toolkit config, runs the training job, and copies the newest safetensors
to OUTPUT_PATH for Igglepixel to import.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
TOOLKIT_REPO = os.environ.get("IGGLEPIXEL_AI_TOOLKIT_REPO", "https://github.com/ostris/ai-toolkit.git")
TOOLKIT_DIR = Path(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_DIR", WORKSPACE / "repos" / "ai-toolkit"))
VENV_DIR = Path(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_VENV", WORKSPACE / "venvs" / "ai-toolkit"))


def log(message: str) -> None:
    print(message, flush=True)


def run(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(shlex.quote(str(a)) for a in args))
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, check=True)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def safe_name(value: str, fallback: str = "qwen_lora") -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in value.strip())
    out = out.strip("._-")
    return out or fallback


def py_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_ai_toolkit() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    TOOLKIT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (TOOLKIT_DIR / "run.py").exists():
        if TOOLKIT_DIR.exists():
            raise SystemExit(f"{TOOLKIT_DIR} exists but does not look like ai-toolkit")
        log(f"Cloning AI Toolkit into {TOOLKIT_DIR}")
        run(["git", "clone", "--depth", "1", TOOLKIT_REPO, str(TOOLKIT_DIR)])
    elif py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_UPDATE"), False):
        log("Updating AI Toolkit")
        run(["git", "fetch", "--depth", "1", "origin", "main"], cwd=TOOLKIT_DIR)
        run(["git", "reset", "--hard", "origin/main"], cwd=TOOLKIT_DIR)

    run(["git", "submodule", "update", "--init", "--recursive"], cwd=TOOLKIT_DIR)
    return TOOLKIT_DIR


def ensure_venv(toolkit_dir: Path) -> Path:
    py = VENV_DIR / "bin" / "python"
    stamp = VENV_DIR / ".igglepixel_ai_toolkit_ready"
    if not py.exists():
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        log(f"Creating AI Toolkit venv at {VENV_DIR}")
        try:
            run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])
        except subprocess.CalledProcessError:
            log("python -m venv failed; trying virtualenv fallback")
            run([sys.executable, "-m", "virtualenv", "--system-site-packages", str(VENV_DIR)])

    reinstall = py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_REINSTALL"), False)
    if reinstall or not stamp.exists():
        log("Installing AI Toolkit requirements. First run can take several minutes.")
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
        run([str(py), "-m", "pip", "install", "-r", str(toolkit_dir / "requirements.txt")])
        stamp.write_text(str(time.time()), encoding="utf-8")
    else:
        log("AI Toolkit requirements already installed")
    ensure_torchaudio(py)
    return py


def py_import_ok(py: Path, module: str) -> bool:
    probe = subprocess.run(
        [str(py), "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def torch_build(py: Path) -> tuple[str, str]:
    code = "import torch; print(torch.__version__)"
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=True)
    version = proc.stdout.strip()
    base, _, cuda = version.partition("+")
    return base, cuda


def ensure_torchaudio(py: Path) -> None:
    if py_import_ok(py, "torchaudio"):
        return
    log("Installing torchaudio to satisfy AI Toolkit startup imports")
    torch_version, cuda_suffix = torch_build(py)
    cmd = [str(py), "-m", "pip", "install", f"torchaudio=={torch_version}"]
    if cuda_suffix.startswith("cu"):
        cmd.extend(["--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}"])
    run(cmd)


def model_arch(base_model: str) -> str:
    if "Edit" in base_model:
        return "qwen_image_edit"
    return "qwen_image"


def model_quant_block(base_model: str) -> str:
    if not py_bool(os.environ.get("TRAIN_QUANTIZE"), True):
        return "      quantize: false\n      quantize_te: false\n"
    if "Edit" in base_model:
        qtype = os.environ.get("TRAIN_QTYPE", "uint3|qwen_image_edit_torchao_uint3.safetensors")
    else:
        qtype = os.environ.get(
            "TRAIN_QTYPE",
            "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
        )
    return (
        "      quantize: true\n"
        f'      qtype: "{qtype}"\n'
        "      quantize_te: true\n"
        f'      qtype_te: "{os.environ.get("TRAIN_QTYPE_TE", "qfloat8")}"\n'
        f"      low_vram: {str(py_bool(os.environ.get('TRAIN_LOW_VRAM'), True)).lower()}\n"
    )


def write_config(toolkit_dir: Path, dataset_dir: Path, output_dir: Path) -> Path:
    output_name = safe_name(os.environ.get("OUTPUT_NAME", "qwen_lora"))
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen-Image")
    trigger = os.environ.get("TRIGGER_PHRASE", "").strip()
    steps = int(float(os.environ.get("TRAIN_STEPS", "3000")))
    rank = int(float(os.environ.get("TRAIN_RANK", "64")))
    lr = float(os.environ.get("TRAIN_LR", "0.0002"))
    resolution = int(float(os.environ.get("TRAIN_RESOLUTION", "1024")))
    batch_size = int(float(os.environ.get("TRAIN_BATCH_SIZE", "1")))
    grad_accum = int(float(os.environ.get("TRAIN_GRAD_ACCUM", "1")))
    save_every = max(250, min(1000, steps // 4 or 250))
    sample_prompt = f"{trigger}, portrait photo, natural skin texture, studio lighting" if trigger else "portrait photo, studio lighting"
    log(
        "Igglepixel training config: "
        f"base_model={base_model}, trigger={trigger or '(none)'}, "
        f"steps={steps}, rank={rank}, lr={lr}, resolution={resolution}, batch={batch_size}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{output_name}.yaml"
    toolkit_output = output_dir / "ai-toolkit-output"

    trigger_line = f'    trigger_word: "{trigger}"\n' if trigger else ""
    config = f"""\
---
job: extension
config:
  name: "{output_name}"
  process:
  - type: "sd_trainer"
    training_folder: "{toolkit_output.as_posix()}"
    device: cuda:0
{trigger_line}    network:
      type: "lora"
      linear: {rank}
      linear_alpha: {rank}
    save:
      dtype: float16
      save_every: {save_every}
      max_step_saves_to_keep: 4
      push_to_hub: false
    datasets:
    - folder_path: "{dataset_dir.as_posix()}"
      caption_ext: "txt"
      caption_dropout_rate: 0.05
      shuffle_tokens: false
      cache_latents_to_disk: true
      resolution: [ {resolution} ]
    train:
      batch_size: {batch_size}
      cache_text_embeddings: true
      steps: {steps}
      gradient_accumulation: {grad_accum}
      train_unet: true
      train_text_encoder: false
      gradient_checkpointing: true
      noise_scheduler: "flowmatch"
      optimizer: "adamw8bit"
      lr: {lr}
      dtype: bf16
      skip_first_sample: true
      disable_sampling: true
    model:
      name_or_path: "{base_model}"
      arch: "{model_arch(base_model)}"
{model_quant_block(base_model)}    sample:
      sampler: "flowmatch"
      sample_every: 250
      width: {resolution}
      height: {resolution}
      prompts:
      - "{sample_prompt}"
      neg: ""
      seed: 42
      walk_seed: true
      guidance_scale: 3
      sample_steps: 25
    meta:
      name: "[name]"
      version: "1.0"
"""
    config_path.write_text(config, encoding="utf-8")
    log(f"Wrote AI Toolkit config: {config_path}")
    return config_path


def check_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {dataset_dir}")
    supported = {".jpg", ".jpeg", ".png"}
    images = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    if not images:
        raise SystemExit("AI Toolkit needs at least one .jpg, .jpeg, or .png image")
    missing = [p.relative_to(dataset_dir).as_posix() for p in images if not p.with_suffix(".txt").exists()]
    if missing:
        raise SystemExit("Missing caption files for: " + ", ".join(missing[:10]))
    unsupported = [
        p.relative_to(dataset_dir).as_posix()
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".webp", ".gif", ".bmp"}
    ]
    if unsupported:
        log("Warning: AI Toolkit ignores non-png/jpg images: " + ", ".join(unsupported[:10]))
    log(f"Dataset ready for AI Toolkit: {len(images)} images")


def run_training(py: Path, toolkit_dir: Path, config_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [str(py), "run.py", str(config_path)]
    log("Starting AI Toolkit training")
    log("$ " + " ".join(shlex.quote(str(a)) for a in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(toolkit_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)


def copy_output(output_dir: Path) -> None:
    output_path = Path(os.environ.get("OUTPUT_PATH", output_dir / f"{safe_name(os.environ.get('OUTPUT_NAME', 'qwen_lora'))}.safetensors"))
    candidates = sorted(output_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"Training finished but no .safetensors was found under {output_dir}")
    src = candidates[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != output_path.resolve():
        shutil.copy2(src, output_path)
    log(f"LoRA output: {output_path}")


def main() -> None:
    dataset_dir = Path(require_env("DATASET_DIR")).resolve()
    output_dir = Path(require_env("OUTPUT_DIR")).resolve()
    check_dataset(dataset_dir)
    toolkit_dir = ensure_ai_toolkit()
    py = ensure_venv(toolkit_dir)
    config_path = write_config(toolkit_dir, dataset_dir, output_dir)
    run_training(py, toolkit_dir, config_path)
    copy_output(output_dir)


if __name__ == "__main__":
    main()
