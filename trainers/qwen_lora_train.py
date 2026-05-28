#!/usr/bin/env python3
"""Igglepixel AI Toolkit LoRA trainer wrapper.

This script is called by backend/main.py with training settings passed through
environment variables. It bootstraps Ostris AI Toolkit into /workspace, writes
an AI Toolkit config, runs the training job, and copies the newest safetensors
to OUTPUT_PATH for Igglepixel to import.
"""

from __future__ import annotations

import json
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
    code = "import torch; print(torch.__version__.split('+', 1)[0]); print(torch.version.cuda or '')"
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True, check=True)
    lines = [line.strip() for line in proc.stdout.splitlines()]
    return (lines[0] if lines else "", lines[1] if len(lines) > 1 else "")


def pytorch_cuda_index(cuda_version: str) -> str:
    parts = (cuda_version or "").split(".")
    if len(parts) < 2:
        return ""
    major, minor = parts[0], parts[1]
    if not (major.isdigit() and minor.isdigit()):
        return ""
    return f"cu{major}{minor}"


def torchaudio_probe(py: Path) -> tuple[bool, str]:
    code = (
        "import torch\n"
        "import torchaudio\n"
        "print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda))\n"
        "print('torchaudio=' + torchaudio.__version__)\n"
    )
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def install_matching_torchaudio(py: Path) -> None:
    torch_version, cuda_version = torch_build(py)
    if not torch_version:
        raise SystemExit("Could not determine AI Toolkit torch version")
    cmd = [
        str(py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        f"torchaudio=={torch_version}",
    ]
    cuda_index = pytorch_cuda_index(cuda_version)
    if cuda_index:
        cmd.extend(["--index-url", f"https://download.pytorch.org/whl/{cuda_index}"])
    run(cmd)


def install_cuda128_torch_stack(py: Path) -> None:
    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
    ])


def ensure_torchaudio(py: Path) -> None:
    ok, details = torchaudio_probe(py)
    if ok:
        log("AI Toolkit torch/torchaudio CUDA stack is aligned")
        return
    log("Repairing AI Toolkit torchaudio install")
    if details:
        log("torchaudio probe failed:")
        for line in details.splitlines()[-8:]:
            log("  " + line)
    try:
        install_matching_torchaudio(py)
    except Exception as exc:
        log(f"Matching torchaudio install failed: {type(exc).__name__}: {exc}")

    ok, details = torchaudio_probe(py)
    if ok:
        log("AI Toolkit torchaudio repaired")
        return

    log("Falling back to known-good CUDA 12.8 torch/torchvision/torchaudio stack")
    install_cuda128_torch_stack(py)
    ok, details = torchaudio_probe(py)
    if ok:
        log("AI Toolkit torch stack repaired")
        return
    if details:
        log("torch stack probe still failed:")
        for line in details.splitlines()[-8:]:
            log("  " + line)
    raise SystemExit("AI Toolkit torch/torchaudio CUDA versions are still mismatched after repair")


def is_flux_klein(base_model: str) -> bool:
    return "flux.2-klein" in (base_model or "").lower()


def is_flux_klein_base(base_model: str) -> bool:
    return "flux.2-klein-base" in (base_model or "").lower()


def model_arch(base_model: str) -> str:
    if is_flux_klein_base(base_model):
        return "flux2_klein_base_9b"
    if is_flux_klein(base_model):
        return "flux2_klein_4b" if "4b" in base_model.lower() else "flux2_klein_9b"
    if "Edit" in base_model:
        return "qwen_image_edit"
    return "qwen_image"


def model_quant_block(base_model: str) -> str:
    if is_flux_klein(base_model):
        if not py_bool(os.environ.get("TRAIN_QUANTIZE"), False):
            return "      quantize: false\n      quantize_te: false\n"
        qtype = os.environ.get("TRAIN_QTYPE", "qfloat8")
        return (
            "      quantize: true\n"
            f'      qtype: "{qtype}"\n'
            "      quantize_te: false\n"
            f"      low_vram: {str(py_bool(os.environ.get('TRAIN_LOW_VRAM'), True)).lower()}\n"
        )
    if not py_bool(os.environ.get("TRAIN_QUANTIZE"), True):
        return "      quantize: false\n      quantize_te: false\n"
    if "Image-2512" in base_model:
        default_qtype = "uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors"
    elif "Edit-2511" in base_model:
        default_qtype = "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors"
    elif "Edit-2509" in base_model:
        default_qtype = "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors"
    elif "Edit" in base_model:
        default_qtype = "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors"
    else:
        default_qtype = "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors"
    qtype = os.environ.get("TRAIN_QTYPE", default_qtype)
    return (
        "      quantize: true\n"
        f'      qtype: "{qtype}"\n'
        "      quantize_te: true\n"
        f'      qtype_te: "{os.environ.get("TRAIN_QTYPE_TE", "qfloat8")}"\n'
        f"      low_vram: {str(py_bool(os.environ.get('TRAIN_LOW_VRAM'), True)).lower()}\n"
    )


def sample_guidance_scale(base_model: str) -> float:
    if is_flux_klein_base(base_model):
        return float(os.environ.get("TRAIN_SAMPLE_GUIDANCE", "4.0"))
    if is_flux_klein(base_model):
        return float(os.environ.get("TRAIN_SAMPLE_GUIDANCE", "1.0"))
    return float(os.environ.get("TRAIN_SAMPLE_GUIDANCE", "3.0"))


def sample_steps(base_model: str) -> int:
    if is_flux_klein_base(base_model):
        return int(float(os.environ.get("TRAIN_SAMPLE_STEPS", "50")))
    if is_flux_klein(base_model):
        return int(float(os.environ.get("TRAIN_SAMPLE_STEPS", "4")))
    return int(float(os.environ.get("TRAIN_SAMPLE_STEPS", "25")))


def _read_sample_prompts(trigger: str) -> list[str]:
    """Resolve sample prompts from env or fall back to a sensible default.

    Backend sets TRAIN_SAMPLES as a JSON-encoded list (one entry per
    prompt) when the wizard's Step 4 sample list is non-empty. Older
    callers don't set it → we synthesize one prompt from the trigger.
    """
    raw = os.environ.get("TRAIN_SAMPLES", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = [str(p).strip() for p in parsed if str(p).strip()]
                if cleaned:
                    return cleaned
        except json.JSONDecodeError:
            log(f"Warning: TRAIN_SAMPLES is not valid JSON; ignoring ({raw[:80]}…)")
    if trigger:
        return [f"{trigger}, portrait photo, natural skin texture, studio lighting"]
    return ["portrait photo, studio lighting"]


def _normalise_optimizer(name: str) -> str:
    """Map wizard-side optimizer ids to AI Toolkit's expected strings.
    Returns the AI Toolkit value or falls back to adamw8bit."""
    table = {
        "adamw8bit": "adamw8bit",
        "adamw":     "adamw",
        "prodigy":   "prodigy",
        "lion":      "lion",
        "adafactor": "adafactor",
    }
    return table.get((name or "").strip().lower(), "adamw8bit")


def _normalise_scheduler(name: str) -> str:
    """AI Toolkit consumes the literal scheduler name. Map our wizard
    options; fall back to 'constant' for anything unknown so training
    still launches rather than hard-erroring on the YAML parse."""
    aliases = {
        "cosine":         "cosine",
        "cosine-restart": "cosine_with_restarts",
        "linear":         "linear",
        "constant":       "constant",
    }
    return aliases.get((name or "").strip().lower(), "constant")


def _normalise_precision(name: str) -> str:
    n = (name or "").strip().lower()
    return n if n in {"bf16", "fp16", "fp32"} else "bf16"


def write_config(toolkit_dir: Path, dataset_dir: Path, output_dir: Path) -> Path:
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen-Image-2512")
    output_name = safe_name(
        os.environ.get("OUTPUT_NAME", "flux_klein_lora" if is_flux_klein(base_model) else "qwen_lora")
    )
    trigger = os.environ.get("TRIGGER_PHRASE", "").strip()
    steps = int(float(os.environ.get("TRAIN_STEPS", "3000")))
    rank = int(float(os.environ.get("TRAIN_RANK", "64")))
    lr = float(os.environ.get("TRAIN_LR", "0.0002"))
    resolution = int(float(os.environ.get("TRAIN_RESOLUTION", "1024")))
    batch_size = int(float(os.environ.get("TRAIN_BATCH_SIZE", "1")))
    grad_accum = int(float(os.environ.get("TRAIN_GRAD_ACCUM", "1")))
    save_every = int(float(os.environ.get("TRAIN_SAVE_EVERY", "0") or "0"))
    if save_every <= 0:
        save_every = max(250, min(1000, steps // 4 or 250))

    # Advanced cfg threaded through by the wizard (Phase 2.5). All optional
    # — defaults match the previous hard-coded behaviour so old callers
    # produce byte-identical configs.
    alpha = int(float(os.environ.get("TRAIN_ALPHA", "0") or "0")) or rank
    optimizer = _normalise_optimizer(os.environ.get("TRAIN_OPTIMIZER", "adamw8bit"))
    scheduler = _normalise_scheduler(os.environ.get("TRAIN_SCHEDULER", "constant"))
    precision = _normalise_precision(os.environ.get("TRAIN_PRECISION", "bf16"))
    grad_ckpt = py_bool(os.environ.get("TRAIN_GRAD_CKPT"), True)
    generate_samples = py_bool(os.environ.get("TRAIN_GENERATE_SAMPLES"), True)
    sample_prompts = _read_sample_prompts(trigger) if generate_samples else _read_sample_prompts(trigger)[:1]

    log(
        "Igglepixel training config: "
        f"base_model={base_model}, trigger={trigger or '(none)'}, "
        f"steps={steps}, save_every={save_every}, rank={rank}, alpha={alpha}, "
        f"lr={lr}, optimizer={optimizer}, scheduler={scheduler}, "
        f"resolution={resolution}, batch={batch_size}, precision={precision}, "
        f"grad_ckpt={'on' if grad_ckpt else 'off'}, "
        f"samples={len(sample_prompts) if generate_samples else 0}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{output_name}.yaml"
    toolkit_output = output_dir / "ai-toolkit-output"

    trigger_line = f"    trigger_word: {json.dumps(trigger)}\n" if trigger else ""

    # Build the YAML sample-prompt block. AI Toolkit accepts a list under
    # sample.prompts; JSON strings are valid YAML scalars and preserve
    # quotes, backslashes, and multi-line text safely.
    sample_block = "\n".join(
        f"      - {json.dumps(p)}"
        for p in sample_prompts
    )
    disable_sampling = not generate_samples

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
      linear_alpha: {alpha}
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
      gradient_checkpointing: {str(grad_ckpt).lower()}
      noise_scheduler: "flowmatch"
      optimizer: "{optimizer}"
      lr_scheduler: "{scheduler}"
      lr: {lr}
      dtype: {precision}
      skip_first_sample: true
      disable_sampling: {str(disable_sampling).lower()}
    model:
      name_or_path: "{base_model}"
      arch: "{model_arch(base_model)}"
{model_quant_block(base_model)}    sample:
      sampler: "flowmatch"
      sample_every: {save_every}
      width: {resolution}
      height: {resolution}
      prompts:
{sample_block}
      neg: ""
      seed: 42
      walk_seed: true
      guidance_scale: {sample_guidance_scale(base_model)}
      sample_steps: {sample_steps(base_model)}
    meta:
      name: "[name]"
      version: "1.0"
"""
    config_path.write_text(config, encoding="utf-8")
    log(f"Wrote AI Toolkit config: {config_path}")
    log(f"AI Toolkit network rank written: linear={rank}, linear_alpha={alpha}")
    return config_path


def check_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {dataset_dir}")
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    if not images:
        raise SystemExit("AI Toolkit needs at least one .jpg, .jpeg, .png, or .webp image")
    missing = [p.relative_to(dataset_dir).as_posix() for p in images if not p.with_suffix(".txt").exists()]
    if missing:
        raise SystemExit("Missing caption files for: " + ", ".join(missing[:10]))
    unsupported = [
        p.relative_to(dataset_dir).as_posix()
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".gif", ".bmp"}
    ]
    if unsupported:
        log("Warning: AI Toolkit ignores gif/bmp images: " + ", ".join(unsupported[:10]))
    log(f"Dataset ready for AI Toolkit: {len(images)} images")


def write_transformers_startup_patch(output_dir: Path) -> Path:
    """Patch a Transformers tokenizer metadata lookup that can require network.

    Recent Transformers builds call huggingface_hub.model_info() inside the
    Mistral tokenizer-regex guard, even for Qwen tokenizers. If the model files
    are cached but the pod temporarily has no network, that unrelated lookup can
    abort training. The patch only intercepts that guard's imported model_info
    for known non-Mistral image-training models; normal HF downloads still use
    huggingface_hub directly.
    """
    patch_dir = output_dir / "python_startup"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "sitecustomize.py"
    patch_path.write_text(
        """
from types import SimpleNamespace

try:
    import transformers.tokenization_utils_tokenizers as _tok_utils

    _orig_model_info = getattr(_tok_utils, "model_info", None)

    def _igglepixel_safe_model_info(model_id, *args, **kwargs):
        mid = str(model_id or "").lower()
        try:
            if _orig_model_info is not None:
                return _orig_model_info(model_id, *args, **kwargs)
        except Exception:
            if mid.startswith("qwen/") or "qwen" in mid or "flux.2-klein" in mid:
                return SimpleNamespace(config={"model_type": "qwen2"})
            raise
        return SimpleNamespace(config={})

    if _orig_model_info is not None:
        _tok_utils.model_info = _igglepixel_safe_model_info
except Exception:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    return patch_dir


def run_training(py: Path, toolkit_dir: Path, config_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    patch_dir = write_transformers_startup_patch(config_path.parents[1])
    env["PYTHONPATH"] = (
        str(patch_dir)
        if not env.get("PYTHONPATH")
        else str(patch_dir) + os.pathsep + env["PYTHONPATH"]
    )
    cmd = [str(py), "run.py", str(config_path)]
    log("Starting AI Toolkit training")
    log(f"Using trainer Python startup patch: {patch_dir}")
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


def is_aux_safetensor(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(
        marker in lower
        for marker in (
            "accuracy_recovery",
            "torchao_uint",
            "qwen_image_torchao",
            "qwen_image_2512_torchao",
            "qwen_image_edit_torchao",
        )
    )


def copy_output(output_dir: Path) -> None:
    output_path = Path(os.environ.get("OUTPUT_PATH", output_dir / f"{safe_name(os.environ.get('OUTPUT_NAME', 'qwen_lora'))}.safetensors"))
    candidates = sorted(
        (p for p in output_dir.rglob("*.safetensors") if not is_aux_safetensor(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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
