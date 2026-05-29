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
# Pin AI Toolkit to a specific commit/tag to freeze BOTH its code and its
# (unpinned) requirements.txt to a known-good snapshot. Defaults to "main"
# (rolling), which is the source of the recurring dependency-drift breakage
# — set IGGLEPIXEL_AI_TOOLKIT_REF=<sha> once a good commit is known to stop
# the drift permanently.
TOOLKIT_REF = os.environ.get("IGGLEPIXEL_AI_TOOLKIT_REF", "main").strip() or "main"
VENV_DIR = Path(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_VENV", WORKSPACE / "venvs" / "ai-toolkit"))
# Narrow pip constraints applied to the AI Toolkit requirements install.
# Bundled with the repo so it deploys via the normal git pull.
AI_TOOLKIT_CONSTRAINTS = Path(__file__).resolve().parent / "ai_toolkit_constraints.txt"


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


def safe_workspace_child(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise SystemExit(f"Refusing to delete {label} outside {root_resolved}: {resolved}")
    if resolved == root_resolved:
        raise SystemExit(f"Refusing to delete {label} root: {resolved}")
    return resolved


def ensure_ai_toolkit() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    TOOLKIT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_RECLONE"), False) and TOOLKIT_DIR.exists():
        safe_workspace_child(TOOLKIT_DIR, WORKSPACE / "repos", "AI Toolkit checkout")
        log(f"Deleting AI Toolkit checkout before bootstrap: {TOOLKIT_DIR}")
        shutil.rmtree(TOOLKIT_DIR)
    if not (TOOLKIT_DIR / "run.py").exists():
        if TOOLKIT_DIR.exists():
            raise SystemExit(f"{TOOLKIT_DIR} exists but does not look like ai-toolkit")
        log(f"Cloning AI Toolkit into {TOOLKIT_DIR}")
        run(["git", "clone", "--depth", "1", TOOLKIT_REPO, str(TOOLKIT_DIR)])
        if TOOLKIT_REF != "main":
            log(f"Checking out AI Toolkit ref {TOOLKIT_REF}")
            run(["git", "fetch", "--depth", "1", "origin", TOOLKIT_REF], cwd=TOOLKIT_DIR)
            run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=TOOLKIT_DIR)
    elif py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_UPDATE"), False):
        log(f"Updating AI Toolkit to {TOOLKIT_REF}")
        run(["git", "fetch", "--depth", "1", "origin", TOOLKIT_REF], cwd=TOOLKIT_DIR)
        run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=TOOLKIT_DIR)

    run(["git", "submodule", "update", "--init", "--recursive"], cwd=TOOLKIT_DIR)
    return TOOLKIT_DIR


def ensure_venv(toolkit_dir: Path) -> Path:
    if py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_DELETE_VENV"), False) and VENV_DIR.exists():
        safe_workspace_child(VENV_DIR, WORKSPACE / "venvs", "AI Toolkit venv")
        log(f"Deleting AI Toolkit venv before bootstrap: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)
    py = VENV_DIR / "bin" / "python"
    stamp = VENV_DIR / ".igglepixel_ai_toolkit_ready"
    if not py.exists():
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        log(f"Creating AI Toolkit venv at {VENV_DIR}")
        venv_args = [sys.executable, "-m", "venv"]
        if py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_SYSTEM_SITE"), False):
            venv_args.append("--system-site-packages")
        try:
            run([*venv_args, str(VENV_DIR)])
        except subprocess.CalledProcessError:
            log("python -m venv failed; trying virtualenv fallback")
            virtualenv_args = [sys.executable, "-m", "virtualenv"]
            if py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_SYSTEM_SITE"), False):
                virtualenv_args.append("--system-site-packages")
            run([*virtualenv_args, str(VENV_DIR)])

    reinstall = py_bool(os.environ.get("IGGLEPIXEL_AI_TOOLKIT_REINSTALL"), False)
    if reinstall or not stamp.exists():
        log("Installing AI Toolkit requirements. First run can take several minutes.")
        constraints = AI_TOOLKIT_CONSTRAINTS if AI_TOOLKIT_CONSTRAINTS.exists() else None
        # Bound setuptools below the pkg_resources.packaging removal (71.x)
        # up front, so the whole install — including CLIP's import — runs
        # against a compatible setuptools rather than the absolute latest.
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<71"])
        req_cmd = [str(py), "-m", "pip", "install", "-r", str(toolkit_dir / "requirements.txt")]
        if constraints:
            # -c freezes the narrow set of drift-prone packages
            # (setuptools, numpy) during resolution so requirements.txt
            # can't bump them back into broken territory.
            log(f"Applying dependency constraints from {constraints}")
            req_cmd.extend(["-c", str(constraints)])
        run(req_cmd)
        stamp.write_text(str(time.time()), encoding="utf-8")
    else:
        log("AI Toolkit requirements already installed")
    # ORDER MATTERS. AI Toolkit's requirements.txt is unpinned, so a fresh
    # install pulls whatever torch is newest on PyPI — currently torch 2.12.0
    # built for CUDA 13, which refuses to import on this pod's CUDA 12.8
    # driver (torch._dynamo trips on a missing `sys.get_int_max_str_digits`
    # / "driver too old"). Every downstream probe that imports torch then
    # fails. We MUST realign torch to the cu128 stack before running the
    # numpy/scipy or transformers probes — otherwise the numpy/scipy repair
    # misreads a broken-torch import as an ABI problem and hard-aborts the
    # whole bootstrap (which is exactly the rebuild loop being hit).
    ensure_cuda_torch_stack(py)
    ensure_numpy_scipy_abi(py)
    ensure_qwen3_vl_transformers(py)
    ensure_torchaudio(py)
    ensure_pkg_resources_packaging(py)
    log_python_stack(py)
    return py


def py_import_ok(py: Path, module: str) -> bool:
    probe = subprocess.run(
        [str(py), "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def ensure_qwen3_vl_transformers(py: Path) -> None:
    module = "transformers.models.qwen3_vl.configuration_qwen3_vl"
    if py_import_ok(py, module):
        return
    log("AI Toolkit transformers install is missing qwen3_vl; upgrading transformers")
    run([str(py), "-m", "pip", "install", "--upgrade", "transformers>=4.57.0"])
    if not py_import_ok(py, module):
        raise SystemExit("AI Toolkit transformers upgrade did not provide qwen3_vl support")


def torch_imports_with_cuda(py: Path) -> tuple[bool, str, bool, str]:
    """Probe torch: does it FUNCTION for training, what CUDA build is it,
    and can the pod's driver run it?

    Returns (import_ok, cuda_version, cuda_usable, raw_output):
      - import_ok:   `import torch` AND `import torch._dynamo` both
                     succeeded. The _dynamo import is the load-bearing
                     check: torch 2.12.0 imports fine and reports CUDA
                     available, but its `_dynamo/polyfills/sys.py`
                     references `sys.get_int_max_str_digits` and crashes
                     on this Python — which then takes down every
                     diffusers/transformers import the trainer needs.
                     A plain `import torch` would falsely pass it.
      - cuda_version: torch.version.cuda string ('12.8', '13.0', ...) or
                      '' for a CPU build / failed import.
      - cuda_usable: torch is functional (import_ok), is a CUDA build,
                     AND torch.cuda.is_available() is True.
    """
    code = (
        "import torch\n"
        # The thing that actually breaks with the latest torch on this
        # Python — import it explicitly so a broken-but-loadable torch
        # fails the probe instead of slipping through.
        "import torch._dynamo  # noqa: F401\n"
        "avail = False\n"
        "try:\n"
        "    avail = torch.cuda.is_available()\n"
        "except Exception:\n"
        "    avail = False\n"
        "print('CUDA=' + str(torch.version.cuda or ''))\n"
        "print('AVAIL=' + ('1' if avail else '0'))\n"
        "print('VER=' + torch.__version__)\n"
    )
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return False, "", False, output
    cuda = ""
    avail = False
    for line in proc.stdout.splitlines():
        if line.startswith("CUDA="):
            cuda = line.split("=", 1)[1].strip()
        elif line.startswith("AVAIL="):
            avail = line.split("=", 1)[1].strip() == "1"
    return True, cuda, (bool(cuda) and avail), output


def ensure_cuda_torch_stack(py: Path) -> None:
    """Make sure torch is actually usable for training; only force the
    known-good cu128 2.8.0 stack when it isn't.

    AI Toolkit's requirements.txt doesn't pin torch, so a clean install
    grabs the newest PyPI torch — currently 2.12.0, which is broken for
    our use on this Python regardless of CUDA: it imports and reports
    CUDA available, but `torch._dynamo` crashes on
    `sys.get_int_max_str_digits`, which takes down every diffusers /
    transformers import the trainer needs. torch 2.8.0 (cu128) works and
    runs on both CUDA 12.8 and 13 drivers (CUDA is backward-compatible).

    "Usable" therefore means: torch AND torch._dynamo import, it's a CUDA
    build, and torch.cuda.is_available() is True (see the probe). We're
    CUDA-version-agnostic — a genuinely-working newer torch on a CUDA-13
    pod would be left alone — but a broken-but-loadable torch (like
    2.12.0) gets replaced rather than slipping through on an import+
    is_available() check.
    """
    import_ok, cuda, usable, output = torch_imports_with_cuda(py)
    if usable:
        log(f"AI Toolkit torch is usable on CUDA {cuda}; leaving torch stack as-is")
        return
    if import_ok:
        log(f"AI Toolkit torch imports + torch._dynamo OK but the GPU isn't usable (build CUDA '{cuda or 'none'}', cuda.is_available=False) — installing the known-good cu128 stack")
    else:
        log("AI Toolkit torch is broken for training (import torch / torch._dynamo failed — likely a too-new torch); installing the known-good cu128 2.8.0 stack")
        for line in output.splitlines()[-6:]:
            log("  " + line)
    install_cuda128_torch_stack(py)
    import_ok, cuda, usable, output = torch_imports_with_cuda(py)
    if not usable:
        log("torch still can't use CUDA after the cu128 install:")
        for line in output.splitlines()[-8:]:
            log("  " + line)
        raise SystemExit("AI Toolkit CUDA torch stack install failed")
    log(f"AI Toolkit CUDA torch stack ready (CUDA {cuda})")


def pkg_resources_packaging_ok(py: Path) -> bool:
    probe = subprocess.run(
        [str(py), "-c", "from pkg_resources import packaging"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def ensure_pkg_resources_packaging(py: Path) -> None:
    """Keep `from pkg_resources import packaging` importable for CLIP.

    AI Toolkit pulls openai-CLIP (via k_diffusion), whose clip.py still
    does `from pkg_resources import packaging`. setuptools >= 71 removed
    the vendored `packaging` re-export from pkg_resources, and the
    bootstrap's `pip install --upgrade ... setuptools` grabs the latest —
    so on a fresh venv that import dies and the whole training job fails
    before it starts. Pin setuptools back below the removal at runtime
    (the venv is already built, so the older setuptools only affects
    runtime pkg_resources behaviour, not wheel building).
    """
    if pkg_resources_packaging_ok(py):
        return
    log("CLIP needs pkg_resources.packaging; pinning setuptools below the 71.x removal")
    run([str(py), "-m", "pip", "install", "--upgrade", "setuptools==69.5.1"])
    if pkg_resources_packaging_ok(py):
        log("pkg_resources.packaging restored (setuptools 69.5.1)")
        return
    # Belt-and-braces: an even older setuptools that definitely vendors it.
    log("setuptools 69.5.1 still missing packaging; falling back to 65.5.1")
    run([str(py), "-m", "pip", "install", "setuptools==65.5.1"])
    if pkg_resources_packaging_ok(py):
        log("pkg_resources.packaging restored (setuptools 65.5.1)")
        return
    raise SystemExit(
        "Could not restore pkg_resources.packaging for CLIP — set "
        "IGGLEPIXEL_AI_TOOLKIT_REINSTALL=1 to rebuild the venv, or pin clip "
        "to a build that doesn't import packaging from pkg_resources"
    )


def numpy_scipy_probe(py: Path) -> tuple[bool, str]:
    code = (
        "import numpy\n"
        "import scipy\n"
        "import scipy.stats\n"
        "from diffusers import DPMSolverMultistepScheduler\n"
        "print('numpy=' + numpy.__version__)\n"
        "print('scipy=' + scipy.__version__)\n"
        "print('diffusers_scheduler=ok')\n"
    )
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def ensure_numpy_scipy_abi(py: Path) -> None:
    ok, details = numpy_scipy_probe(py)
    if ok:
        return
    # Guard against misdiagnosis: if the probe failed because torch itself
    # can't import (not because numpy/scipy have an ABI mismatch),
    # reinstalling numpy/scipy is futile and we'd burn a rebuild cycle on
    # the wrong fix. ensure_cuda_torch_stack() runs before us so this
    # should already be healthy, but fail loudly with the real cause if not.
    torch_ok, _, _, torch_out = torch_imports_with_cuda(py)
    if not torch_ok:
        log("NumPy/SciPy probe failed because torch will not import — this is a torch problem, not a numpy/scipy ABI one:")
        for line in torch_out.splitlines()[-8:]:
            log("  " + line)
        raise SystemExit("AI Toolkit torch import is broken; fix the torch stack before the numpy/scipy ABI step")
    log("Repairing AI Toolkit NumPy/SciPy binary compatibility")
    if details:
        log("NumPy/SciPy probe failed:")
        for line in details.splitlines()[-12:]:
            log("  " + line)
    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "numpy==1.26.4",
        "scipy==1.11.4",
    ])
    ok, details = numpy_scipy_probe(py)
    if ok:
        log("AI Toolkit NumPy/SciPy ABI repaired")
        return
    if details:
        log("NumPy/SciPy probe still failed:")
        for line in details.splitlines()[-12:]:
            log("  " + line)
    raise SystemExit("AI Toolkit NumPy/SciPy binary compatibility repair failed")


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


def log_python_stack(py: Path) -> None:
    code = (
        "import importlib.metadata as md\n"
        "import torch\n"
        "print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda))\n"
        "for pkg in ('numpy', 'scipy', 'diffusers', 'transformers', 'accelerate', 'bitsandbytes', 'torchvision', 'torchaudio'):\n"
        "    try:\n"
        "        print(pkg + '=' + md.version(pkg))\n"
        "    except Exception as exc:\n"
        "        print(pkg + '=missing:' + type(exc).__name__)\n"
    )
    proc = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip()
    if output:
        log("AI Toolkit Python stack:")
        for line in output.splitlines()[-10:]:
            log("  " + line)


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
        if "cuda=None" in details or "cuda=" not in details:
            log("AI Toolkit torch install is CPU-only; installing CUDA 12.8 torch stack")
            install_cuda128_torch_stack(py)
            ok, details = torchaudio_probe(py)
            if ok:
                if "cuda=None" not in details and "cuda=" in details:
                    log("AI Toolkit CUDA torch stack installed")
                    return
                raise SystemExit("AI Toolkit torch install is still CPU-only after CUDA stack repair")
        else:
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
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
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
