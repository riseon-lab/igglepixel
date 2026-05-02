"""Shared runtime-profile virtual environment lifecycle.

Some model runners need a different Python version or pinned deps that
conflict with the shared /usr/bin/python image. LTX-2.3 is the first profile
consumer — its `ltx-pipelines` stack brings Torch ~2.7-era dependencies that
we do not want to force onto every other runner. Rather than bumping the
entire image (which would force every other runner to re-install its deps),
we let models point at shared `runtime_profiles` in the registry and spin up
isolated venvs for those dependency profiles.

The launcher reads `model.runtime["id"]`, asks `runtime_python(id)` for
the venv's Python interpreter, and spawns the runner subprocess against
that. Other runners are unaffected.

Tooling: prefers `uv` (single static binary, manages Python versions +
fast pip). Falls back to `python -m venv`, then `virtualenv`, when uv
isn't installed.

Persistence: venvs live under WORKSPACE/venvs/<id>/ — same persistent
volume that holds models and pip cache, so a re-installed runtime is a
near-instant cache hit on a warm pod.

This module is process-shared state (in-memory job dicts on the FastAPI
app), so import once at module level from main.py — never per-request.
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
VENVS_DIR = WORKSPACE / "venvs"
REPOS_DIR = WORKSPACE / "repos"
HF_HOME_DIR   = WORKSPACE / ".cache" / "huggingface"
PIP_CACHE_DIR = WORKSPACE / ".cache" / "pip"
SPEC_MARKER = ".igglepixel-runtime-spec.sha256"

os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME_DIR / "datasets"))
os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))

# Initialised on import so callers can rely on the dirs existing. WORKSPACE
# may not exist locally (dev sandbox); endpoints handle that on demand.
for d in (VENVS_DIR, REPOS_DIR, HF_HOME_DIR, PIP_CACHE_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _has_uv() -> bool:
    return shutil.which("uv") is not None


def _venv_dir(runtime_id: str) -> Path:
    return VENVS_DIR / runtime_id


def _venv_python(runtime_id: str) -> Path:
    """Path to the venv's Python interpreter. Doesn't check existence —
    pair with `is_runtime_ready()` for that."""
    return _venv_dir(runtime_id) / "bin" / "python"


def _spec_marker(runtime_id: str) -> Path:
    return _venv_dir(runtime_id) / SPEC_MARKER


def _spec_hash(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _spec_matches(runtime_id: str, spec: Optional[dict]) -> bool:
    if not spec:
        return True
    marker = _spec_marker(runtime_id)
    if not marker.exists():
        return False
    try:
        return marker.read_text().strip() == _spec_hash(spec)
    except OSError:
        return False


def is_runtime_ready(runtime_id: str, spec: Optional[dict] = None) -> bool:
    """True iff the venv exists, is executable, and matches the spec."""
    p = _venv_python(runtime_id)
    return p.exists() and os.access(p, os.X_OK) and _spec_matches(runtime_id, spec)


def runtime_python(runtime_id: str, spec: Optional[dict] = None) -> Optional[Path]:
    """Returns the venv's Python path if ready, else None.

    Used by the launcher to decide whether to spawn a runner against its
    own venv or fall through to sys.executable.
    """
    return _venv_python(runtime_id) if is_runtime_ready(runtime_id, spec) else None


def runtime_status(runtime_id: str, spec: Optional[dict] = None) -> dict:
    """Snapshot for the UI: state + the resolved Python path when ready.

    The active install job (if any) is tracked separately in main.py's
    `runtime_install_jobs` dict — this function only reports the on-disk
    state, not in-progress work.
    """
    if is_runtime_ready(runtime_id, spec):
        return {
            "state":       "ready",
            "python_path": str(_venv_python(runtime_id)),
        }
    if _venv_dir(runtime_id).exists():
        if _venv_python(runtime_id).exists() and spec and not _spec_matches(runtime_id, spec):
            return {"state": "stale", "note": "runtime dependencies changed; reinstall required"}
        # Directory exists but the python binary doesn't — half-installed.
        # Treat as missing so the user can re-run install to repair it.
        return {"state": "missing", "note": "venv directory exists but python missing"}
    return {"state": "missing"}


def _run(cmd: list[str], log: Callable[[str], None], cwd: Optional[Path] = None) -> int:
    """Run a subprocess, streaming each stdout/stderr line to `log`.

    Why streaming: pip installs for ML deps take 3–5 minutes and can
    download multi-GB wheels. The UI status endpoint reads the latest
    log line as a "currently installing X" hint, so we need them
    available as the install runs, not buffered to the end.
    """
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    return proc.returncode


def _ensure_git_clone(spec: dict, log: Callable[[str], None]) -> None:
    """Clone the repo if missing, fetch + checkout the requested ref otherwise."""
    repo = spec.get("repo")
    ref  = spec.get("ref", "main")
    dest = Path(spec.get("dest") or (REPOS_DIR / Path(repo).stem))
    required_paths = [dest / p for p in (spec.get("required_paths") or [])]

    def _has_required_checkout() -> bool:
        if required_paths:
            return all(p.exists() for p in required_paths)
        return dest.exists() and any(dest.iterdir())

    def _is_sha_ref() -> bool:
        return len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref)

    def _cached_checkout_ok() -> bool:
        if not _has_required_checkout():
            return False
        if not _is_sha_ref():
            return True
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(dest),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return head.startswith(ref) or ref.startswith(head)
        except Exception:
            return False

    def _checkout_ref() -> int:
        rc = _run(["git", "fetch", "--depth", "50", "origin", ref], log, cwd=dest)
        if rc != 0:
            if _is_sha_ref():
                cached_rc = _run(["git", "checkout", "--force", ref], log, cwd=dest)
                if cached_rc == 0:
                    return 0
            return rc
        return _run(["git", "checkout", "--force", "FETCH_HEAD"], log, cwd=dest)

    def _clone(url: str) -> int:
        rc = _run(["git", "clone", "--depth", "50", url, str(dest)], log)
        if rc != 0:
            return rc
        return _checkout_ref()

    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = _clone(repo)
        if rc != 0 and repo and not repo.endswith(".git"):
            shutil.rmtree(dest, ignore_errors=True)
            alt_repo = repo.rstrip("/") + ".git"
            log(f"WARN: clone failed; retrying with {alt_repo}")
            rc = _clone(alt_repo)
        if rc != 0:
            raise RuntimeError(f"git clone failed: {repo}@{ref}")
        return

    # Existing clone — fetch the requested branch/tag/SHA and detach there.
    rc = _checkout_ref()
    if rc != 0:
        if _cached_checkout_ok():
            log(f"WARN: git fetch failed for {repo}@{ref}; using cached checkout at {dest}")
            return
        log("WARN: cached checkout incomplete; recloning")
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = _clone(repo)
        if rc != 0 and repo and not repo.endswith(".git"):
            alt_repo = repo.rstrip("/") + ".git"
            log(f"WARN: clone failed; retrying with {alt_repo}")
            rc = _clone(alt_repo)
        if rc == 0:
            return
        raise RuntimeError(f"git fetch failed: {repo}@{ref}")


def _create_venv(runtime_id: str, python_version: Optional[str], log: Callable[[str], None]) -> None:
    """Create the venv with the requested Python version.

    Prefers uv (handles Python version installs + venv in one step).
    Falls back to `python -m venv`, then `virtualenv`, against the best
    available Python. The virtualenv fallback covers slim containers where
    the stdlib venv module cannot seed pip.
    """
    venv = _venv_dir(runtime_id)
    if venv.exists():
        # Wipe a partial venv so a re-install starts clean. We hold off on
        # this for "ready" state (handled by is_runtime_ready upstream).
        log(f"removing partial venv at {venv}")
        shutil.rmtree(venv, ignore_errors=True)

    # Tee log lines into a local tail buffer so we can fold the most
    # recent stderr into the RuntimeError message — without that, the
    # toast just says "uv venv failed" and the user has to dig the actual
    # cause out of the install log endpoint.
    from collections import deque
    tail: deque = deque(maxlen=12)

    def tee(line: str) -> None:
        tail.append(line)
        log(line)

    def _fail(step: str) -> None:
        snippet = "\n".join(tail) if tail else "(no log captured)"
        raise RuntimeError(
            f"{step} failed for runtime '{runtime_id}'. Last log lines:\n{snippet}"
        )

    if _has_uv():
        # Surface the uv version up front so a "uv venv failed" log is
        # easier to triage (different uv versions handle --python very
        # differently — older builds didn't auto-fetch managed Python).
        _run(["uv", "--version"], tee)

        # Pre-fetch the requested managed Python as a discrete step. This
        # separates "download interpreter" from "make venv" so the user can
        # see exactly which step failed in the install log. uv's `python
        # install` is idempotent — already-installed versions are a no-op.
        if python_version:
            log(f"== ensuring managed Python {python_version} ==")
            rc = _run(["uv", "python", "install", python_version], tee)
            if rc != 0:
                _fail(f"uv python install {python_version}")

        cmd = ["uv", "venv", str(venv)]
        if python_version:
            cmd.extend(["--python", python_version])
        rc = _run(cmd, tee)
        if rc != 0:
            _fail("uv venv")
        return

    # No uv — try the system python -m venv. Only honest Python version
    # we can pick is whatever python<version> binary the host has.
    py_bin = sys.executable
    if python_version:
        # Look for an interpreter named exactly that version (e.g. python3.12).
        candidate = shutil.which(f"python{python_version}")
        if candidate:
            py_bin = candidate
        else:
            log(f"WARN: no python{python_version} binary found; using {py_bin}")
    rc = _run([py_bin, "-m", "venv", str(venv)], log)
    if rc == 0:
        return

    log("WARN: python -m venv failed; trying virtualenv fallback")
    shutil.rmtree(venv, ignore_errors=True)
    if _run([py_bin, "-m", "virtualenv", "--version"], log) != 0:
        log("virtualenv not available; installing it into the base Python")
        install_rc = _run([py_bin, "-m", "pip", "install", "--user", "virtualenv"], log)
        if install_rc != 0:
            install_rc = _run([py_bin, "-m", "pip", "install", "virtualenv"], log)
        if install_rc != 0:
            raise RuntimeError(f"python -m venv failed and virtualenv could not be installed for runtime '{runtime_id}'")

    rc = _run([py_bin, "-m", "virtualenv", str(venv)], log)
    if rc != 0:
        raise RuntimeError(f"python -m venv and virtualenv failed for runtime '{runtime_id}'")


def _pip_install(runtime_id: str, packages: list[str], log: Callable[[str], None]) -> None:
    """Install pip packages into the venv.

    `packages` is a flat list passed as-is to pip (so the registry can
    include `-e /path/to/local`, version pins, etc. transparently).
    """
    if not packages:
        return
    py = _venv_python(runtime_id)
    if _has_uv():
        # uv pip install --python <venv_py> uses the venv as the install
        # target without activating it.
        cmd = ["uv", "pip", "install", "--python", str(py), *packages]
    else:
        cmd = [str(py), "-m", "pip", "install", *packages]
    rc = _run(cmd, log)
    if rc != 0:
        raise RuntimeError(f"pip install failed for runtime '{runtime_id}'")


def _verify_imports(runtime_id: str, imports: list[str], log: Callable[[str], None]) -> None:
    """Fail fast if the prepared venv cannot import the runtime's essentials."""
    if not imports:
        return
    py = _venv_python(runtime_id)
    code = (
        "import importlib\n"
        f"mods = {imports!r}\n"
        "for name in mods:\n"
        "    importlib.import_module(name)\n"
        "print('verified imports: ' + ', '.join(mods))\n"
    )
    rc = _run([str(py), "-c", code], log)
    if rc != 0:
        raise RuntimeError(f"runtime import verification failed for '{runtime_id}'")


def ensure_runtime(spec: dict, log: Callable[[str], None] = print) -> Path:
    """Idempotently install a runtime venv described by a registry spec.

    Spec shape (from model_registry.json's `runtime` block):
        {
          "id":      "ltx23",                         // venv directory
          "python":  "3.12",                          // optional version
          "git":     {                                // optional source clone
            "repo": "https://github.com/...",
            "ref":  "main",
            "dest": "/workspace/repos/ltx-2"
          },
          "pip":     ["-e", "/path/to/pkg", "transformers", ...]
        }

    Returns the Python interpreter path on success; raises on failure so
    the caller can mark the install job 'error' with the exception
    message intact.
    """
    runtime_id = spec.get("id")
    if not runtime_id:
        raise ValueError("runtime spec missing 'id'")

    # Step 1: optional git clone (must run before pip so `-e <local>`
    # references resolve to a real directory).
    git = spec.get("git")
    if git:
        log(f"== git: {git.get('repo')}@{git.get('ref', 'main')} ==")
        _ensure_git_clone(git, log)

    # Step 2: venv. We always recreate when the venv isn't already ready
    # (caller checks is_runtime_ready before invoking ensure_runtime when
    # they want a fast no-op).
    log(f"== venv: {runtime_id} (python={spec.get('python', 'system')}) ==")
    _create_venv(runtime_id, spec.get("python"), log)

    # Step 3: pip install.
    pip_packages = spec.get("pip") or []
    if pip_packages:
        log(f"== pip: {len(pip_packages)} package(s) ==")
        _pip_install(runtime_id, pip_packages, log)

    verify_imports = spec.get("verify_imports") or []
    if verify_imports:
        log(f"== verify imports: {len(verify_imports)} module(s) ==")
        _verify_imports(runtime_id, verify_imports, log)

    _spec_marker(runtime_id).write_text(_spec_hash(spec))
    log("== done ==")
    return _venv_python(runtime_id)
