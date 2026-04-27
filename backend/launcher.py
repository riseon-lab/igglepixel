"""Model subprocess launcher.

Each model declares a `runner_module` (e.g. "backend.runners.flux1_dev"). We
spawn it with `python -m backend.runner_host <module> <port>`, which exposes
`/healthz` and `/generate` on a localhost-only port. The runner inherits:

    WORKSPACE       /workspace mount root
    HF_TOKEN        for gated repos (optional)
    LORAS, LORAS_DIR comma-joined LoRA filenames + their dir

User-supplied params from the drawer travel as the JSON body of /generate, NOT
env vars — runners read them straight from the request.

Logs go to $WORKSPACE/logs/<model_id>.log and are streamed via SSE.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator, Optional


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
LOGS_DIR  = WORKSPACE / "logs"
PORT_BASE = int(os.environ.get("RUNNER_PORT_BASE", "17000"))

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # WORKSPACE may not exist locally; deferred to launch time.


def _free_port(start: int) -> int:
    """Find the next free TCP port on localhost starting at `start`."""
    p = start
    while p < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    raise RuntimeError("No free runner port in range")


class ModelLauncher:
    def __init__(self):
        self._procs: dict[str, dict] = {}     # model_id -> {proc, name, port, pid, status, log_path}
        self._downloads: dict[str, subprocess.Popen] = {}

    # ── Launch ──────────────────────────────────────────────────────────
    async def launch(self, model: dict, loras: list[str], hf_token: Optional[str], quant: Optional[str] = None, variant: Optional[str] = None) -> dict:
        mid = model["id"]
        if mid in self._procs and self._is_alive(self._procs[mid]["proc"]):
            return {
                "status": "already_running",
                "pid":    self._procs[mid]["pid"],
                "port":   self._procs[mid]["port"],
            }

        runner_module = model.get("runner_module")
        if not runner_module:
            return {"status": "error", "message": f"Model '{mid}' has no runner_module"}

        port = _free_port(PORT_BASE)
        env = os.environ.copy()
        env["WORKSPACE"] = str(WORKSPACE)
        env["LORAS_DIR"] = str(WORKSPACE / "loras")
        env["LORAS"]     = ",".join(loras)
        if hf_token:
            env["HF_TOKEN"] = hf_token
        # Quantisation choice (bf16 | int8 | nf4). Runner reads FORGE_QUANT
        # in load() and applies bitsandbytes config accordingly. Defaults to
        # bf16 in the runner if not set.
        if quant:
            env["FORGE_QUANT"] = quant
        # Size variant for models with multiple sizes (e.g. Wan 2.2 14B vs 5B).
        # Runner reads FORGE_VARIANT in load() and resolves the right HF repo.
        if variant:
            env["FORGE_VARIANT"] = variant
        # Pass the at-rest data key to the runner subprocess (hex-encoded).
        # Without this, the runner can't decrypt user refs or encrypt outputs.
        # Use sys.modules['__main__'] — the backend runs as __main__ (not 'main'),
        # so `from main import auth` would re-import a fresh module with no key.
        try:
            _backend_auth = sys.modules["__main__"].auth
            if _backend_auth.is_unlocked():
                env["FORGE_DATA_KEY"] = _backend_auth.data_key.hex()
        except Exception:
            pass

        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            return {"status": "error", "message": f"Cannot create logs dir at {LOGS_DIR}"}

        log_path = LOGS_DIR / f"{mid}.log"
        log_f = open(log_path, "wb")

        # Run from repo root so `python -m backend.runner_host` resolves.
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, "-m", "backend.runner_host", runner_module, str(port)]

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(repo_root),
            start_new_session=True,
        )
        self._procs[mid] = {
            "proc":     proc,
            "model_id": mid,
            "name":     model["name"],
            "pid":      proc.pid,
            "port":     port,
            "status":   "running",
            "log_path": str(log_path),
        }
        return {"status": "launched", "pid": proc.pid, "port": port}

    async def stop(self, model_id: str) -> dict:
        info = self._procs.get(model_id)
        if not info:
            return {"status": "not_running"}
        try:
            os.killpg(os.getpgid(info["proc"].pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        info["status"] = "stopped"
        return {"status": "stopped", "pid": info["pid"]}

    def status(self) -> dict:
        for mid, info in list(self._procs.items()):
            if info["status"] == "running" and not self._is_alive(info["proc"]):
                info["status"] = "exited"
        return {
            mid: {k: v for k, v in info.items() if k != "proc"}
            for mid, info in self._procs.items()
        }

    def get(self, model_id: str) -> Optional[dict]:
        return self._procs.get(model_id)

    # ── Logs (SSE) ───────────────────────────────────────────────────────
    async def stream_logs(self, model_id: str) -> AsyncIterator[str]:
        info = self._procs.get(model_id)
        if not info:
            yield "[no such process]"
            return
        path = Path(info["log_path"])
        last_size = 0
        for _ in range(10_000):
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        last_size = f.tell()
                    if chunk:
                        for line in chunk.decode("utf-8", errors="replace").splitlines():
                            yield line
                except Exception:
                    pass
            if info["status"] != "running":
                break
            await asyncio.sleep(0.6)

    # ── Misc ────────────────────────────────────────────────────────────
    def track_download(self, key: str, proc: subprocess.Popen) -> None:
        self._downloads[key] = proc

    @staticmethod
    def _is_alive(proc: subprocess.Popen) -> bool:
        return proc.poll() is None
