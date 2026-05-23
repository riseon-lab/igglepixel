"""Qwen2.5-VL vision captioner runner for Dataset Studio.

Exposes a standard Runner class that spawns an internal vLLM OpenAI-compatible
API server process group and maps generate() requests to it.

Eviction and VRAM hygiene are managed automatically by ModelLauncher.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import socket
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


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
    raise RuntimeError("No free vLLM port in range")


class Runner(RunnerBase):
    model_id            = "qwen25-vl-captioner"
    model_name          = "Qwen2.5-VL Captioner"
    category            = "vision"
    supports_lora       = False
    min_vram_gb         = 16
    recommended_vram_gb = 24

    def __init__(self) -> None:
        self._vllm_proc: Optional[subprocess.Popen] = None
        self._vllm_port: Optional[int] = None
        self._cancel = False

    def load(self) -> None:
        import httpx

        # vLLM is executed inside the profile isolated virtualenv
        print("[runner] preparing to launch isolated vLLM vision server…", flush=True)
        self._vllm_port = _free_port(18000)
        
        # Build vLLM OpenAI-compatible server command
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
            "--host", "127.0.0.1",
            "--port", str(self._vllm_port),
            "--trust-remote-code",
            "--dtype", "auto",
            "--gpu-memory-utilization", os.environ.get("IGGLEPIXEL_VISION_GPU_MEMORY", "0.82"),
            "--max-model-len", os.environ.get("IGGLEPIXEL_VISION_MAX_MODEL_LEN", "8192"),
        ]

        # Support optional low-VRAM cache dtype override
        kv_cache_dtype = os.environ.get("IGGLEPIXEL_VISION_KV_CACHE_DTYPE")
        if kv_cache_dtype:
            cmd.extend(["--kv-cache-dtype", kv_cache_dtype])

        print(f"[runner] starting vLLM command: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)

        env = os.environ.copy()
        # Confirm the auth state without leaking the secret — vLLM logs an
        # "unauthenticated requests" warning when HF_TOKEN is unset, which
        # then rate-limits the weight download and silently stalls.
        hf_tok = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
        print(
            f"[runner] HF_TOKEN {'set' if hf_tok else 'unset'}"
            + (f" (len={len(hf_tok)})" if hf_tok else "")
            + " — vLLM will use this for HF Hub downloads",
            flush=True,
        )
        # Direct logs straight to stdout so they stream into the runner's log file
        self._vllm_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            start_new_session=True,
        )

        # Poll `/v1/models` until the API server is healthy and fully loaded.
        # First-launch weight download for Qwen2.5-VL-7B is ~16 GB, so even
        # on a fast HF connection cold-start runs 4-6 min. The previous
        # 2.5 min cap killed the runner mid-download. Overridable via
        # IGGLEPIXEL_VISION_LOAD_TIMEOUT seconds for slow links / larger
        # models.
        load_timeout_s = int(os.environ.get("IGGLEPIXEL_VISION_LOAD_TIMEOUT", "900"))
        print(
            f"[runner] waiting up to {load_timeout_s}s for vLLM vision server to bind to port {self._vllm_port}…",
            flush=True,
        )
        ready = False
        start_time = time.time()
        last_heartbeat = start_time
        deadline = start_time + load_timeout_s
        while time.time() < deadline:
            if self._vllm_proc.poll() is not None:
                rc = self._vllm_proc.poll()
                raise RuntimeError(f"vLLM server subprocess exited unexpectedly with code {rc}")
            try:
                res = httpx.get(f"http://127.0.0.1:{self._vllm_port}/v1/models", timeout=2.0)
                if res.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            now = time.time()
            if now - last_heartbeat >= 30:
                print(
                    f"[runner] still waiting for vLLM ({now - start_time:.0f}s elapsed of {load_timeout_s}s budget) — vLLM is downloading/loading weights",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(1.0)

        if not ready:
            self.terminate_vllm()
            raise RuntimeError(f"vLLM vision server failed to respond within {load_timeout_s}s (elapsed: {time.time() - start_time:.1f}s)")

        print(f"[runner] vLLM vision server is ready and serving on port {self._vllm_port}", flush=True)

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import httpx
        import base64

        if not self._vllm_proc or self._vllm_proc.poll() is not None or not self._vllm_port:
            raise RuntimeError("vLLM vision runner is not loaded or has crashed")

        self._cancel = False
        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        image_base64 = params.get("image_base64")
        image_path = params.get("image_path")

        # Resolve image to base64. Dataset images live under
        # /workspace/datasets/ and are intentionally stored as plain
        # bytes — AI Toolkit reads them directly during training and
        # the `/api/trainers/file` endpoint serves them without going
        # through the encryption layer. So we read raw bytes here too,
        # rather than calling self.load_image() (which routes through
        # crypto.read_decrypted and refuses non-.enc files). Path is
        # sandboxed to /workspace so a malicious payload can't escape
        # the workspace root.
        if not image_base64 and image_path:
            from PIL import Image
            img_path = Path(image_path)
            if not img_path.is_absolute():
                img_path = WORKSPACE / img_path
            try:
                img_path = img_path.resolve()
                img_path.relative_to(WORKSPACE.resolve())
            except ValueError:
                raise RuntimeError(f"Refusing to read image outside workspace: {image_path}")
            if not img_path.exists() or not img_path.is_file():
                raise FileNotFoundError(f"Dataset image not found: {image_path}")
            try:
                pil_img = Image.open(img_path).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=95)
                image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset image at {img_path}: {e}")

        if not image_base64:
            raise ValueError("Either `image_base64` or `image_path` must be provided")

        # Strip header if it contains data URL format
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Call the local vLLM OpenAI endpoint
        temperature = float(params.get("temperature", 0.1))
        body = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "temperature": temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
        }

        headers = {"Content-Type": "application/json"}
        try:
            res = httpx.post(
                f"http://127.0.0.1:{self._vllm_port}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=120.0,
            )
            if res.status_code != 200:
                raise RuntimeError(f"vLLM server returned error code {res.status_code}: {res.text}")
            
            json_res = res.json()
            response_text = json_res.get("choices", [{}])[0].get("message", {}).get("content", "") or json_res.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with local vLLM vision server: {e}")

        return self.asset_response([], meta={
            "model": self.model_id,
            "text": response_text,
        })

    def cancel(self) -> None:
        self._cancel = True

    def terminate_vllm(self) -> None:
        if self._vllm_proc:
            print("[runner] terminating internal vLLM vision server…", flush=True)
            self._vllm_proc.terminate()
            try:
                self._vllm_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._vllm_proc.kill()
                self._vllm_proc.wait(timeout=3)
            self._vllm_proc = None

    def __del__(self) -> None:
        self.terminate_vllm()
