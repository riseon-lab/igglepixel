"""Generic FastAPI shell that wraps any backend.runners.<module>:Runner.

Spawned by backend.launcher as:
    python -m backend.runner_host <runner_module> <port>

Endpoints (bound to 127.0.0.1 only):
    GET  /healthz   → {ready, loading, model_id}
    POST /generate  → forwards body to Runner.generate(params, loras)
"""

from __future__ import annotations

import importlib
import sys
import threading
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class GenerateRequest(BaseModel):
    params: dict = {}
    loras:  list = []     # [{filename: str, strength: float}]
    hf_token: Optional[str] = None


def _last_meaningful_frame(tb: str) -> Optional[str]:
    """Pick the deepest stack frame from a traceback string, preferring our
    own backend code over library internals so the UI toast points at the
    actual call site that raised — not a deep diffusers/peft/torch frame.

    Returns something like 'backend/runners/wan22_i2v.py:142' or
    'site-packages/diffusers/.../something.py:88' if nothing in our code
    appears in the trace.
    """
    if not tb:
        return None
    # traceback.format_exc emits lines like:
    #   File "/workspace/forge-src/backend/runners/wan22_i2v.py", line 142, in generate
    import re
    pat = re.compile(r'File "([^"]+)", line (\d+)')
    matches = pat.findall(tb)
    if not matches:
        return None
    own_match = None
    last_match = None
    for path, line in matches:
        last_match = (path, line)
        # "Our code" = anything under backend/ in the runtime checkout, but
        # not under site-packages.
        if "site-packages" in path:
            continue
        if "/backend/" in path or path.endswith("/main.py"):
            own_match = (path, line)
    chosen = own_match or last_match
    if not chosen:
        return None
    path, line = chosen
    # Trim noisy absolute prefix; keep the last 2-3 path components for context.
    parts = path.split("/")
    short = "/".join(parts[-3:]) if len(parts) > 3 else path
    return f"{short}:{line}"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python -m backend.runner_host <runner_module> <port>")

    runner_module_name = sys.argv[1]
    port               = int(sys.argv[2])

    print(f"[host] importing {runner_module_name}…", flush=True)
    module = importlib.import_module(runner_module_name)
    runner = module.Runner()
    print(f"[host] runner = {runner.model_id}", flush=True)

    app = FastAPI()

    state = {"ready": False, "loading": True, "load_error": None}

    def _load_in_background() -> None:
        try:
            runner.load()
            # Pre-load the moderation model so it's in VRAM by the time
            # the first image generate() call fires. Text runners should not
            # pay this VRAM/dependency cost.
            if getattr(runner, "category", None) in ("image", "video"):
                from backend import moderator
                moderator.init()
            state["ready"] = True
        except Exception as e:
            state["load_error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            state["loading"] = False

    threading.Thread(target=_load_in_background, daemon=True).start()

    @app.get("/healthz")
    def healthz():
        return {
            "ready":      state["ready"],
            "loading":    state["loading"],
            "load_error": state["load_error"],
            "model_id":   runner.model_id,
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if state["load_error"]:
            raise HTTPException(503, f"Model failed to load: {state['load_error']}")
        if not state["ready"]:
            raise HTTPException(503, "Model still loading")
        try:
            return runner.generate(req.params or {}, req.loras or [])
        except Exception as e:
            tb = traceback.format_exc()
            traceback.print_exc()
            # Surface the deepest non-library frame in our error response so
            # the UI toast points at the actual file:line that raised — not
            # just the type/message which on its own usually doesn't tell us
            # where in our code (or which library) failed.
            origin = _last_meaningful_frame(tb)
            detail = f"{type(e).__name__}: {e}"
            if origin:
                detail = f"{detail} · at {origin}"
            raise HTTPException(500, detail)

    @app.post("/cancel")
    def cancel():
        """Asks the runner to interrupt at the next step boundary.

        Runners opt-in by exposing a `cancel()` method. Models that don't
        support cancellation simply return `{supported: False}`.
        """
        if hasattr(runner, "cancel"):
            try:
                runner.cancel()
                return {"supported": True}
            except Exception as e:
                raise HTTPException(500, f"{type(e).__name__}: {e}")
        return {"supported": False}

    print(f"[host] listening on 127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
