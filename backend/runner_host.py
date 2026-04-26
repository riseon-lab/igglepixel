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
    loras:  list[str] = []
    hf_token: Optional[str] = None


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
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")

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
