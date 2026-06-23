#!/usr/bin/env python3
import argparse
import base64
import gc
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

PIPE = None
LOADED_LORAS = ()
# Loading happens on a background thread so POST /load returns immediately and the
# UI can poll /health for progress instead of holding a multi-minute request open.
LOADING = False
LOAD_ERROR = None
_LOAD_LOCK = threading.Lock()

MODE = os.getenv("RUNNER_MODE", "txt2img")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image")
PORT = int(os.getenv("PORT", "8000"))
WORKSPACE = Path(os.getenv("CITIVIA_DATA_DIR", "/workspace"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", WORKSPACE / "outputs" / MODE))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", WORKSPACE / "models"))
LORA_DIR = Path(os.getenv("LORA_DIR", WORKSPACE / "loras"))
SAVE_OUTPUTS = os.getenv("RUNNER_SAVE_OUTPUTS") == "1"


def json_response(handler, status, body):
  data = json.dumps(body).encode()
  handler.send_response(status)
  handler.send_header("content-type", "application/json")
  handler.send_header("content-length", str(len(data)))
  handler.end_headers()
  handler.wfile.write(data)


def decode_json(handler):
  length = int(handler.headers.get("content-length", "0"))
  if length > 32 * 1024 * 1024:
    raise ValueError("request too large")
  return json.loads(handler.rfile.read(length) or b"{}")


def clamp_int(value, default, low, high):
  try:
    value = int(value)
  except (TypeError, ValueError):
    value = default
  return max(low, min(high, value))


def safe_lora_path(value):
  if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value):
    raise ValueError("invalid LoRA path")
  path = (LORA_DIR / value).resolve()
  if LORA_DIR.resolve() not in path.parents and path != LORA_DIR.resolve():
    raise ValueError("LoRA path escapes LORA_DIR")
  return path


def load_pipe():
  global PIPE
  if PIPE is not None:
    return PIPE

  # Serialise loads so a background /load and an on-demand /generate can't both
  # build the pipeline at once.
  with _LOAD_LOCK:
    if PIPE is not None:
      return PIPE

    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if MODE == "edit":
      try:
        from diffusers import QwenImageEditPipeline
        cls = QwenImageEditPipeline
      except ImportError:
        from diffusers import DiffusionPipeline
        cls = DiffusionPipeline
    else:
      from diffusers import DiffusionPipeline
      cls = DiffusionPipeline

    kwargs = {"torch_dtype": dtype, "cache_dir": str(MODEL_CACHE_DIR)}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
      kwargs["token"] = token

    pipe = cls.from_pretrained(MODEL_ID, **kwargs)
    if os.getenv("RUNNER_CPU_OFFLOAD") == "1" and hasattr(pipe, "enable_model_cpu_offload"):
      pipe.enable_model_cpu_offload()
    else:
      pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    PIPE = pipe
  return PIPE


def _background_load():
  global LOADING, LOAD_ERROR
  LOADING = True
  LOAD_ERROR = None
  try:
    load_pipe()
  except Exception as exc:  # surfaced via /health so the UI can show it
    LOAD_ERROR = str(exc)
  finally:
    LOADING = False


def start_load():
  """Kick off a non-blocking load; returns the current state immediately."""
  if PIPE is not None:
    return {"ok": True, "loaded": True, "loading": False}
  if not LOADING:
    threading.Thread(target=_background_load, daemon=True).start()
  return {"ok": True, "loaded": False, "loading": True, "load_error": LOAD_ERROR}


def unload_pipe():
  global PIPE, LOADED_LORAS, LOAD_ERROR
  PIPE = None
  LOADED_LORAS = ()
  LOAD_ERROR = None
  gc.collect()
  try:
    import torch
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  except Exception:
    pass


def configure_loras(pipe, loras):
  global LOADED_LORAS
  paths = tuple(str(safe_lora_path(item)) for item in (loras or []))
  if paths == LOADED_LORAS:
    return
  if hasattr(pipe, "unload_lora_weights"):
    pipe.unload_lora_weights()
  if not paths:
    LOADED_LORAS = ()
    return
  for index, path in enumerate(paths):
    pipe.load_lora_weights(path, adapter_name=f"lora_{index}")
  if hasattr(pipe, "set_adapters"):
    pipe.set_adapters([f"lora_{i}" for i in range(len(paths))])
  LOADED_LORAS = paths


def image_from_data_url(value):
  if not isinstance(value, str):
    raise ValueError("image_base64 is required")
  raw = value.split(",", 1)[-1]
  from PIL import Image
  return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")


def generate(payload):
  import torch

  prompt = str(payload.get("prompt") or "").strip()
  if not prompt:
    raise ValueError("prompt is required")

  seed = clamp_int(payload.get("seed"), int(time.time()), 0, 2**31 - 1)
  steps = clamp_int(payload.get("steps"), 30, 1, 100)
  width = clamp_int(payload.get("width"), 1024, 256, 2048)
  height = clamp_int(payload.get("height"), 1024, 256, 2048)
  cfg = float(payload.get("cfg") or 4.0)
  negative = str(payload.get("negative_prompt") or " ")

  pipe = load_pipe()
  configure_loras(pipe, payload.get("loras"))

  device = "cuda" if torch.cuda.is_available() else "cpu"
  args = {
    "prompt": prompt,
    "negative_prompt": negative,
    "num_inference_steps": steps,
    "true_cfg_scale": cfg,
    "generator": torch.Generator(device=device).manual_seed(seed),
  }
  if MODE == "edit":
    args["image"] = image_from_data_url(payload.get("image_base64"))
  else:
    args["width"] = width
    args["height"] = height

  with torch.inference_mode():
    image = pipe(**args).images[0]

  buf = BytesIO()
  image.save(buf, format="PNG")
  out = None
  if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{int(time.time())}-{seed}.png"
    image.save(out)
  return {
    "path": str(out) if out else None,
    "mime": "image/png",
    "width": image.width,
    "height": image.height,
    "seed": seed,
    "image_base64": base64.b64encode(buf.getvalue()).decode(),
  }


class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path != "/health":
      return json_response(self, 404, {"error": "not found"})
    body = {
      "ok": True,
      "mode": MODE,
      "model": MODEL_ID,
      "loaded": PIPE is not None,
      "loading": LOADING,
      "load_error": LOAD_ERROR,
    }
    try:
      import torch
      body["torch"] = torch.__version__
      body["cuda"] = torch.cuda.is_available()
      if torch.cuda.is_available():
        try:
          free, total = torch.cuda.mem_get_info()
          body["device"] = torch.cuda.get_device_name(0)
          body["vram_total_gb"] = round(total / 1e9, 1)
          body["vram_used_gb"] = round((total - free) / 1e9, 1)
        except Exception:
          pass
    except Exception:
      body["cuda"] = False
    json_response(self, 200, body)

  def do_POST(self):
    try:
      if self.path == "/load":
        # Non-blocking: returns immediately, loads on a background thread.
        return json_response(self, 202, start_load())
      if self.path == "/unload":
        unload_pipe()
        return json_response(self, 200, {"ok": True, "loaded": False, "loading": False})
      if self.path == "/generate":
        return json_response(self, 200, generate(decode_json(self)))
      json_response(self, 404, {"error": "not found"})
    except Exception as exc:
      json_response(self, 500, {"error": str(exc)})

  def log_message(self, fmt, *args):
    if self.command == "GET" and os.getenv("RUNNER_ACCESS_LOG") != "1":
      return
    print(f"{self.address_string()} - {fmt % args}", flush=True)


def self_test():
  assert clamp_int("9", 1, 0, 10) == 9
  assert clamp_int("999", 1, 0, 10) == 10
  assert clamp_int("nope", 3, 0, 10) == 3
  assert safe_lora_path("a/b.safetensors").is_absolute()
  try:
    safe_lora_path("../x")
  except ValueError:
    pass
  else:
    raise AssertionError("path escape was accepted")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.self_test:
    self_test()
    return
  if SAVE_OUTPUTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  print(f"runner {MODE} {MODEL_ID} on :{PORT}", flush=True)
  ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
  main()
