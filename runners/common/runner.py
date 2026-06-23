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
_LOG_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()
LOGS = []
GENERATION = {
  "active": False,
  "step": 0,
  "steps": 0,
  "progress": 0,
  "seed": None,
  "preview_mime": "image/png",
  "preview_base64": None,
  "error": None,
}

MODE = os.getenv("RUNNER_MODE", "txt2img")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image")
PORT = int(os.getenv("PORT", "8000"))
WORKSPACE = Path(os.getenv("CITIVIA_DATA_DIR", "/workspace"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", WORKSPACE / "outputs" / MODE))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", WORKSPACE / "models"))
LORA_DIR = Path(os.getenv("LORA_DIR", WORKSPACE / "loras"))
SAVE_OUTPUTS = os.getenv("RUNNER_SAVE_OUTPUTS") == "1"


def log_event(message):
  line = f"{time.strftime('%H:%M:%S')} {message}"
  with _LOG_LOCK:
    LOGS.append(line)
    del LOGS[:-120]
  print(f"[{MODE}] {message}", flush=True)


def recent_logs():
  with _LOG_LOCK:
    return list(LOGS)


def generation_snapshot():
  with _GEN_LOCK:
    return dict(GENERATION)


def set_generation(**updates):
  with _GEN_LOCK:
    GENERATION.update(updates)


def image_base64(image):
  buf = BytesIO()
  image.save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode()


def latent_preview_base64(latents):
  try:
    import torch
    from PIL import Image

    if isinstance(latents, (list, tuple)):
      latents = latents[0]
    if not isinstance(latents, torch.Tensor):
      return None
    tensor = latents.detach()
    while tensor.ndim > 4:
      tensor = tensor[0]
    if tensor.ndim == 4:
      tensor = tensor[0]
    if tensor.ndim == 2:
      tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
      return None
    if tensor.shape[0] > 4 and tensor.shape[-1] <= 4:
      tensor = tensor.permute(2, 0, 1)
    tensor = tensor[:3].float().cpu()
    if tensor.shape[0] == 1:
      tensor = tensor.repeat(3, 1, 1)
    low = tensor.min()
    high = tensor.max()
    tensor = (tensor - low) / (high - low).clamp_min(1e-6)
    array = (tensor * 255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(array, "RGB")
    image.thumbnail((512, 512))
    return image_base64(image)
  except Exception:
    return None


def progress_callback(total_steps, seed):
  def callback(_pipe, step, _timestep, callback_kwargs):
    preview = latent_preview_base64(callback_kwargs.get("latents"))
    current = int(step) + 1
    set_generation(
      active=True,
      step=current,
      steps=total_steps,
      progress=round((current / max(total_steps, 1)) * 100),
      seed=seed,
      preview_base64=preview,
      error=None,
    )
    return callback_kwargs

  return callback


def legacy_progress_callback(total_steps, seed):
  def callback(step, _timestep, latents):
    preview = latent_preview_base64(latents)
    current = int(step) + 1
    set_generation(
      active=True,
      step=current,
      steps=total_steps,
      progress=round((current / max(total_steps, 1)) * 100),
      seed=seed,
      preview_base64=preview,
      error=None,
    )

  return callback


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
    log_event("model already loaded")
    return PIPE

  # Serialise loads so a background /load and an on-demand /generate can't both
  # build the pipeline at once.
  with _LOAD_LOCK:
    if PIPE is not None:
      log_event("model already loaded")
      return PIPE

    log_event("importing torch")
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_event(f"selected {device} with dtype {dtype}")
    if MODE == "edit":
      try:
        log_event("importing Qwen image edit pipeline")
        from diffusers import QwenImageEditPipeline
        cls = QwenImageEditPipeline
      except ImportError:
        log_event("edit pipeline unavailable; falling back to DiffusionPipeline")
        from diffusers import DiffusionPipeline
        cls = DiffusionPipeline
    else:
      log_event("importing diffusion pipeline")
      from diffusers import DiffusionPipeline
      cls = DiffusionPipeline

    kwargs = {"torch_dtype": dtype, "cache_dir": str(MODEL_CACHE_DIR)}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
      kwargs["token"] = token

    log_event(f"loading {MODEL_ID}")
    log_event(f"using model cache {MODEL_CACHE_DIR}")
    pipe = cls.from_pretrained(MODEL_ID, **kwargs)
    if os.getenv("RUNNER_CPU_OFFLOAD") == "1" and hasattr(pipe, "enable_model_cpu_offload"):
      log_event("enabling CPU offload")
      pipe.enable_model_cpu_offload()
    else:
      log_event(f"moving pipeline to {device}")
      pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    PIPE = pipe
    log_event("model loaded")
  return PIPE


def _background_load():
  global LOADING, LOAD_ERROR
  LOADING = True
  LOAD_ERROR = None
  log_event("load started")
  try:
    load_pipe()
  except Exception as exc:  # surfaced via /health so the UI can show it
    LOAD_ERROR = str(exc)
    log_event(f"load failed: {LOAD_ERROR}")
  finally:
    LOADING = False
    if LOAD_ERROR is None:
      log_event("load finished")


def start_load():
  """Kick off a non-blocking load; returns the current state immediately."""
  if PIPE is not None:
    return {"ok": True, "loaded": True, "loading": False, "logs": recent_logs()}
  if not LOADING:
    threading.Thread(target=_background_load, daemon=True).start()
  return {
    "ok": True,
    "loaded": False,
    "loading": True,
    "load_error": LOAD_ERROR,
    "logs": recent_logs(),
  }


def unload_pipe():
  global PIPE, LOADED_LORAS, LOAD_ERROR
  log_event("unloading model")
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
  log_event("model unloaded")


def lora_selection(value):
  if isinstance(value, str):
    return safe_lora_path(value), 1.0
  if isinstance(value, dict):
    path = safe_lora_path(value.get("path"))
    try:
      strength = float(value.get("strength", 1.0))
    except (TypeError, ValueError):
      strength = 1.0
    return path, max(0.1, strength)
  raise ValueError("invalid LoRA selection")


def configure_loras(pipe, loras):
  global LOADED_LORAS
  selections = tuple(
    (str(path), strength) for path, strength in (lora_selection(item) for item in (loras or []))
  )
  if selections == LOADED_LORAS:
    return
  if hasattr(pipe, "unload_lora_weights"):
    pipe.unload_lora_weights()
  if not selections:
    LOADED_LORAS = ()
    return
  for index, (path, _strength) in enumerate(selections):
    pipe.load_lora_weights(path, adapter_name=f"lora_{index}")
  if hasattr(pipe, "set_adapters"):
    names = [f"lora_{i}" for i in range(len(selections))]
    weights = [strength for _path, strength in selections]
    try:
      pipe.set_adapters(names, adapter_weights=weights)
    except TypeError:
      pipe.set_adapters(names)
  LOADED_LORAS = selections


def image_from_data_url(value):
  if not isinstance(value, str):
    raise ValueError("image_base64 is required")
  raw = value.split(",", 1)[-1]
  from PIL import Image
  return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")


def generate(payload):
  import inspect
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
  set_generation(
    active=True,
    step=0,
    steps=steps,
    progress=0,
    seed=seed,
    preview_mime="image/png",
    preview_base64=None,
    error=None,
  )

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

  params = inspect.signature(pipe.__call__).parameters
  if "callback_on_step_end" in params:
    args["callback_on_step_end"] = progress_callback(steps, seed)
    if "callback_on_step_end_tensor_inputs" in params:
      args["callback_on_step_end_tensor_inputs"] = ["latents"]
  elif "callback" in params:
    args["callback"] = legacy_progress_callback(steps, seed)
    if "callback_steps" in params:
      args["callback_steps"] = 1

  try:
    with torch.inference_mode():
      image = pipe(**args).images[0]

    final_base64 = image_base64(image)
    set_generation(
      active=False,
      step=steps,
      steps=steps,
      progress=100,
      seed=seed,
      preview_base64=final_base64,
      error=None,
    )
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
      "image_base64": final_base64,
    }
  except Exception as exc:
    set_generation(active=False, error=str(exc))
    raise


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
      "logs": recent_logs(),
      "generation": generation_snapshot(),
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
        return json_response(
          self,
          200,
          {"ok": True, "loaded": False, "loading": False, "logs": recent_logs()},
        )
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
  assert lora_selection({"path": "a/b.safetensors", "strength": "0.05"})[1] == 0.1
  assert lora_selection("a/b.safetensors")[1] == 1.0
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
  log_event(f"runner {MODEL_ID} listening on :{PORT}")
  ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
  main()
