# QA & Diagnostic Report: IgglePixel Dataset Captioning Model (Qwen2.5-VL)

This report details a complete architectural audit, dependency analysis, and QA inspection of the **Dataset Studio Vision Captioning Runtime** (`Qwen/Qwen2.5-VL-7B-Instruct`) and its relationship to the local-captioner utility.

---

## 🔍 1. How the Vision Runtime Works

Unlike the other generation models in IgglePixel, the captioning/vision model does **not** use the standard runner class interface or the launcher process eviction pipeline. Instead:

1. **Naked Backend Subprocess**: It is managed via global variables (`vision_runtime_proc`, `vision_runtime_log`) inside the main `backend/main.py` FastAPI server.
2. **vLLM OpenAI-Compatible API**: It launches either the `vllm` binary or `vllm.entrypoints.openai.api_server` directly to serve an OpenAI-compatible vision completion endpoint on port `8000` (or the configured endpoint).
3. **Vision Proxy Intermediary**: The front-end (`local-captioner/index.html`) makes requests to `/api/trainers/dataset/vision-proxy` on the main FastAPI backend. The backend forwards the prompt and base64-encoded image to the local `vLLM` server, bypassing CORS and mixed-content restrictions.

---

## 🚨 2. The Dependency Issues (Why It's Mismatch-Prone)

The primary reason users encounter "deps issues" when attempting to run the captioner locally or in production is the highly delicate relationship between **PyTorch**, **CUDA**, and **vLLM**:

* **Strict CUDA Bindings**: `vllm` compiles custom CUDA kernels (specifically under `vllm._C`). If the system PyTorch wheel, system CUDA driver, and the installed `vllm` package version are mismatched by even a minor version, the compiled `_C` extension will throw a fatal `ImportError` or `OSError: undefined symbol` at launch.
* **Aggressive Upgrades**: When clicking the "Runtime deps install" button in Settings, the backend runs:
  ```bash
  uv pip install --python <sys.executable> --torch-backend=cu128 --upgrade --reinstall-package vllm -r requirements-runtime.txt
  ```
  If `requirements-runtime.txt` pins `torch==2.8.0` and `torchaudio==2.8.0`, but the system Python already has a pre-installed `torch==2.10.0` or `2.1.2`, the installer will attempt to force-reinstall or downgrade PyTorch, causing broken bindings across all other active components (e.g. `diffusers`, `peft`, and downstream `trainers/qwen_lora_train.py`).
* **Lack of Isolation**: Because the vision runtime is launched in the primary backend interpreter rather than a profile-isolated virtual environment, any package conflicts introduced by `vllm` instantly bleed into the main web application and other single-interpreter runners.

---

## ⚠️ 3. Critical Architectural Edge Cases & Risks

### A. The VRAM Allocation OOM Trap (82% VRAM Lock)
By default, the vision server command uses `IGGLEPIXEL_VISION_GPU_MEMORY` which defaults to `0.82`:
```python
"--gpu-memory-utilization", os.environ.get("IGGLEPIXEL_VISION_GPU_MEMORY", "0.82")
```
When vLLM starts up, it immediately pre-allocates **82% of the total VRAM** of the GPU to store KV caches and model weights.
* **The Risk**: Because the vision runtime is **not** managed by `launcher.py`'s process sweep, it is **never evicted** when you select a FLUX or Wan 2.2 model and click "Generate". 
* **The Outcome**: Spawning a FLUX/Wan runner on top of an active 82% VRAM-locked vLLM process results in an **instant, catastrophic CUDA Out of Memory (OOM) crash** for both models.

### B. Mismatched Fallbacks
If `vllm` fails the `_vllm_available()` check, the launcher raises a `500` error or falls back to looking for an external command (`IGGLEPIXEL_VISION_SERVER_CMD`), which can lead to silent startup failures and unhelpful frontend toasts.

---

## 🛠️ 4. Actionable Recommendations & Fixes

To "QA the hell out of it" and stabilize this critical subsystem, we should implement these adjustments:

### 1. Enable VRAM-Hygiene Integration in `backend/launcher.py`
We should hook the vision runtime lifecycle into the main `launcher.py` process manager:
* When a generator runner is launched (e.g. FLUX/Wan), automatically call `/api/trainers/dataset/vision-runtime/stop` to release the 82% VRAM block.
* When the vision runtime is started, sweep active `ModelLauncher` processes and evict them.

### 2. Isolate the Vision Runtime in its Own Profile Venv
Just like `ltx-pipelines` and `comfy-ltx` declare an isolated `"runtime_profile"` in `model_registry.json`, we should create a profile-specific virtual environment (e.g. `vision-vllm`) dedicated entirely to vLLM, `qwen-vl-utils`, and compatible torch wheels. This guarantees that installing/updating vLLM will **never** break the main system interpreter or corrupt other model runners.

### 3. Relax Hard Torch Pins in `requirements-runtime.txt`
Instead of forcing a rigid, potentially incompatible PyTorch downgrade/upgrade, let `uv` resolve the closest compatible wheel matching the host's existing CUDA environment:
```text
# Avoid pinning hard versions in requirements-runtime if the host already provides them.
# Keep the pins flexible to allow natural alignment:
vllm>=0.10.2
qwen-vl-utils
```

### 4. Provide a Low-VRAM CPU/FP8 Option
Add a `--kv-cache-dtype fp8` or `--gpu-memory-utilization 0.4` toggle in the Settings panel so that users on lower-VRAM cards (e.g. 16GB or 24GB RTX UIs) can run the captioner without completely choking the GPU.
