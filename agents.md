# Developer & AI Agent Playbook (`agents.md`)

This document is a mandatory operational guide for any developer, AI agent, or automated system contributing to the **IgglePixel (PleoAI)** codebase. 

---

## 🚨 Critical Rules & Constraints

> [!IMPORTANT]
> **1. DO NOT edit the `Dockerfile` or base container configs without explicit user permission.**
> The Docker image is a heavy, stable runtime layer. Adding dependencies directly to the Dockerfile forces a slow, high-friction rebuilding and tag-pushing cycle. Always defer to **Runtime Self-Healing (Section 3)** or `requirements-runtime.txt`.
>
> **2. Maintain the Separation of Runner Subprocesses.**
> Never import heavy machine learning frameworks (like PyTorch, Diffusers, PEFT, or Transformers) at the top-level of `backend/main.py` or `backend/launcher.py`. Keep them inside their respective runner classes under `backend/runners/`. Top-level imports in the backend will cause cold-start delays, slow down health checks, and trigger VRAM fragmentation.
>
> **3. Preserve Plaintext Fallbacks and Transparent Encryption.**
> Any user-visible asset read/write operation MUST pass through `save_image()`, `load_image()`, or `save_bytes()` from `backend/runners/base.py`. Never call direct file-system writes for user assets, as this bypasses the AES-256-GCM browser encryption layer.
>
> **4. Do Not Break Client-Side Preview Mode (`?preview`).**
> When adding a new model or category to the registry (`backend/model_registry.json`), always replicate its skeleton structure in `mockModels()` in `ui/app.js` so that developers can test UI layouts offline without active GPU backends.

---

## 📦 1. Multi-Model Dependency Landscape

IgglePixel runs a wide array of models across multiple domains:
* **Image**: Qwen-Image-Edit, FLUX.1-dev, FLUX.2 [klein].
* **Video**: Wan 2.2 (T2V/I2V), LTX-2.3 (T2V/I2V), HunyuanVideo.
* **Audio**: Kyutai Pocket TTS.
* **Text**: Qwen 2.5 Chat.

These models have conflicting, volatile, or highly version-sensitive Python dependencies (e.g., custom VAE libraries, specialized attention kernels, or pinned deep-learning packages like `vllm`). 

To prevent dependency hell, IgglePixel uses a layered isolation strategy:
1. **Isolated Runner Processes**: Model execution is sandboxed using localhost subprocesses.
2. **Runtime Profiles**: In `model_registry.json`, models can declare a `"runtime"` dependency profile. If configured, the launcher will delegate the subprocess to a profile-specific virtual environment rather than the system-wide interpreter.
3. **Startup Script**: `docker/entrypoint.sh` automatically checks for a `requirements-runtime.txt` at boot and installs additions into a persistent pip cache volume.

---

## 🛠️ 2. The Self-Healing Pattern (Avoiding Dependency Collisions)

When modifying or adding runner code, **never assume a dependency is present globally**. Follow these self-healing and defensive coding practices:

### Try-Imports & Lazy Imports
Always import model-specific dependencies *inside* the `load()` or `generate()` methods of your runner. 

```python
# GOOD: Imports are deferred until the runner is explicitly initialized
class Runner(RunnerBase):
    def load(self) -> None:
        import torch
        from diffusers import FluxPipeline
        ...
```

If a package is only needed for an optional utility (like upscaling or prompt formatting), wrap it in a `try...except ImportError` block and offer a graceful fallback or a structured user alert.

### Dynamic Runtime Dependency Resolution
If your runner requires a library that isn't shipped in the base image, you can check its availability at runtime and trigger a dynamic setup if missing:

```python
# Self-healing dependency check pattern
def load(self) -> None:
    try:
        import decord
    except ImportError:
        print("[runner] decord is missing! Attempting runtime self-heal...", flush=True)
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "decord>=0.6.0"])
        import decord
```

*Note: This dynamic install works seamlessly because the container environment caches pip packages to `/workspace/.cache/pip`, making subsequent launches instant.*

---

## 🔐 3. Transparent Asset Encryption & Privacy

IgglePixel encrypts generated outputs and user uploads at rest on the persistent workspace volume.

### How Key Exchange Works
1. Upon login or unlocking, the user's password derives the 32-byte AES key via PBKDF2-SHA256.
2. The key is held strictly in-memory by the `FastAPI` process.
3. When `launcher.py` spawns a runner, it extracts the unlocked key and passes it as a hex string via the `FORGE_DATA_KEY` environment variable.
4. The key is used by the runner to read reference images and write generated outputs.

### Crucial File I/O Helpers
Never write assets directly to disk. Use the base class methods:
* **Writing Images**: `self.save_image(pil_image, output_path, format="PNG")`
* **Writing Videos**: `self.save_video(list_of_pil_frames, output_path, fps=24)`
* **Writing Raw Bytes**: `self.save_bytes(raw_bytes, output_path)`

These helpers automatically encrypt the data as a `[12-byte nonce][ciphertext + 16-byte GCM tag]` with an `.enc` suffix if the key is available, falling back to legacy plaintext only when no key is set.

---

## ⚡ 4. Edge Cases & Multi-Model Pitfalls

### VRAM Hygiene & Subprocess Eviction
* **The Pitfall**: Multiple runners can easily double-allocate VRAM, leading to catastrophic CUDA Out Of Memory (OOM) crashes.
* **The Rule**: When launching a model, `launcher.py` evaluates the state of running processes. Any active runner that does not match the requested `model_id` should be cleanly terminated via `SIGTERM` (evicting it and freeing its GPU footprint) before the new runner subprocess is spawned.

### Dual-Expert (High/Low Noise) LoRA Mechanics
* **The Pitfall**: Paired MoE denoisers (like Wan 2.2) require different LoRA weights applied to separate sub-transformers (`transformer` vs `transformer_2`). Bypassing the pipeline loader and loading directly to a submodule skips necessary key-mapping utilities, rendering the LoRA inert.
* **The Rule**: For dual-expert pipelines, always load LoRA adapters at the **pipeline-level** (`pipe.load_lora_weights`). For single-transformer pipelines (like Qwen or FLUX), you may load directly onto the submodule (`submodule.load_lora_adapter`) to bypass target module validation errors.

### Safe Video Dims for H.264
* **The Pitfall**: H.264 video compression requires both width and height to be even integers. Passing odd dimensions to ffmpeg causes immediate pipeline errors or corrupted output files.
* **The Rule**: In `save_video()`, always crop or pad the frame dimensions to the nearest even integer before encoding:
  ```python
  w0, h0 = frames[0].size
  ew = w0 if w0 % 2 == 0 else w0 - 1
  eh = h0 if h0 % 2 == 0 else h0 - 1
  ```

---

## 🧪 5. Testing & Verification Checklist

Before pushing changes or asking the user to deploy, execute these verification commands:

```bash
# 1. Lint the javascript shell for structural syntax errors
node --check ui/app.js

# 2. Validate registry format correctness
python3 -m json.tool backend/model_registry.json >/dev/null

# 3. Verify all Python backends compile clean
python3 -m py_compile backend/runners/*.py backend/*.py
```

### UI Testing Checklist
* Open local preview mode (`http://localhost:4175/?preview`) and verify navigation works.
* Test responsiveness under both desktop and mobile viewports.
* Verify that long labels or multi-select dropdown fields in the configuration drawer do not overflow boundaries or overlap buttons.
