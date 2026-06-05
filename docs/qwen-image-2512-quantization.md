# Qwen-Image-2512 8-bit runtime

Date: 2026-06-05

This project runs Qwen-Image-2512 through Diffusers, but the 8-bit path must be
more selective than a generic "quantize the pipeline" hook. Qwen's prompt
conditioning depends on a multimodal text stack, so the default runtime keeps
the text encoder and VAE in BF16 and quantizes only the DiT transformer.

## Implementation pattern

Use `FORGE_QUANT=fp8` or `FORGE_QUANT=int8`.

The runner first tries Diffusers pipeline-level quantization:

```python
PipelineQuantizationConfig(
    quant_mapping={
        "transformer": TorchAoConfig(Float8WeightOnlyConfig(...))
    }
)
```

For INT8, it uses `TorchAoConfig(Int8WeightOnlyConfig())`. If the pipeline-level
hook fails, the runner falls back to explicitly loading:

```python
QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-2512",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    quantization_config=TorchAoConfig(...),
)
```

and injects that transformer into `QwenImagePipeline.from_pretrained(...)`.

This avoids quantizing alignment-sensitive text-encoder layers while still
cutting the 20B diffusion transformer weight footprint to an 8-bit state.

## CUDA 12.8 / 13 package rules

The isolated profile `qwen-diffusers-quant` pins the quant stack locally:

- `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0`
- `diffusers==0.38.0`, `transformers==4.57.6`, `accelerate>=1.11,<2`
- `torchao==0.17.0`
- `bitsandbytes>=0.48.0`

`pip_extra_args: ["--torch-backend=auto"]` lets uv pick `cu128` or `cu130`
wheels from the visible driver/toolkit. If uv is unavailable, the pip fallback
maps the detected CUDA version to the matching PyTorch wheel index.

Known failure mode: older bitsandbytes wheels do not contain matching CUDA
12.8/13 binaries or newer architecture kernels, so import can succeed while
8-bit CUDA kernels fail later. Use bitsandbytes 0.48+ for CUDA 13 coverage, or
set `BNB_CUDA_VERSION=128`/`130` only when the matching bnb shared library and
CUDA runtime are actually present on `LD_LIBRARY_PATH`.

## Venv isolation

The registry profile sets `system_site_packages: false`, so the runner uses:

```text
/workspace/venvs/qwen-diffusers-quant/bin/python
```

The launcher sets `VIRTUAL_ENV`, prepends the venv `bin` directory to `PATH`,
sets `PYTHONNOUSERSITE=1`, and preserves system-level CUDA/driver locations
through `LD_LIBRARY_PATH` without copying driver libraries into the venv.

## Alternative backends

If Diffusers + TorchAO still cannot initialize on a given CUDA image, prefer a
dedicated diffusion quant backend instead of broadening generic bnb hooks:

- Nunchaku: replaces `QwenImageTransformer2DModel` with a Nunchaku Qwen
  transformer and uses Qwen-specific low-bit kernels. This is the best next
  backend when 4-bit/FP4 performance is acceptable.
- Pre-quantized FP8 Diffusers repo: only use a trusted repo that publishes
  TorchAO-compatible serialized weights for Qwen-Image-2512.
- GGUF/Unsloth: good for runtimes that already support Qwen diffusion GGUF
  weights, but this is a separate loader family and should live in its own
  runtime profile rather than being mixed into the Diffusers runner.

References:

- Qwen model card: https://huggingface.co/Qwen/Qwen-Image-2512
- Diffusers quantization API: https://huggingface.co/docs/diffusers/main/api/quantization
- Diffusers TorchAO guide: https://huggingface.co/docs/diffusers/quantization/torchao
- TorchAO inference docs: https://docs.pytorch.org/ao/stable/workflows/inference.html
- PyTorch CUDA wheels: https://pytorch.org/get-started/previous-versions/
- bitsandbytes install matrix: https://huggingface.co/docs/bitsandbytes/installation
- Nunchaku Qwen docs: https://nunchaku.tech/docs/nunchaku/usage/qwen-image.html
