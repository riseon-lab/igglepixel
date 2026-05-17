"""Shared Diffusers pipeline-level quantization helpers."""

from __future__ import annotations


def _check_bnb_blackwell_compat(torch) -> None:
    """Emit a clear error if we're about to int8/NF4 on a Blackwell GPU
    with a too-old bitsandbytes. Blackwell (sm_120, e.g. RTX 5090, RTX
    Pro 6000) requires bnb >= 0.46 — older versions ship int8 kernels
    that call CUDA stream APIs that don't exist on sm_120, surfacing as
    `AttributeError: ... getCurrentStream` deep inside bnb.functional
    rather than as a clean "unsupported GPU" message.

    BF16 doesn't go through bnb so the issue only bites the int8/NF4
    paths — easy to miss until someone tries to quantise on a 5090.
    """
    try:
        if not torch.cuda.is_available():
            return
        capability = torch.cuda.get_device_capability(0)
    except Exception:
        return
    major = capability[0] if capability else 0
    if major < 12:
        return
    try:
        import bitsandbytes as bnb
        version = getattr(bnb, "__version__", "0.0")
    except Exception:
        return
    try:
        parts = [int(p) for p in version.split(".")[:2]]
        major_v, minor_v = (parts + [0, 0])[:2]
    except Exception:
        return
    if (major_v, minor_v) < (0, 46):
        raise RuntimeError(
            f"bitsandbytes {version} doesn't support Blackwell (sm_{capability[0]}{capability[1]}) "
            f"int8/NF4 kernels. Upgrade to bitsandbytes>=0.46 on this pod. "
            f"requirements-runtime.txt already pins it; the runtime install "
            f"likely needs to re-run, or the running runner subprocess needs a restart."
        )


def pipeline_bnb_quantization_config(quant: str, torch, *, components_to_quantize="transformer"):
    """Return a Diffusers pipeline quantization config for bnb int8/NF4.

    Newer Diffusers pipelines require PipelineQuantizationConfig at the
    pipeline level. Older builds accepted BitsAndBytesConfig directly, so this
    helper keeps a fallback for older runtimes.
    """
    quant = (quant or "bf16").lower()
    if quant in ("int8", "nf4"):
        _check_bnb_blackwell_compat(torch)
    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError:
        from diffusers import BitsAndBytesConfig

        if quant == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if quant == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        raise ValueError(f"unsupported bitsandbytes quantization mode: {quant}")

    if quant == "int8":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=components_to_quantize,
        )
    if quant == "nf4":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=components_to_quantize,
        )
    raise ValueError(f"unsupported bitsandbytes quantization mode: {quant}")


def seed_torch_for_pipeline(torch, seed: int) -> None:
    """Seed PyTorch without passing an explicit generator to Diffusers.

    Quantized/offloaded pipelines can keep large modules on CUDA while their
    pipeline execution device remains CPU. Passing an explicit generator can
    trigger CPU/CUDA tensor or stream mismatches inside pipeline internals.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
