"""Shared Diffusers pipeline-level quantization helpers."""

from __future__ import annotations


def pipeline_bnb_quantization_config(quant: str, torch, *, components_to_quantize="transformer"):
    """Return a Diffusers pipeline quantization config for bnb int8/NF4.

    Newer Diffusers pipelines require PipelineQuantizationConfig at the
    pipeline level. Older builds accepted BitsAndBytesConfig directly, so this
    helper keeps a fallback for older runtimes.
    """
    quant = (quant or "bf16").lower()
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


def pipeline_generator(pipe, torch, seed: int):
    """Create a generator on the device Diffusers will use for latents.

    Quantized/offloaded pipelines can keep large modules on CUDA while their
    pipeline execution device remains CPU. Passing a CUDA generator into a CPU
    latent allocation raises "can't generate a tensor on a CPU from CUDA".
    """
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None)
    if device is None or str(device) == "meta":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception as e:
        print(f"[runner] WARN: generator device {device!r} failed ({type(e).__name__}: {e}); falling back to CPU", flush=True)
        return torch.Generator(device="cpu").manual_seed(seed)
