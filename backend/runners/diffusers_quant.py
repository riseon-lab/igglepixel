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


def seed_torch_for_pipeline(torch, seed: int) -> None:
    """Seed PyTorch without passing an explicit generator to Diffusers.

    Quantized/offloaded pipelines can keep large modules on CUDA while their
    pipeline execution device remains CPU. Passing an explicit generator can
    trigger CPU/CUDA tensor or stream mismatches inside pipeline internals.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
