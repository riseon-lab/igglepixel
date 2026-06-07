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


def pipeline_torchao_int8_quantization_config(*, components_to_quantize="transformer"):
    """Return a Diffusers pipeline-level TorchAO int8 config.

    bitsandbytes int8 is useful, but it can fail deep in CUDA stream handling
    on some pod images. TorchAO's weight-only int8 path avoids bnb entirely
    and is the preferred Qwen text-to-image INT8 backend.
    """
    return _pipeline_torchao_quantization_config(
        _torchao_int8_component_config,
        components_to_quantize=components_to_quantize,
    )


def pipeline_torchao_fp8_quantization_config(torch, *, components_to_quantize="transformer", dynamic=False):
    """Return a Diffusers pipeline-level TorchAO FP8 config.

    This intentionally targets the Qwen DiT transformer by default. Qwen's
    text encoder has multimodal alignment-sensitive layers; leaving it in
    BF16 avoids the startup failures and prompt-conditioning drift seen when
    generic pipeline hooks try to quantize every large component.
    """
    _check_fp8_runtime_compat(torch)
    return _pipeline_torchao_quantization_config(
        lambda: _torchao_fp8_component_config(torch, dynamic=dynamic),
        components_to_quantize=components_to_quantize,
    )


def torchao_int8_component_quantization_config():
    """Return a component-level Diffusers TorchAoConfig for explicit loaders."""
    return _torchao_int8_component_config()


def torchao_fp8_component_quantization_config(torch, *, dynamic=False):
    """Return a component-level Diffusers TorchAoConfig for explicit loaders."""
    _check_fp8_runtime_compat(torch)
    return _torchao_fp8_component_config(torch, dynamic=dynamic)


def _torchao_int8_component_config():
    try:
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
    except Exception as e:
        raise RuntimeError(
            "TorchAO INT8 is not installed. Run the runtime dependency install "
            "or `python -m pip install 'torchao>=0.15'`, then restart the runner."
        ) from e

    return TorchAoConfig(Int8WeightOnlyConfig())


def _torchao_fp8_component_config(torch, *, dynamic=False):
    try:
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
    except Exception as e:
        raise RuntimeError(
            "TorchAO FP8 is not installed. Use the Qwen quant runtime profile "
            "or install a torch/torchao pair that exposes Float8WeightOnlyConfig."
        ) from e

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn; install a current CUDA PyTorch wheel.")

    if dynamic:
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        except Exception as e:
            raise RuntimeError(
                "TorchAO dynamic FP8 is unavailable in this torchao build; use weight-only FP8 or INT8."
            ) from e
        return TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())

    return TorchAoConfig(Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn))


def _component_names(components_to_quantize) -> list[str]:
    if isinstance(components_to_quantize, str):
        raw = components_to_quantize.replace(";", ",").split(",")
    else:
        raw = list(components_to_quantize or [])
    names: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names or ["transformer"]


def _pipeline_torchao_quantization_config(component_config_factory, *, components_to_quantize="transformer"):
    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except Exception as e:
        raise RuntimeError(
            "Diffusers PipelineQuantizationConfig is not installed. Use the Qwen "
            "quant runtime profile or update diffusers, then restart the runner."
        ) from e

    names = _component_names(components_to_quantize)
    return PipelineQuantizationConfig(
        quant_mapping={
            name: component_config_factory() for name in names
        }
    )


def _check_fp8_runtime_compat(torch) -> None:
    """Fail early on GPUs without useful FP8 inference support."""
    try:
        if not torch.cuda.is_available():
            return
        capability = torch.cuda.get_device_capability(0)
    except Exception:
        return
    if not capability:
        return
    major, minor = capability[:2]
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"TorchAO FP8 needs an Ada/Hopper/Blackwell-class GPU (compute capability >= 8.9); "
            f"this device reports sm_{major}{minor}. Use FORGE_QUANT=int8 on this GPU."
        )


def seed_torch_for_pipeline(torch, seed: int) -> None:
    """Seed PyTorch without passing an explicit generator to Diffusers.

    Quantized/offloaded pipelines can keep large modules on CUDA while their
    pipeline execution device remains CPU. Passing an explicit generator can
    trigger CPU/CUDA tensor or stream mismatches inside pipeline internals.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
