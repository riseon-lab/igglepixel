"""FLUX.2 [klein] 9B Base runner."""

from __future__ import annotations

from .flux2_klein import Flux2KleinRunner


class Runner(Flux2KleinRunner):
    model_id = "flux2-klein-base-9b"
    model_name = "FLUX.2 [klein] 9B Base"
    min_vram_gb = 24
    recommended_vram_gb = 36
    HF_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
    DEFAULT_STEPS = 50
    DEFAULT_CFG = 4.0
    GPU_DIRECT_THRESHOLD_GB = 36
