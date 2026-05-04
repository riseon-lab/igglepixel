"""FLUX.2 [klein] 9B runner."""

from __future__ import annotations

from .flux2_klein import Flux2KleinRunner


class Runner(Flux2KleinRunner):
    model_id = "flux2-klein-9b"
    model_name = "FLUX.2 [klein] 9B"
    min_vram_gb = 24
    recommended_vram_gb = 36
    HF_REPO = "black-forest-labs/FLUX.2-klein-9B"
    DEFAULT_STEPS = 4
    DEFAULT_CFG = 1.0
    GPU_DIRECT_THRESHOLD_GB = 36
