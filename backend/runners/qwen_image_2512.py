"""Qwen-Image-2512 runner.

The December refresh uses the same QwenImagePipeline contract as the original
Qwen-Image model, so this runner reuses the base text-to-image implementation
with a different official HF repo and model id.
"""

from __future__ import annotations

from .qwen_image import Runner as QwenImageRunner


class Runner(QwenImageRunner):
    model_id            = "qwen-image-2512"
    model_name          = "Qwen-Image-2512"
    HF_REPO             = "Qwen/Qwen-Image-2512"
