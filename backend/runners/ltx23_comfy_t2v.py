"""ComfyUI-backed LTX-2.3 text-to-video runner."""

from __future__ import annotations

from .ltx23_comfy import ComfyLTXRunner


class Runner(ComfyLTXRunner):
    model_id = "ltx23-comfy-t2v"
    model_name = "LTX-2.3 Comfy — Text to Video"
    requires_ref = False
    mode = "t2v"
