"""LTX-2.3 text-to-video runner.

Shares the LTX-2.3 runtime and memory policy with the I2V runner, but launches
without image conditioning so the prompt drives the clip.
"""

from .ltx23 import Runner as LTX23Runner


class Runner(LTX23Runner):
    model_id = "ltx23-t2v"
    model_name = "LTX-2.3 — Text to Video"
    requires_ref = False
