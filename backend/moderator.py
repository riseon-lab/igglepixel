"""NSFW image/video moderation gate for generated outputs.

Loads Falconsai/nsfw_image_detection (88M-param ViT, ~350 MB VRAM) once
and keeps it on GPU. Lightweight enough that we don't bother offloading
— the cost is noise next to a 47 GB diffusion model.

Runs in the runner subprocess after the pipeline returns the PIL image or
video frames, but before save_image()/save_video(). When flagged: nothing
persists, runner returns {flagged: true} and the UI shows a neutral toast.

This module owns the canonical `is_enabled()` for the whole project — the
prompt moderator delegates here so a single switch toggles every gate
(prompt + ref + output + CivitAI browse) consistently.

## Disabling moderation: by design, deliberately friction-y

A boolean env flag is too easy to flip casually, and worse — a fork can
bake `=false` into its RunPod template defaults so end users never even
see the choice. So there is no boolean. Two intentional paths exist:

  Path 1 — fork operator declaration (visible in pod config):
      Set env var `IGGLEPIXEL_MODERATION_DISABLE_ACK` to the value of
      the DISABLE_TOKEN constant below. The token reads as a written
      acknowledgement of liability — copying it is the operator
      asserting they accept responsibility for outputs.

  Path 2 — runtime override (Settings UI on the pod):
      A logged-in operator pastes the same token into the Settings
      dialog. Backend writes `<workspace>/.moderation-override.disabled`.
      Survives reboot. NOT visible as a pod env var, so it lives on
      the workspace volume only. Auth-gated endpoints prevent a
      logged-out request from flipping it.

Either path discloses *why* moderation is off in the boot log so an
inspector reading logs can see the exact source. Both paths check the
SAME constant — one place to audit, one place to find.

This is friction-as-a-feature, not security-by-obscurity. A determined
operator finds the token in 30 seconds. The goal is to ensure no one
disables moderation *without realising they did*.

## Other env

    IGGLEPIXEL_VIDEO_MODERATION_FRAMES=7
        max number of evenly-spaced video frames to score before saving

Failure mode is fail-closed: if the moderation model can't load or
errors during inference, we block the output.
"""

from __future__ import annotations

import os
from pathlib import Path

NSFW_THRESHOLD = 0.85         # tune after observing real flag patterns
MODEL_REPO     = "Falconsai/nsfw_image_detection"
DEFAULT_VIDEO_SAMPLE_FRAMES = 7

# The acknowledgement token. Copying this string is the operator stating
# the words "I accept full liability for outputs". Friction-by-design;
# changing the wording requires a code change, which is the point.
DISABLE_TOKEN = "i-am-a-fork-operator-and-i-accept-full-liability-for-all-outputs-from-this-pod"

# Runtime override marker. Lives on the persistent workspace volume so
# the override survives reboot but does NOT show up in pod env config.
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
OVERRIDE_MARKER = WORKSPACE / ".moderation-override.disabled"

_processor = None
_model     = None
_device    = None
_load_failed = False
_state_logged = False


def disable_source() -> str:
    """Identify *why* moderation is currently off, for status/log surfaces.

    Returns one of: 'default' (it's on), 'env' (fork-operator declaration),
    'runtime_override' (UI override), 'default_fallback' (env or marker
    held the wrong token — treated as on).
    """
    env_value = os.environ.get("IGGLEPIXEL_MODERATION_DISABLE_ACK")
    if env_value is not None:
        if env_value == DISABLE_TOKEN:
            return "env"
        # Anything-else-not-the-token is treated as on; we surface this
        # so the operator's logs say their wrong token didn't take effect.
        return "default_fallback"
    try:
        if OVERRIDE_MARKER.exists():
            stored = OVERRIDE_MARKER.read_text(encoding="utf-8").strip()
            if stored == DISABLE_TOKEN:
                return "runtime_override"
            return "default_fallback"
    except OSError:
        pass
    return "default"


def is_enabled() -> bool:
    src = disable_source()
    enabled = src not in ("env", "runtime_override")
    global _state_logged
    if not _state_logged:
        if enabled:
            print("[moderator] enabled", flush=True)
        elif src == "env":
            print("[moderator] DISABLED via fork-operator env var (IGGLEPIXEL_MODERATION_DISABLE_ACK)", flush=True)
        elif src == "runtime_override":
            print(f"[moderator] DISABLED via runtime override ({OVERRIDE_MARKER})", flush=True)
        _state_logged = True
    return enabled


def reset_state_log() -> None:
    """Force the next is_enabled() call to re-announce the source. Called
    by the API endpoints after toggling so log readers see the transition."""
    global _state_logged
    _state_logged = False


def init() -> None:
    """Pre-load the moderation model on GPU. Safe to call multiple times."""
    global _processor, _model, _device, _load_failed
    if not is_enabled() or _model is not None or _load_failed:
        return
    try:
        import torch
        from transformers import AutoModelForImageClassification, ViTImageProcessor
        print(f"[moderator] loading {MODEL_REPO}…", flush=True)
        _processor = ViTImageProcessor.from_pretrained(MODEL_REPO)
        _model     = AutoModelForImageClassification.from_pretrained(MODEL_REPO)
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        _model     = _model.to(_device).eval()
        print(f"[moderator] ready on {_device}", flush=True)
    except Exception as e:
        _load_failed = True
        print(f"[moderator] load failed — fail-closed will block all outputs: {e}", flush=True)


def is_flagged(image) -> bool:
    """Run NSFW classification on a PIL image. Returns True if flagged."""
    if not is_enabled():
        return False
    if _model is None:
        init()
    if _model is None:                # init failed → fail closed
        return True
    try:
        import torch
        with torch.no_grad():
            inputs  = _processor(images=image, return_tensors="pt").to(_device)
            outputs = _model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1)
            # id2label = {0: "normal", 1: "nsfw"}
            nsfw    = probs[0][1].item()
        if nsfw > NSFW_THRESHOLD:
            print(f"[moderator] flagged (nsfw={nsfw:.3f})", flush=True)
            return True
        return False
    except Exception as e:
        print(f"[moderator] inference error, blocking: {e}", flush=True)
        return True


def is_video_flagged(frames) -> bool:
    """Run NSFW classification on sampled PIL frames from a generated video.

    Video-native moderation is still overkill for our current stack. Sampling
    evenly across the clip catches content that appears before/after the
    middle frame while reusing the exact same image moderation model and
    fail-closed behaviour as still image generation.
    """
    if not is_enabled():
        return False
    if frames is None:
        print("[moderator] video produced no frames, blocking", flush=True)
        return True
    try:
        frame_count = len(frames)
    except TypeError:
        frames = list(frames)
        frame_count = len(frames)

    if frame_count <= 0:
        print("[moderator] video produced no frames, blocking", flush=True)
        return True

    sample_indices = _video_sample_indices(frame_count)
    print(
        f"[moderator] scanning {len(sample_indices)}/{frame_count} video frames",
        flush=True,
    )
    for idx in sample_indices:
        if is_flagged(frames[idx]):
            print(f"[moderator] flagged video frame {idx + 1}/{frame_count}", flush=True)
            return True
    return False


def _video_sample_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    try:
        sample_count = int(os.environ.get("IGGLEPIXEL_VIDEO_MODERATION_FRAMES", DEFAULT_VIDEO_SAMPLE_FRAMES))
    except ValueError:
        sample_count = DEFAULT_VIDEO_SAMPLE_FRAMES
    sample_count = max(1, min(frame_count, sample_count))
    if sample_count == 1:
        return [frame_count // 2]
    return sorted({
        round(i * (frame_count - 1) / (sample_count - 1))
        for i in range(sample_count)
    })
