"""NSFW image/video moderation gate for generated outputs.

Loads Falconsai/nsfw_image_detection (88M-param ViT, ~350 MB VRAM) once
and keeps it on GPU. Lightweight enough that we don't bother offloading
— the cost is noise next to a 47 GB diffusion model.

Runs in the runner subprocess after the pipeline returns the PIL image or
video frames, but before save_image()/save_video(). When flagged: nothing
persists, runner returns {flagged: true} and the UI shows a neutral toast.

Toggle via env:
    IGGLEPIXEL_MODERATION=true   (default — production)
    IGGLEPIXEL_MODERATION=false  (private dev pods, or fork operators
                                  taking their own responsibility)
    IGGLEPIXEL_VIDEO_MODERATION_FRAMES=7
                                  max number of evenly-spaced video frames
                                  to score before saving

Failure mode is fail-closed: if the moderation model can't load or
errors during inference, we block the output. If you really want no
moderation, set the env var to false.
"""

from __future__ import annotations

import os

NSFW_THRESHOLD = 0.85         # tune after observing real flag patterns
MODEL_REPO     = "Falconsai/nsfw_image_detection"
DEFAULT_VIDEO_SAMPLE_FRAMES = 7

_processor = None
_model     = None
_device    = None
_load_failed = False


def is_enabled() -> bool:
    return os.environ.get("IGGLEPIXEL_MODERATION", "true").lower() != "false"


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
