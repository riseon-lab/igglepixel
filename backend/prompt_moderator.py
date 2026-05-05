"""Prompt-side moderation for generation requests.

Loads KoalaAI/Text-Moderation (DistilRoBERTa, ~330 MB on CPU) once and keeps
it on CPU — text classification is small and we never want it competing with
the diffusion model for VRAM. Runs at the `/api/generate` boundary in the
backend (not inside the runner subprocess) so a flagged prompt never even
spins up a model load.

The on/off switch is shared with `backend.moderator` (one source of truth
for prompt + ref + output gates), so disabling moderation requires the
fork-operator env var or the runtime UI override documented in moderator.py.

Thresholds via env:
    IGGLEPIXEL_PROMPT_THRESHOLD=0.6
                                  global confidence threshold (0..1)
    IGGLEPIXEL_PROMPT_THRESHOLDS='{"S":0.6,"S3":0.05,"V":0.7}'
                                  per-category overrides; S3 is capped at 0.05

Categories (KoalaAI taxonomy):
    OK  ok                   ← passes
    S   sexual               H   hate                 V   violence
    HR  harassment           SH  self-harm            S3  sexual/minors
    H2  hate/threatening     V2  violence/graphic

Failure mode is fail-closed: load or inference errors block the prompt. The
operator can disable moderation entirely with the env flag if that matters.
"""

from __future__ import annotations

import json
import os
from typing import Optional

MODEL_REPO = "KoalaAI/Text-Moderation"
DEFAULT_THRESHOLD = 0.5
S3_THRESHOLD = 0.05

# Human-readable labels for the categories the model emits — used in the
# rejection message the UI renders. Keys match KoalaAI's id2label.
CATEGORY_LABELS = {
    "S":  "sexual",
    "H":  "hate",
    "V":  "violence",
    "HR": "harassment",
    "SH": "self-harm",
    "S3": "sexual/minors",
    "H2": "hate/threatening",
    "V2": "violence/graphic",
}

_pipe = None
_load_failed = False


def is_enabled() -> bool:
    # Single source of truth — never make these gates disagree.
    try:
        import moderator
    except ImportError:
        from backend import moderator
    return moderator.is_enabled()


def _thresholds() -> dict:
    """Build the per-category threshold table. Categories not in the table
    fall back to the global threshold."""
    raw = os.environ.get("IGGLEPIXEL_PROMPT_THRESHOLDS")
    overrides: dict = {}
    if raw:
        try:
            overrides = json.loads(raw)
            if not isinstance(overrides, dict):
                overrides = {}
        except json.JSONDecodeError:
            print("[prompt_moderator] WARN: IGGLEPIXEL_PROMPT_THRESHOLDS is not valid JSON; ignoring", flush=True)
    try:
        global_thr = float(os.environ.get("IGGLEPIXEL_PROMPT_THRESHOLD", DEFAULT_THRESHOLD))
    except ValueError:
        global_thr = DEFAULT_THRESHOLD
    table = {cat: global_thr for cat in CATEGORY_LABELS}
    for k, v in overrides.items():
        try:
            table[k] = float(v)
        except (TypeError, ValueError):
            continue
    # Sexual/minors should never have a slack threshold, but a zero threshold
    # would flag every prompt because softmax class scores are nearly always
    # greater than 0. Cap any operator override at a strict low threshold.
    table["S3"] = min(float(table.get("S3", S3_THRESHOLD)), S3_THRESHOLD)
    return table


def init() -> None:
    """Pre-load the classifier. Safe to call multiple times."""
    global _pipe, _load_failed
    if not is_enabled() or _pipe is not None or _load_failed:
        return
    try:
        from transformers import pipeline
        print(f"[prompt_moderator] loading {MODEL_REPO}…", flush=True)
        _pipe = pipeline(
            "text-classification",
            model=MODEL_REPO,
            top_k=None,        # return all class scores, not just argmax
            device=-1,         # CPU — text classification is tiny
        )
        print("[prompt_moderator] ready on cpu", flush=True)
    except Exception as e:
        _load_failed = True
        print(f"[prompt_moderator] load failed — fail-closed will block prompts: {e}", flush=True)


def is_flagged(prompt: str) -> Optional[dict]:
    """Run moderation on a prompt string. Returns None if OK, else a dict:
        {"category": "S", "label": "sexual", "score": 0.87}

    Empty / whitespace-only prompts pass through (the runner will reject
    them with a clearer "prompt is required" error).
    """
    if not is_enabled():
        return None
    if not prompt or not prompt.strip():
        return None
    if _pipe is None:
        init()
    if _pipe is None:                         # init failed → fail closed
        return {"category": "ERR", "label": "moderation unavailable", "score": 1.0}
    try:
        # KoalaAI returns [[{label: "OK", score: ...}, {label: "S", score: ...}, …]]
        results = _pipe(prompt[:2000])        # cap input length defensively
        scores = results[0] if results and isinstance(results[0], list) else results
        if not scores:
            return None
        # Find the highest non-OK category score and check it against the
        # per-category threshold. argmax-style is wrong here: a borderline
        # prompt where OK=0.45 and S=0.40 should still flag if S exceeds
        # its category threshold.
        thresholds = _thresholds()
        worst = None
        for entry in scores:
            label = entry.get("label", "")
            if label == "OK":
                continue
            score = float(entry.get("score", 0.0))
            thr = thresholds.get(label, DEFAULT_THRESHOLD)
            if score >= thr and (worst is None or score > worst["score"]):
                worst = {"category": label, "label": CATEGORY_LABELS.get(label, label), "score": score}
        if worst:
            print(f"[prompt_moderator] flagged ({worst['category']}={worst['score']:.3f})", flush=True)
        return worst
    except Exception as e:
        print(f"[prompt_moderator] inference error, blocking: {e}", flush=True)
        return {"category": "ERR", "label": "moderation error", "score": 1.0}
