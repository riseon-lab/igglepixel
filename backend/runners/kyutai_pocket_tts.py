"""Kyutai Pocket TTS runner.

CPU-first text-to-speech using Kyutai's 100M parameter Pocket TTS package.
The runner keeps the model and voice state in memory between generations, so
the first request pays the load cost and later turns are quick.

Sources:
  - https://kyutai-labs.github.io/pocket-tts/
  - https://huggingface.co/kyutai/pocket-tts
"""

from __future__ import annotations

import io
from typing import Optional

from .base import Runner as RunnerBase


DEFAULT_VOICE = "alba"
SUPPORTED_LANGUAGE = "english"


class Runner(RunnerBase):
    model_id = "kyutai-pocket-tts"
    model_name = "Kyutai Pocket TTS"
    category = "audio"
    supports_lora = False
    min_vram_gb = 0
    recommended_vram_gb = 0

    def __init__(self) -> None:
        self._model = None
        self._voice_cache: dict[str, object] = {}

    def load(self) -> None:
        from pocket_tts import TTSModel

        print("[runner] loading Kyutai Pocket TTS on CPU...", flush=True)
        self._model = TTSModel.load_model()
        print("[runner] ready", flush=True)

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        if self._model is None:
            raise RuntimeError("Runner not loaded")

        text = str(params.get("prompt") or params.get("text") or "").strip()
        if not text:
            raise ValueError("Prompt text is required")

        language = str(params.get("language") or SUPPORTED_LANGUAGE).strip().lower()
        if language != SUPPORTED_LANGUAGE:
            raise ValueError("This runner currently supports Pocket TTS English only")

        voice = str(params.get("voice") or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        voice_state = self._voice_state(voice)

        print(f"[gen] generating speech voice={voice!r} chars={len(text)}", flush=True)
        audio = self._model.generate_audio(voice_state, text)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu()

        import scipy.io.wavfile

        sample_rate = int(getattr(self._model, "sample_rate", 24_000))
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sample_rate, audio.numpy())

        dest = self.new_output_path("wav", prefix="kyutai_tts")
        on_disk = self.save_bytes(buf.getvalue(), dest)
        samples = int(audio.shape[-1]) if hasattr(audio, "shape") else 0
        duration = samples / sample_rate if sample_rate and samples else None

        return self.asset_response([on_disk], meta={
            "model": self.model_id,
            "voice": voice,
            "language": language,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
        })

    def _voice_state(self, voice: str):
        if voice not in self._voice_cache:
            self._voice_cache[voice] = self._model.get_state_for_audio_prompt(voice)
        return self._voice_cache[voice]
