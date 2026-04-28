"""Qwen 2.5 chat runner — capable instruction-following LLM for the chat workspace.

Three size variants from one registry entry:
  - 7B   (Qwen/Qwen2.5-7B-Instruct)   — fits any modern card
  - 14B  (Qwen/Qwen2.5-14B-Instruct)  — better quality, ~28 GB BF16
  - 32B  (Qwen/Qwen2.5-32B-Instruct)  — workstation-class, ~64 GB BF16

Launcher passes FORGE_VARIANT and FORGE_QUANT; we resolve the right HF repo
and quantisation config here. Apache 2.0 licence on all variants.

Thinking toggle: changes the system prompt to either ask the model to reason
internally before answering ("Think through internally, then answer clearly")
or to respond directly. This is a prompt-engineering nudge — Qwen 2.5 will
follow the instruction reliably, but it's not a native reasoning model with
hidden CoT tokens. For that, see the DeepSeek-R1-Distill / QwQ runners.

Native context window: 32K tokens. Far above the runner-host's keep-recent
slice, so the workspace's compact-summary path handles long sessions.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import Runner as RunnerBase


VARIANTS = {
    "7b":  "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
}


class Runner(RunnerBase):
    model_id            = "qwen25-chat"
    model_name          = "Qwen 2.5 Chat"
    category            = "llm"
    supports_lora       = False
    min_vram_gb         = 4
    recommended_vram_gb = 64

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._variant = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token   = os.environ.get("HF_TOKEN")
        variant = os.environ.get("FORGE_VARIANT", "7b").lower()
        quant   = os.environ.get("FORGE_QUANT",   "bf16").lower()
        cuda    = torch.cuda.is_available()

        if variant not in VARIANTS:
            print(f"[runner] unknown variant '{variant}', falling back to 7b", flush=True)
            variant = "7b"
        self._variant = variant
        repo = VARIANTS[variant]

        tokenizer = AutoTokenizer.from_pretrained(repo, token=token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"token": token}
        if cuda:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32

        if cuda and quant in ("int8", "nf4"):
            from transformers import BitsAndBytesConfig
            kwargs.pop("torch_dtype", None)
            if quant == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            print(f"[runner] loading Qwen 2.5 {variant} with {quant} quantisation…", flush=True)
        else:
            print(f"[runner] loading Qwen 2.5 {variant} (bf16)…", flush=True)

        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        print("[runner] ready", flush=True)

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Runner not loaded")

        prompt = (params.get("prompt") or "").strip()
        messages = self._normalise_messages(params.get("messages") or [])
        if prompt and (not messages or messages[-1].get("content") != prompt):
            messages.append({"role": "user", "content": prompt})
        if not messages:
            raise ValueError("`prompt` is required")

        thinking = bool(params.get("thinking", True))
        # Qwen 2.5 follows direct instructions reliably. The "thinking" prompt
        # tells it to reason internally before producing the answer; the
        # user's chat will just see the polished response. Without it the
        # model goes straight to the answer.
        system = (
            "You are a helpful, concise assistant. Think through the request "
            "internally before responding. Don't expose your reasoning steps "
            "or use phrases like 'Internal Thought Process'. Just give the "
            "final answer clearly."
            if thinking else
            "You are a helpful, concise assistant. Answer directly without "
            "preamble."
        )
        chat = [{"role": "system", "content": system}, *messages[-32:]]
        tokenizer = self._tokenizer

        rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = int(params.get("max_new_tokens", 1024))
        temperature    = float(params.get("temperature", 0.7))
        top_p          = float(params.get("top_p", 0.9))
        do_sample      = temperature > 0

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self._model.generate(**generate_kwargs)

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return self.asset_response([], meta={
            "model":   self.model_id,
            "variant": self._variant,
            "text":    text,
            "tokens":  int(generated.shape[-1]),
        })

    @staticmethod
    def _normalise_messages(raw: list) -> list[dict]:
        messages = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role not in {"user", "assistant", "system"}:
                continue
            content = item.get("content", item.get("text", ""))
            content = str(content).strip()
            if content:
                messages.append({"role": role, "content": content})
        return messages
