"""TinyLlama chat runner.

Small text-generation runner for prototyping the LLM workspace. It is not
intended to be the final model quality bar; it gives the UI a real local model
that fits on modest GPUs while heavier Llama / GPT-OSS options are designed.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import Runner as RunnerBase


class Runner(RunnerBase):
    model_id = "tinyllama-chat"
    model_name = "TinyLlama Chat 1.1B"
    category = "llm"
    supports_lora = False
    min_vram_gb = 4
    recommended_vram_gb = 8

    HF_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")
        quant = os.environ.get("FORGE_QUANT", "bf16").lower()
        cuda = torch.cuda.is_available()

        tokenizer = AutoTokenizer.from_pretrained(self.HF_REPO, token=token)
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
            print(f"[runner] loading TinyLlama with {quant} quantisation...", flush=True)
        else:
            print("[runner] loading TinyLlama...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(self.HF_REPO, **kwargs)
        if cuda and quant not in ("int8", "nf4"):
            model.eval()
        elif not cuda:
            model.to("cpu").eval()
        else:
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
        system = (
            "You are a concise local AI assistant. Think through the task internally, "
            "then answer clearly without exposing hidden reasoning."
            if thinking else
            "You are a concise local AI assistant. Answer directly and clearly."
        )
        # Preserve every system message (compact summary, saved-context block,
        # etc.) and trim only user/assistant turns. Otherwise a long thread
        # would push the saved-context system message out of the slice window.
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        non_sys  = [m for m in messages if m.get("role") != "system"]
        chat = [{"role": "system", "content": system}, *sys_msgs, *non_sys[-16:]]
        tokenizer = self._tokenizer

        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            rendered = self._fallback_chat_template(chat)

        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = int(params.get("max_new_tokens", 512))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.9))
        do_sample = temperature > 0

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
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = self._clean_text(text)
        return self.asset_response([], meta={
            "model": self.model_id,
            "text": text,
            "tokens": int(generated.shape[-1]),
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

    @staticmethod
    def _fallback_chat_template(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    @staticmethod
    def _clean_text(text: str) -> str:
        return text.replace("</s>", "").strip()
