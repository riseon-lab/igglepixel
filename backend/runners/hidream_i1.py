"""HiDream-I1 text-to-image runner.

Repo:     HiDream-ai/HiDream-I1-Full / Dev / Fast
Pipeline: diffusers.HiDreamImagePipeline
Text enc: meta-llama/Meta-Llama-3.1-8B-Instruct
Licence: MIT for HiDream transformer weights; component licences also apply.
"""

from __future__ import annotations

from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE, save_latent_preview


class Runner(RunnerBase):
    model_id = "hidream-i1"
    model_name = "HiDream-I1"
    category = "image"
    supports_lora = False
    min_vram_gb = 24
    recommended_vram_gb = 80

    LLAMA_REPO = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    VARIANTS = {
        "full": {"repo": "HiDream-ai/HiDream-I1-Full", "steps": 50, "cfg": 5.0},
        "dev": {"repo": "HiDream-ai/HiDream-I1-Dev", "steps": 28, "cfg": 0.0},
        "fast": {"repo": "HiDream-ai/HiDream-I1-Fast", "steps": 16, "cfg": 0.0},
    }

    def __init__(self) -> None:
        self._pipe = None
        self._variant = "full"
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            from diffusers import HiDreamImagePipeline
            from transformers import AutoTokenizer, LlamaForCausalLM
        except ImportError as e:
            raise RuntimeError(
                "HiDreamImagePipeline is missing. Install a current diffusers release with HiDream support."
            ) from e

        token = os.environ.get("HF_TOKEN")
        variant = (os.environ.get("FORGE_VARIANT") or "full").lower()
        spec = self.VARIANTS.get(variant, self.VARIANTS["full"])
        repo = spec["repo"]

        print(f"[runner] loading HiDream-I1 {variant} from {repo}...", flush=True)
        if not token:
            print("[runner] note: HiDream needs access to meta-llama/Meta-Llama-3.1-8B-Instruct; set HF_TOKEN if gated", flush=True)

        tokenizer_4 = AutoTokenizer.from_pretrained(self.LLAMA_REPO, token=token)
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            self.LLAMA_REPO,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            token=token,
        )

        pipe = HiDreamImagePipeline.from_pretrained(
            repo,
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            torch_dtype=torch.bfloat16,
            token=token,
        )

        if torch.cuda.is_available():
            try:
                _, total = torch.cuda.mem_get_info()
                total_gb = total / 1024 ** 3
                if total_gb >= 80:
                    pipe.to("cuda")
                elif hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_sequential_cpu_offload()
            except Exception:
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_sequential_cpu_offload()

        print("[runner] ready", flush=True)
        self._variant = variant
        self._pipe = pipe

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 16) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import secrets
        import torch

        if self._pipe is None:
            raise RuntimeError("Runner not loaded")
        self._cancel = False
        if loras:
            print("[runner] HiDream diffusers LoRA support is not wired yet; ignoring selected LoRAs", flush=True)

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        spec = self.VARIANTS.get(self._variant, self.VARIANTS["full"])
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = secrets.randbits(31)
        steps = int(params.get("steps", spec["steps"]))
        cfg = float(params.get("cfg", spec["cfg"]))
        width = self._round_to_multiple(int(params.get("width", 1024)))
        height = self._round_to_multiple(int(params.get("height", 1024)))
        negative = (params.get("negative_prompt") or "").strip()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)
        preview_path = WORKSPACE / "assets" / f".preview_{self.model_id}.jpg"
        runner = self

        def _on_step(pipe, step, timestep, callback_kwargs):
            if runner._cancel:
                pipe._interrupt = True
            print(f"[gen] step {step + 1}/{steps}", flush=True)
            if step % 5 == 0 and "latents" in callback_kwargs:
                save_latent_preview(pipe, callback_kwargs["latents"], height, width, preview_path)
            return callback_kwargs

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative or None,
            height=height,
            width=width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=gen,
            callback_on_step_end=_on_step,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        try:
            result = self._pipe(**call_kwargs)
        except TypeError:
            call_kwargs.pop("callback_on_step_end", None)
            call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
            result = self._pipe(**call_kwargs)

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        out_image = result.images[0]
        out_image = self._upscale_if_requested(out_image, params)
        from backend import moderator

        if moderator.is_flagged(out_image):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(prefix=f"{self.model_id}_{self._variant}_{seed}")
        on_disk = self.save_image(out_image, out_path, format="PNG")
        return self.asset_response([on_disk], meta={
            "model": self.model_id,
            "variant": self._variant,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
        })

    def cancel(self) -> None:
        self._cancel = True
