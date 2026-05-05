"""SenseNova-U1 runner.

Repo:     sensenova/SenseNova-U1-8B-MoT
Runtime:  OpenSenseNova/SenseNova-U1 package in an isolated venv.
Modes:    text-to-image, plus image-edit when a reference image is provided.
Licence: Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


class Runner(RunnerBase):
    model_id = "sensenova-u1"
    model_name = "SenseNova-U1"
    category = "image"
    supports_lora = False
    min_vram_gb = 48
    recommended_vram_gb = 80

    REPOS = {
        "base": "sensenova/SenseNova-U1-8B-MoT",
        "8step": "sensenova/SenseNova-U1-8B-MoT-8step-preview",
        "sft": "sensenova/SenseNova-U1-8B-MoT-SFT",
    }

    NORM_MEAN = (0.5, 0.5, 0.5)
    NORM_STD = (0.5, 0.5, 0.5)

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._variant = "base"
        self._cancel = False

    def load(self) -> None:
        import os
        import torch

        try:
            import sensenova_u1
            from sensenova_u1 import check_checkpoint_compatibility
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "SenseNova-U1 runtime is missing. Install the model runtime profile before starting this runner."
            ) from e

        variant = (os.environ.get("FORGE_VARIANT") or "base").lower()
        repo = self.REPOS.get(variant, self.REPOS["base"])
        token = os.environ.get("HF_TOKEN")
        dtype = torch.bfloat16
        if os.environ.get("FORGE_QUANT", "bf16").lower() == "fp16":
            dtype = torch.float16

        print(f"[runner] loading SenseNova-U1 ({variant}) from {repo}...", flush=True)
        sensenova_u1.set_attn_backend("auto")
        config = AutoConfig.from_pretrained(repo, token=token)
        check_checkpoint_compatibility(config)
        tokenizer = AutoTokenizer.from_pretrained(repo, token=token)
        model = AutoModel.from_pretrained(repo, config=config, torch_dtype=dtype, token=token)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        print(f"[runner] ready on {device}; attention={sensenova_u1.effective_attn_backend()}", flush=True)

        self._variant = variant
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def _to_pil(cls, batch):
        import numpy as np
        import torch
        from PIL import Image

        mean = torch.tensor(cls.NORM_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor(cls.NORM_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
        arr = (batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        return [Image.fromarray(item) for item in arr]

    @staticmethod
    def _round_to_multiple(value: int, multiple: int = 32) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    @staticmethod
    def _load_ref(path: str):
        from PIL import Image

        rp = Path(path)
        if not rp.is_absolute():
            rp = WORKSPACE / rp
        img = RunnerBase.load_image(rp)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        import secrets
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Runner not loaded")
        self._cancel = False
        if loras:
            print("[runner] SenseNova-U1 does not support diffusers LoRAs; ignoring selected LoRAs", flush=True)

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        ref_path = params.get("ref_image") or params.get("ref")
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = secrets.randbits(31)

        default_steps = 8 if self._variant == "8step" else 50
        default_cfg = 1.0 if self._variant == "8step" else 4.0
        steps = int(params.get("steps", default_steps))
        cfg = float(params.get("cfg", default_cfg))
        width = self._round_to_multiple(int(params.get("width", 2048)))
        height = self._round_to_multiple(int(params.get("height", 2048)))
        think = bool(params.get("think", False))
        timestep_shift = float(params.get("timestep_shift", 3.0))
        cfg_norm = str(params.get("cfg_norm") or "none")
        cfg_interval = (0.0, 1.0)

        print(f"[gen] SenseNova-U1 {width}x{height}, steps={steps}, cfg={cfg}, seed={seed}", flush=True)
        with torch.inference_mode():
            if ref_path:
                ref_img = self._load_ref(ref_path)
                img_cfg = float(params.get("img_cfg", 1.0))
                if cfg_norm == "cfg_zero_star":
                    cfg_norm = "none"
                out = self._model.it2i_generate(
                    self._tokenizer,
                    prompt,
                    [ref_img],
                    image_size=(width, height),
                    cfg_scale=cfg,
                    img_cfg_scale=img_cfg,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=steps,
                    batch_size=1,
                    think_mode=think,
                    seed=seed,
                )
                tensor, think_text = out if think else (out, "")
            else:
                out = self._model.t2i_generate(
                    self._tokenizer,
                    prompt,
                    image_size=(width, height),
                    cfg_scale=cfg,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=steps,
                    batch_size=1,
                    seed=seed,
                    think_mode=think,
                )
                tensor, think_text = out if think else (out, "")

        if self._cancel:
            return self.asset_response([], meta={"cancelled": True, "model": self.model_id})

        images = self._to_pil(tensor)
        out_image = self._upscale_if_requested(images[0], params)
        from backend import moderator

        if moderator.is_flagged(out_image):
            return self.asset_response([], meta={"flagged": True, "model": self.model_id, "reason": "moderation"})

        out_path = self.new_output_path(prefix=f"{self.model_id}_{seed}")
        on_disk = self.save_image(out_image, out_path, format="PNG")
        meta = {
            "model": self.model_id,
            "variant": self._variant,
            "prompt": prompt,
            "reference": bool(ref_path),
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "cfg_norm": cfg_norm,
            "timestep_shift": timestep_shift,
            "width": width,
            "height": height,
        }
        if ref_path:
            meta["ref"] = ref_path
            meta["img_cfg"] = float(params.get("img_cfg", 1.0))
        if think_text:
            meta["think"] = think_text
        return self.asset_response([on_disk], meta=meta)

    def cancel(self) -> None:
        self._cancel = True
