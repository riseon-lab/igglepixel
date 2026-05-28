"""API-backed image runner.

This runner lets hosted image models sit behind the same launch/generate
surface as local diffusion runners. It intentionally loads no ML frameworks:
all work happens inside provider HTTP calls during generate().
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase


PROFILES = {
    "openai-gpt-image-2": {
        "provider": "openai",
        "model": "gpt-image-2",
        "env": "OPENAI_API_KEY",
    },
    "openai-gpt-image-15": {
        "provider": "openai",
        "model": "gpt-image-1.5",
        "env": "OPENAI_API_KEY",
    },
    "openai-gpt-image-1-mini": {
        "provider": "openai",
        "model": "gpt-image-1-mini",
        "env": "OPENAI_API_KEY",
    },
    "gemini-nano-banana": {
        "provider": "gemini",
        "model": "gemini-2.5-flash-image",
        "env": "GEMINI_API_KEY",
    },
    "gemini-nano-banana-2": {
        "provider": "gemini",
        "model": "gemini-3.1-flash-image-preview",
        "env": "GEMINI_API_KEY",
    },
    "gemini-nano-banana-pro": {
        "provider": "gemini",
        "model": "gemini-3-pro-image-preview",
        "env": "GEMINI_API_KEY",
    },
    "stability-core": {
        "provider": "stability",
        "model": "core",
        "env": "STABILITY_API_KEY",
    },
    "stability-ultra": {
        "provider": "stability",
        "model": "ultra",
        "env": "STABILITY_API_KEY",
    },
    "replicate-flux-schnell": {
        "provider": "replicate",
        "model": "black-forest-labs/flux-schnell",
        "env": "REPLICATE_API_TOKEN",
    },
    "fal-flux-2-turbo": {
        "provider": "fal",
        "model": "fal-ai/flux-2/turbo",
        "env": "FAL_KEY",
    },
}


class Runner(RunnerBase):
    model_id = "api-image"
    model_name = "API Image Models"
    category = "image"
    supports_lora = False
    min_vram_gb = 0
    recommended_vram_gb = 0

    def __init__(self) -> None:
        self._profile = PROFILES.get(os.environ.get("FORGE_VARIANT") or "", PROFILES["openai-gpt-image-2"])

    def load(self) -> None:
        print(
            f"[runner] API image runner ready provider={self._profile['provider']} model={self._profile['model']}",
            flush=True,
        )

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        profile = self._profile_for_params(params)
        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        api_key = (params.get("api_key") or os.environ.get(profile["env"]) or "").strip()
        if not api_key:
            raise ValueError(f"{profile['env']} is required. Save the provider key in API keys, or set the environment variable.")

        refs = self._reference_images(params)
        provider = profile["provider"]
        if provider == "openai":
            image_bytes = self._generate_openai(profile, prompt, params, refs, api_key)
        elif provider == "gemini":
            image_bytes = self._generate_gemini(profile, prompt, params, refs, api_key)
        elif provider == "stability":
            image_bytes = self._generate_stability(profile, prompt, params, refs, api_key)
        elif provider == "replicate":
            image_bytes = self._generate_replicate(profile, prompt, params, refs, api_key)
        elif provider == "fal":
            image_bytes = self._generate_fal(profile, prompt, params, refs, api_key)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")

        ext = self._image_ext(image_bytes, params.get("output_format") or "png")
        out = self.new_output_path(ext=ext, prefix=self.model_id)
        saved = self.save_bytes(image_bytes, out)
        return self.asset_response([saved], meta={
            "provider": provider,
            "model": profile["model"],
            "prompt": prompt,
            "references": len(refs),
        })

    def _profile_for_params(self, params: dict) -> dict:
        variant = str(params.get("api_model") or "").strip()
        if not variant or variant == "auto":
            return self._profile
        if variant not in PROFILES:
            raise ValueError(f"Unsupported API image model: {variant}")
        return PROFILES[variant]

    def _reference_images(self, params: dict) -> list[tuple[str, bytes]]:
        refs = []
        seen = set()
        for key in ("ref_image", "ref", "ref2", "ref3", "ref4", "ref5", "ref6", "ref7", "ref8"):
            raw = (params.get(key) or "").strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            img = self.load_image(Path(raw))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            refs.append(("image/png", buf.getvalue()))
        return refs

    @staticmethod
    def _closest_openai_size(params: dict) -> str:
        w = int(params.get("width") or 1024)
        h = int(params.get("height") or 1024)
        if w > h * 1.15:
            return "1536x1024"
        if h > w * 1.15:
            return "1024x1536"
        return "1024x1024"

    @staticmethod
    def _image_ext(data: bytes, fallback: str) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        ext = str(fallback or "png").lower()
        if ext == "jpg":
            return "jpeg"
        return ext if ext in {"png", "jpeg", "webp"} else "png"

    def _generate_openai(self, profile: dict, prompt: str, params: dict, refs: list[tuple[str, bytes]], api_key: str) -> bytes:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"}
        model = profile["model"]
        quality = params.get("quality") or "auto"
        fmt = (params.get("output_format") or "png").lower()
        size = self._closest_openai_size(params)
        if refs:
            data = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
            }
            files = [("image[]", (f"reference_{i}.png", data_bytes, mime)) for i, (mime, data_bytes) in enumerate(refs, 1)]
            response = httpx.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
                timeout=180,
            )
        else:
            body = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "output_format": "jpeg" if fmt == "jpg" else fmt,
            }
            response = httpx.post(
                "https://api.openai.com/v1/images/generations",
                headers={**headers, "Content-Type": "application/json"},
                json=body,
                timeout=180,
            )
        if response.status_code >= 400:
            raise RuntimeError(self._http_error(response))
        data = response.json()
        b64 = (data.get("data") or [{}])[0].get("b64_json")
        if not b64:
            raise RuntimeError(f"OpenAI response did not include image bytes: {data}")
        return base64.b64decode(b64)

    @staticmethod
    def _gemini_reference_prompt(prompt: str, params: dict, ref_count: int) -> str:
        if ref_count <= 0:
            return prompt
        mode = str(params.get("reference_mode") or "auto").strip().lower()
        cues = {
            "edit image": "Edit the reference image according to the prompt. Preserve the subject identity, composition, and important details unless the prompt explicitly changes them.",
            "combine images": "Combine the supplied reference images into one coherent final image. Use each reference for its visible subject, object, scene, or material identity.",
            "style transfer": "Use the reference image or images as style, texture, lighting, palette, and rendering guidance while following the prompt content.",
            "product mockup": "Preserve product shape, labels, logos, text, materials, and proportions from the references. Integrate them naturally into the requested scene.",
            "preserve details": "Prioritize high-fidelity detail preservation from the reference images, including small text, markings, clothing, texture, and facial or object identity.",
            "inpaint by instruction": "Change only the regions implied by the instruction. Keep unaffected areas from the reference image visually consistent.",
            "sketch to finished": "Treat sketch, pose, or layout references as structural guidance. Produce a polished finished image that follows the same composition.",
            "character consistency": "Keep the character identity consistent across the generated image, including face, hair, outfit cues, proportions, and distinguishing features.",
        }
        cue = cues.get(mode)
        if not cue:
            return prompt
        return f"{cue}\n\nUser prompt:\n{prompt}"

    def _generate_gemini(self, profile: dict, prompt: str, params: dict, refs: list[tuple[str, bytes]], api_key: str) -> bytes:
        import httpx

        parts = [{"text": self._gemini_reference_prompt(prompt, params, len(refs))}]
        for mime, data in refs:
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode("ascii")}})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{profile['model']}:generateContent"
        response = httpx.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            },
            timeout=180,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._http_error(response))
        data = response.json()
        for candidate in data.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise RuntimeError(f"Gemini response did not include image bytes: {data}")

    def _generate_stability(self, profile: dict, prompt: str, params: dict, refs: list[tuple[str, bytes]], api_key: str) -> bytes:
        import httpx

        if refs:
            raise ValueError("Stability API profiles in this runner are text-to-image only. Use OpenAI, Gemini, or fal for reference-image runs.")
        endpoint = profile["model"]
        url = f"https://api.stability.ai/v2beta/stable-image/generate/{endpoint}"
        data = {
            "prompt": prompt,
            "output_format": "jpeg" if (params.get("output_format") or "").lower() == "jpg" else (params.get("output_format") or "png"),
        }
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "image/*"},
            files={"none": ""},
            data=data,
            timeout=180,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._http_error(response))
        ctype = response.headers.get("content-type", "")
        if ctype.startswith("image/"):
            return response.content
        return self._image_from_payload(response.json(), api_key=None)

    def _generate_replicate(self, profile: dict, prompt: str, params: dict, refs: list[tuple[str, bytes]], api_key: str) -> bytes:
        import httpx

        input_body = {
            "prompt": prompt,
            "width": int(params.get("width") or 1024),
            "height": int(params.get("height") or 1024),
            "num_outputs": 1,
        }
        if refs:
            input_body["image"] = self._data_uri(refs[0])
        url = f"https://api.replicate.com/v1/models/{profile['model']}/predictions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
        }
        response = httpx.post(url, headers=headers, json={"input": input_body}, timeout=90)
        if response.status_code >= 400:
            raise RuntimeError(self._http_error(response))
        payload = response.json()
        payload = self._poll_replicate(payload, headers)
        return self._image_from_payload(payload, api_key=None)

    def _poll_replicate(self, payload: dict, headers: dict) -> dict:
        import httpx

        get_url = (payload.get("urls") or {}).get("get")
        deadline = time.time() + 180
        while get_url and payload.get("status") not in {"succeeded", "successful", "failed", "canceled"} and time.time() < deadline:
            time.sleep(2)
            response = httpx.get(get_url, headers=headers, timeout=30)
            if response.status_code >= 400:
                raise RuntimeError(self._http_error(response))
            payload = response.json()
        if payload.get("status") in {"failed", "canceled"}:
            raise RuntimeError(str(payload.get("error") or f"Replicate prediction {payload.get('status')}"))
        return payload

    def _generate_fal(self, profile: dict, prompt: str, params: dict, refs: list[tuple[str, bytes]], api_key: str) -> bytes:
        import httpx

        body = {
            "prompt": prompt,
            "image_size": {
                "width": int(params.get("width") or 1024),
                "height": int(params.get("height") or 1024),
            },
            "num_images": 1,
        }
        if refs:
            body["image_url"] = self._data_uri(refs[0])
        response = httpx.post(
            f"https://queue.fal.run/{profile['model']}",
            headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._http_error(response))
        payload = response.json()
        result_url = payload.get("response_url")
        status_url = payload.get("status_url")
        deadline = time.time() + 180
        while status_url and time.time() < deadline:
            status = httpx.get(status_url, headers={"Authorization": f"Key {api_key}"}, timeout=30)
            if status.status_code >= 400:
                raise RuntimeError(self._http_error(status))
            status_payload = status.json()
            if status_payload.get("status") in {"COMPLETED", "completed"}:
                break
            if status_payload.get("status") in {"FAILED", "failed"}:
                raise RuntimeError(str(status_payload))
            time.sleep(1.5)
        if result_url:
            result = httpx.get(result_url, headers={"Authorization": f"Key {api_key}"}, timeout=60)
            if result.status_code >= 400:
                raise RuntimeError(self._http_error(result))
            payload = result.json()
        return self._image_from_payload(payload, api_key=api_key)

    @staticmethod
    def _data_uri(ref: tuple[str, bytes]) -> str:
        mime, data = ref
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _image_from_payload(self, payload, api_key: Optional[str] = None) -> bytes:
        import httpx

        candidate = self._find_image_value(payload)
        if not candidate:
            raise RuntimeError(f"Provider response did not include an image: {payload}")
        if isinstance(candidate, str) and candidate.startswith("data:"):
            return base64.b64decode(candidate.split(",", 1)[1])
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            headers = {"Authorization": f"Key {api_key}"} if api_key else {}
            response = httpx.get(candidate, headers=headers, timeout=120)
            if response.status_code >= 400:
                raise RuntimeError(self._http_error(response))
            return response.content
        if isinstance(candidate, str):
            return base64.b64decode(candidate)
        raise RuntimeError(f"Unsupported image output: {candidate!r}")

    def _find_image_value(self, value):
        if isinstance(value, str):
            if value.startswith(("http://", "https://", "data:")) or len(value) > 200:
                return value
            return None
        if isinstance(value, list):
            for item in value:
                found = self._find_image_value(item)
                if found:
                    return found
            return None
        if isinstance(value, dict):
            for key in ("url", "image", "image_url", "output", "data", "b64_json", "base64"):
                if key in value:
                    found = self._find_image_value(value[key])
                    if found:
                        return found
            for item in value.values():
                found = self._find_image_value(item)
                if found:
                    return found
        return None

    @staticmethod
    def _http_error(response) -> str:
        try:
            body = response.json()
            detail = body.get("error", body)
            if isinstance(detail, dict):
                return detail.get("message") or str(detail)
            return str(detail)
        except Exception:
            return response.text
