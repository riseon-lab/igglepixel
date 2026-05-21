"""ComfyUI-backed LTX-2.3 runner.

This path keeps IgglePixel's UI while delegating LTX execution to ComfyUI's
native LTX nodes. It is intentionally separate from the original
ltx-pipelines runner because Comfy's model placement, offload, and tiled decode
story is much stronger on 48 GB cards.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .base import Runner as RunnerBase, WORKSPACE


COMFY_DIR = Path(os.environ.get("FORGE_COMFY_DIR", WORKSPACE / "repos" / "ComfyUI"))
COMFY_LTX_NODES = COMFY_DIR / "custom_nodes" / "ComfyUI-LTXVideo"
COMFY_INPUT_DIR = Path(os.environ.get("FORGE_COMFY_INPUT_DIR", WORKSPACE / "tmp" / "comfy-ltx" / "input"))
COMFY_OUTPUT_DIR = Path(os.environ.get("FORGE_COMFY_OUTPUT_DIR", WORKSPACE / "tmp" / "comfy-ltx" / "output"))
COMFY_PORT_BASE = int(os.environ.get("FORGE_COMFY_PORT_BASE", "18100"))
DEFAULT_WORKFLOW = "LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json"
DEFAULT_NEGATIVE = "low quality, distorted, jittery motion, malformed hands, bad anatomy, artifacts"


VARIANTS = {
    "distilled-fp8": {
        "checkpoint_repo": "Lightricks/LTX-2.3-fp8",
        "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
        "lora_repo": "Lightricks/LTX-2.3",
        "lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "lora_strength": 1.0,
        "steps": 8,
        "cfg": 1.0,
    },
    "dev-fp8": {
        "checkpoint_repo": "Lightricks/LTX-2.3-fp8",
        "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
        "lora_repo": "Lightricks/LTX-2.3",
        "lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "lora_strength": 0.0,
        "steps": 30,
        "cfg": 3.0,
    },
    "dev-bf16": {
        "checkpoint_repo": "Lightricks/LTX-2.3",
        "checkpoint": "ltx-2.3-22b-dev.safetensors",
        "lora_repo": "Lightricks/LTX-2.3",
        "lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "lora_strength": 0.0,
        "steps": 30,
        "cfg": 3.0,
    },
}

TEXT_ENCODER = {
    "repo": "Comfy-Org/ltx-2",
    "filename": "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
}
LATENT_UPSCALER = {
    "repo": "Lightricks/LTX-2.3",
    "filename": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
}
REQUIRED_LTX_NODE_TYPES = {
    "ClownSampler_Beta",
    "LTXVImgToVideoConditionOnly",
    "LTXVPreprocess",
    "LTXVTiledVAEDecode",
    "LTXVAudioVAEDecode",
    "LTXVConditioning",
    "LTXVScheduler",
    "MultimodalGuider",
    "GuiderParameters",
    "LTXFloatToInt",
}


def _json_request(method: str, url: str, payload: Optional[dict] = None, timeout: float = 30) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            detail = json.dumps(parsed, ensure_ascii=False)[:4000]
        except Exception:
            detail = body[:4000]
        raise RuntimeError(f"ComfyUI {method} {url} failed with HTTP {e.code}: {detail}") from e
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ComfyUI port found")


def _round_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _frame_count(duration: float, fps: int) -> int:
    frames = max(17, int(round(float(duration) * int(fps))))
    remainder = (frames - 1) % 8
    if remainder:
        frames += 8 - remainder
    return frames


class ComfyLTXRunner(RunnerBase):
    model_id = "ltx23-comfy-i2v"
    model_name = "LTX-2.3 Comfy"
    category = "video"
    supports_lora = False
    min_vram_gb = 48
    recommended_vram_gb = 48
    requires_ref = True
    mode = "i2v"

    def __init__(self) -> None:
        self._comfy_url = ""
        self._comfy_proc: Optional[subprocess.Popen] = None
        self._object_info: dict = {}
        self._comfy_log_tail: list[str] = []
        self._cancel = False

    def load(self) -> None:
        COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        COMFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._prepare_variant(VARIANTS["distilled-fp8"])
        self._comfy_url = self._start_or_reuse_comfy()
        self._object_info = self._get_json("/object_info", timeout=20)
        self._validate_ltx_nodes()
        print(f"[runner] Comfy LTX ready at {self._comfy_url}", flush=True)

    def generate(self, params: dict, loras: Optional[list] = None) -> dict:
        self._cancel = False
        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        variant_id = str(params.get("variant") or os.environ.get("FORGE_VARIANT") or "distilled-fp8")
        variant = VARIANTS.get(variant_id, VARIANTS["distilled-fp8"])
        self._prepare_variant(variant)
        # Variant files can be linked after Comfy has started. Refreshing
        # object_info asks Comfy to rebuild its current node schemas/choices
        # before prompt validation, which avoids stale checkpoint/LoRA lists.
        self._object_info = self._get_json("/object_info", timeout=20)
        self._validate_ltx_nodes()

        width = _round_multiple(int(params.get("width", 832)))
        height = _round_multiple(int(params.get("height", 480)))
        fps = int(params.get("fps", 16))
        frames = _frame_count(float(params.get("duration", 2.0)), fps)
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        steps = int(params.get("steps", variant["steps"]))
        cfg = float(params.get("cfg", variant["cfg"]))
        negative = (params.get("negative_prompt") or DEFAULT_NEGATIVE).strip()

        input_name = self._prepare_input_image(params, width, height)
        workflow = self._load_workflow()
        self._patch_workflow(
            workflow,
            mode=self.mode,
            prompt=prompt,
            negative=negative,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            input_name=input_name,
            variant=variant,
        )
        api_prompt = self._workflow_to_api_prompt(workflow)
        prompt_id = self._queue_prompt(api_prompt, workflow)
        print(
            f"[runner] LTX Comfy queued {prompt_id} "
            f"(mode={self.mode}, variant={variant_id}, {width}x{height}, frames={frames}, steps={steps})",
            flush=True,
        )
        outputs = self._wait_for_outputs(prompt_id)
        out_path = self._import_first_video(outputs, seed)
        return self.asset_response([out_path], meta={
            "model": self.model_id,
            "engine": "comfyui",
            "variant": variant_id,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
        })

    def cancel(self) -> None:
        self._cancel = True
        try:
            self._post_json("/interrupt", {}, timeout=5)
        except Exception:
            pass

    def _start_or_reuse_comfy(self) -> str:
        override = os.environ.get("FORGE_COMFY_URL", "").strip().rstrip("/")
        if override:
            self._probe_comfy(override)
            print(f"[runner] using external ComfyUI at {override}", flush=True)
            return override
        if not (COMFY_DIR / "main.py").exists():
            raise RuntimeError(
                f"ComfyUI runtime is missing at {COMFY_DIR}. Install the comfy-ltx runtime profile first."
            )
        port = _free_port(COMFY_PORT_BASE)
        url = f"http://127.0.0.1:{port}"
        cmd = [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--disable-auto-launch",
            "--input-directory", str(COMFY_INPUT_DIR),
            "--output-directory", str(COMFY_OUTPUT_DIR),
        ]
        print("[runner] starting ComfyUI: " + " ".join(cmd), flush=True)
        self._comfy_proc = subprocess.Popen(
            cmd,
            cwd=str(COMFY_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._wait_for_comfy(url)
        return url

    def _wait_for_comfy(self, url: str) -> None:
        deadline = time.time() + 180
        tail: list[str] = []
        while time.time() < deadline:
            if self._comfy_proc and self._comfy_proc.stdout:
                while True:
                    line = self._read_available_line()
                    if line is None:
                        break
                    self._remember_comfy_log(line)
                    tail.append(line.rstrip())
                    tail = tail[-20:]
                    print("[comfy] " + line.rstrip(), flush=True)
            try:
                self._probe_comfy(url)
                return
            except Exception:
                if self._comfy_proc and self._comfy_proc.poll() is not None:
                    raise RuntimeError("ComfyUI exited during startup. Log tail:\n" + "\n".join(tail[-12:]))
                time.sleep(1)
        raise RuntimeError("Timed out waiting for ComfyUI to start. Log tail:\n" + "\n".join(tail[-12:]))

    def _read_available_line(self) -> Optional[str]:
        if not self._comfy_proc or not self._comfy_proc.stdout:
            return None
        # Text pipes are blocking; use select when available.
        try:
            import select
            ready, _, _ = select.select([self._comfy_proc.stdout], [], [], 0)
            if not ready:
                return None
        except Exception:
            return None
        return self._comfy_proc.stdout.readline()

    def _remember_comfy_log(self, line: str) -> None:
        self._comfy_log_tail.append(line.rstrip())
        self._comfy_log_tail = self._comfy_log_tail[-60:]

    def _drain_comfy_logs(self) -> None:
        while True:
            line = self._read_available_line()
            if line is None:
                break
            self._remember_comfy_log(line)
            print("[comfy] " + line.rstrip(), flush=True)

    @staticmethod
    def _probe_comfy(url: str) -> None:
        _json_request("GET", f"{url}/system_stats", timeout=5)

    def _get_json(self, path: str, timeout: float = 30) -> dict:
        return _json_request("GET", self._comfy_url + path, timeout=timeout)

    def _post_json(self, path: str, payload: dict, timeout: float = 30) -> dict:
        return _json_request("POST", self._comfy_url + path, payload, timeout=timeout)

    def _validate_ltx_nodes(self) -> None:
        missing = sorted(REQUIRED_LTX_NODE_TYPES.difference(self._object_info.keys()))
        if not missing:
            return
        self._drain_comfy_logs()
        install_hint = (
            f"Expected Lightricks ComfyUI-LTXVideo under {COMFY_LTX_NODES}. "
            "Install or reinstall the ComfyUI LTX runtime profile, then stop and relaunch the LTX Comfy model."
        )
        if os.environ.get("FORGE_COMFY_URL", "").strip():
            install_hint = (
                "The external ComfyUI server in FORGE_COMFY_URL is missing the Lightricks "
                "ComfyUI-LTXVideo custom nodes. Install ComfyUI-LTXVideo into that ComfyUI instance "
                "and restart it."
            )
        exists = COMFY_LTX_NODES.exists()
        tail = "\n".join(self._comfy_log_tail[-20:])
        raise RuntimeError(
            "ComfyUI started without required LTXVideo custom nodes: "
            + ", ".join(missing)
            + f". Custom node folder exists: {exists}. {install_hint}"
            + (f"\nComfy log tail:\n{tail}" if tail else "")
        )

    def _queue_prompt(self, prompt: dict, workflow: dict) -> str:
        client_id = str(uuid.uuid4())
        response = self._post_json(
            "/prompt",
            {"prompt": prompt, "client_id": client_id, "extra_data": {"extra_pnginfo": {"workflow": workflow}}},
            timeout=30,
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {response}")
        return prompt_id

    def _wait_for_outputs(self, prompt_id: str) -> dict:
        deadline = time.time() + float(os.environ.get("FORGE_COMFY_TIMEOUT_SECONDS", "1800"))
        last_status = ""
        while time.time() < deadline:
            if self._cancel:
                raise RuntimeError("Generation cancelled")
            hist = self._get_json(f"/history/{prompt_id}", timeout=20)
            item = hist.get(prompt_id)
            if item:
                status = item.get("status") or {}
                if status and status != last_status:
                    print(f"[runner] Comfy status: {status}", flush=True)
                    last_status = status
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    raise RuntimeError("ComfyUI workflow failed: " + json.dumps(messages[-3:])[-1200:])
                outputs = item.get("outputs") or {}
                if outputs:
                    return outputs
            time.sleep(1.5)
        raise RuntimeError("Timed out waiting for ComfyUI generation to finish")

    def _import_first_video(self, outputs: dict, seed: int) -> Path:
        candidates = []
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("videos", "gifs", "images"):
                for item in node_out.get(key) or []:
                    filename = item.get("filename")
                    if not filename:
                        continue
                    ext = Path(filename).suffix.lower()
                    if ext in {".mp4", ".webm", ".mov", ".mkv", ".gif", ".png", ".jpg", ".jpeg"}:
                        candidates.append(item)
        if not candidates:
            raise RuntimeError("ComfyUI finished without a video/image output")
        priority = {".mp4": 0, ".webm": 1, ".mov": 2, ".mkv": 3, ".gif": 4}
        candidates.sort(key=lambda item: priority.get(Path(item["filename"]).suffix.lower(), 9))
        item = candidates[0]
        data = self._read_comfy_output(item)
        ext = Path(item["filename"]).suffix.lower().lstrip(".") or "mp4"
        dest = self.new_output_path(ext=ext, prefix=f"{self.model_id}_{seed}")
        return self.save_bytes(data, dest)

    def _read_comfy_output(self, item: dict) -> bytes:
        filename = item.get("filename", "")
        subfolder = item.get("subfolder", "") or ""
        kind = item.get("type", "output") or "output"
        base = COMFY_OUTPUT_DIR if kind == "output" else COMFY_INPUT_DIR if kind == "input" else COMFY_DIR / kind
        local = (base / subfolder / filename).resolve()
        try:
            local.relative_to((base / subfolder).resolve())
        except ValueError:
            raise RuntimeError("Comfy output path escaped its folder")
        if local.exists():
            return local.read_bytes()
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
        with urllib.request.urlopen(f"{self._comfy_url}/view?{query}", timeout=120) as resp:
            return resp.read()

    def _prepare_variant(self, variant: dict) -> None:
        self._link_hf_file(variant["checkpoint_repo"], variant["checkpoint"], COMFY_DIR / "models" / "checkpoints")
        if variant.get("lora"):
            self._link_hf_file(variant["lora_repo"], variant["lora"], COMFY_DIR / "models" / "loras")
        self._link_hf_file(TEXT_ENCODER["repo"], TEXT_ENCODER["filename"], COMFY_DIR / "models" / "text_encoders")
        self._link_hf_file(LATENT_UPSCALER["repo"], LATENT_UPSCALER["filename"], COMFY_DIR / "models" / "latent_upscale_models")

    @staticmethod
    def _link_hf_file(repo: str, filename: str, dest_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        src = Path(hf_hub_download(repo_id=repo, filename=filename, token=token))
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(filename).name
        if dest.exists():
            return dest
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        return dest

    def _prepare_input_image(self, params: dict, width: int, height: int) -> str:
        ref_path = params.get("ref_image") or params.get("ref")
        if self.mode == "i2v" and not ref_path:
            raise ValueError("`ref_image` is required for LTX-2.3 image-to-video")
        if ref_path:
            p = Path(ref_path)
            if not p.is_absolute():
                p = WORKSPACE / p
            image = self.load_image(p).convert("RGB")
        else:
            from PIL import Image
            image = Image.new("RGB", (width, height), (8, 10, 12))
        name = f"iggle_ltx_{uuid.uuid4().hex}.png"
        image.save(COMFY_INPUT_DIR / name)
        return name

    @staticmethod
    def _load_workflow() -> dict:
        explicit = os.environ.get("FORGE_LTX_COMFY_WORKFLOW", "").strip()
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.extend([
            COMFY_LTX_NODES / "example_workflows" / "2.3" / DEFAULT_WORKFLOW,
            COMFY_LTX_NODES / "example_workflows" / DEFAULT_WORKFLOW,
        ])
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(
            "LTX Comfy workflow template was not found. Expected "
            f"{COMFY_LTX_NODES / 'example_workflows' / '2.3' / DEFAULT_WORKFLOW}"
        )

    @staticmethod
    def _patch_workflow(workflow: dict, *, mode: str, prompt: str, negative: str, seed: int,
                        width: int, height: int, frames: int, fps: int, steps: int,
                        cfg: float, input_name: str, variant: dict) -> None:
        checkpoint = variant["checkpoint"]
        lora = variant.get("lora", "")
        lora_strength = float(variant.get("lora_strength", 0.0))
        for node in workflow.get("nodes", []):
            ntype = str(node.get("type") or "")
            title = str(node.get("title") or "")
            widgets = node.get("widgets_values")
            if not isinstance(widgets, list):
                continue
            lower_title = title.lower()
            if ntype == "CLIPTextEncode" and "negative" in lower_title and widgets:
                widgets[0] = negative
            elif ntype == "CLIPTextEncode" and widgets:
                widgets[0] = prompt
            elif ntype == "RandomNoise" and widgets:
                widgets[0] = seed
            elif ntype == "LoadImage" and widgets:
                widgets[0] = input_name
            elif ntype == "PrimitiveBoolean" and "bypass_i2v" in lower_title and widgets:
                widgets[0] = mode != "i2v"
            elif ntype == "PrimitiveFloat" and "fps" in lower_title and widgets:
                widgets[0] = fps
            elif ntype == "PrimitiveInt" and "number of frames" in lower_title and widgets:
                widgets[0] = frames
            elif ntype == "EmptyLTXVLatentVideo" and len(widgets) >= 3:
                widgets[0], widgets[1], widgets[2] = width, height, frames
                if len(widgets) >= 4:
                    widgets[3] = 1
            elif ntype == "LTXVEmptyLatentAudio" and len(widgets) >= 2:
                widgets[0], widgets[1] = frames, fps
            elif ntype == "LTXVConditioning" and widgets:
                widgets[0] = fps
            elif ntype == "LTXVScheduler" and widgets:
                widgets[0] = steps
            elif ntype in {"CheckpointLoaderSimple", "LTXVAudioVAELoader"} and widgets:
                widgets[0] = checkpoint
            elif ntype == "LTXAVTextEncoderLoader" and widgets:
                widgets[0] = Path(TEXT_ENCODER["filename"]).name
                if len(widgets) > 1:
                    widgets[1] = checkpoint
            elif "lora" in ntype.lower() and widgets:
                for i, value in enumerate(widgets):
                    if isinstance(value, str) and value.endswith(".safetensors") and lora:
                        widgets[i] = lora
                    elif isinstance(value, (int, float)):
                        widgets[i] = lora_strength
            elif ntype == "LTXVImgToVideoConditionOnly" and widgets:
                if len(widgets) >= 2:
                    widgets[1] = mode != "i2v"
            elif ntype == "GuiderParameters" and len(widgets) >= 2 and str(widgets[0]).upper() == "VIDEO":
                widgets[1] = cfg
            elif ntype in {"SaveVideo", "VHS_VideoCombine", "VideoCombine"} and widgets:
                for i, value in enumerate(widgets):
                    if isinstance(value, str):
                        widgets[i] = f"igglepixel_ltx23_{seed}"
                        break

    def _workflow_to_api_prompt(self, workflow: dict) -> dict:
        nodes = {int(n["id"]): n for n in workflow.get("nodes", []) if "id" in n}
        links = {int(l[0]): l for l in workflow.get("links", []) if isinstance(l, list) and len(l) >= 6}
        reachable = self._reachable_nodes(nodes, links)
        prompt = {}
        for node_id in sorted(reachable):
            node = nodes[node_id]
            class_type = str(node.get("type") or "")
            inputs = self._node_inputs(node, links, reachable)
            prompt[str(node_id)] = {
                "class_type": class_type,
                "inputs": inputs,
            }
        return prompt

    @staticmethod
    def _reachable_nodes(nodes: dict[int, dict], links: dict[int, list]) -> set[int]:
        output_types = ("savevideo", "videocombine", "saveanimated", "saveimage")
        roots = [
            nid for nid, n in nodes.items()
            if any(t in str(n.get("type", "")).lower() for t in output_types)
        ]
        if not roots:
            return set(nodes)
        seen: set[int] = set()

        def visit(nid: int) -> None:
            if nid in seen or nid not in nodes:
                return
            seen.add(nid)
            for inp in nodes[nid].get("inputs") or []:
                link_id = inp.get("link")
                if link_id is None:
                    continue
                link = links.get(int(link_id))
                if link:
                    visit(int(link[1]))

        for root in roots:
            visit(root)
        return seen

    def _node_inputs(self, node: dict, links: dict[int, list], reachable: set[int]) -> dict:
        out = {}
        widget_names = []
        for inp in node.get("inputs") or []:
            name = inp.get("name")
            if not name:
                continue
            link_id = inp.get("link")
            if link_id is not None:
                link = links.get(int(link_id))
                if link and int(link[1]) in reachable:
                    out[name] = [str(link[1]), int(link[2])]
                continue
            widget = inp.get("widget") or {}
            widget_name = widget.get("name")
            if widget_name:
                widget_names.append(widget_name)

        widgets = list(node.get("widgets_values") or [])
        consumed = 0
        for name in widget_names:
            if consumed < len(widgets) and name not in out:
                out[name] = widgets[consumed]
                consumed += 1

        class_type = str(node.get("type") or "")
        schema_inputs = self._schema_input_names(class_type)
        for name in schema_inputs:
            if consumed >= len(widgets):
                break
            if name not in out:
                out[name] = widgets[consumed]
                consumed += 1
        return out

    def _schema_input_names(self, class_type: str) -> list[str]:
        info = self._object_info.get(class_type) or {}
        inputs = (info.get("input") or {})
        names = []
        for section in ("required", "optional"):
            values = inputs.get(section) or {}
            if isinstance(values, dict):
                names.extend(values.keys())
        return names


class Runner(ComfyLTXRunner):
    model_id = "ltx23-comfy-i2v"
    model_name = "LTX-2.3 Comfy — Image to Video"
    requires_ref = True
    mode = "i2v"
