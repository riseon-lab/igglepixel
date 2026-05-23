from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import cv2
except Exception:  # pragma: no cover - surfaced through /api/health
    cv2 = None

from PIL import Image, ImageEnhance, ImageFilter


BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("LOCAL_CAPTIONER_WORK_DIR", BASE_DIR / "workspace")).resolve()
VIDEO_DIR = WORK_DIR / "video-jobs"
ENHANCE_CMD = os.environ.get("LOCAL_CAPTIONER_ENHANCE_CMD", "").strip()

app = FastAPI(title="IgglePixel Local Captioner")


class RankRequest(BaseModel):
    job_id: str
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    model: str = "llama3.2-vision:11b"
    temperature: float = 0.1
    threshold: int = 72
    max_rank: int = 80
    filenames: Optional[list[str]] = None


class EnhanceRequest(BaseModel):
    job_id: str
    filenames: list[str]
    mode: str = "conservative"


class ExportRequest(BaseModel):
    job_id: str
    filenames: list[str]
    prefer_enhanced: bool = True
    export_name: str = "video_lora_frames"
    captions: Optional[dict[str, str]] = None


class BatchExportFrame(BaseModel):
    job_id: str
    filename: str
    caption: str = ""


class BatchExportRequest(BaseModel):
    frames: list[BatchExportFrame]
    prefer_enhanced: bool = True
    export_name: str = "video_lora_frames"


def safe_name(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return cleaned or fallback


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", job_id):
        raise HTTPException(400, "Invalid video job id")
    path = VIDEO_DIR / job_id
    if not path.exists():
        raise HTTPException(404, "Video job not found")
    return path


def frame_records(job_path: Path) -> list[dict]:
    manifest = job_path / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text(encoding="utf-8")).get("frames", [])


def write_manifest(job_path: Path, payload: dict) -> None:
    (job_path / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def relative_frame_url(job_id: str, filename: str) -> str:
    return f"/api/video/jobs/{job_id}/frames/{filename}"


def jpeg_bytes_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_jsonish(text: str) -> dict:
    raw = (text or "").strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"score": 0, "keep": False, "reason": raw[:240] or "No structured response"}


def laplacian_blur_score(image) -> float:
    if cv2 is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "opencv": cv2 is not None,
        "work_dir": str(WORK_DIR),
        "external_enhancer": bool(ENHANCE_CMD),
    }


@app.post("/api/video/extract")
async def extract_video_frames(
    video: UploadFile = File(...),
    sample_every: float = 0.5,
    max_frames: int = 120,
    min_scene_delta: float = 3.0,
):
    if cv2 is None:
        raise HTTPException(500, "opencv-python-headless is required. Run: pip install -r local-captioner/requirements.txt")
    if max_frames < 1 or max_frames > 1000:
        raise HTTPException(400, "max_frames must be between 1 and 1000")
    sample_every = max(0.1, min(float(sample_every or 0.5), 30.0))
    min_scene_delta = max(0.0, min(float(min_scene_delta or 0.0), 100.0))

    job_id = secrets.token_hex(8)
    out_dir = VIDEO_DIR / job_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 and total_frames else 0
        stride = max(1, int(round(fps * sample_every)))
        records = []
        last_small = None
        frame_index = 0

        while len(records) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            delta = 100.0 if last_small is None else float(cv2.absdiff(small, last_small).mean())
            last_small = small
            if delta < min_scene_delta:
                frame_index += 1
                continue

            timestamp = frame_index / fps if fps else 0
            blur = laplacian_blur_score(frame)
            filename = f"frame_{len(records) + 1:04d}_{timestamp:08.2f}s.jpg"
            path = frames_dir / filename
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            h, w = frame.shape[:2]
            records.append({
                "filename": filename,
                "url": relative_frame_url(job_id, filename),
                "frame_index": frame_index,
                "timestamp": round(timestamp, 3),
                "width": w,
                "height": h,
                "blur_score": round(blur, 2),
                "scene_delta": round(delta, 2),
                "score": None,
                "keep": None,
                "reason": "",
                "enhanced": False,
            })
            frame_index += 1

        cap.release()
        payload = {
            "job_id": job_id,
            "source_name": video.filename,
            "created_at": time.time(),
            "fps": fps,
            "duration": duration,
            "frames": records,
        }
        write_manifest(out_dir, payload)
        return payload
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.get("/api/video/jobs/{job_id}/frames/{filename}")
def get_video_frame(job_id: str, filename: str):
    root = job_dir(job_id)
    enhanced = root / "enhanced" / safe_name(filename)
    path = enhanced if enhanced.exists() else root / "frames" / safe_name(filename)
    if not path.exists():
        raise HTTPException(404, "Frame not found")
    return FileResponse(path)


@app.post("/api/video/rank")
def rank_video_frames(req: RankRequest):
    path = job_dir(req.job_id)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    wanted = set(req.filenames or [])
    candidates = [f for f in frames if not wanted or f["filename"] in wanted]
    candidates = candidates[: max(1, min(req.max_rank, 300))]

    prompt = f"""You are selecting frames for a single-character LoRA dataset.
Score this frame from 0 to 100 for training usefulness.
Prefer exactly one visible person, readable face, close or medium framing, natural detail, useful expression, useful outfit variation, and low blur.
Reject object-only frames, multi-person frames, distant unreadable faces, heavy motion blur, extreme occlusion, closed eyes unless expressive, and frames that would confuse identity.

Return compact JSON only:
{{"score": 0, "keep": false, "reason": "short reason", "caption_hint": "short visual notes"}}"""

    ranked = []
    with httpx.Client(timeout=120.0) as client:
        for frame in candidates:
            image_path = path / "frames" / frame["filename"]
            data_url = jpeg_bytes_to_data_url(image_path)
            body = {
                "model": req.model,
                "stream": False,
                "options": {"temperature": req.temperature, "num_predict": 180},
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [data_url.split(",", 1)[1]],
                }],
            }
            try:
                res = client.post(req.endpoint, json=body)
                if res.status_code != 200:
                    raise HTTPException(res.status_code, f"Vision model error: {res.text}")
                text = res.json().get("message", {}).get("content", "") or res.json().get("response", "")
            except httpx.TimeoutException:
                frame.update({
                    "score": 0,
                    "keep": False,
                    "reason": "Vision model timed out on this frame.",
                    "caption_hint": "",
                })
                ranked.append(frame)
                continue
            except httpx.RequestError as exc:
                frame.update({
                    "score": 0,
                    "keep": False,
                    "reason": f"Vision request failed: {exc.__class__.__name__}",
                    "caption_hint": "",
                })
                ranked.append(frame)
                continue
            parsed = parse_jsonish(text)
            score = max(0, min(100, int(float(parsed.get("score") or 0))))
            keep = bool(parsed.get("keep", score >= req.threshold)) and score >= req.threshold
            frame.update({
                "score": score,
                "keep": keep,
                "reason": str(parsed.get("reason") or ""),
                "caption_hint": str(parsed.get("caption_hint") or ""),
            })
            ranked.append(frame)

    manifest["frames"] = frames
    write_manifest(path, manifest)
    ranked.sort(key=lambda item: (item.get("score") or 0), reverse=True)
    return {"job_id": req.job_id, "ranked": ranked, "frames": frames}


def conservative_enhance(src: Path, dest: Path) -> None:
    image = Image.open(src).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.04)
    image = ImageEnhance.Sharpness(image).enhance(1.08)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=75, threshold=5))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, quality=96, subsampling=0)


def external_enhance(src: Path, dest: Path) -> None:
    if not ENHANCE_CMD:
        conservative_enhance(src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = ENHANCE_CMD.format(input=shlex.quote(str(src)), output=shlex.quote(str(dest)))
    subprocess.run(command, shell=True, check=True, cwd=str(BASE_DIR))


@app.post("/api/video/enhance")
def enhance_video_frames(req: EnhanceRequest):
    path = job_dir(req.job_id)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    wanted = set(req.filenames)
    if not wanted:
        raise HTTPException(400, "No frames selected for enhancement")

    enhanced = []
    for frame in frames:
        if frame["filename"] not in wanted:
            continue
        src = path / "frames" / frame["filename"]
        dest = path / "enhanced" / frame["filename"]
        try:
            if req.mode == "external":
                external_enhance(src, dest)
            else:
                conservative_enhance(src, dest)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"External enhancer failed with code {exc.returncode}") from exc
        frame["enhanced"] = True
        frame["enhanced_url"] = relative_frame_url(req.job_id, frame["filename"])
        enhanced.append(frame)

    write_manifest(path, manifest)
    return {"job_id": req.job_id, "enhanced": enhanced, "frames": frames}


@app.post("/api/video/export")
def export_video_frames(req: ExportRequest):
    path = job_dir(req.job_id)
    frames = frame_records(path)
    wanted = set(req.filenames)
    selected = [f for f in frames if f["filename"] in wanted]
    if not selected:
        raise HTTPException(400, "No frames selected for export")

    zip_path = path / f"{safe_name(req.export_name, 'video_lora_frames')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, frame in enumerate(selected, 1):
            enhanced_path = path / "enhanced" / frame["filename"]
            original_path = path / "frames" / frame["filename"]
            src = enhanced_path if req.prefer_enhanced and enhanced_path.exists() else original_path
            base = f"{idx:04d}"
            zf.write(src, f"{base}{src.suffix.lower()}")
            caption = (req.captions or {}).get(frame["filename"]) or frame.get("caption") or frame.get("caption_hint") or ""
            zf.writestr(f"{base}.txt", caption.strip() + "\n")
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.post("/api/video/export-batch")
def export_video_frame_batch(req: BatchExportRequest):
    if not req.frames:
        raise HTTPException(400, "No frames selected for export")

    export_root = WORK_DIR / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    zip_path = export_root / f"{safe_name(req.export_name, 'video_lora_frames')}_{secrets.token_hex(4)}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, frame in enumerate(req.frames, 1):
            root = job_dir(frame.job_id)
            filename = safe_name(frame.filename)
            enhanced_path = root / "enhanced" / filename
            original_path = root / "frames" / filename
            src = enhanced_path if req.prefer_enhanced and enhanced_path.exists() else original_path
            if not src.exists():
                raise HTTPException(404, f"Frame not found: {frame.job_id}/{frame.filename}")
            base = f"{idx:04d}"
            zf.write(src, f"{base}{src.suffix.lower()}")
            zf.writestr(f"{base}.txt", frame.caption.strip() + "\n")
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("LOCAL_CAPTIONER_PORT", "8778")))
