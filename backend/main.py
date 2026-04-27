import asyncio
import hmac
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gpu_detect import detect_gpu
from launcher import ModelLauncher
import crypto as fcrypto

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
BACKEND_DIR   = Path(__file__).resolve().parent
UI_DIR        = BASE_DIR / "ui"
REGISTRY_PATH = BACKEND_DIR / "model_registry.json"

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
LORAS_DIR        = WORKSPACE / "loras"
MODELS_DIR       = WORKSPACE / "models"
ASSETS_DIR       = WORKSPACE / "assets"
ASSET_UPLOADS    = ASSETS_DIR / "uploads"
ASSET_GENERATED  = ASSETS_DIR / "generated"
COMFY_OUTPUT     = WORKSPACE / "ComfyUI" / "output"  # also scanned as 'generated'

for d in (LORAS_DIR, MODELS_DIR, ASSET_UPLOADS, ASSET_GENERATED):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # WORKSPACE may not exist locally — endpoints handle that on demand.

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".mkv"}

app = FastAPI(title="Forge — RunPod Launcher")
launcher = ModelLauncher()


# ── Auth (persisted to a writable volume) ────────────────────────────────
# We persist a hashed password + the active token to a JSON file. That way:
#   - Page refresh: token in localStorage still matches → no re-login.
#   - Process restart with persistent /workspace: creds + token reload from
#     disk, frontend's stored token still works.
#   - Pod recreate (workspace wiped): file is gone, user does setup again.
#
# We try /workspace first (the persistent volume on RunPod) and fall back
# to ~/.forge/ on dev machines where /workspace doesn't exist. Without the
# fallback, every uvicorn --reload would silently wipe auth and the user
# would be told to sign in again on every page refresh.
def _resolve_auth_file() -> Path:
    for candidate in [WORKSPACE / ".forge_auth.json",
                      Path.home() / ".forge" / "auth.json"]:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # Probe writability: if we can touch a sibling, this dir works.
            probe = candidate.parent / ".write_probe"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    return WORKSPACE / ".forge_auth.json"   # last resort; will fail silently


AUTH_FILE = _resolve_auth_file()
print(f"[forge] auth file: {AUTH_FILE}", flush=True)


class _Auth:
    def __init__(self):
        self.username:    Optional[str]   = None
        self.pw_hash:     Optional[bytes] = None
        self.token:       Optional[str]   = None
        # Persistent secret for signing asset URLs. Generated once at first
        # boot and reused thereafter so signed URLs survive a process restart.
        self.signing_key: bytes = secrets.token_bytes(32)
        # Encryption-at-rest: derived from the password at setup/login/unlock,
        # never persisted. After process restart the salt+canary are loaded
        # but the key isn't — the user has to unlock with their password.
        self.salt:        Optional[bytes] = None
        self.canary_ct:   Optional[bytes] = None
        self.data_key:    Optional[bytes] = None   # ← in RAM only
        self._load()

    def is_setup(self) -> bool:
        return self.pw_hash is not None

    @staticmethod
    def _hash(pw: str) -> bytes:
        # Best-effort: convert to a mutable bytearray, hash, then zero the
        # buffer. Doesn't help against the original `pw` string (Python
        # interns/copies strings outside our reach) but reduces the number
        # of plaintext-password copies hanging around in our process.
        buf = bytearray(pw.encode("utf-8"))
        try:
            return hashlib.sha256(buf).digest()
        finally:
            for i in range(len(buf)):
                buf[i] = 0

    def _load(self) -> None:
        if not AUTH_FILE.exists():
            return
        try:
            with open(AUTH_FILE) as f:
                d = json.load(f)
            self.username = d.get("username")
            ph = d.get("pw_hash")
            if ph:
                self.pw_hash = bytes.fromhex(ph)
            self.token = d.get("token")
            sk = d.get("signing_key")
            if sk:
                self.signing_key = bytes.fromhex(sk)
            s = d.get("salt")
            if s:
                self.salt = bytes.fromhex(s)
            cc = d.get("canary_ct")
            if cc:
                self.canary_ct = bytes.fromhex(cc)
            # Note: data_key is NEVER persisted — that's the whole point.
        except Exception:
            # Corrupt file — leave fields empty so user re-runs setup.
            pass

    def _save(self) -> None:
        try:
            AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(AUTH_FILE, "w") as f:
                json.dump({
                    "username":    self.username,
                    "pw_hash":     self.pw_hash.hex() if self.pw_hash else None,
                    "token":       self.token,
                    "signing_key": self.signing_key.hex(),
                    "salt":        self.salt.hex()      if self.salt      else None,
                    "canary_ct":   self.canary_ct.hex() if self.canary_ct else None,
                }, f)
            try:
                os.chmod(AUTH_FILE, 0o600)
            except OSError:
                pass
        except OSError:
            # /workspace not writable in dev — auth still works in memory.
            pass

    # ── Setup/Login/Unlock/Verify ───────────────────────────────────────
    def setup(self, username: str, password: str) -> str:
        if self.is_setup():
            raise HTTPException(409, "Already set up. Use /api/auth/login.")
        self.username   = username
        self.pw_hash    = self._hash(password)
        self.token      = secrets.token_urlsafe(32)
        # Generate a fresh salt and bind the password to a known plaintext
        # canary so a future unlock attempt can detect a wrong password
        # without ever storing the plaintext key.
        self.salt       = fcrypto.new_salt()
        self.data_key   = fcrypto.derive_key(password, self.salt)
        self.canary_ct  = fcrypto.make_canary(self.data_key)
        self._save()
        return self.token

    def login(self, username: str, password: str) -> str:
        """Verify creds, rotate the token, and re-derive the data key in RAM."""
        if not self.is_setup():
            raise HTTPException(409, "Setup required.")
        if username != self.username or not hmac.compare_digest(self.pw_hash, self._hash(password)):
            raise HTTPException(401, "Invalid credentials")
        self._derive_and_check(password)   # raises 401 if salt/canary mismatch
        self.token = secrets.token_urlsafe(32)
        self._save()
        return self.token

    def unlock(self, password: str) -> None:
        """Re-derive the data key from password (no token rotation).

        Used after a process restart: the existing token is still valid for
        non-encrypted endpoints, but encrypted asset I/O needs the data key.
        Calling this with the right password puts the key back in RAM.
        """
        if not self.is_setup():
            raise HTTPException(409, "Setup required.")
        if not hmac.compare_digest(self.pw_hash, self._hash(password)):
            raise HTTPException(401, "Invalid password")
        self._derive_and_check(password)

    def _derive_and_check(self, password: str) -> None:
        """Derive the data key and verify it via the canary; raise 401 if it fails."""
        if not self.salt or not self.canary_ct:
            # Pre-Phase-2 auth file: bootstrap a fresh salt + canary using
            # this login's password. Existing assets stay plaintext until
            # the user re-uploads (no in-place migration for now).
            self.salt      = fcrypto.new_salt()
            self.data_key  = fcrypto.derive_key(password, self.salt)
            self.canary_ct = fcrypto.make_canary(self.data_key)
            self._save()
            return
        key = fcrypto.derive_key(password, self.salt)
        if not fcrypto.verify_canary(key, self.canary_ct):
            raise HTTPException(401, "Invalid password")
        self.data_key = key

    def verify(self, token: Optional[str]) -> bool:
        if not self.is_setup() or not self.token:
            return False
        if not token:
            return False
        return hmac.compare_digest(token, self.token)

    def is_unlocked(self) -> bool:
        return self.data_key is not None


# ── Signed-URL helpers ───────────────────────────────────────────────────
# `<img>` tags can't send Authorization headers, so we issue ephemeral
# HMAC-signed URLs from the asset-listing endpoints. The route below accepts
# either a valid Bearer token OR a valid signature+expiry, so curl-with-token
# and src=signed-url both work.
#
# Signature = first 32 hex chars of HMAC-SHA256(signing_key, "{path}|{exp}").
# `path` is the asset's path relative to WORKSPACE.
def _sign_url(rel_path: str, ttl_seconds: int = 3600) -> str:
    exp = int(time.time()) + ttl_seconds
    msg = f"{rel_path}|{exp}".encode("utf-8")
    sig = hmac.new(auth.signing_key, msg, hashlib.sha256).hexdigest()[:32]
    return f"/api/assets/file/{rel_path}?sig={sig}&exp={exp}"


def _verify_signature(rel_path: str, sig: Optional[str], exp: Optional[int]) -> bool:
    if not sig or not exp:
        return False
    if int(exp) < int(time.time()):
        return False
    msg = f"{rel_path}|{exp}".encode("utf-8")
    expected = hmac.new(auth.signing_key, msg, hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


auth = _Auth()


def _request_token(request: Request, authorization: Optional[str] = None) -> Optional[str]:
    """Pull the bearer token from the HttpOnly cookie or, as a fallback, the
    Authorization header. Cookies are preferred because they can't be read
    by JavaScript — that's the XSS-token-theft mitigation."""
    cookie = request.cookies.get("forge_token")
    if cookie:
        return cookie
    if authorization:
        return authorization.removeprefix("Bearer ").strip() or None
    return None


def require_token(request: Request, authorization: Optional[str] = Header(None)) -> None:
    """Dependency that gates every /api/* path except auth + signed assets."""
    path = request.url.path
    if (path.startswith("/api/auth/")
            or path.startswith("/api/assets/file/")
            or not path.startswith("/api/")):
        return
    if not auth.is_setup():
        raise HTTPException(401, "Setup required")
    token = _request_token(request, authorization)
    if not auth.verify(token):
        raise HTTPException(401, "Invalid or missing token")


# Lifetime of the token cookie. RunPod usually keeps pods running for days
# at a time; 30 days lets users return without re-login but expires session
# cookies on lost devices reasonably soon.
_COOKIE_MAX_AGE = 30 * 24 * 3600


def _set_token_cookie(response, token: str, request: Request) -> None:
    """Set the HttpOnly+Secure cookie carrying the bearer token.

    Secure flag mirrors the request scheme so dev (`http://localhost`) still
    works while RunPod (HTTPS proxy) gets the strict version.
    """
    response.set_cookie(
        key="forge_token",
        value=token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        require_token(request, request.headers.get("authorization"))
    except HTTPException as e:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    return await call_next(request)


# Security headers — CSP is the highest-leverage XSS mitigation: even if an
# attacker injects HTML, the browser refuses to run any inline / external
# scripts. Inline `style=…` is allowed because we use it everywhere; that's
# safe (styles can't execute code).
#
# Image policy is intentionally loose (`https:`) so external thumbnails from
# CivitAI / HF previews load. If we ever pin to specific CDNs, tighten here.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob: https:; "
    "media-src 'self' blob:; "
    "connect-src 'self'; "
    "worker-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", _CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "interest-cohort=()")
    return response


class AuthBody(BaseModel):
    username: str
    password: str


@app.get("/api/auth/status")
def auth_status(request: Request):
    # `locked` = setup has happened on disk but the data key isn't in RAM.
    # `authenticated` lets the frontend skip the gated-endpoint ping when it
    # already knows the cookie isn't valid (e.g. fresh tab, expired cookie).
    cookie_token = request.cookies.get("forge_token")
    return {
        "needs_setup":   not auth.is_setup(),
        "username":      auth.username,
        "locked":        auth.is_setup() and not auth.is_unlocked(),
        "authenticated": bool(cookie_token and auth.verify(cookie_token)),
    }


def _key_material() -> dict:
    """Salt + canary for the browser's PBKDF2 derivation. Both are public
    by design — knowing them doesn't help an attacker without the password.
    Browser uses the salt to derive the same AES key the backend would, and
    decrypts the canary to confirm the password matches."""
    return {
        "salt":      auth.salt.hex()      if auth.salt      else None,
        "canary_ct": auth.canary_ct.hex() if auth.canary_ct else None,
        "iterations": fcrypto.PBKDF2_ITERATIONS,
    }


@app.post("/api/auth/setup")
def auth_setup(body: AuthBody, request: Request, response: Response):
    token = auth.setup(body.username, body.password)
    _set_token_cookie(response, token, request)
    return {"token": token, "username": auth.username, **_key_material()}


@app.post("/api/auth/login")
def auth_login(body: AuthBody, request: Request, response: Response):
    token = auth.login(body.username, body.password)
    _set_token_cookie(response, token, request)
    return {"token": token, "username": auth.username, **_key_material()}


class UnlockBody(BaseModel):
    password: str


@app.post("/api/auth/unlock")
def auth_unlock(body: UnlockBody):
    """Re-derive the data key from the password without rotating the token.

    Called after a process restart: the existing token is still valid (it's
    persisted on disk), but the encryption key isn't. The user proves they
    know the password, the key gets put back in RAM, encrypted asset I/O
    starts working again. No token rotation = no cookie churn.

    Returns the key material so the BROWSER can also re-derive its own key
    in IndexedDB (browser-side decryption is independent of backend's key).
    """
    auth.unlock(body.password)
    return {"status": "unlocked", "username": auth.username, **_key_material()}


@app.post("/api/auth/logout")
def auth_logout(response: Response):
    """Clear the auth cookie on the client. Backend keeps the token (and any
    later session that holds the same cookie value) valid until rotated, so
    `login` again from the same browser still gets a fresh cookie."""
    response.delete_cookie("forge_token", path="/")
    return {"status": "logged_out"}


# ── Weight downloads (separate from runner launch) ───────────────────────
# UX wants Download → Start as two distinct steps so users can pull weights
# without committing GPU memory until they're ready to generate.
class _DownloadTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}      # model_id -> {downloading, downloaded, error, progress}

    def get(self, model_id: str) -> dict:
        with self.lock:
            return dict(self.state.get(model_id, {"downloading": False, "downloaded": False, "progress": 0.0, "error": None}))

    def set(self, model_id: str, **patch) -> None:
        with self.lock:
            cur = self.state.setdefault(model_id, {"downloading": False, "downloaded": False, "progress": 0.0, "error": None})
            cur.update(patch)


downloads = _DownloadTracker()


def _is_repo_cached(repo_id: str) -> bool:
    """Heuristic: HF cache layout is ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/<sha>/."""
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.exists():
        return False
    snaps = repo_dir / "snapshots"
    return snaps.exists() and any(snaps.iterdir())


@app.get("/api/models/{model_id}/weight-status")
def weight_status(model_id: str):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    tracked = downloads.get(model_id)
    cached  = _is_repo_cached(model.get("hf_repo", ""))
    return {
        "downloading": tracked["downloading"],
        "downloaded":  tracked["downloaded"] or cached,
        "progress":    tracked["progress"],
        "error":       tracked["error"],
    }


class DownloadBody(BaseModel):
    hf_token: Optional[str] = None


@app.post("/api/models/{model_id}/download")
def model_download(model_id: str, body: DownloadBody):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    if not model.get("hf_repo"):
        raise HTTPException(400, "Model has no hf_repo to download")
    if downloads.get(model_id)["downloading"]:
        return {"status": "already_in_progress"}

    repo = model["hf_repo"]
    token = body.hf_token or os.environ.get("HF_TOKEN")

    def _worker():
        downloads.set(model_id, downloading=True, downloaded=False, error=None, progress=0.05)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, token=token)
            downloads.set(model_id, downloading=False, downloaded=True, progress=1.0)
        except Exception as e:
            downloads.set(model_id, downloading=False, error=f"{type(e).__name__}: {e}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "repo": repo}


# ── Registry ─────────────────────────────────────────────────────────────
@app.get("/api/models")
def get_models():
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    gpu = detect_gpu()
    for m in data["models"]:
        m["gpu_compatible"]   = gpu["type"] in m.get("gpu_support", [])
        m["vram_ok"]          = gpu["vram_gb"] >= m.get("min_vram_gb", 0)
        m["vram_recommended"] = gpu["vram_gb"] >= m.get("recommended_vram_gb", 0)
    return {
        "models":    data["models"],
        "upscalers": data.get("upscalers", []),
        "gpu":       gpu,
    }


@app.get("/api/gpu")
def get_gpu():
    return detect_gpu()


# ── Launcher ─────────────────────────────────────────────────────────────
class LaunchRequest(BaseModel):
    model_id: str
    loras: list[str] = []
    hf_token: Optional[str] = None
    quant: Optional[str] = None      # bf16 | int8 | nf4 — runner reads FORGE_QUANT
    variant: Optional[str] = None    # 14b | 5b etc. — runner reads FORGE_VARIANT


@app.post("/api/launch")
async def launch_model(req: LaunchRequest):
    # Runners need the data key to read encrypted refs and encrypt outputs.
    # If we're locked, refuse the launch and tell the frontend to unlock.
    _require_unlocked()
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == req.model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    return await launcher.launch(model, req.loras, req.hf_token, req.quant, req.variant)


@app.post("/api/stop/{model_id}")
async def stop_model(model_id: str):
    return await launcher.stop(model_id)


@app.get("/api/status")
def get_status():
    return launcher.status()


# ── Inference (proxies to the runner subprocess) ─────────────────────────
class GenerateRequest(BaseModel):
    model_id: str
    params: dict = {}
    loras: list = []      # [{filename: str, strength: float}]
    hf_token: Optional[str] = None


@app.get("/api/runner/{model_id}/healthz")
async def runner_health(model_id: str):
    info = launcher.get(model_id)
    if not info:
        raise HTTPException(404, "Runner not running")
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"http://127.0.0.1:{info['port']}/healthz", timeout=3)
            return r.json()
        except httpx.HTTPError as e:
            return {"ready": False, "loading": True, "error": str(e), "model_id": model_id}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    info = launcher.get(req.model_id)
    if not info or info["status"] != "running":
        raise HTTPException(409, "Runner not running. Launch the model first.")
    preview_path = WORKSPACE / "assets" / f".preview_{req.model_id}.jpg"
    try:
        preview_path.unlink(missing_ok=True)
    except OSError:
        pass
    payload = {"params": req.params, "loras": req.loras, "hf_token": req.hf_token}
    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(f"http://127.0.0.1:{info['port']}/generate", json=payload)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        result = r.json()
    # Re-sign asset URLs emitted by the runner. Runners use the legacy
    # /workspace-assets/... shape from base.py; we override here so the
    # frontend never sees the unsigned form.
    for a in result.get("assets", []):
        try:
            p = Path(a.get("path", ""))
            rel = p.relative_to(WORKSPACE.resolve()) if p.is_absolute() else p
            a["url"] = _sign_url(rel.as_posix())
        except (ValueError, OSError):
            pass
    return result


@app.post("/api/cancel/{model_id}")
async def cancel_generate(model_id: str):
    info = launcher.get(model_id)
    if not info or info["status"] != "running":
        raise HTTPException(404, "Runner not running")
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(f"http://127.0.0.1:{info['port']}/cancel", timeout=5)
            return r.json()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Cancel failed: {e}")


@app.get("/api/logs/{model_id}")
async def stream_logs(model_id: str, tail: bool = False):
    async def gen():
        async for line in launcher.stream_logs(model_id, tail=tail):
            yield f"data: {line}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/runner/{model_id}/preview")
async def runner_preview(model_id: str):
    """Return the latest in-progress generation preview JPEG, if available.

    Runners write a low-quality JPEG to .preview_<model_id>.jpg every 5 steps.
    The frontend polls this during generation to show a live ComfyUI-style preview.
    Returns 404 if no preview exists yet (start of generation or model idle).
    """
    path = WORKSPACE / "assets" / f".preview_{model_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "No preview available")
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@app.delete("/api/models/{model_id}")
def delete_model_weights(model_id: str):
    """Remove cached weights for a model so the pod can free disk."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")

    removed = []
    repo = model.get("hf_repo", "")
    # HF cache layout: ~/.cache/huggingface/hub/models--{org}--{name}
    if repo:
        cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed.append(str(cache_dir))
        # workspace-local copy if launcher mirrored it there
        local = MODELS_DIR / repo.replace("/", "__")
        if local.exists():
            shutil.rmtree(local, ignore_errors=True)
            removed.append(str(local))
    return {"status": "deleted", "removed": removed}


# ── HuggingFace downloads ────────────────────────────────────────────────
class HFDownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None
    target_dir: str = "models"
    hf_token: Optional[str] = None


@app.post("/api/download/hf")
async def download_hf(req: HFDownloadRequest):
    target = WORKSPACE / req.target_dir
    target.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if req.hf_token:
        env["HF_TOKEN"] = req.hf_token
    cmd = ["huggingface-cli", "download", req.repo_id]
    if req.filename:
        cmd.append(req.filename)
    cmd += ["--local-dir", str(target)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    launcher.track_download("hf", proc)
    return {"status": "started", "pid": proc.pid}


# ── CivitAI ──────────────────────────────────────────────────────────────
CIVITAI_BASE = "https://civitai.com/api/v1"


@app.get("/api/civitai/search")
async def civitai_search(
    query: str = Query(""),
    types: str = Query("LORA"),
    limit: int = Query(20),
    page: int = Query(1),
    nsfw: bool = Query(False),
    api_key: Optional[str] = Query(None),
):
    params = {"query": query, "types": types, "limit": limit, "page": page, "nsfw": str(nsfw).lower()}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CIVITAI_BASE}/models", params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()


@app.get("/api/civitai/model/{model_id}")
async def civitai_model(model_id: int, api_key: Optional[str] = Query(None)):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CIVITAI_BASE}/models/{model_id}", headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()


class CivitaiDownloadRequest(BaseModel):
    model_version_id: int
    filename: str
    model_id: str
    tags: list[str] = []
    api_key: Optional[str] = None


@app.post("/api/civitai/download")
async def civitai_download(req: CivitaiDownloadRequest):
    url = f"https://civitai.com/api/download/models/{req.model_version_id}"
    dest = LORAS_DIR / req.filename
    headers = {"Authorization": f"Bearer {req.api_key}"} if req.api_key else {}

    async def do_download():
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in r.aiter_bytes(8192):
                        f.write(chunk)
        meta_path = LORAS_DIR / (req.filename + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "civitai_version_id": req.model_version_id,
                    "model_id": req.model_id,
                    "tags": req.tags,
                    "filename": req.filename,
                },
                f,
            )

    asyncio.create_task(do_download())
    return {"status": "downloading", "dest": str(dest)}


# ── LoRAs ────────────────────────────────────────────────────────────────
@app.get("/api/loras")
def list_loras():
    """List every .safetensors under LORAS_DIR (recursive).

    `rglob` (rather than `glob`) so multi-file HF downloads, which keep their
    repo's directory structure, still surface their LoRA files. The meta.json
    sidecar is looked up next to each file.
    """
    loras = []
    for f in LORAS_DIR.rglob("*.safetensors"):
        meta_path = f.with_suffix(f.suffix + ".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
            except Exception:
                pass
        try:
            rel = f.relative_to(LORAS_DIR).as_posix()
        except ValueError:
            rel = f.name
        loras.append(
            {
                # `filename` stays the basename so DELETE/PATCH routes (which
                # can't span slashes) keep working. `rel_path` carries the
                # full subpath for display when nested.
                "filename":   f.name,
                "rel_path":   rel,
                "size_mb":    round(f.stat().st_size / 1024 / 1024, 1),
                "path":       str(f),
                **meta,
            }
        )
    loras.sort(key=lambda l: -Path(l["path"]).stat().st_mtime)
    return {"loras": loras}


def _find_lora(filename: str) -> Optional[Path]:
    """Find a LoRA by basename anywhere under LORAS_DIR (handles nested HF layouts)."""
    direct = LORAS_DIR / filename
    if direct.exists():
        return direct
    # filename can be a path like "Folder/high_noise_model.safetensors" — try
    # that as a relative path first, then fall back to basename match.
    rel = LORAS_DIR / filename.lstrip("/")
    if rel.exists():
        return rel
    base = Path(filename).name
    matches = list(LORAS_DIR.rglob(base))
    return matches[0] if matches else None


class LoraInstallBody(BaseModel):
    files:    list      # [{hf_repo: str, filename: str}, ...]
    hf_token: Optional[str] = None


@app.post("/api/loras/install")
def install_loras(body: LoraInstallBody):
    """Pull a list of {hf_repo, filename} LoRA files from HuggingFace into
    LORAS_DIR. Skips files that are already on disk. Returns a per-file
    status report so the UI can update its install state.
    """
    from huggingface_hub import hf_hub_download
    token = body.hf_token or os.environ.get("HF_TOKEN")
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in body.files or []:
        repo     = entry.get("hf_repo")
        filename = entry.get("filename")
        if not repo or not filename:
            results.append({"filename": filename or "", "status": "error", "error": "missing hf_repo or filename"})
            continue
        # If the file is already present (anywhere under LORAS_DIR), no-op.
        if _find_lora(filename) is not None:
            results.append({"filename": filename, "status": "already_installed"})
            continue
        try:
            local = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(LORAS_DIR), token=token)
            results.append({"filename": filename, "status": "installed", "path": str(local)})
        except Exception as e:
            results.append({"filename": filename, "status": "error", "error": f"{type(e).__name__}: {e}"})
    return {"results": results}


@app.delete("/api/loras/{filename}")
def delete_lora(filename: str):
    target = _find_lora(filename)
    if not target:
        raise HTTPException(404, "LoRA not found")
    target.unlink()
    meta = target.with_suffix(target.suffix + ".meta.json")
    if meta.exists():
        meta.unlink()
    return {"status": "deleted"}


class LoraTagRequest(BaseModel):
    tags: list[str] = []
    model_id: Optional[str] = None


@app.patch("/api/loras/{filename}")
def update_lora(filename: str, req: LoraTagRequest):
    target = _find_lora(filename)
    if not target:
        raise HTTPException(404, "LoRA not found")
    meta_path = target.with_suffix(target.suffix + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass
    meta["tags"] = req.tags
    if req.model_id is not None:
        # Empty string clears the assignment.
        if req.model_id == "":
            meta.pop("model_id", None)
        else:
            meta["model_id"] = req.model_id
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return {"status": "updated", "meta": meta}


# ── Assets ───────────────────────────────────────────────────────────────
# All asset I/O is encrypted at rest. Files on disk live at <name>.enc as
# AES-GCM blobs (see backend/crypto.py). The user-facing path stays as
# <name> — i.e. "assets/uploads/photo.png" — and the crypto layer transparently
# resolves the real .enc file. URLs are issued via _sign_url() against the
# visible name so the frontend never sees the encryption suffix.

def _require_unlocked():
    """Raise 423 Locked if the data key isn't in RAM yet."""
    if not auth.is_unlocked():
        raise HTTPException(423, "Locked. POST /api/auth/unlock with your password.")


def _scan_assets(root: Path, source: str):
    """List every encrypted asset under `root`, presenting visible names."""
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix != ".enc":
            continue
        visible = fcrypto.visible_path(p)
        ext = visible.suffix.lower()
        if ext in IMAGE_EXTS:
            kind = "image"
        elif ext in VIDEO_EXTS:
            kind = "video"
        else:
            continue
        rel = visible.relative_to(WORKSPACE)
        out.append(
            {
                "name": visible.name,
                "path": str(rel),
                "url":  _sign_url(rel.as_posix()),
                "size_kb": round(p.stat().st_size / 1024, 1),  # ciphertext size
                "kind": kind,
                "source": source,
                "mtime": p.stat().st_mtime,
            }
        )
    return out


@app.get("/api/assets")
def list_assets():
    items = (
        _scan_assets(ASSET_UPLOADS, "upload")
        + _scan_assets(ASSET_GENERATED, "generated")
        + _scan_assets(COMFY_OUTPUT, "generated")
    )
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"assets": items}


@app.post("/api/assets/upload")
async def upload_asset(
    file: UploadFile = File(...),
    x_forge_encrypted: Optional[str] = Header(None),
):
    """Store an uploaded asset.

    Phase 3 / E2E flow: the browser encrypts the file with its in-memory
    AES-GCM key BEFORE upload and sets `X-Forge-Encrypted: 1`. The backend
    just stores the bytes as `<name>.enc`. Plaintext never crosses the wire
    or touches our process beyond TLS framing.

    Legacy / no-key fallback: if the header is absent and the backend has
    its own data key in RAM (Phase 2 path — process started with FORGE
    auth unlocked), encrypt server-side. Else just store plaintext bytes.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(400, "Unsupported file type")

    # Pick a free visible name; the on-disk file is <visible>.enc.
    visible = ASSET_UPLOADS / file.filename
    i = 1
    while fcrypto.encrypted_path(visible).exists() or visible.exists():
        visible = ASSET_UPLOADS / f"{Path(file.filename).stem}_{i}{ext}"
        i += 1

    chunks = []
    while chunk := await file.read(64 * 1024):
        chunks.append(chunk)
    body = b"".join(chunks)

    if x_forge_encrypted == "1":
        # Browser already encrypted — store as-is at <name>.enc. We never
        # see plaintext server-side. Closes the internal-hop attack.
        dest = fcrypto.encrypted_path(visible)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    elif auth.is_unlocked():
        # Phase 2 fallback: backend has the key, do the encryption here.
        fcrypto.write_encrypted(auth.data_key, visible, body)
    else:
        # No keys anywhere — store plaintext. (Dev / debug only.)
        visible.parent.mkdir(parents=True, exist_ok=True)
        visible.write_bytes(body)

    rel = visible.relative_to(WORKSPACE)
    return {
        "status":    "uploaded",
        "path":      str(rel),
        "url":       _sign_url(rel.as_posix()),
        "name":      visible.name,
        "encrypted": x_forge_encrypted == "1",
    }


@app.delete("/api/assets")
def delete_asset(path: str = Query(...)):
    """Delete the encrypted file behind a visible asset path."""
    visible = (WORKSPACE / path).resolve()
    try:
        visible.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")
    on_disk = fcrypto.find_on_disk(visible)
    if on_disk is None:
        raise HTTPException(404, "Asset not found")
    on_disk.unlink()
    return {"status": "deleted"}


# ── Authenticated asset stream ───────────────────────────────────────────
# Replaces the old `/workspace-assets/...` static mount, which was outside
# the auth chain — anyone with the proxy URL could fetch assets without a
# token. This route accepts EITHER a valid Bearer token OR a signed query
# string, so `<img src="/api/assets/file/...?sig=…&exp=…">` still works.
@app.get("/api/assets/file/{rel_path:path}")
def get_asset_file(
    rel_path: str,
    sig: Optional[str] = None,
    exp: Optional[int] = None,
    authorization: Optional[str] = Header(None),
):
    has_sig = _verify_signature(rel_path, sig, exp)
    has_token = False
    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
        has_token = auth.verify(token)
    if not (has_sig or has_token):
        raise HTTPException(401, "Authentication required")

    # Resolve against WORKSPACE (sandbox check).
    visible = (WORKSPACE / rel_path).resolve()
    try:
        visible.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")

    on_disk = fcrypto.find_on_disk(visible)
    if on_disk is None:
        raise HTTPException(404, "Asset not found")

    # Always serve raw bytes. The browser's service worker decrypts on the
    # fly using its in-memory CryptoKey — see ui/sw.js. We pick the media
    # type from the VISIBLE filename so the SW can announce the right MIME
    # to the <img> tag after decrypting.
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".m4v": "video/x-m4v", ".mkv": "video/x-matroska",
    }.get(visible.suffix.lower(), "application/octet-stream")
    response = FileResponse(on_disk, media_type=media_type)
    # Hint to the SW so it knows whether to attempt decryption. Plaintext
    # files (legacy / dev) get served through unchanged.
    if on_disk.suffix == ".enc":
        response.headers["X-Forge-Encrypted"] = "1"
    return response


# ── Static UI ────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UI_PORT", 3000)))
