import asyncio
import copy
import hmac
import hashlib
import json
import os
import secrets
import shutil
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
HF_HOME_DIR      = WORKSPACE / ".cache" / "huggingface"
PIP_CACHE_DIR    = WORKSPACE / ".cache" / "pip"
TMP_DIR          = WORKSPACE / "tmp"
UV_CACHE_DIR     = WORKSPACE / ".cache" / "uv"
UV_PYTHON_DIR    = WORKSPACE / ".cache" / "uv-python"
DOWNLOAD_MARKERS_DIR = WORKSPACE / ".cache" / "igglepixel-downloads"
LORAS_DIR        = WORKSPACE / "loras"
MODELS_DIR       = WORKSPACE / "models"
COMPONENTS_DIR   = WORKSPACE / "components"     # split-file transformer/VAE/text-encoder swaps
ASSETS_DIR       = WORKSPACE / "assets"
ASSET_UPLOADS    = ASSETS_DIR / "uploads"
ASSET_GENERATED  = ASSETS_DIR / "generated"
COMFY_OUTPUT     = WORKSPACE / "ComfyUI" / "output"  # also scanned as 'generated'

# Components live one dir per target so collisions across models can't happen
# and the launcher resolves a path by target+filename without ambiguity.
COMPONENT_TARGETS = ("transformer", "vae", "text_encoder")

os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME_DIR / "datasets"))
os.environ.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ.setdefault("UV_CACHE_DIR", str(UV_CACHE_DIR))
os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(UV_PYTHON_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(WORKSPACE / ".cache"))

for d in (LORAS_DIR, MODELS_DIR, COMPONENTS_DIR, ASSET_UPLOADS, ASSET_GENERATED, HF_HOME_DIR, PIP_CACHE_DIR, TMP_DIR, UV_CACHE_DIR, UV_PYTHON_DIR, DOWNLOAD_MARKERS_DIR):
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


def _is_secure_request(request: Request) -> bool:
    """Treat proxied HTTPS requests as secure even if Uvicorn sees HTTP."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    forwarded = request.headers.get("forwarded", "").lower()
    return (
        request.url.scheme == "https"
        or proto == "https"
        or request.headers.get("x-forwarded-ssl", "").lower() == "on"
        or "proto=https" in forwarded
    )


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
        secure=_is_secure_request(request),
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
def auth_status(request: Request, response: Response):
    # `locked` = setup has happened on disk but the data key isn't in RAM.
    # `authenticated` lets the frontend skip the gated-endpoint ping when it
    # already knows the cookie isn't valid (e.g. fresh tab, expired cookie).
    response.headers["Cache-Control"] = "no-store"
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


def _is_repo_file_cached(repo_id: str, filename: str) -> bool:
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if not snaps.exists():
        return False
    return any((snap / filename).exists() for snap in snaps.iterdir())


def _download_marker_path(repo_id: str) -> Path:
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
    return DOWNLOAD_MARKERS_DIR / f"{digest}.json"


def _mark_repo_downloaded(repo_id: str) -> None:
    try:
        DOWNLOAD_MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        _download_marker_path(repo_id).write_text(json.dumps({"repo": repo_id, "ts": time.time()}))
    except OSError:
        pass


def _clear_repo_download_marker(repo_id: str) -> None:
    try:
        _download_marker_path(repo_id).unlink(missing_ok=True)
    except OSError:
        pass


def _repo_snapshot_downloaded(repo_id: str) -> bool:
    # HF may create a snapshot dir for partial from_pretrained fetches. For
    # whole-repo downloads, trust our own completion marker plus the cache.
    return _download_marker_path(repo_id).exists() and _is_repo_cached(repo_id)


def _download_key(model_id: str, variant: Optional[str] = None) -> str:
    return f"{model_id}:{variant}" if variant else model_id


def _download_plan(model: dict, variant: Optional[str] = None) -> tuple[str, list[dict], list[str], bool]:
    selected = None
    if variant:
        selected = next((v for v in model.get("variants", []) if v.get("id") == variant), None)
    repo = (selected or {}).get("hf_repo") or model.get("hf_repo", "")
    extra_files = list(model.get("download_files") or [])
    extra_repos = list(model.get("download_repos") or [])
    snapshot = bool(model.get("download_snapshot", True))
    if selected:
        extra_files.extend(selected.get("download_files") or [])
        extra_repos.extend(selected.get("download_repos") or [])
        snapshot = bool(selected.get("download_snapshot", snapshot))
    return repo, extra_files, extra_repos, snapshot


def _download_plan_cached(repo: str, extra_files: list[dict], extra_repos: list[str], snapshot: bool) -> bool:
    if not repo:
        return False
    repo_cached = (not snapshot) or _repo_snapshot_downloaded(repo)
    files_cached = all(
        _is_repo_file_cached(f.get("hf_repo", ""), f.get("filename", ""))
        for f in extra_files
        if f.get("hf_repo") and f.get("filename")
    )
    repos_cached = all(_repo_snapshot_downloaded(repo_id) for repo_id in extra_repos)
    return repo_cached and files_cached and repos_cached


@app.get("/api/models/{model_id}/weight-status")
def weight_status(model_id: str, variant: Optional[str] = Query(None)):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    repo, extra_files, extra_repos, snapshot = _download_plan(model, variant)
    tracked = downloads.get(_download_key(model_id, variant))
    cached = _download_plan_cached(repo, extra_files, extra_repos, snapshot)
    return {
        "downloading": tracked["downloading"] and not cached,
        "downloaded":  cached,
        "progress":    1.0 if cached else tracked["progress"],
        "error":       None if cached else tracked["error"],
    }


class DownloadBody(BaseModel):
    hf_token: Optional[str] = None
    variant: Optional[str] = None


@app.post("/api/models/{model_id}/download")
def model_download(model_id: str, body: DownloadBody):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")
    repo, extra_files, extra_repos, snapshot = _download_plan(model, body.variant)
    if not repo:
        raise HTTPException(400, "Model has no hf_repo to download")
    key = _download_key(model_id, body.variant)
    if downloads.get(key)["downloading"]:
        return {"status": "already_in_progress"}
    if _download_plan_cached(repo, extra_files, extra_repos, snapshot):
        downloads.set(key, downloading=False, downloaded=True, error=None, progress=1.0)
        return {"status": "already_cached", "repo": repo, "variant": body.variant}

    token = body.hf_token or os.environ.get("HF_TOKEN")

    def _worker():
        downloads.set(key, downloading=True, downloaded=False, error=None, progress=0.05)
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            total_units = (1 if snapshot else 0) + len(extra_files) + len(extra_repos)
            completed_units = 0

            def _mark_unit_done() -> None:
                nonlocal completed_units
                completed_units += 1
                if total_units:
                    downloads.set(key, progress=min(0.95, 0.05 + 0.90 * (completed_units / total_units)))

            if snapshot:
                snapshot_download(repo_id=repo, token=token)
                _mark_repo_downloaded(repo)
                _mark_unit_done()
            for entry in extra_files:
                hf_hub_download(
                    repo_id=entry["hf_repo"],
                    filename=entry["filename"],
                    token=token,
                )
                _mark_unit_done()
            for repo_id in extra_repos:
                snapshot_download(repo_id=repo_id, token=token)
                _mark_repo_downloaded(repo_id)
                _mark_unit_done()
            downloads.set(key, downloading=False, downloaded=True, progress=1.0)
        except Exception as e:
            downloads.set(key, downloading=False, error=f"{type(e).__name__}: {e}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "repo": repo, "variant": body.variant}


# ── Registry ─────────────────────────────────────────────────────────────
def _resolve_runtime_spec(registry: dict, model: dict) -> Optional[dict]:
    """Resolve a model's runtime profile into the concrete runtime spec.

    Models may still declare a legacy inline `runtime` block; profiles are the
    preferred path because multiple models can share one dependency runtime.
    """
    if model.get("runtime"):
        return copy.deepcopy(model["runtime"])
    profile_id = model.get("runtime_profile")
    if not profile_id:
        return None
    profile = (registry.get("runtime_profiles") or {}).get(profile_id)
    if not profile:
        return None
    spec = copy.deepcopy(profile)
    spec.setdefault("id", profile_id)
    return spec


def _with_resolved_runtime(registry: dict, model: dict) -> dict:
    model = copy.deepcopy(model)
    spec = _resolve_runtime_spec(registry, model)
    if spec:
        model["runtime"] = spec
    return model


@app.get("/api/models")
def get_models():
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    gpu = detect_gpu()
    models = [_with_resolved_runtime(data, m) for m in data["models"]]
    for m in models:
        m["gpu_compatible"]   = gpu["type"] in m.get("gpu_support", [])
        m["vram_ok"]          = gpu["vram_gb"] >= m.get("min_vram_gb", 0)
        m["vram_recommended"] = gpu["vram_gb"] >= m.get("recommended_vram_gb", 0)
    return {
        "models":    models,
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
    # {target: filename} for split-file component swaps (e.g.
    # {"transformer": "qwen_image_edit_2511_bf16.safetensors", "vae": "..."}).
    # Launcher resolves each to a path under WORKSPACE/components/<target>/
    # and passes via FORGE_COMPONENT_<TARGET>.
    components: Optional[dict] = None


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
    model = _with_resolved_runtime(registry, model)
    return await launcher.launch(model, req.loras, req.hf_token, req.quant, req.variant, req.components)


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


class PromptEnhanceRequest(BaseModel):
    prompt: str
    model_id: Optional[str] = None
    category: Optional[str] = None
    hf_token: Optional[str] = None


generation_jobs: dict[str, dict] = {}


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


async def _generate_result(req: GenerateRequest) -> dict:
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


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    return await _generate_result(req)


def _clean_enhanced_prompt(text: str) -> str:
    text = (text or "").strip()
    for prefix in ("Improved prompt:", "Enhanced prompt:", "Prompt:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text.strip()


@app.post("/api/prompt/enhance")
async def enhance_prompt(req: PromptEnhanceRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is empty")

    enhancer_id = "qwen25-chat"
    info = launcher.get(enhancer_id)
    if not info or info.get("status") != "running":
        raise HTTPException(409, "Prompt enhancer needs Qwen 2.5 Chat running. Start it first, then try Enhance.")

    system = (
        "You are a prompt editor for image and video generation models. "
        "Rewrite the user's prompt to be clearer and more useful, while preserving "
        "the subject, intent, style, and constraints. Do not add new characters, "
        "brands, camera moves, moods, or story beats unless they are already implied. "
        "Output only the improved prompt, with no heading, no notes, and no quotes."
    )
    user = (
        f"Target model category: {req.category or 'generation'}\n"
        f"Target model id: {req.model_id or 'unknown'}\n\n"
        f"Original prompt:\n{prompt}"
    )
    result = await _generate_result(GenerateRequest(
        model_id=enhancer_id,
        params={
            "prompt": user,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "thinking": False,
            "temperature": 0.25,
            "top_p": 0.8,
            "max_new_tokens": 220,
        },
        loras=[],
        hf_token=req.hf_token,
    ))
    enhanced = _clean_enhanced_prompt(result.get("meta", {}).get("text") or result.get("text") or "")
    return {"prompt": enhanced or prompt, "model_id": enhancer_id}


async def _run_generation_job(job_id: str, req: GenerateRequest):
    generation_jobs[job_id].update({"status": "running", "started_at": time.time()})
    try:
        result = await _generate_result(req)
        generation_jobs[job_id].update({
            "status": "done",
            "finished_at": time.time(),
            "result": result,
        })
    except HTTPException as e:
        generation_jobs[job_id].update({
            "status": "error",
            "finished_at": time.time(),
            "error": str(e.detail),
        })
    except Exception as e:
        generation_jobs[job_id].update({
            "status": "error",
            "finished_at": time.time(),
            "error": f"{type(e).__name__}: {e}",
        })


@app.post("/api/generate-jobs")
async def start_generate_job(req: GenerateRequest):
    job_id = secrets.token_urlsafe(12)
    generation_jobs[job_id] = {
        "id": job_id,
        "model_id": req.model_id,
        "status": "queued",
        "created_at": time.time(),
    }
    asyncio.create_task(_run_generation_job(job_id, req))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/generate-jobs/{job_id}")
async def get_generate_job(job_id: str):
    job = generation_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Generation job not found")
    return job


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


def _dir_size_bytes(p: Path) -> int:
    """Sum of sizes of every file under p, following symlinks (HF caches use them)."""
    total = 0
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        try:
            if f.is_file() or f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _purge_dir(p: Path) -> tuple[int, Optional[str]]:
    """Delete a directory and return (freed_bytes, error). Doesn't swallow errors —
    ignore_errors hid real failures and made delete look like it worked when it didn't."""
    if not p.exists():
        return (0, None)
    before = _dir_size_bytes(p)
    try:
        shutil.rmtree(p)
    except Exception as e:
        # If something is holding a file (a still-alive runner subprocess for
        # example), report it. The UI surfaces this so the user knows what
        # to do.
        return (before - _dir_size_bytes(p), f"{type(e).__name__}: {e}")
    return (before, None)


@app.delete("/api/models/{model_id}")
async def delete_model_weights(model_id: str):
    """Remove cached weights for a model so the pod can free disk.

    Stops the runner first (a live runner holds open handles to weight files
    and rm will fail on those). Then deletes the HF cache directory for the
    repo, plus any workspace-local mirror, and tracks bytes freed so the UI
    can show real numbers — not just "deleted" while the volume stays full.

    Variants: a single registry entry can declare multiple HF repos via the
    `variants` array (Wan 2.2 has 14B + 5B repos for the same model). Delete
    sweeps all of them so "delete weights" actually clears everything that
    model can pull.
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    model = next((m for m in registry["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, "Model not found")

    # Stop the runner first — a live process holds open handles to the
    # weight files and rm fails silently on those.
    await launcher.stop(model_id)

    # Build the full set of HF repos this model owns. variants extend this
    # so a single model card can manage multiple HF repos.
    repos = set()
    if model.get("hf_repo"):
        repos.add(model["hf_repo"])
    _, extra_files, extra_repos, _ = _download_plan(model)
    repos.update(f.get("hf_repo") for f in extra_files if f.get("hf_repo"))
    repos.update(extra_repos)
    for v in (model.get("variants") or []):
        if v.get("hf_repo"):
            repos.add(v["hf_repo"])
        _, extra_files, extra_repos, _ = _download_plan(model, v.get("id"))
        repos.update(f.get("hf_repo") for f in extra_files if f.get("hf_repo"))
        repos.update(extra_repos)
    repos.discard(None)

    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    removed: list = []
    errors:  list = []
    freed_bytes = 0

    for repo in repos:
        _clear_repo_download_marker(repo)
        # HF cache layout: hub/models--{org}--{name}/
        cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
        freed, err = _purge_dir(cache_dir)
        if freed or err:
            removed.append({"path": str(cache_dir), "freed_bytes": freed, "error": err})
            freed_bytes += freed
            if err:
                errors.append(err)
        # Workspace-local copy if a runner mirrored it there
        local = MODELS_DIR / repo.replace("/", "__")
        freed, err = _purge_dir(local)
        if freed or err:
            removed.append({"path": str(local), "freed_bytes": freed, "error": err})
            freed_bytes += freed
            if err:
                errors.append(err)

    downloads.set(model_id, downloading=False, downloaded=False, progress=0.0, error=None)
    return {
        "status":      "deleted" if not errors else "partial",
        "removed":     removed,
        "freed_bytes": freed_bytes,
        "freed_mb":    round(freed_bytes / 1024 / 1024, 1),
        "errors":      errors,
    }


# ── HuggingFace downloads ────────────────────────────────────────────────
# The previous implementation shelled out to `huggingface-cli download` with
# stdout/stderr piped but never read. The CLI's progress output filled the
# 64 KB pipe buffer, the subprocess deadlocked on its next write, and the
# user saw "Started: …" with no actual download. Errors (bad token, private
# repo, missing CLI) were swallowed silently. Replaced here with the Python
# API + a background worker thread that surfaces progress, errors, and
# cancellation through a polled job dict — same shape as components install.

# Files we treat as "weights" in the browser. Anything else is hidden from
# the flat list; advanced users can still pass an exact path through the
# legacy /api/download/hf endpoint.
HF_WEIGHT_EXTENSIONS = {".safetensors", ".ckpt", ".bin", ".gguf", ".pth", ".pt"}

# Allowed target dirs for downloaded files. Restricting prevents path-
# traversal trickery via `../` in `target_dir` payloads.
HF_TARGET_DIRS = {"loras", "models", "checkpoints", "components"}

# In-memory job table. Keyed by job_id. We cap at the most recent 200 jobs
# so a long-running session doesn't grow unbounded.
hf_download_jobs: dict[str, dict] = {}
HF_JOB_MAX = 200


def _trim_hf_jobs() -> None:
    """Keep only the most recent HF_JOB_MAX jobs, sorted by created_at."""
    if len(hf_download_jobs) <= HF_JOB_MAX:
        return
    extras = sorted(hf_download_jobs.values(), key=lambda j: j.get("created_at", 0))[: -HF_JOB_MAX]
    for j in extras:
        hf_download_jobs.pop(j["id"], None)


def _hf_repo_path(repo: str, rel_path: str) -> Path:
    """Local on-disk basename layout for HF downloads. We store under
    <target>/<repo_basename>/<rel_path> so different files from different
    repos don't collide when they share a basename."""
    safe_repo = repo.replace("/", "__")
    return Path(safe_repo) / rel_path


def _flatten_lora_rel_path(rel_path: str) -> str:
    """Use the repo's filename for LoRA downloads.

    LoRAs are user-managed assets, not model snapshots. Keeping the full HF
    folder path under /workspace/loras makes the library hard to patch/delete
    and can hide files behind nested repo layouts. Store these flat by
    basename; collisions still keep the repo namespace through local_dir.
    """
    return Path(rel_path).name


def _format_hf_error(e: Exception) -> str:
    """Produce a useful message for the UI. huggingface_hub raises a few
    different exception types; we surface the relevant ones plainly so
    users don't have to copy-paste a stacktrace."""
    name = type(e).__name__
    msg  = str(e) or repr(e)
    # 401/403 from gated/private repos lands as HfHubHTTPError. Highlight
    # the auth dimension since that's the action the user can take.
    if "401" in msg or "403" in msg or "Unauthorized" in msg or "gated" in msg.lower():
        return f"Auth required — set an HF token with read access to this repo. ({name})"
    if "404" in msg:
        return f"Repo or revision not found. ({name})"
    return f"{name}: {msg}"


@app.get("/api/hf/files")
def list_hf_files(
    repo: str = Query(...),
    revision: str = Query("main"),
    hf_token: Optional[str] = Query(None),
):
    """Flat list of weight files in a repo. Recursive — folders aren't
    surfaced; the user just sees the safetensors/ckpt/bin/gguf/pth files
    with their full `rel_path` so subfolder copies disambiguate."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    token = hf_token or os.environ.get("HF_TOKEN") or None
    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id=repo, revision=revision, token=token)
    except RepositoryNotFoundError:
        raise HTTPException(404, f"Repo not found: {repo}")
    except HfHubHTTPError as e:
        raise HTTPException(502, _format_hf_error(e))

    weights = [p for p in all_files if Path(p).suffix.lower() in HF_WEIGHT_EXTENSIONS]
    weights.sort()

    # Sizes: get_paths_info is one API call for the whole list.
    sizes: dict[str, int] = {}
    lfs_flags: dict[str, bool] = {}
    if weights:
        try:
            infos = api.get_paths_info(repo_id=repo, paths=weights, revision=revision, token=token)
            for info in infos:
                # `info` is RepoFile (size, lfs) or RepoFolder. We only asked
                # for files but defensive-isinstance is cheap. Different
                # huggingface_hub versions return either objects or dict-ish
                # payloads, and LFS files may expose the real byte count under
                # `lfs.size` while `size` can be the pointer size.
                if isinstance(info, dict):
                    p = info.get("path") or info.get("rfilename")
                    lfs = info.get("lfs")
                    raw_size = info.get("size")
                else:
                    p = getattr(info, "path", None) or getattr(info, "rfilename", None)
                    lfs = getattr(info, "lfs", None)
                    raw_size = getattr(info, "size", 0)
                if p is None:
                    continue
                lfs_size = None
                if isinstance(lfs, dict):
                    lfs_size = lfs.get("size")
                elif lfs is not None:
                    lfs_size = getattr(lfs, "size", None)
                sizes[p] = int(lfs_size or raw_size or 0)
                lfs_flags[p] = bool(lfs)
        except Exception:
            # Sizes are best-effort — UI shows "?" and download still works.
            pass

    return {
        "repo":     repo,
        "revision": revision,
        "files":    [
            {"rel_path": p, "size": sizes.get(p, 0), "lfs": lfs_flags.get(p, False)}
            for p in weights
        ],
    }


class HFDownloadFile(BaseModel):
    rel_path:   str
    target_dir: str = "models"


class HFDownloadRequest(BaseModel):
    repo_id:    str
    files:      list[HFDownloadFile] = []
    revision:   Optional[str]        = "main"
    hf_token:   Optional[str]        = None

    # Back-compat fields. Old single-file callers send {repo_id, filename, target_dir}.
    # `files` takes precedence when present.
    filename:   Optional[str] = None
    target_dir: Optional[str] = None


class HFLoraImportDoneRequest(BaseModel):
    repo_id:   str
    rel_paths: list[str] = []
    revision:  Optional[str] = "main"
    hf_token:  Optional[str] = None


def _resolve_target_dir(name: str) -> Path:
    if name not in HF_TARGET_DIRS:
        raise HTTPException(400, f"target_dir must be one of {sorted(HF_TARGET_DIRS)}")
    return WORKSPACE / name


def _hf_job_worker(job_id: str, repo: str, revision: str, rel_path: str,
                   target_dir_name: str, token: Optional[str]) -> None:
    """Run one HF download in a worker thread. Polls partial file size so
    the UI can render a live progress bar without depending on huggingface_hub
    surfacing a callback (it doesn't expose one cleanly)."""
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    job = hf_download_jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = time.time()

    target_root = _resolve_target_dir(target_dir_name)
    repo_subdir = _hf_repo_path(repo, "")
    local_dir   = target_root / repo_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    local_filename = _flatten_lora_rel_path(rel_path) if target_dir_name == "loras" else rel_path
    expected_dest = local_dir / local_filename

    # Pre-flight size so the UI has a denominator before bytes start moving.
    try:
        api = HfApi()
        infos = api.get_paths_info(repo_id=repo, paths=[rel_path], revision=revision, token=token)
        if infos:
            job["total_bytes"] = int(getattr(infos[0], "size", 0) or 0)
    except Exception:
        pass  # Size unknown; UI shows "? MB". Download still works.

    # Background poller that watches the partial file size on disk while the
    # blocking hf_hub_download call runs. huggingface_hub doesn't expose a
    # progress callback we can rely on across versions, so this is the
    # least fragile signal we have. Threads share the job dict by reference.
    cancel_flag = {"cancel": False}
    job["_cancel_flag"] = cancel_flag

    def _poll_size() -> None:
        # Look for the partial file under local_dir; HF either creates it
        # at the final path with `.incomplete` suffix or, in newer versions,
        # writes directly. Either is reliable enough as a progress proxy.
        while job["status"] == "running" and not cancel_flag["cancel"]:
            try:
                if expected_dest.exists():
                    job["downloaded_bytes"] = expected_dest.stat().st_size
                else:
                    # Incomplete files HF leaves around mid-download.
                    candidates = sorted(
                        local_dir.rglob(Path(rel_path).name + "*"),
                        key=lambda p: p.stat().st_mtime, reverse=True,
                    )
                    if candidates:
                        job["downloaded_bytes"] = candidates[0].stat().st_size
            except OSError:
                pass
            time.sleep(1.0)

    poll_thread = threading.Thread(target=_poll_size, daemon=True)
    poll_thread.start()

    try:
        if cancel_flag["cancel"]:
            job["status"] = "cancelled"
            return
        dest = hf_hub_download(
            repo_id=repo,
            filename=rel_path,
            revision=revision,
            local_dir=str(local_dir),
            token=token,
        )
        if target_dir_name == "loras":
            final_dest = local_dir / _flatten_lora_rel_path(rel_path)
            if Path(dest) != final_dest:
                final_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, final_dest)
                dest = str(final_dest)
        if cancel_flag["cancel"]:
            # Cancellation arrived after the file landed. Treat as cancelled
            # but DON'T delete a fully-completed file — the user can choose
            # to remove it from the LoRAs/models tab.
            job["status"] = "cancelled"
            return
        # Final size = on-disk size, in case our poller missed the last
        # write before this thread raced to flip the status.
        try:
            job["downloaded_bytes"] = Path(dest).stat().st_size
        except OSError:
            pass
        job["dest_path"] = str(dest)
        job["status"]    = "done"
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        job["status"] = "error"
        job["error"]  = _format_hf_error(e)
    except Exception as e:
        job["status"] = "error"
        job["error"]  = f"{type(e).__name__}: {e}"
    finally:
        job["finished_at"] = time.time()
        # If cancelled mid-download, sweep the partial file. Done/error
        # states leave the file in place (done = wanted; error = let the
        # user inspect what got partially fetched).
        if job["status"] == "cancelled" and not job.get("dest_path"):
            try:
                if expected_dest.exists():
                    expected_dest.unlink()
            except OSError:
                pass


def _enqueue_hf_job(repo: str, revision: str, rel_path: str,
                    target_dir_name: str, token: Optional[str]) -> str:
    job_id = secrets.token_urlsafe(12)
    hf_download_jobs[job_id] = {
        "id":               job_id,
        "repo":             repo,
        "revision":         revision,
        "rel_path":         rel_path,
        "target_dir":       target_dir_name,
        "status":           "queued",
        "created_at":       time.time(),
        "downloaded_bytes": 0,
        "total_bytes":      0,
    }
    _trim_hf_jobs()
    threading.Thread(
        target=_hf_job_worker,
        args=(job_id, repo, revision, rel_path, target_dir_name, token),
        daemon=True,
    ).start()
    return job_id


@app.post("/api/hf/download")
def hf_download_multi(req: HFDownloadRequest):
    """Queue one job per requested file. Files run sequentially per request
    but multiple requests can run in parallel — the limiting factor is the
    user's bandwidth, not our scheduling.

    Back-compat: a body with `filename` + `target_dir` and no `files` array
    is treated as a single-file request. Old `/api/download/hf` callers
    are forwarded here so this is the one source of truth.
    """
    token = req.hf_token or os.environ.get("HF_TOKEN") or None
    revision = req.revision or "main"

    files = req.files
    if not files and req.filename:
        # Back-compat single-file shape.
        files = [HFDownloadFile(rel_path=req.filename, target_dir=req.target_dir or "models")]
    if not files:
        raise HTTPException(400, "No files specified")

    job_ids: list[str] = []
    for f in files:
        if not f.rel_path:
            continue
        # Validate target_dir up front so the user gets one clear error,
        # not N identical errors per file.
        _resolve_target_dir(f.target_dir)
        job_ids.append(_enqueue_hf_job(req.repo_id, revision, f.rel_path, f.target_dir, token))
    return {"job_ids": job_ids}


@app.post("/api/hf/import-loras/done")
def import_completed_hf_loras(req: HFLoraImportDoneRequest):
    """Best-effort repair/import for HF LoRA files after browser downloads.

    HuggingFace's local_dir behaviour changed across versions and can leave
    files nested under repo paths or cache pointers. After the UI sees jobs
    complete, it calls this endpoint so /workspace/loras has a flat,
    library-visible copy of each selected LoRA filename.
    """
    token = req.hf_token or os.environ.get("HF_TOKEN") or None
    revision = req.revision or "main"
    local_dir = LORAS_DIR / req.repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for rel_path in req.rel_paths:
        if not rel_path:
            continue
        dest = local_dir / Path(rel_path).name
        try:
            matches = list(local_dir.rglob(Path(rel_path).name))
            src = next((p for p in matches if p.exists() and p != dest), None)
            if src and src.exists():
                shutil.copy2(src, dest)
            elif not dest.exists():
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id=req.repo_id,
                    filename=rel_path,
                    revision=revision,
                    local_dir=str(local_dir),
                    token=token,
                )
                if Path(downloaded) != dest:
                    shutil.copy2(downloaded, dest)
            results.append({"rel_path": rel_path, "filename": dest.name, "status": "ok", "path": str(dest)})
        except Exception as e:
            results.append({"rel_path": rel_path, "filename": Path(rel_path).name, "status": "error", "error": f"{type(e).__name__}: {e}"})
    return {"results": results}


@app.get("/api/hf/jobs")
def hf_jobs(since: Optional[float] = Query(None)):
    """Snapshot of the job table. UI polls this every ~2s while there are
    active jobs to render the queue panel."""
    items = []
    for j in hf_download_jobs.values():
        if since and (j.get("finished_at") or j.get("created_at", 0)) < since:
            continue
        # Strip the cancel-flag dict — internal-only and not JSON-clean.
        items.append({k: v for k, v in j.items() if not k.startswith("_")})
    items.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return {"jobs": items}


@app.delete("/api/hf/jobs/{job_id}")
def hf_job_cancel(job_id: str):
    """Cancel active jobs or dismiss finished jobs from the queue.

    For running jobs we set a flag the poller checks; if the file is already
    mid-write to disk, hf_hub_download finishes the current call. Finished
    jobs are removed from the in-memory table so dismissed rows don't come
    back on the next UI poll.
    """
    job = hf_download_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("done", "error", "cancelled"):
        hf_download_jobs.pop(job_id, None)
        return {"status": "dismissed"}
    flag = job.get("_cancel_flag")
    if flag:
        flag["cancel"] = True
    if job["status"] == "queued":
        job["status"] = "cancelled"
        job["finished_at"] = time.time()
    return {"status": "cancelling"}


@app.post("/api/hf/jobs/{job_id}/retry")
def hf_job_retry(job_id: str):
    """Re-queue an errored or cancelled job with the same params."""
    job = hf_download_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("error", "cancelled"):
        raise HTTPException(409, f"Cannot retry a job in '{job['status']}' state")
    token = os.environ.get("HF_TOKEN") or None
    new_id = _enqueue_hf_job(job["repo"], job["revision"], job["rel_path"], job["target_dir"], token)
    return {"job_id": new_id}


@app.post("/api/download/hf")
async def download_hf(req: HFDownloadRequest):
    """Legacy single-file endpoint. Forwards to the new multi-file flow so
    we have one worker, one job table, and one set of error semantics. UI
    callers should prefer /api/hf/download going forward.
    """
    return hf_download_multi(req)


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
def _guess_lora_target(name: str) -> Optional[str]:
    """Best-effort high/low detection from a Wan-style filename.

    CivitAI ships dual-expert Wan LoRAs as a pair like ``foo_high.safetensors``
    + ``foo_low.safetensors`` (or ``high_noise`` / ``low_noise``). When the
    user hasn't manually tagged the file we surface a guess so the workspace
    can apply it to the right expert without forcing a manual step.
    """
    n = name.lower()
    # Order matters — match the more-specific patterns first.
    if "high_noise" in n or "highnoise" in n or "_high" in n or "-high" in n or n.endswith("high.safetensors"):
        return "high"
    if "low_noise"  in n or "lownoise"  in n or "_low"  in n or "-low"  in n or n.endswith("low.safetensors"):
        return "low"
    return None


@app.get("/api/loras")
def list_loras():
    """List every .safetensors under LORAS_DIR (recursive).

    `rglob` (rather than `glob`) so multi-file HF downloads, which keep their
    repo's directory structure, still surface their LoRA files. The meta.json
    sidecar is looked up next to each file.
    """
    loras = []
    seen: set[tuple[str, str]] = set()
    files = sorted(
        LORAS_DIR.rglob("*.safetensors"),
        key=lambda p: (
            len(p.relative_to(LORAS_DIR).parts) if p.is_relative_to(LORAS_DIR) else 999,
            -p.stat().st_mtime,
        ),
    )
    for f in files:
        try:
            rel = f.relative_to(LORAS_DIR).as_posix()
        except ValueError:
            rel = f.name
        parts = Path(rel).parts
        dedupe_key = (parts[0] if len(parts) > 1 else "", f.name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        meta_path = f.with_suffix(f.suffix + ".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
            except Exception:
                pass
        # Auto-detect dual-expert target from filename when the user hasn't
        # set one explicitly. UI can show this as a guess they can override.
        target_guess = _guess_lora_target(f.name)
        loras.append(
            {
                # `filename` stays the basename so DELETE/PATCH routes (which
                # can't span slashes) keep working. `rel_path` carries the
                # full subpath for display when nested.
                "filename":     f.name,
                "rel_path":     rel,
                "size_mb":      round(f.stat().st_size / 1024 / 1024, 1),
                "path":         str(f),
                "target_guess": target_guess,
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


def _lora_delete_targets(filename: str) -> list[Path]:
    """Resolve all physical files represented by a LoRA library row.

    HF imports may leave both a flat library-visible copy and the original
    nested local_dir copy. Removing only one makes the row immediately
    reappear, which feels like delete is broken.
    """
    targets: list[Path] = []

    def add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        try:
            resolved.relative_to(LORAS_DIR.resolve())
        except ValueError:
            return
        if resolved.exists() and resolved not in targets:
            targets.append(resolved)

    rel = Path(filename.lstrip("/"))
    exact = LORAS_DIR / rel
    add(exact)

    base = rel.name
    parts = rel.parts
    if len(parts) > 1:
        namespace = LORAS_DIR / parts[0]
        if namespace.exists():
            for match in namespace.rglob(base):
                add(match)
    else:
        for match in LORAS_DIR.rglob(base):
            add(match)

    found = _find_lora(filename)
    if found:
        add(found)
    return targets


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


@app.delete("/api/loras/{filename:path}")
def delete_lora(filename: str):
    targets = _lora_delete_targets(filename)
    if not targets:
        raise HTTPException(404, "LoRA not found")
    deleted = []
    for target in targets:
        target.unlink(missing_ok=True)
        deleted.append(str(target))
        meta = target.with_suffix(target.suffix + ".meta.json")
        if meta.exists():
            meta.unlink()
    return {"status": "deleted", "deleted": deleted}


# ── Components (split-file transformer / VAE / text-encoder swaps) ────────
# Each component lives at WORKSPACE/components/<target>/<basename>. Targets
# match diffusers component names so the runner can pass the loaded object
# straight into Pipeline.from_pretrained(transformer=…, vae=…). The registry
# entry on a model declares which components it supports; the user installs
# them via /api/components/install (HF) and the launcher passes the chosen
# paths through env vars (FORGE_COMPONENT_<TARGET>) to the runner.

def _component_path(target: str, filename: str) -> Path:
    """Resolve a component file by target and basename. Filename can be a
    nested path (e.g. 'split_files/diffusion_models/foo.safetensors') — we
    strip directories and store flat under components/<target>/."""
    if target not in COMPONENT_TARGETS:
        raise HTTPException(400, f"Unknown component target: {target}")
    return COMPONENTS_DIR / target / Path(filename).name


def _find_component(target: str, filename: str) -> Optional[Path]:
    p = _component_path(target, filename)
    return p if p.exists() else None


class ComponentInstallBody(BaseModel):
    files:    list      # [{target, hf_repo, filename}, ...]
    hf_token: Optional[str] = None


# In-memory job table for component installs. Components are 20-40 GB so a
# synchronous request is killed by RunPod's public-proxy timeout long before
# hf_hub_download finishes — we run the download as a background task and
# let the UI poll for completion.
component_install_jobs: dict[str, dict] = {}


def _install_one_component(entry: dict, token: Optional[str]) -> dict:
    """Sync helper run inside a thread by the background job."""
    from huggingface_hub import hf_hub_download
    target   = (entry.get("target") or "").strip()
    repo     = entry.get("hf_repo")
    filename = entry.get("filename")
    if target not in COMPONENT_TARGETS:
        return {"filename": filename or "", "target": target, "status": "error",
                "error": f"target must be one of {COMPONENT_TARGETS}"}
    if not repo or not filename:
        return {"filename": filename or "", "target": target, "status": "error",
                "error": "missing hf_repo or filename"}
    dest_dir = COMPONENTS_DIR / target
    dest_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(filename).name
    dest = dest_dir / basename
    if dest.exists():
        return {"filename": basename, "target": target, "status": "already_installed",
                "path": str(dest)}
    try:
        # hf_hub_download writes into local_dir preserving the repo's internal
        # path. We then move the file flat into <target>/ so the launcher
        # always finds it at a predictable location.
        staged = hf_hub_download(repo_id=repo, filename=filename,
                                 local_dir=str(dest_dir), token=token)
        staged_path = Path(staged)
        if staged_path != dest:
            staged_path.rename(dest)
            try:
                parent = staged_path.parent
                while parent != dest_dir and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            except OSError:
                pass
        return {"filename": basename, "target": target, "status": "installed",
                "path": str(dest)}
    except Exception as e:
        return {"filename": basename, "target": target, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


async def _run_component_install_job(job_id: str, files: list, token: Optional[str]) -> None:
    job = component_install_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()
    results: list = []
    job["results"] = results
    try:
        for entry in files or []:
            target   = (entry.get("target") or "").strip()
            basename = Path(str(entry.get("filename") or "")).name
            job["current"] = {"target": target, "filename": basename}
            # Run the (blocking, multi-GB) HF download in a worker thread so
            # the FastAPI event loop stays responsive for status polls.
            res = await asyncio.to_thread(_install_one_component, entry, token)
            results.append(res)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["finished_at"] = time.time()
        job.pop("current", None)


@app.post("/api/components/install")
async def install_components(body: ComponentInstallBody):
    """Start a background component install job. Returns a job_id that the
    UI polls via /api/components/install-status/{job_id} until done."""
    token = body.hf_token or os.environ.get("HF_TOKEN")
    job_id = secrets.token_urlsafe(12)
    component_install_jobs[job_id] = {
        "id":         job_id,
        "status":     "queued",
        "created_at": time.time(),
        "files":      [{"target": e.get("target"),
                        "filename": Path(str(e.get("filename") or "")).name}
                       for e in (body.files or [])],
    }
    asyncio.create_task(_run_component_install_job(job_id, body.files, token))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/components/install-status/{job_id}")
def install_components_status(job_id: str):
    job = component_install_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Install job not found")
    # Best-effort progress: peek at partial file size on disk for the file
    # currently downloading. HF writes a `.tmp` or `.incomplete` sibling.
    cur = job.get("current")
    if cur and job.get("status") == "running":
        try:
            d = COMPONENTS_DIR / cur["target"]
            if d.exists():
                # Prefer the most recently-modified non-final file.
                partials = sorted(
                    [p for p in d.iterdir() if p.is_file() and p.name != cur["filename"]],
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if partials:
                    job["downloaded_mb"] = round(partials[0].stat().st_size / 1024 / 1024, 1)
        except OSError:
            pass
    return job


@app.get("/api/components")
def list_components():
    """Return installed components grouped by target — UI uses this to mark
    registry entries as installed and show file size."""
    items = []
    for target in COMPONENT_TARGETS:
        d = COMPONENTS_DIR / target
        if not d.exists():
            continue
        for f in d.glob("*.safetensors"):
            try:
                size_mb = round(f.stat().st_size / 1024 / 1024, 1)
            except OSError:
                size_mb = 0
            items.append({
                "target":   target,
                "filename": f.name,
                "size_mb":  size_mb,
                "path":     str(f),
            })
    items.sort(key=lambda c: (c["target"], c["filename"]))
    return {"components": items}


@app.delete("/api/components/{target}/{filename}")
def delete_component(target: str, filename: str):
    if target not in COMPONENT_TARGETS:
        raise HTTPException(400, f"Unknown component target: {target}")
    p = _component_path(target, filename)
    if not p.exists():
        raise HTTPException(404, "Component not found")
    p.unlink()
    return {"status": "deleted", "path": str(p)}


# ── Runtime profiles ────────────────────────────────────────────────────
# A model can point at a shared `runtime_profile` in the registry. Installing
# the profile (creating the venv, cloning source, pip-installing packages)
# can take 5+ minutes and downloads multi-GB wheels, so it runs as a
# background job with the same shape as components/loras.
import venv_manager  # noqa: E402  (imported here so launcher.py's top-level import works in subprocess too)

runtime_install_jobs: dict[str, dict] = {}


def _registry_runtime_for(model_id: str) -> Optional[dict]:
    """Look up a model's resolved runtime profile/spec, or None."""
    try:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    model = next((m for m in registry.get("models", []) if m.get("id") == model_id), None)
    if not model:
        return None
    return _resolve_runtime_spec(registry, model)


def _run_runtime_install_job(job_id: str, spec: dict) -> None:
    """Background-thread worker — spawns subprocesses for git + pip via
    venv_manager.ensure_runtime. Streams every log line into the job dict
    so the UI status endpoint can show a live tail."""
    job = runtime_install_jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = time.time()
    log_lines: list = []
    job["log"] = log_lines

    def _log(line: str) -> None:
        # Cap the log buffer at 2000 lines — pip can emit a lot of progress
        # and we don't want unbounded memory on long installs.
        log_lines.append(line)
        if len(log_lines) > 2000:
            del log_lines[: len(log_lines) - 2000]
        job["last_line"] = line

    try:
        venv_manager.ensure_runtime(spec, _log)
        job["status"] = "done"
        job["python_path"] = str(venv_manager.runtime_python(spec["id"], spec))
    except Exception as e:
        job["status"] = "error"
        job["error"]  = f"{type(e).__name__}: {e}"
    finally:
        job["finished_at"] = time.time()


@app.post("/api/runtime/{model_id}/install")
async def install_runtime(model_id: str):
    """Kick off runtime-profile preparation for a model.

    Returns a job_id immediately; UI polls the status endpoint until done.
    """
    spec = _registry_runtime_for(model_id)
    if not spec or not spec.get("id"):
        raise HTTPException(400, f"Model '{model_id}' has no runtime profile in the registry")

    # Already ready? No-op success — the UI can flip its install button
    # to "Installed" without scheduling a job.
    if venv_manager.is_runtime_ready(spec["id"], spec):
        return {"status": "already_installed", "runtime": spec["id"]}

    for job in runtime_install_jobs.values():
        if (
            job.get("model_id") == model_id
            and job.get("runtime") == spec["id"]
            and job.get("status") in ("queued", "running")
        ):
            return {"job_id": job["id"], "status": job["status"], "runtime": spec["id"]}

    job_id = secrets.token_urlsafe(12)
    runtime_install_jobs[job_id] = {
        "id":         job_id,
        "model_id":   model_id,
        "runtime":    spec["id"],
        "status":     "queued",
        "created_at": time.time(),
    }
    # Run in a worker thread (not asyncio.create_task) because
    # venv_manager.ensure_runtime makes blocking subprocess.Popen calls
    # that would otherwise stall other endpoints during the 3-5 minute install.
    threading.Thread(
        target=_run_runtime_install_job,
        args=(job_id, spec),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "queued", "runtime": spec["id"]}


@app.get("/api/runtime/{model_id}/install-status/{job_id}")
def runtime_install_status(model_id: str, job_id: str):
    job = runtime_install_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Install job not found")
    if job.get("model_id") != model_id:
        raise HTTPException(404, "Install job not for this model")
    return job


@app.get("/api/runtime/{model_id}/status")
def runtime_status(model_id: str):
    """Lightweight pre-launch check for auto-preparing dependency profiles."""
    spec = _registry_runtime_for(model_id)
    if not spec or not spec.get("id"):
        return {"required": False}
    for job in runtime_install_jobs.values():
        if (
            job.get("model_id") == model_id
            and job.get("runtime") == spec["id"]
            and job.get("status") in ("queued", "running")
        ):
            return {
                "required": True,
                "runtime": spec["id"],
                "state": "installing",
                "job_id": job["id"],
                "last_line": job.get("last_line") or "Preparing runtime profile...",
            }
    return {
        "required": True,
        "runtime":  spec["id"],
        **venv_manager.runtime_status(spec["id"], spec),
    }


class LoraTagRequest(BaseModel):
    tags: list[str] = []
    model_id: Optional[str] = None
    # Dual-expert (MoE) target. CivitAI Wan LoRAs ship as one file per expert
    # and the user has to tell us which is which. Values: "high" | "low" | None.
    target: Optional[str] = None


@app.patch("/api/loras/{filename:path}")
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
    if req.target is not None:
        # Empty string clears; "high"/"low" set; anything else rejected.
        if req.target == "":
            meta.pop("target", None)
        elif req.target in ("high", "low"):
            meta["target"] = req.target
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
