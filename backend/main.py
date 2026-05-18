import asyncio
import copy
import hmac
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
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
import moderator
import prompt_moderator

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
TRAINING_DIR     = WORKSPACE / "training"
DATASETS_DIR     = WORKSPACE / "datasets"
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

for d in (LORAS_DIR, MODELS_DIR, COMPONENTS_DIR, TRAINING_DIR, DATASETS_DIR, ASSET_UPLOADS, ASSET_GENERATED, HF_HOME_DIR, PIP_CACHE_DIR, TMP_DIR, UV_CACHE_DIR, UV_PYTHON_DIR, DOWNLOAD_MARKERS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # WORKSPACE may not exist locally — endpoints handle that on demand.

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

app = FastAPI(title="Forge — RunPod Launcher")
launcher = ModelLauncher()

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Prompt moderation now lazy-loads on first generation request. Preloading at
# boot is nice on a settled pod, but on RunPod it can make startup look like a
# repo/moderation loop if the process is restarted while the classifier is
# still downloading. Operators who want the old warm path can opt in.
if _truthy_env("IGGLEPIXEL_PROMPT_MODERATION_WARMUP"):
    threading.Thread(target=prompt_moderator.init, daemon=True).start()
else:
    print("[prompt_moderator] lazy startup; set IGGLEPIXEL_PROMPT_MODERATION_WARMUP=true to preload", flush=True)


def _access_logs_enabled() -> bool:
    return _truthy_env("IGGLEPIXEL_ACCESS_LOGS")


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
            or path == "/api/moderation/status"
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


def _selected_variant(model: dict, variant: Optional[str] = None) -> Optional[dict]:
    if variant:
        return next((v for v in model.get("variants", []) if v.get("id") == variant), None)
    return None


def _download_plan(model: dict, variant: Optional[str] = None) -> tuple[str, list[dict], list[str], bool]:
    selected = _selected_variant(model, variant)
    repo = (selected or {}).get("hf_repo") or model.get("hf_repo", "")
    extra_files = list(model.get("download_files") or [])
    extra_repos = list(model.get("download_repos") or [])
    snapshot = bool(model.get("download_snapshot", True))
    if selected:
        extra_files.extend(selected.get("download_files") or [])
        extra_repos.extend(selected.get("download_repos") or [])
        snapshot = bool(selected.get("download_snapshot", snapshot))
    return repo, extra_files, extra_repos, snapshot


def _download_access(model: dict, variant: Optional[str] = None) -> dict:
    """Metadata for repos that require HF terms/contact acceptance.

    This is intentionally registry-driven. Hugging Face exposes gating at
    runtime, but checking it live before every download would add another
    network request to a path that already has enough ways to be slow.
    """
    selected = _selected_variant(model, variant)
    merged: dict = {
        "gated": False,
        "requires_hf_token": False,
        "repos": [],
        "note": "",
    }
    seen: set[str] = set()

    def _add_repo(entry) -> None:
        if not entry:
            return
        if isinstance(entry, str):
            item = {"repo": entry, "url": f"https://huggingface.co/{entry}"}
        elif isinstance(entry, dict):
            repo = entry.get("repo")
            if not repo:
                return
            item = {
                "repo": repo,
                "url": entry.get("url") or f"https://huggingface.co/{repo}",
                "label": entry.get("label") or repo,
            }
        else:
            return
        if item["repo"] in seen:
            return
        seen.add(item["repo"])
        merged["repos"].append(item)

    def _merge(access: Optional[dict]) -> None:
        if not isinstance(access, dict):
            return
        merged["gated"] = bool(merged["gated"] or access.get("gated"))
        merged["requires_hf_token"] = bool(merged["requires_hf_token"] or access.get("requires_hf_token"))
        if access.get("note"):
            merged["note"] = access["note"]
        for entry in access.get("repos") or access.get("gated_repos") or []:
            _add_repo(entry)

    _merge(model.get("access"))
    if selected:
        _merge(selected.get("access"))
    if model.get("gated") or model.get("requires_hf_token"):
        merged["gated"] = bool(model.get("gated", True))
        merged["requires_hf_token"] = bool(model.get("requires_hf_token", True))
        if model.get("hf_repo"):
            _add_repo(model["hf_repo"])
    return merged


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
        "access":      _download_access(model, variant),
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
    access = _download_access(model, body.variant)
    if access.get("gated") and not token:
        repos = ", ".join(r.get("repo", "") for r in access.get("repos", []) if r.get("repo")) or repo
        raise HTTPException(
            400,
            f"HF access required for {repos}. Accept the repo terms on Hugging Face, then paste a read HF token before downloading.",
        )

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
            downloads.set(key, downloading=False, error=_format_hf_error(e), progress=0.0)

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
    # `hidden: true` on a registry entry omits the model from the catalogue
    # response. Runner files stay on disk so a manual launch via the API
    # still works for testing; just nothing surfaces in the UI list.
    models = [
        _with_resolved_runtime(data, m)
        for m in data["models"]
        if not m.get("hidden")
    ]
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


# ── Moderation status + runtime override ────────────────────────────────
class ModerationOverrideRequest(BaseModel):
    token: str


@app.get("/api/moderation/status")
def moderation_status():
    """Lightweight, unauthenticated status endpoint so the lock-screen and
    Settings tab can render the current state. Returns enabled + the
    source of the current state (default | env | runtime_override |
    default_fallback).
    """
    return {
        "enabled": moderator.is_enabled(),
        "source":  moderator.disable_source(),
    }


@app.post("/api/moderation/override")
def moderation_override(req: ModerationOverrideRequest):
    """Disable moderation at runtime by writing the override marker file.

    Auth-required so a logged-out request can never flip the policy. The
    submitted token must equal moderator.DISABLE_TOKEN exactly — the same
    constant the env-var path checks against.
    """
    _require_unlocked()
    if (req.token or "").strip() != moderator.DISABLE_TOKEN:
        raise HTTPException(403, "Invalid acknowledgement token")
    try:
        moderator.OVERRIDE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        moderator.OVERRIDE_MARKER.write_text(moderator.DISABLE_TOKEN, encoding="utf-8")
    except OSError as e:
        raise HTTPException(500, f"Could not write marker: {e}")
    moderator.reset_state_log()
    # Force a re-announce on the next is_enabled() call so the audit log
    # captures the transition.
    return {
        "enabled": moderator.is_enabled(),
        "source":  moderator.disable_source(),
    }


@app.delete("/api/moderation/override")
def moderation_override_clear():
    """Re-enable moderation by deleting the runtime override marker. Has no
    effect on the env-var path — that's the fork operator's declaration
    and only the env removal can revoke it. Auth-required.
    """
    _require_unlocked()
    try:
        moderator.OVERRIDE_MARKER.unlink(missing_ok=True)
    except OSError as e:
        raise HTTPException(500, f"Could not remove marker: {e}")
    moderator.reset_state_log()
    return {
        "enabled": moderator.is_enabled(),
        "source":  moderator.disable_source(),
    }


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
    # Prompt moderation runs here — before any model work — so flagged
    # prompts never spin up a runner call. The 422 carries category +
    # label so the UI can render an inline "your prompt was flagged as X"
    # error instead of a silent failure.
    prompt_text = (req.params or {}).get("prompt") or ""
    flagged = prompt_moderator.is_flagged(prompt_text)
    if flagged:
        raise HTTPException(422, {
            "moderated": True,
            "stage":     "prompt",
            "category":  flagged["category"],
            "label":     flagged["label"],
            "score":     flagged["score"],
            "message":   f"Prompt was flagged as {flagged['label']}. Edit the prompt and try again.",
        })
    preview_path = WORKSPACE / "assets" / f".preview_{req.model_id}.jpg"
    try:
        preview_path.unlink(missing_ok=True)
    except OSError:
        pass
    payload = {"params": req.params, "loras": req.loras, "hf_token": req.hf_token}
    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(f"http://127.0.0.1:{info['port']}/generate", json=payload)
        if r.status_code >= 400:
            # Preserve structured FastAPI error bodies (e.g. the ref-image
            # moderation 422 from runner_host) so the UI gets a parsed dict
            # instead of a string-encoded JSON in detail.
            try:
                body = r.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except Exception:
                detail = r.text
            raise HTTPException(r.status_code, detail)
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
    Returns 204 if no preview exists yet (start of generation, model idle, or
    a runner that does not emit step previews). This keeps expected polling
    misses out of access/error logs.
    """
    path = WORKSPACE / "assets" / f".preview_{model_id}.jpg"
    if not path.exists():
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    try:
        body = path.read_bytes()
    except OSError:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    if not body:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(content=body, media_type="image/jpeg",
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
        return f"HF access required — accept the repo terms on Hugging Face, then set a read HF token with access to this repo. ({name})"
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


def _moderation_on() -> bool:
    # Centralised in backend.moderator so prompt + ref + output + CivitAI
    # gates can never silently disagree. See moderator.is_enabled.
    return moderator.is_enabled()


def _nsfw_level(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _strip_nsfw_civitai(payload: dict) -> dict:
    """Drop NSFW items from a CivitAI search response.

    CivitAI flags content via `nsfw: bool` and a graded `nsfwLevel: int`. We
    drop anything with `nsfw: true` or a level above the safe band (1).
    Applied server-side so a tampered client can't bypass it.
    """
    items = payload.get("items") or []
    payload["items"] = [
        m for m in items
        if not m.get("nsfw") and _nsfw_level(m.get("nsfwLevel")) <= 1
    ]
    return payload


@app.get("/api/civitai/search")
async def civitai_search(
    query: str = Query(""),
    types: str = Query("LORA"),
    limit: int = Query(48),
    page: int = Query(1),
    sort: str = Query("Most Downloaded"),
    base_model: Optional[str] = Query(None),
    nsfw: bool = Query(False),
    api_key: Optional[str] = Query(None),
):
    if _moderation_on():
        nsfw = False
    # Over-fetch when moderation is on so the post-strip count is roughly
    # the requested limit. Otherwise a query where ~half of results carry
    # NSFW tags shows the user a sparse 8-item grid for a 24-item request.
    upstream_limit = min(100, limit * 2 if _moderation_on() else limit)
    params: dict = {
        "query":  query,
        "types":  types,
        "limit":  upstream_limit,
        "page":   page,
        "sort":   sort,
        "nsfw":   str(nsfw).lower(),
    }
    # CivitAI's `baseModels` filter accepts an array. The string values
    # must match the labels CivitAI uses in their own UI filter exactly
    # (e.g. "Flux.1 D", "Qwen", "Wan Video") — we surface a curated list
    # in the frontend so callers don't have to guess.
    if base_model:
        bm_list = [s.strip() for s in base_model.split(",") if s.strip()]
        if bm_list:
            params["baseModels"] = bm_list
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CIVITAI_BASE}/models", params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    raw_count = len(data.get("items") or [])
    if _moderation_on():
        data = _strip_nsfw_civitai(data)
    visible = len(data.get("items") or [])
    if not isinstance(data.get("metadata"), dict):
        data["metadata"] = {}
    # Surface the strip cost so the UI can show "12 hidden by moderation",
    # plus pagination hints used by the Load More button.
    data["metadata"]["forge_raw_count"] = raw_count
    data["metadata"]["forge_hidden"]    = max(0, raw_count - visible)
    data["metadata"]["forge_page"]      = page
    data["metadata"]["forge_has_more"]  = bool(data["metadata"].get("nextPage")) or raw_count >= upstream_limit
    if visible > limit:
        data["items"] = data["items"][:limit]
    return data


@app.get("/api/civitai/model/{model_id}")
async def civitai_model(model_id: int, api_key: Optional[str] = Query(None)):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CIVITAI_BASE}/models/{model_id}", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    # Block fetching the detail page for an NSFW model — UI shouldn't be able
    # to bypass the search filter by deep-linking a known model id.
    if _moderation_on() and (data.get("nsfw") or _nsfw_level(data.get("nsfwLevel")) > 1):
        raise HTTPException(403, "Model unavailable under current moderation policy")
    return data


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


# ── Trainers ─────────────────────────────────────────────────────────────
TRAINER_ID_QWEN_CHARACTER = "qwen-character-lora"
QWEN_TRAINER_COMMAND_ENV = "IGGLEPIXEL_QWEN_LORA_TRAIN_CMD"
QWEN_TRAINER_SCRIPT = BASE_DIR / "trainers" / "qwen_lora_train.py"
TRAINING_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TRAINER_OPTIMIZERS = {"adamw8bit", "adamw", "prodigy", "lion", "adafactor"}
TRAINER_SCHEDULERS = {"cosine", "constant", "linear", "cosine-restart"}
TRAINER_PRECISIONS = {"fp16", "bf16", "fp32"}
QWEN_TRAINER_BASE_MODELS = (
    "Qwen/Qwen-Image-2512",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image-Edit-2511",
)
train_jobs: dict[str, dict] = {}


class TrainerDatasetRequest(BaseModel):
    dataset_path: str


class TrainerDatasetDownloadRequest(BaseModel):
    hf_repo: str
    target_name: Optional[str] = None
    repo_type: str = "dataset"
    revision: Optional[str] = None
    hf_token: Optional[str] = None


class TrainJobRequest(BaseModel):
    trainer_id: str = TRAINER_ID_QWEN_CHARACTER
    dataset_path: str
    output_name: str = "kerry_qwen_lora"
    trigger_phrase: str = "A woman named Kerry"
    base_model: str = "Qwen/Qwen-Image-2512"
    steps: int = 3000
    rank: int = 64
    learning_rate: float = 0.0002
    resolution: int = 1024
    batch_size: int = 1
    repeats: int = 1
    hf_token: Optional[str] = None
    # Advanced cfg + sample prompts surfaced by the wizard's Step 4/5.
    # All optional with sensible defaults so older callers (legacy form,
    # automation scripts) keep working unchanged.
    optimizer: Optional[str] = None       # adamw8bit | prodigy | lion | adafactor
    scheduler: Optional[str] = None       # cosine | constant | linear | cosine-restart
    network_alpha: Optional[int] = None   # defaults to rank if unset
    save_every: Optional[int] = None      # default 500
    precision: Optional[str] = None       # fp16 | bf16 | fp32
    gradient_checkpointing: Optional[bool] = None
    sample_prompts: Optional[list[str]] = None
    generate_samples: bool = True
    auto_import_lora: bool = True


class HFLoraUploadRequest(BaseModel):
    filename: str
    hf_repo: str
    private: bool = True
    hf_token: Optional[str] = None
    commit_message: Optional[str] = None


def _safe_name(value: str, fallback: str = "qwen_lora") -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in (value or "").strip())
    out = out.strip("._-")
    return out or fallback


def _safe_upload_filename(value: str) -> str:
    name = Path(value or "").name
    stem = _safe_name(Path(name).stem, "file")
    ext = Path(name).suffix.lower()
    return f"{stem}{ext}"


def _resolve_training_path(value: str) -> Path:
    if not value or not value.strip():
        raise HTTPException(400, "Dataset path is required")
    raw = Path(value.strip()).expanduser()
    path = raw if raw.is_absolute() else WORKSPACE / raw
    path = path.resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(400, "Dataset path must be inside /workspace")
    return path


def _read_caption_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(errors="ignore").strip()


def _training_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in TRAINING_IMAGE_EXTS
    )


def _caption_key(dataset_dir: Path, path: Path) -> tuple[str, str]:
    return (path.parent.relative_to(dataset_dir).as_posix(), path.stem)


def _resolve_dataset_image(dataset_dir: Path, image_rel: str) -> tuple[Path, str]:
    rel = Path(image_rel or "")
    if rel.is_absolute() or not image_rel or ".." in rel.parts:
        raise HTTPException(400, "Image path must stay inside the dataset folder")
    image_path = (dataset_dir / rel).resolve()
    try:
        image_path.relative_to(dataset_dir.resolve())
    except ValueError:
        raise HTTPException(400, "Image path escapes the dataset folder")
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(404, "Image does not exist")
    if image_path.suffix.lower() not in TRAINING_IMAGE_EXTS:
        raise HTTPException(400, "Unsupported training image type")
    return image_path, image_path.relative_to(dataset_dir).as_posix()


def _scan_training_dataset(dataset_dir: Path) -> dict:
    if not dataset_dir.exists():
        return {
            "valid": False,
            "dataset_path": str(dataset_dir),
            "error": "Dataset folder does not exist",
            "image_count": 0,
            "caption_count": 0,
            "pairs": [],
            "missing_captions": [],
            "orphan_captions": [],
            "empty_captions": [],
        }
    if not dataset_dir.is_dir():
        return {
            "valid": False,
            "dataset_path": str(dataset_dir),
            "error": "Dataset path is not a folder",
            "image_count": 0,
            "caption_count": 0,
            "pairs": [],
            "missing_captions": [],
            "orphan_captions": [],
            "empty_captions": [],
        }

    images = _training_images(dataset_dir)
    captions = sorted(p for p in dataset_dir.rglob("*.txt") if p.is_file())
    excludes = _load_excludes(dataset_dir)
    included_images = [p for p in images if p.relative_to(dataset_dir).as_posix() not in excludes]
    image_keys = {_caption_key(dataset_dir, p): p for p in images}
    included_image_keys = {_caption_key(dataset_dir, p): p for p in included_images}
    caption_keys = {_caption_key(dataset_dir, p): p for p in captions}

    pairs = []
    missing = []
    empty = []
    for key, img in included_image_keys.items():
        cap = caption_keys.get(key)
        rel_img = img.relative_to(dataset_dir).as_posix()
        if cap is None:
            missing.append(rel_img)
            continue
        rel_cap = cap.relative_to(dataset_dir).as_posix()
        caption_text = _read_caption_text(cap)
        if not caption_text:
            empty.append(rel_cap)
        pairs.append({"image": rel_img, "caption": rel_cap})

    orphan = [p.relative_to(dataset_dir).as_posix() for key, p in caption_keys.items() if key not in image_keys]
    valid = bool(included_images) and not missing and not empty
    return {
        "valid": valid,
        "dataset_path": str(dataset_dir),
        "image_count": len(included_images),
        "caption_count": sum(1 for key in included_image_keys if key in caption_keys),
        "pair_count": len(pairs),
        "total_image_count": len(images),
        "total_caption_count": len(captions),
        "excluded_count": len(images) - len(included_images),
        "pairs": pairs[:12],
        "missing_captions": missing[:25],
        "orphan_captions": orphan[:25],
        "empty_captions": empty[:25],
        "error": None if valid else "Dataset needs one non-empty .txt caption beside each image",
    }


def _trainer_command() -> str:
    if QWEN_TRAINER_SCRIPT.exists() and not _truthy_env("IGGLEPIXEL_USE_CUSTOM_QWEN_LORA_TRAIN_CMD"):
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(QWEN_TRAINER_SCRIPT))}"
    command = os.environ.get(QWEN_TRAINER_COMMAND_ENV, "").strip()
    if command:
        return command
    if QWEN_TRAINER_SCRIPT.exists():
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(QWEN_TRAINER_SCRIPT))}"
    return ""


def _trainer_command_configured() -> bool:
    return bool(_trainer_command())


@app.get("/api/trainers")
def list_trainers():
    return {
        "trainers": [
            {
                "id": TRAINER_ID_QWEN_CHARACTER,
                "name": "Qwen Character LoRA",
                "category": "lora",
                "description": "Train a Qwen-compatible character LoRA from a curated image/caption folder.",
                "configured": _trainer_command_configured(),
                "command_env": QWEN_TRAINER_COMMAND_ENV,
                "default_command": _trainer_command(),
                "dataset_root": str(WORKSPACE),
                "output_root": str(TRAINING_DIR),
                "base_models": [
                    {"id": model_id, "label": model_id.replace("Qwen/", "").replace("-2511", " 2511")}
                    for model_id in QWEN_TRAINER_BASE_MODELS
                ],
            }
        ],
        "jobs": [_public_train_job(j) for j in sorted(train_jobs.values(), key=lambda x: x.get("created_at", 0), reverse=True)],
    }


@app.post("/api/trainers/validate")
def validate_trainer_dataset(req: TrainerDatasetRequest):
    _require_unlocked()
    return _scan_training_dataset(_resolve_training_path(req.dataset_path))


# ── Dataset curation (wizard Step 3) ─────────────────────────────────────
# A separate "list" endpoint that returns the FULL pair set (not capped at
# 12 like the scan path) plus per-image flags. The /api/trainers/file
# route serves the raw bytes — training datasets aren't encrypted at rest
# so we can stream them directly without the SW decrypt dance.

class TrainerCaptionUpdate(BaseModel):
    dataset_path: str
    image_rel: str          # path of the image inside the dataset
    caption: str


class TrainerExcludeUpdate(BaseModel):
    dataset_path: str
    image_rel: str
    excluded: bool


def _excludes_path(dataset_dir: Path) -> Path:
    """Per-dataset .igglepixel_excludes.json — lists image rel_paths the
    operator marked excluded in the curate step. The launch path builds a
    job-local snapshot from included pairs only, so the listed paths are
    dropped before training starts. Lives inside the dataset folder so it
    travels with it."""
    return dataset_dir / ".igglepixel_excludes.json"


def _load_excludes(dataset_dir: Path) -> set[str]:
    p = _excludes_path(dataset_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception:
        pass
    return set()


def _save_excludes(dataset_dir: Path, excludes: set[str]) -> None:
    p = _excludes_path(dataset_dir)
    p.write_text(json.dumps(sorted(excludes), indent=2), encoding="utf-8")


@app.post("/api/trainers/dataset/list")
def list_trainer_dataset(req: TrainerDatasetRequest):
    """Return the full pair list with caption text + per-image flags. The
    wizard Step 3 reads this once on open and renders the grid + inspector
    from it. Caption edits + exclude toggles go through the dedicated
    endpoints below and the client patches local state — we don't re-fetch
    the whole list on every interaction."""
    _require_unlocked()
    dataset_dir = _resolve_training_path(req.dataset_path)
    if not dataset_dir.is_dir():
        raise HTTPException(404, "Dataset folder does not exist")

    images = _training_images(dataset_dir)
    caption_map = {
        _caption_key(dataset_dir, p): p
        for p in dataset_dir.rglob("*.txt") if p.is_file()
    }
    excludes = _load_excludes(dataset_dir)

    pairs = []
    for img in images:
        rel_img = img.relative_to(dataset_dir).as_posix()
        key = _caption_key(dataset_dir, img)
        cap_path = caption_map.get(key)
        caption_text = ""
        if cap_path is not None:
            caption_text = _read_caption_text(cap_path)

        # Lightweight flags computed without ML. Phase 2.5 adds sharpness,
        # near-dupe detection, and subject-framing scores via a separate
        # analysis pass.
        flags = []
        size = img.stat().st_size if img.exists() else 0
        if size < 30 * 1024:            # <30 KB usually means corrupt / web-thumb-sized
            flags.append({"label": "LOW-RES", "tone": "warn", "detail": f"file size only {size // 1024} KB — likely too small for 1024px training"})
        if cap_path is None:
            flags.append({"label": "NO CAP", "tone": "warn", "detail": "no .txt caption file alongside this image"})
        elif not caption_text:
            flags.append({"label": "EMPTY CAP", "tone": "warn", "detail": "caption file exists but is empty"})
        elif len(caption_text) < 12:
            flags.append({"label": "SHORT CAP", "tone": "warn", "detail": f"caption is only {len(caption_text)} chars — usually you want >30"})

        try:
            ws_rel = img.relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            ws_rel = str(img)
        pairs.append({
            "image": rel_img,
            "caption_path": cap_path.relative_to(dataset_dir).as_posix() if cap_path else None,
            "caption": caption_text,
            "size_bytes": size,
            "flags": flags,
            "excluded": rel_img in excludes,
            "url": _sign_trainer_url(ws_rel),
        })

    # Health metrics — coverage (have caption), captions (non-empty), balance
    # (caption length stddev), variety (unique file size buckets as a cheap
    # proxy). All produced without ML so the bar fills meaningfully even
    # without the Phase 2.5 analysis layer.
    included_pairs = [p for p in pairs if not p["excluded"]]
    total = len(included_pairs)
    denom = total or 1
    covered  = sum(1 for p in included_pairs if p["caption_path"])
    has_text = sum(1 for p in included_pairs if p["caption"])
    avg_len  = sum(len(p["caption"]) for p in included_pairs) / denom
    var_buckets = len({(p["size_bytes"] // (200 * 1024)) for p in included_pairs})  # 200 KB buckets
    health = {
        "coverage": round(100 * covered / denom) if total else 0,
        "captions": round(100 * has_text / denom) if total else 0,
        "balance":  round(max(0, min(100, 100 * (1 - abs(avg_len - 90) / 90)))) if avg_len else 0,
        "variety":  min(100, round(100 * var_buckets / max(8, total // 6))) if total else 0,
        "ready":    covered == total and has_text == total and total >= 8,
    }

    return {
        "dataset_path": req.dataset_path,
        "pair_count":   len(pairs),
        "included_count": len(included_pairs),
        "excluded_count": len(pairs) - len(included_pairs),
        "pairs":        pairs,
        "health":       health,
    }


@app.post("/api/trainers/dataset/caption")
def update_trainer_caption(req: TrainerCaptionUpdate):
    """Write a caption to <dataset>/<image>.txt. Creates the file if
    missing. Returns the persisted caption + updated flags for the row
    so the client can patch its local state without re-fetching."""
    _require_unlocked()
    dataset_dir = _resolve_training_path(req.dataset_path)
    image_path, rel_img = _resolve_dataset_image(dataset_dir, req.image_rel)
    caption_path = image_path.with_suffix(".txt")
    text = (req.caption or "").strip()
    caption_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return {
        "status":   "saved",
        "image":    rel_img,
        "caption":  text,
        "length":   len(text),
    }


@app.post("/api/trainers/dataset/exclude")
def update_trainer_exclude(req: TrainerExcludeUpdate):
    """Toggle an image into/out of the dataset's excluded set. Persisted
    to .igglepixel_excludes.json in the dataset folder. Launch creates a
    curated snapshot and trains against that snapshot."""
    _require_unlocked()
    dataset_dir = _resolve_training_path(req.dataset_path)
    if not dataset_dir.is_dir():
        raise HTTPException(404, "Dataset folder does not exist")
    _, rel_img = _resolve_dataset_image(dataset_dir, req.image_rel)
    excludes = _load_excludes(dataset_dir)
    if req.excluded:
        excludes.add(rel_img)
    else:
        excludes.discard(rel_img)
    _save_excludes(dataset_dir, excludes)
    return {"status": "saved", "image": rel_img, "excluded": req.excluded, "total_excluded": len(excludes)}


def _sign_trainer_url(rel_path: str, ttl_seconds: int = 3600) -> str:
    """Mirror of `_sign_url` for the trainer/file endpoint. Used by the
    dataset-list response so the wizard's <img src> tags can fetch
    thumbnails without an Authorization header (which images can't set).
    """
    exp = int(time.time()) + ttl_seconds
    msg = f"trainer:{rel_path}|{exp}".encode("utf-8")
    sig = hmac.new(auth.signing_key, msg, hashlib.sha256).hexdigest()[:32]
    return f"/api/trainers/file?path={rel_path}&sig={sig}&exp={exp}"


def _verify_trainer_signature(rel_path: str, sig: Optional[str], exp: Optional[int]) -> bool:
    if not sig or not exp:
        return False
    try:
        if int(exp) < int(time.time()):
            return False
    except (TypeError, ValueError):
        return False
    msg = f"trainer:{rel_path}|{exp}".encode("utf-8")
    expected = hmac.new(auth.signing_key, msg, hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


@app.get("/api/trainers/file")
def get_trainer_dataset_file(
    path: str = Query(..., description="Path relative to /workspace"),
    sig: Optional[str] = Query(None),
    exp: Optional[int] = Query(None),
    authorization: Optional[str] = Header(None),
):
    """Serve a plain-bytes file from a training dataset. Sandboxed to
    /workspace, no encryption indirection — training images aren't
    encrypted at rest so we stream them straight.

    Accepts either a signed URL (sig + exp, issued by _sign_trainer_url)
    or an Authorization bearer token. Signed URLs are the normal path
    because <img src> can't set headers.

    The service worker route at /api/assets/file/{path} assumes encrypted
    bytes and decrypts them client-side; using a separate path means we
    bypass that layer entirely.
    """
    has_sig = _verify_trainer_signature(path, sig, exp)
    has_token = False
    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
        has_token = auth.verify(token)
    if not (has_sig or has_token):
        raise HTTPException(401, "Authentication required")
    visible = (WORKSPACE / path).resolve()
    try:
        visible.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not visible.exists() or not visible.is_file():
        raise HTTPException(404, "Not found")
    media_type = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }.get(visible.suffix.lower(), "application/octet-stream")
    return FileResponse(str(visible), media_type=media_type)


@app.post("/api/trainers/datasets/download")
def download_trainer_dataset(req: TrainerDatasetDownloadRequest):
    _require_unlocked()
    repo = (req.hf_repo or "").strip()
    if not repo or " " in repo or repo.startswith("/") or repo.endswith("/"):
        raise HTTPException(400, "Enter a Hugging Face repo id like username/kerry-dataset")
    repo_type = (req.repo_type or "dataset").strip()
    if repo_type not in ("dataset", "model"):
        raise HTTPException(400, "Dataset source must be a dataset or model repo")

    from huggingface_hub import snapshot_download

    target_name = _safe_name(req.target_name or repo.split("/")[-1], "dataset")
    target_dir = (DATASETS_DIR / target_name).resolve()
    try:
        target_dir.relative_to(DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid dataset folder name")
    target_dir.mkdir(parents=True, exist_ok=True)

    token = req.hf_token or os.environ.get("HF_TOKEN")
    try:
        snapshot_download(
            repo_id=repo,
            repo_type=repo_type,
            revision=req.revision or None,
            token=token,
            local_dir=str(target_dir),
        )
    except Exception as e:
        raise HTTPException(400, _format_hf_error(e))

    scan = _scan_training_dataset(target_dir)
    try:
        rel = target_dir.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        rel = str(target_dir)
    return {
        "status": "downloaded",
        "repo": repo,
        "repo_type": repo_type,
        "dataset_path": rel,
        "path": str(target_dir),
        "scan": scan,
    }


@app.post("/api/trainers/dataset/upload")
async def upload_trainer_dataset(
    target_name: str = Query("upload"),
    files: list[UploadFile] = File(...),
):
    _require_unlocked()
    if not files:
        raise HTTPException(400, "Choose at least one image or caption file")

    safe_target = _safe_name(target_name, "upload")
    target_dir = (DATASETS_DIR / "uploads" / f"{safe_target}_{int(time.time())}_{secrets.token_hex(3)}").resolve()
    try:
        target_dir.relative_to(DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid dataset folder name")
    target_dir.mkdir(parents=True, exist_ok=False)

    saved = []
    for item in files:
        filename = _safe_upload_filename(item.filename or "")
        ext = Path(filename).suffix.lower()
        if ext not in TRAINING_IMAGE_EXTS | {".txt"}:
            continue
        dest = (target_dir / filename).resolve()
        try:
            dest.relative_to(target_dir)
        except ValueError:
            raise HTTPException(400, "Invalid uploaded filename")
        i = 1
        base = dest
        while dest.exists():
            dest = target_dir / f"{base.stem}_{i}{base.suffix}"
            i += 1
        body = await item.read()
        if not body:
            continue
        dest.write_bytes(body)
        saved.append(dest.name)

    if not saved:
        raise HTTPException(400, "No supported JPG, PNG, WebP, or TXT files were uploaded")

    scan = _scan_training_dataset(target_dir)
    rel = target_dir.relative_to(WORKSPACE.resolve()).as_posix()
    return {
        "status": "uploaded",
        "dataset_path": rel,
        "path": str(target_dir),
        "files": saved,
        "scan": scan,
    }


def _public_train_job(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    out["log_tail"] = out.get("log_tail", [])[-80:]
    return out


def _mark_train_job_cancelled(job: dict, message: str = "Training cancelled") -> None:
    job["_cancel"] = True
    job["status"] = "cancelled"
    job["phase"] = "Cancelled"
    job["finished_at"] = time.time()
    if message:
        tail = job.setdefault("log_tail", [])
        if not tail or tail[-1] != message:
            tail.append(message)
        job["log_tail"] = tail[-120:]


def _set_train_job_error(job: dict, message: str) -> None:
    if job.get("_cancel") or job.get("status") == "cancelled":
        _mark_train_job_cancelled(job)
        return
    job["status"] = "error"
    job["error"] = message
    job["finished_at"] = time.time()


def _trainer_save_every(steps: int, requested: Optional[int] = None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    return max(250, min(1000, steps // 4 or 250))


def _clean_sample_prompts(prompts: Optional[list[str]]) -> list[str]:
    if not prompts:
        return []
    cleaned = []
    for prompt in prompts:
        text = str(prompt or "").strip()
        if not text:
            continue
        cleaned.append(text[:1000])
        if len(cleaned) >= 12:
            break
    return cleaned


def _prepare_curated_training_dataset(source_dir: Path, target_dir: Path) -> dict:
    """Copy the included image/caption pairs into a job-local snapshot.

    Training receives this snapshot as DATASET_DIR, so excluded rows cannot
    leak into custom trainer commands or AI Toolkit's folder scan.
    """
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    excludes = _load_excludes(source_dir)
    copied = []
    skipped = []
    for image_path in _training_images(source_dir):
        rel_img = image_path.relative_to(source_dir).as_posix()
        if rel_img in excludes:
            skipped.append(rel_img)
            continue
        caption_path = image_path.with_suffix(".txt")
        rel_cap = caption_path.relative_to(source_dir).as_posix()
        dest_img = target_dir / rel_img
        dest_cap = target_dir / rel_cap
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_cap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_img)
        shutil.copy2(caption_path, dest_cap)
        copied.append({"image": rel_img, "caption": rel_cap})

    (target_dir / ".igglepixel_curated_manifest.json").write_text(
        json.dumps(
            {
                "source_dataset_path": str(source_dir),
                "created_at": time.time(),
                "included_count": len(copied),
                "excluded_count": len(skipped),
                "included": copied,
                "excluded": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scan = _scan_training_dataset(target_dir)
    scan["source_dataset_path"] = str(source_dir)
    scan["training_dataset_path"] = str(target_dir)
    return scan


def _is_trainer_aux_safetensor(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(
        marker in lower
        for marker in (
            "accuracy_recovery",
            "torchao_uint",
            "qwen_image_torchao",
            "qwen_image_2512_torchao",
            "qwen_image_edit_torchao",
        )
    )


def _extract_trainer_checkpoint_step(path: Path, output_name: str, total_steps: int) -> Optional[int]:
    stem = path.stem.lower()
    safe_output = re.escape(output_name.lower())
    patterns = [
        r"(?:step|steps|checkpoint|global_step)[-_ ]*0*(\d{1,8})",
        rf"{safe_output}[_-]0*(\d{{3,8}})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        step = int(match.group(1))
        if 0 < step <= max(total_steps * 2, total_steps + 1000):
            return step
    return None


def _trainer_output_candidates(output_dir: Path, output_name: str, total_steps: int) -> list[dict]:
    expected_output = output_dir / f"{output_name}.safetensors"
    raw = sorted(output_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    filtered = [p for p in raw if not _is_trainer_aux_safetensor(p)]
    if not filtered:
        return []

    rows = []
    for path in filtered:
        is_expected = path.resolve() == expected_output.resolve()
        step = total_steps if is_expected else _extract_trainer_checkpoint_step(path, output_name, total_steps)
        rows.append({"path": path, "step": step, "is_expected": is_expected, "mtime": path.stat().st_mtime})

    by_step: dict[int, dict] = {}
    no_step: list[dict] = []
    for row in rows:
        step = row["step"]
        if not step:
            no_step.append(row)
            continue
        existing = by_step.get(step)
        if existing is None:
            by_step[step] = row
            continue
        # Prefer the trainer's native step file over our copied final alias.
        if existing["is_expected"] and not row["is_expected"]:
            by_step[step] = row
        elif row["mtime"] > existing["mtime"] and existing["is_expected"] == row["is_expected"]:
            by_step[step] = row

    if by_step:
        return sorted(by_step.values(), key=lambda r: (r["step"] or total_steps + 1, r["mtime"]))
    return [max(no_step, key=lambda r: r["mtime"])]


def _unique_lora_destination(base_name: str) -> Path:
    dest = LORAS_DIR / f"{base_name}.safetensors"
    i = 1
    while dest.exists():
        dest = LORAS_DIR / f"{base_name}_{i}.safetensors"
        i += 1
    return dest


def _import_trainer_outputs(job: dict, req: TrainJobRequest, output_dir: Path) -> list[dict]:
    output_name = job["output_name"]
    candidates = _trainer_output_candidates(output_dir, output_name, req.steps)
    if not candidates:
        return []

    width = max(4, len(str(req.steps)))
    multi = len(candidates) > 1
    imported = []
    for row in candidates:
        src: Path = row["path"]
        step = row.get("step")
        if multi and step:
            dest_base = f"{output_name}_step{int(step):0{width}d}"
        else:
            dest_base = output_name
        dest = _unique_lora_destination(_safe_name(dest_base, output_name))
        shutil.copy2(src, dest)
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        meta = {
            "tags": ["trained", "qwen", "character"],
            "model_id": "",
            "trainer_id": req.trainer_id,
            "base_model": req.base_model,
            "trigger_phrase": req.trigger_phrase,
            "dataset_path": job["dataset_path"],
            "steps": req.steps,
            "checkpoint_step": step,
            "checkpoint_count": len(candidates),
            "rank": req.rank,
            "learning_rate": req.learning_rate,
            "resolution": req.resolution,
            "batch_size": req.batch_size,
            "source_path": str(src),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        imported.append({"path": str(dest), "filename": dest.name, "step": step})
    return imported


def _trainer_output_records(output_dir: Path, output_name: str, total_steps: int) -> list[dict]:
    records = []
    for row in _trainer_output_candidates(output_dir, output_name, total_steps):
        path: Path = row["path"]
        records.append({
            "path": str(path),
            "filename": path.name,
            "step": row.get("step"),
        })
    return records


def _update_train_job_from_log(job: dict, req: TrainJobRequest, line: str) -> None:
    lower = line.lower()
    if "grabbing lora from the hub" in lower:
        job["_next_rank_is_quant_adapter"] = True

    phase_markers = [
        ("installing ai toolkit", "Installing trainer"),
        ("cloning ai toolkit", "Installing trainer"),
        ("loading checkpoint shards", "Loading base model"),
        ("quantizing transformer", "Quantizing transformer"),
        ("grabbing lora from the hub", "Loading quant adapter"),
        ("create lora network", "Creating LoRA network"),
        ("enable lora", "Creating LoRA network"),
        ("cache latents", "Caching latents"),
        ("caching latents", "Caching latents"),
        ("cache text", "Caching captions"),
        ("caching text", "Caching captions"),
        ("starting ai toolkit training", "Starting training"),
        ("starting trainer command", "Starting trainer"),
    ]
    for marker, phase in phase_markers:
        if marker in lower:
            job["phase"] = phase
            break

    setup_progress = {
        "Installing trainer": 5,
        "Starting trainer": 8,
        "Loading base model": 15,
        "Quantizing transformer": 22,
        "Loading quant adapter": 26,
        "Creating LoRA network": 30,
        "Caching latents": 38,
        "Caching captions": 45,
        "Starting training": 50,
    }
    if job.get("phase") in setup_progress:
        job["progress"] = max(float(job.get("progress", 0)), setup_progress[job["phase"]])

    rank_match = re.search(r"base dim \(rank\):\s*(\d+),\s*alpha:\s*(\d+)", line, re.I)
    if rank_match:
        rank = int(rank_match.group(1))
        alpha = int(rank_match.group(2))
        if job.pop("_next_rank_is_quant_adapter", False):
            job["quant_adapter_rank"] = rank
            job["quant_adapter_alpha"] = alpha
            return
        job["observed_rank"] = rank
        job["observed_alpha"] = alpha
        if job["observed_rank"] != req.rank:
            job["rank_warning"] = f"Trainer reported rank {job['observed_rank']} while UI requested rank {req.rank}"
        else:
            job.pop("rank_warning", None)

    percent_match = re.search(r"(\d{1,3})%\|", line)
    if percent_match and "Loading base model" == job.get("phase"):
        job["phase_progress"] = min(100, int(percent_match.group(1)))

    for current, total in re.findall(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])", line):
        cur = int(current)
        tot = int(total)
        if tot <= 0:
            continue
        # Trainer logs can contain many tqdm counters: dataset items,
        # checkpoint shards, token windows, etc. Only treat n/total as
        # optimizer-step progress when the total is close to the requested
        # training step count.
        step_total_min = max(1, int(req.steps * 0.8))
        step_total_max = max(req.steps + 1000, int(req.steps * 1.2))
        if step_total_min <= tot <= step_total_max:
            job["phase"] = "Training"
            job["current_step"] = cur
            job["total_steps"] = req.steps if abs(tot - req.steps) <= max(10, int(req.steps * 0.02)) else tot
            job["phase_progress"] = min(100, round((cur / tot) * 100, 1))
            job["progress"] = min(98, round(50 + 48 * (cur / tot), 1))
            # Capture loss + lr on the same line if AI Toolkit emitted them.
            # Common shapes: `loss 0.42`, `loss=0.42`, `lr 1.96e-4`, `lr=2.0e-04`.
            # We append to bounded ring buffers so the Monitor sparkline has
            # a windowed history without unbounded memory growth.
            _record_train_metric(job, "loss", line)
            _record_train_metric(job, "lr",   line)
            return
        if "loading checkpoint shards" in lower:
            job["phase"] = "Loading base model"
            job["phase_progress"] = min(100, round((cur / tot) * 100, 1))


TRAIN_METRIC_HISTORY_MAX = 240   # ~4 min of samples at 1 step/sec


def _record_train_metric(job: dict, name: str, line: str) -> None:
    """Append a single metric reading parsed from a trainer log line.

    `name` is 'loss' or 'lr'. We tolerate either `key value` or `key=value`
    forms and either decimal or scientific notation. Values land in
    job["metrics"][name] as a list of (step, value) tuples capped at
    TRAIN_METRIC_HISTORY_MAX so the Monitor sparkline has bounded memory.
    """
    pattern = rf"{name}\s*[=:]?\s*([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)"
    m = re.search(pattern, line, re.I)
    if not m:
        return
    try:
        value = float(m.group(1))
    except ValueError:
        return
    metrics = job.setdefault("metrics", {})
    series = metrics.setdefault(name, [])
    step = job.get("current_step") or (series[-1][0] + 1 if series else 0)
    series.append([int(step), value])
    if len(series) > TRAIN_METRIC_HISTORY_MAX:
        # Slice in-place to preserve the dict's reference.
        del series[: len(series) - TRAIN_METRIC_HISTORY_MAX]


def _run_train_job(job_id: str) -> None:
    job = train_jobs[job_id]
    req: TrainJobRequest = job["_request"]
    if job.get("_cancel") or job.get("status") == "cancelled":
        _mark_train_job_cancelled(job)
        return
    job["status"] = "running"
    job["phase"] = "Validating dataset"
    job["started_at"] = time.time()
    job["log_tail"].append("Validating dataset")

    source_dataset_dir = Path(job["dataset_path"])
    scan = _scan_training_dataset(source_dataset_dir)
    job["dataset"] = scan
    if not scan["valid"]:
        _set_train_job_error(job, scan.get("error") or "Dataset validation failed")
        return
    if job.get("_cancel"):
        _mark_train_job_cancelled(job)
        return

    command = _trainer_command()
    if not command:
        _set_train_job_error(
            job,
            f"Trainer command is not configured. Set {QWEN_TRAINER_COMMAND_ENV} on the pod to enable training.",
        )
        return

    output_dir = TRAINING_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dataset_dir = output_dir / "dataset_curated"
    training_scan = _prepare_curated_training_dataset(source_dataset_dir, training_dataset_dir)
    if job.get("_cancel"):
        _mark_train_job_cancelled(job)
        return
    job["training_dataset_path"] = str(training_dataset_dir)
    job["training_dataset"] = training_scan
    job["log_tail"].append(
        f"Prepared curated dataset: {training_scan.get('image_count', 0)} included, "
        f"{scan.get('excluded_count', 0)} excluded"
    )
    manifest_path = output_dir / "manifest.json"
    save_every = _trainer_save_every(req.steps, req.save_every)
    sample_prompts = _clean_sample_prompts(req.sample_prompts)
    manifest = {
        "trainer_id": req.trainer_id,
        "dataset_path": job["dataset_path"],
        "training_dataset_path": str(training_dataset_dir),
        "output_name": job["output_name"],
        "trigger_phrase": req.trigger_phrase,
        "base_model": req.base_model,
        "steps": req.steps,
        "rank": req.rank,
        "learning_rate": req.learning_rate,
        "resolution": req.resolution,
        "batch_size": req.batch_size,
        "repeats": req.repeats,
        "save_every": save_every,
        "optimizer": req.optimizer or "adamw8bit",
        "scheduler": req.scheduler or "constant",
        "network_alpha": req.network_alpha or req.rank,
        "precision": req.precision or "bf16",
        "gradient_checkpointing": True if req.gradient_checkpointing is None else req.gradient_checkpointing,
        "generate_samples": bool(req.generate_samples),
        "sample_prompts": sample_prompts,
        "auto_import_lora": bool(req.auto_import_lora),
        "dataset": scan,
        "training_dataset": training_scan,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    job["manifest_path"] = str(manifest_path)
    job["output_dir"] = str(output_dir)
    job["log_tail"].append("Starting trainer command")
    job["log_tail"].append(
        f"Requested settings: rank {req.rank}, steps {req.steps}, lr {req.learning_rate}, "
        f"resolution {req.resolution}, batch {req.batch_size}, checkpoints every {save_every} steps"
    )
    job["log_tail"].append(
        "Advanced settings: "
        f"optimizer {req.optimizer or 'adamw8bit'}, scheduler {req.scheduler or 'constant'}, "
        f"alpha {req.network_alpha or req.rank}, precision {req.precision or 'bf16'}, "
        f"samples {'on' if req.generate_samples else 'off'}, "
        f"auto-import {'on' if req.auto_import_lora else 'off'}"
    )

    env = os.environ.copy()
    if req.hf_token:
        env["HF_TOKEN"] = req.hf_token
    env.update(
        {
            "DATASET_DIR": str(training_dataset_dir),
            "SOURCE_DATASET_DIR": job["dataset_path"],
            "TRAINING_DATASET_DIR": str(training_dataset_dir),
            "OUTPUT_DIR": str(output_dir),
            "OUTPUT_NAME": job["output_name"],
            "OUTPUT_PATH": str(output_dir / f"{job['output_name']}.safetensors"),
            "BASE_MODEL": req.base_model,
            "TRIGGER_PHRASE": req.trigger_phrase,
            "TRAIN_STEPS": str(req.steps),
            "TRAIN_RANK": str(req.rank),
            "TRAIN_LR": str(req.learning_rate),
            "TRAIN_RESOLUTION": str(req.resolution),
            "TRAIN_BATCH_SIZE": str(req.batch_size),
            "TRAIN_REPEATS": str(req.repeats),
            "TRAIN_SAVE_EVERY": str(save_every),
            "TRAIN_GENERATE_SAMPLES": "1" if req.generate_samples else "0",
            "TRAIN_MANIFEST": str(manifest_path),
        }
    )
    # Advanced cfg from the wizard. Each is optional — wrapper falls back
    # to its own defaults when not set, so legacy callers stay unaffected.
    if req.optimizer:                       env["TRAIN_OPTIMIZER"] = req.optimizer
    if req.scheduler:                       env["TRAIN_SCHEDULER"] = req.scheduler
    if req.network_alpha is not None:       env["TRAIN_ALPHA"] = str(req.network_alpha)
    if req.precision:                       env["TRAIN_PRECISION"] = req.precision
    if req.gradient_checkpointing is not None:
        env["TRAIN_GRAD_CKPT"] = "1" if req.gradient_checkpointing else "0"
    if sample_prompts:
        # JSON-encoded so multi-line prompts and special chars survive the
        # round-trip cleanly. Wrapper json.loads on read.
        env["TRAIN_SAMPLES"] = json.dumps(sample_prompts)
    if job.get("_cancel"):
        _mark_train_job_cancelled(job)
        return

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job["_process"] = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            job["log_tail"].append(line)
            job["log_tail"] = job["log_tail"][-120:]
            _update_train_job_from_log(job, req, line)
            if job.get("_cancel"):
                proc.terminate()
                _mark_train_job_cancelled(job)
                return
        rc = proc.wait()
        if job.get("_cancel"):
            _mark_train_job_cancelled(job)
            return
        if rc != 0:
            _set_train_job_error(job, f"Trainer command exited with code {rc}")
            return
    except Exception as e:
        _set_train_job_error(job, f"{type(e).__name__}: {e}")
        return
    finally:
        job.pop("_process", None)

    if job.get("_cancel"):
        _mark_train_job_cancelled(job)
        return

    if not req.auto_import_lora:
        outputs = _trainer_output_records(output_dir, job["output_name"], req.steps)
        if not outputs:
            _set_train_job_error(job, "Training finished but no exported LoRA checkpoint was found in the output folder")
            return
        job["status"] = "done"
        job["phase"] = "Done"
        job["progress"] = 100
        job["finished_at"] = time.time()
        job["imported_to_library"] = False
        job["lora_paths"] = [item["path"] for item in outputs]
        job["lora_filenames"] = [item["filename"] for item in outputs]
        job["lora_checkpoints"] = [{"filename": item["filename"], "step": item.get("step")} for item in outputs]
        job["lora_path"] = outputs[-1]["path"]
        job["lora_filename"] = outputs[-1]["filename"]
        job["log_tail"].append("LoRA checkpoints left in training output: " + ", ".join(item["filename"] for item in outputs))
        return

    imported = _import_trainer_outputs(job, req, output_dir)
    if not imported:
        _set_train_job_error(job, "Training finished but no exported LoRA checkpoint was found in the output folder")
        return
    job["status"] = "done"
    job["phase"] = "Done"
    job["progress"] = 100
    job["finished_at"] = time.time()
    job["imported_to_library"] = True
    job["lora_paths"] = [item["path"] for item in imported]
    job["lora_filenames"] = [item["filename"] for item in imported]
    job["lora_checkpoints"] = [{"filename": item["filename"], "step": item.get("step")} for item in imported]
    job["lora_path"] = imported[-1]["path"]
    job["lora_filename"] = imported[-1]["filename"]
    if len(imported) == 1:
        job["log_tail"].append(f"LoRA imported: {imported[0]['filename']}")
    else:
        job["log_tail"].append("LoRA checkpoints imported: " + ", ".join(item["filename"] for item in imported))


@app.post("/api/train-jobs")
def create_train_job(req: TrainJobRequest):
    _require_unlocked()
    if req.trainer_id != TRAINER_ID_QWEN_CHARACTER:
        raise HTTPException(404, "Trainer not found")
    dataset_dir = _resolve_training_path(req.dataset_path)
    scan = _scan_training_dataset(dataset_dir)
    if not scan["valid"]:
        raise HTTPException(400, scan.get("error") or "Dataset validation failed")
    if req.steps < 1 or req.steps > 100000:
        raise HTTPException(400, "Steps must be between 1 and 100000")
    if req.rank not in (4, 8, 16, 32, 64, 128):
        raise HTTPException(400, "Rank must be one of 4, 8, 16, 32, 64, 128")
    if req.base_model not in QWEN_TRAINER_BASE_MODELS:
        raise HTTPException(400, "Unsupported Qwen base model")
    if req.learning_rate <= 0 or req.learning_rate > 1:
        raise HTTPException(400, "Learning rate must be greater than 0 and at most 1")
    if req.resolution not in (512, 768, 1024, 1328):
        raise HTTPException(400, "Resolution must be one of 512, 768, 1024, 1328")
    if req.batch_size < 1 or req.batch_size > 16:
        raise HTTPException(400, "Batch size must be between 1 and 16")
    if req.repeats < 1 or req.repeats > 100:
        raise HTTPException(400, "Repeats must be between 1 and 100")
    if req.optimizer and req.optimizer not in TRAINER_OPTIMIZERS:
        raise HTTPException(400, "Unsupported optimizer")
    if req.scheduler and req.scheduler not in TRAINER_SCHEDULERS:
        raise HTTPException(400, "Unsupported scheduler")
    if req.network_alpha is not None and (req.network_alpha < 1 or req.network_alpha > 256):
        raise HTTPException(400, "Network alpha must be between 1 and 256")
    if req.save_every is not None and (req.save_every < 1 or req.save_every > 100000):
        raise HTTPException(400, "Save every must be between 1 and 100000")
    if req.precision and req.precision not in TRAINER_PRECISIONS:
        raise HTTPException(400, "Unsupported precision")
    req.sample_prompts = _clean_sample_prompts(req.sample_prompts) or None

    job_id = secrets.token_hex(8)
    output_name = _safe_name(req.output_name, "qwen_lora")
    save_every = _trainer_save_every(req.steps, req.save_every)
    job = {
        "id": job_id,
        "trainer_id": req.trainer_id,
        "status": "queued",
        "created_at": time.time(),
        "dataset_path": str(dataset_dir),
        "dataset": scan,
        "output_name": output_name,
        "trigger_phrase": req.trigger_phrase,
        "base_model": req.base_model,
        "steps": req.steps,
        "rank": req.rank,
        "learning_rate": req.learning_rate,
        "resolution": req.resolution,
        "batch_size": req.batch_size,
        "repeats": req.repeats,
        "save_every": save_every,
        "optimizer": req.optimizer or "adamw8bit",
        "scheduler": req.scheduler or "constant",
        "network_alpha": req.network_alpha or req.rank,
        "precision": req.precision or "bf16",
        "gradient_checkpointing": True if req.gradient_checkpointing is None else req.gradient_checkpointing,
        "generate_samples": bool(req.generate_samples),
        "sample_prompts": req.sample_prompts or [],
        "auto_import_lora": bool(req.auto_import_lora),
        "progress": 0,
        "log_tail": [],
        "_request": req,
        "_cancel": False,
    }
    train_jobs[job_id] = job
    threading.Thread(target=_run_train_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "job": _public_train_job(job)}


@app.get("/api/train-jobs")
def list_train_jobs():
    items = sorted(train_jobs.values(), key=lambda x: x.get("created_at", 0), reverse=True)
    return {"jobs": [_public_train_job(j) for j in items]}


@app.get("/api/train-jobs/{job_id}")
def get_train_job(job_id: str):
    job = train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _public_train_job(job)


@app.get("/api/train-jobs/{job_id}/samples")
def get_train_job_samples(job_id: str):
    """List checkpoint sample images written by AI Toolkit during training.

    AI Toolkit writes samples to <output_dir>/ai-toolkit-output/<name>/samples/
    using filenames like `<step>__<prompt-slug>__<seed>.jpg`. We group by step
    so the wizard's Samples tab can render a step × prompt matrix.
    """
    _require_unlocked()
    job = train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    out_dir = Path(job.get("output_dir") or "")
    if not out_dir.exists():
        return {"checkpoints": [], "prompts": []}

    # Walk output_dir for any folder named `samples` (AI Toolkit nests one
    # level deeper under ai-toolkit-output/<name>). Collect image files,
    # parse the step out of the filename, and bucket per step.
    sample_files = list(out_dir.rglob("samples/*.jpg")) + list(out_dir.rglob("samples/*.png"))
    by_step: dict[int, list[dict]] = {}
    prompt_order: list[str] = []
    seen_prompts: set[str] = set()
    for p in sample_files:
        m = re.match(r"^(\d+)__(.+?)__\d+\.(?:jpg|png)$", p.name)
        if not m:
            # Fall back to leading-int parse so non-standard names still appear.
            mi = re.match(r"^(\d+)", p.name)
            if not mi:
                continue
            step = int(mi.group(1))
            prompt_slug = "default"
        else:
            step = int(m.group(1))
            prompt_slug = m.group(2)
        try:
            ws_rel = p.relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            ws_rel = str(p)
        if prompt_slug not in seen_prompts:
            seen_prompts.add(prompt_slug)
            prompt_order.append(prompt_slug)
        by_step.setdefault(step, []).append({
            "prompt_slug": prompt_slug,
            "url":         _sign_trainer_url(ws_rel),
            "name":        p.name,
        })

    checkpoints = sorted(by_step.keys())
    return {
        "checkpoints": [
            {"step": s, "samples": by_step[s]} for s in checkpoints
        ],
        "prompts": prompt_order,
    }


@app.delete("/api/train-jobs/{job_id}")
def cancel_or_dismiss_train_job(job_id: str):
    job = train_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("done", "error", "cancelled"):
        train_jobs.pop(job_id, None)
        return {"status": "dismissed"}
    job["_cancel"] = True
    proc = job.get("_process")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    if job["status"] == "queued":
        _mark_train_job_cancelled(job)
    else:
        job["phase"] = "Cancelling"
        tail = job.setdefault("log_tail", [])
        if not tail or tail[-1] != "Cancellation requested":
            tail.append("Cancellation requested")
        job["log_tail"] = tail[-120:]
    return {"status": "cancelling"}


@app.post("/api/loras/upload-hf")
def upload_lora_to_hf(req: HFLoraUploadRequest):
    _require_unlocked()
    repo = (req.hf_repo or "").strip()
    if not repo or " " in repo or repo.startswith("/") or repo.endswith("/"):
        raise HTTPException(400, "Enter a Hugging Face model repo id like username/kerry-qwen-lora")
    token = req.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise HTTPException(400, "A Hugging Face token with write access is required")

    lora_path = _find_lora(req.filename)
    if not lora_path or not lora_path.exists():
        raise HTTPException(404, "LoRA not found")
    try:
        lora_path.resolve().relative_to(LORAS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "LoRA must be inside /workspace/loras")

    meta_path = lora_path.with_suffix(lora_path.suffix + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo, repo_type="model", private=bool(req.private), exist_ok=True)
        commit = req.commit_message or f"Upload {lora_path.name}"
        api.upload_file(
            path_or_fileobj=str(lora_path),
            path_in_repo=lora_path.name,
            repo_id=repo,
            repo_type="model",
            commit_message=commit,
        )
        if meta_path.exists():
            api.upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo=meta_path.name,
                repo_id=repo,
                repo_type="model",
                commit_message="Upload Igglepixel metadata",
            )
        tags = ["lora", "qwen-image", "igglepixel"]
        trigger = meta.get("trigger_phrase") or ""
        base_model = meta.get("base_model") or ""
        readme = "\n".join(
            [
                "---",
                "tags:",
                *[f"- {tag}" for tag in tags],
                *(["base_model: " + base_model] if base_model else []),
                "---",
                "",
                f"# {repo.split('/')[-1]}",
                "",
                "LoRA trained and exported from Igglepixel.",
                "",
                *(["Trigger phrase: `" + trigger + "`", ""] if trigger else []),
                *(["Base model: `" + base_model + "`", ""] if base_model else []),
                f"File: `{lora_path.name}`",
                "",
            ]
        ).encode("utf-8")
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="Add model card",
        )
    except Exception as e:
        raise HTTPException(400, _format_hf_error(e))

    return {
        "status": "uploaded",
        "repo": repo,
        "filename": lora_path.name,
        "url": f"https://huggingface.co/{repo}",
    }


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
        elif ext in AUDIO_EXTS:
            kind = "audio"
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
    if ext not in IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS:
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
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac",
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("UI_PORT", 3000)),
        access_log=_access_logs_enabled(),
    )
