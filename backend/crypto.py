"""AES-256-GCM helpers for at-rest asset encryption.

Used by both the FastAPI backend and runner subprocesses so refs and outputs
never touch disk in plaintext. The 32-byte data key is derived from the
user's password via PBKDF2 and only ever lives in RAM:

  - Backend keeps it on the `_Auth` singleton after setup/login/unlock.
  - Runners receive it as `FORGE_DATA_KEY` (hex) on spawn.

On-disk format for an encrypted file:
    [12-byte nonce] [ciphertext + 16-byte GCM tag]

There's no version header — `.enc` files are assumed to be this format.
If you ever change the format, bump the file extension.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# `cryptography` is a hard dep when encryption is enabled. Imported lazily
# so the module still loads on hosts that haven't installed it yet (the
# legacy unencrypted path will still work for asset listing, etc.).
def _aesgcm():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM


# ── Key derivation ────────────────────────────────────────────────────────
PBKDF2_ITERATIONS = 200_000  # ~200ms on a modern CPU; tune up if too fast
SALT_LEN          = 16
NONCE_LEN         = 12
CANARY_PLAINTEXT  = b"FORGE-CANARY-V1"   # known plaintext, encrypted at setup


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from a password + salt via PBKDF2-SHA256.

    Best-effort: we copy the password into a mutable bytearray and zero
    that buffer after derivation. Doesn't reach the original `password`
    string (Python's str immutability + interning) but reduces the number
    of plaintext copies hanging around in our process.
    """
    if len(salt) < 8:
        raise ValueError("salt too short")
    import hashlib
    pw = bytearray(password.encode("utf-8"))
    try:
        return hashlib.pbkdf2_hmac("sha256", pw, salt, PBKDF2_ITERATIONS, dklen=32)
    finally:
        for i in range(len(pw)):
            pw[i] = 0


def new_salt() -> bytes:
    return os.urandom(SALT_LEN)


# ── Bytes-level encrypt / decrypt ─────────────────────────────────────────
def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt with AES-GCM. Returns nonce || ciphertext || tag."""
    nonce = os.urandom(NONCE_LEN)
    ct_with_tag = _aesgcm()(key).encrypt(nonce, plaintext, None)
    return nonce + ct_with_tag


def decrypt_bytes(key: bytes, blob: bytes) -> bytes:
    """Decrypt the format produced by encrypt_bytes(). Raises on bad key/tampering."""
    if len(blob) < NONCE_LEN + 16:
        raise ValueError("ciphertext too short")
    nonce, ct_with_tag = blob[:NONCE_LEN], blob[NONCE_LEN:]
    return _aesgcm()(key).decrypt(nonce, ct_with_tag, None)


# ── Canary helpers (verifies the password without exposing the key) ──────
def make_canary(key: bytes) -> bytes:
    """Encrypt a known plaintext so a future unlock attempt can verify the key."""
    return encrypt_bytes(key, CANARY_PLAINTEXT)


def verify_canary(key: bytes, blob: bytes) -> bool:
    try:
        return decrypt_bytes(key, blob) == CANARY_PLAINTEXT
    except Exception:
        return False


# ── File-level encrypt / decrypt ──────────────────────────────────────────
def encrypted_path(p: Path) -> Path:
    """Append .enc unless already present."""
    return p if p.suffix == ".enc" else p.with_suffix(p.suffix + ".enc")


def visible_path(p: Path) -> Path:
    """Strip the trailing .enc, if any, for display/UI use."""
    return p.with_suffix("") if p.suffix == ".enc" else p


def find_on_disk(visible: Path) -> Optional[Path]:
    """Given the user-facing path, return whichever of <name>.enc or <name> exists."""
    enc = encrypted_path(visible)
    if enc.exists():
        return enc
    if visible.exists():
        return visible
    return None


def write_encrypted(key: bytes, dest_visible: Path, plaintext: bytes) -> Path:
    """Encrypt and write to <dest>.enc atomically. Returns the on-disk path."""
    dest = encrypted_path(dest_visible)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = encrypt_bytes(key, plaintext)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, dest)
    return dest


def read_decrypted(key: bytes, source_visible: Path) -> bytes:
    """Read whichever of <name>.enc or <name> exists; decrypt the encrypted form."""
    on_disk = find_on_disk(source_visible)
    if on_disk is None:
        raise FileNotFoundError(source_visible)
    raw = on_disk.read_bytes()
    if on_disk.suffix == ".enc":
        return decrypt_bytes(key, raw)
    return raw   # legacy plaintext (kept for migration / dev convenience)


def looks_encrypted(path: Path) -> bool:
    """True if the file is at <name>.enc on disk (not a content check — by name)."""
    return path.suffix == ".enc"
