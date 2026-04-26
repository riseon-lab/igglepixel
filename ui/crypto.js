// ─── Browser-side AES-GCM helpers (Web Crypto) ─────────────────────────────
//
// The data key is derived from the user's password + the backend-supplied
// salt via PBKDF2-SHA256 (matching backend/crypto.py exactly). Both sides
// derive identical keys without ever transmitting the key itself.
//
// The CryptoKey is created `extractable: false`, so even an XSS-injected
// script can't read its raw bytes — only call `subtle.encrypt/decrypt`
// while the JS context is alive. On logout we drop the IDB record and the
// key reference; the browser GC's the underlying material soon after.
//
// File format on the wire is identical to backend/crypto.py:
//     [12-byte IV][ciphertext + 16-byte GCM tag]
// ─────────────────────────────────────────────────────────────────────────

export const PBKDF2_DEFAULT_ITERATIONS = 200_000;
const NONCE_LEN = 12;
const CANARY_PLAINTEXT = new TextEncoder().encode("FORGE-CANARY-V1");

const _utf8 = new TextEncoder();

// Convert a hex string (the form the backend serialises bytes in) into a
// Uint8Array.
export function hexToBytes(hex) {
  if (!hex) return new Uint8Array();
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
  return out;
}

export async function deriveKey(password, salt, iterations = PBKDF2_DEFAULT_ITERATIONS) {
  // Use a Uint8Array we can zero after PBKDF2. The `password` string
  // parameter is still in the JS heap somewhere outside our reach — that's
  // a Python/JS limitation, not a bug. This still beats keeping a buffer
  // alive across the rest of the function.
  const pwBytes = _utf8.encode(password);
  let baseKey;
  try {
    baseKey = await crypto.subtle.importKey(
      "raw", pwBytes, { name: "PBKDF2" }, /* extractable */ false, ["deriveKey"]
    );
  } finally {
    pwBytes.fill(0);   // best-effort wipe of the raw password buffer
  }
  return crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: salt instanceof Uint8Array ? salt : new Uint8Array(salt),
      iterations,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    /* extractable */ false,
    ["encrypt", "decrypt"]
  );
}

// Encrypt arbitrary bytes (Uint8Array or ArrayBuffer). Returns Uint8Array
// matching the backend's "[iv][ct+tag]" format.
export async function encryptBytes(key, plaintext) {
  const iv = crypto.getRandomValues(new Uint8Array(NONCE_LEN));
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plaintext);
  const out = new Uint8Array(NONCE_LEN + ct.byteLength);
  out.set(iv, 0);
  out.set(new Uint8Array(ct), NONCE_LEN);
  return out;
}

export async function decryptBytes(key, blob) {
  const data = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
  if (data.byteLength < NONCE_LEN + 16) throw new Error("ciphertext too short");
  const iv = data.subarray(0, NONCE_LEN);
  const ct = data.subarray(NONCE_LEN);
  return await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
}

// Encrypt a Blob (file picker output) and return a new Blob containing the
// canonical "[iv][ct+tag]" bytes. Used by the upload path so plaintext
// never crosses the wire.
export async function encryptBlob(key, blob) {
  const buf = await blob.arrayBuffer();
  const out = await encryptBytes(key, buf);
  return new Blob([out], { type: "application/octet-stream" });
}

// Verify that a freshly-derived key matches the stored canary. Returns
// true if decryption succeeds AND yields the canonical canary plaintext.
export async function verifyCanary(key, canaryHex) {
  if (!canaryHex) return true;   // no canary stored yet (fresh setup)
  try {
    const pt = new Uint8Array(await decryptBytes(key, hexToBytes(canaryHex)));
    if (pt.length !== CANARY_PLAINTEXT.length) return false;
    for (let i = 0; i < pt.length; i++) {
      if (pt[i] !== CANARY_PLAINTEXT[i]) return false;
    }
    return true;
  } catch {
    return false;
  }
}
