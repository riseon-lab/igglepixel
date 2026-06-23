// AES-256-GCM core for Citivia Studio's "encrypted at rest" requirement.
//
// Environment-agnostic: uses the standard Web Crypto API (`crypto.subtle`),
// which is identical in modern browsers and Node 20+. That lets the exact same
// code path be unit-tested under `node --test` and run inside a browser Worker.
//
// Design (see plan.md "Encryption"):
//   - Per-file random 96-bit IV (never reused with a given key).
//   - Authenticated encryption: GCM's 128-bit tag detects any tampering.
//   - Self-describing envelope so a stored blob carries everything needed to
//     decrypt except the key (which never reaches the server).
//   - The key is derived from the account password via PBKDF2; the server only
//     ever sees ciphertext.

export const MAGIC = new Uint8Array([0x43, 0x49, 0x54, 0x56]); // "CITV"
export const VERSION = 1;
export const IV_BYTES = 12; // 96-bit IV, the GCM-recommended size
export const SALT_BYTES = 16;
export const KEY_BITS = 256;
export const PBKDF2_ITERATIONS = 600_000;
export const HEADER_BYTES = MAGIC.length + 1 /*version*/ + 1 /*flags*/ + IV_BYTES;

function subtle(): SubtleCrypto {
  const s = globalThis.crypto?.subtle;
  if (!s) {
    throw new Error(
      "Web Crypto (crypto.subtle) is unavailable in this environment.",
    );
  }
  return s;
}

/** Cryptographically secure random bytes. */
export function randomBytes(length: number): Uint8Array {
  const out = new Uint8Array(length);
  globalThis.crypto.getRandomValues(out);
  return out;
}

export function generateSalt(): Uint8Array {
  return randomBytes(SALT_BYTES);
}

/** Derive a 256-bit AES-GCM key from a password + salt using PBKDF2-SHA-256. */
export async function deriveKey(
  password: string,
  salt: Uint8Array,
  iterations: number = PBKDF2_ITERATIONS,
): Promise<CryptoKey> {
  const baseKey = await subtle().importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  return subtle().deriveKey(
    {
      name: "PBKDF2",
      salt: salt as BufferSource,
      iterations,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: KEY_BITS },
    true, // extractable so it can be handed to a Worker as raw bytes
    ["encrypt", "decrypt"],
  );
}

/** Generate a fresh random AES-256-GCM key (used by the self-test / ephemeral data keys). */
export function generateKey(): Promise<CryptoKey> {
  return subtle().generateKey({ name: "AES-GCM", length: KEY_BITS }, true, [
    "encrypt",
    "decrypt",
  ]);
}

export async function exportRawKey(key: CryptoKey): Promise<Uint8Array> {
  return new Uint8Array(await subtle().exportKey("raw", key));
}

export function importRawKey(raw: Uint8Array): Promise<CryptoKey> {
  return subtle().importKey("raw", raw as BufferSource, { name: "AES-GCM" }, true, [
    "encrypt",
    "decrypt",
  ]);
}

/**
 * Encrypt `plaintext`, returning a self-describing envelope:
 *   [ MAGIC(4) | version(1) | flags(1) | iv(12) | ciphertext+tag ]
 */
export async function encrypt(
  key: CryptoKey,
  plaintext: Uint8Array,
): Promise<Uint8Array> {
  const iv = randomBytes(IV_BYTES);
  const cipher = new Uint8Array(
    await subtle().encrypt(
      { name: "AES-GCM", iv: iv as BufferSource },
      key,
      plaintext as BufferSource,
    ),
  );

  const out = new Uint8Array(HEADER_BYTES + cipher.length);
  let o = 0;
  out.set(MAGIC, o);
  o += MAGIC.length;
  out[o++] = VERSION;
  out[o++] = 0; // flags (reserved)
  out.set(iv, o);
  o += IV_BYTES;
  out.set(cipher, o);
  return out;
}

/** Decrypt an envelope produced by {@link encrypt}. Throws on tampering or wrong key. */
export async function decrypt(
  key: CryptoKey,
  envelope: Uint8Array,
): Promise<Uint8Array> {
  if (envelope.length < HEADER_BYTES) {
    throw new Error("Ciphertext is too short to be a valid envelope.");
  }
  for (let i = 0; i < MAGIC.length; i++) {
    if (envelope[i] !== MAGIC[i]) {
      throw new Error("Not a Citivia encrypted file (bad magic header).");
    }
  }
  const version = envelope[MAGIC.length];
  if (version !== VERSION) {
    throw new Error(`Unsupported envelope version: ${version}.`);
  }
  const iv = envelope.subarray(MAGIC.length + 2, HEADER_BYTES);
  const cipher = envelope.subarray(HEADER_BYTES);

  // GCM verifies the auth tag; a wrong key or modified bytes reject here.
  const plain = await subtle().decrypt(
    { name: "AES-GCM", iv: iv as BufferSource },
    key,
    cipher as BufferSource,
  );
  return new Uint8Array(plain);
}

/** True if `bytes` begins with the Citivia envelope magic + a supported version. */
export function isEnvelope(bytes: Uint8Array): boolean {
  if (bytes.length < HEADER_BYTES) return false;
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) return false;
  }
  return bytes[MAGIC.length] === VERSION;
}
