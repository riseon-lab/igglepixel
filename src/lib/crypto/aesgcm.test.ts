// Run with:  npm run test:crypto   (node --test, no extra deps)
//
// These tests exercise the exact code path the browser Worker uses, so a green
// run here means the crypto layer itself is sound before we ever open a browser.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  HEADER_BYTES,
  MAGIC,
  decrypt,
  deriveKey,
  encrypt,
  exportRawKey,
  generateKey,
  generateSalt,
  importRawKey,
  isEnvelope,
} from "./aesgcm.ts";

const enc = (s: string) => new TextEncoder().encode(s);
const dec = (b: Uint8Array) => new TextDecoder().decode(b);

test("round-trips a UTF-8 string", async () => {
  const key = await generateKey();
  const plain = enc("hello, encrypted world — café 🔐");
  const envelope = await encrypt(key, plain);
  const out = await decrypt(key, envelope);
  assert.equal(dec(out), "hello, encrypted world — café 🔐");
});

test("envelope carries the magic header and is recognised", async () => {
  const key = await generateKey();
  const envelope = await encrypt(key, enc("x"));
  for (let i = 0; i < MAGIC.length; i++) assert.equal(envelope[i], MAGIC[i]);
  assert.equal(isEnvelope(envelope), true);
  assert.equal(isEnvelope(enc("not encrypted")), false);
});

test("ciphertext differs from plaintext and is longer by header + GCM tag", async () => {
  const key = await generateKey();
  const plain = enc("the quick brown fox");
  const envelope = await encrypt(key, plain);
  // header + 16-byte GCM tag overhead
  assert.equal(envelope.length, HEADER_BYTES + plain.length + 16);
  assert.notDeepEqual([...envelope.subarray(HEADER_BYTES)], [...plain]);
});

test("a fresh IV is used per call (same key + plaintext ⇒ different ciphertext)", async () => {
  const key = await generateKey();
  const plain = enc("deterministic input");
  const a = await encrypt(key, plain);
  const b = await encrypt(key, plain);
  assert.notDeepEqual([...a], [...b]); // IV differs ⇒ whole envelope differs
  assert.deepEqual([...(await decrypt(key, a))], [...(await decrypt(key, b))]);
});

test("decryption with the wrong key is rejected (auth tag fails)", async () => {
  const k1 = await generateKey();
  const k2 = await generateKey();
  const envelope = await encrypt(k1, enc("secret"));
  await assert.rejects(() => decrypt(k2, envelope));
});

test("tampering with the ciphertext is detected", async () => {
  const key = await generateKey();
  const envelope = await encrypt(key, enc("integrity matters"));
  envelope[envelope.length - 1] ^= 0xff; // flip a byte in the tag/ciphertext
  await assert.rejects(() => decrypt(key, envelope));
});

test("a corrupt/foreign blob is rejected with a clear error", async () => {
  const key = await generateKey();
  await assert.rejects(
    () => decrypt(key, enc("definitely not an envelope")),
    /magic header|too short/i,
  );
});

test("password-derived keys round-trip with the same salt", async () => {
  const salt = generateSalt();
  // Lower iteration count keeps the test fast; production uses 600k.
  const key1 = await deriveKey("correct horse battery staple", salt, 50_000);
  const key2 = await deriveKey("correct horse battery staple", salt, 50_000);
  const envelope = await encrypt(key1, enc("derived-key payload"));
  assert.equal(dec(await decrypt(key2, envelope)), "derived-key payload");
});

test("a wrong password derives a key that cannot decrypt", async () => {
  const salt = generateSalt();
  const good = await deriveKey("right-password", salt, 50_000);
  const bad = await deriveKey("wrong-password", salt, 50_000);
  const envelope = await encrypt(good, enc("password gated"));
  await assert.rejects(() => decrypt(bad, envelope));
});

test("raw key export/import preserves the key (the Worker hand-off path)", async () => {
  const key = await generateKey();
  const raw = await exportRawKey(key);
  assert.equal(raw.length, 32); // 256-bit
  const reimported = await importRawKey(raw);
  const envelope = await encrypt(key, enc("worker transfer"));
  assert.equal(dec(await decrypt(reimported, envelope)), "worker transfer");
});

test("handles a 2 MiB binary payload without corruption", async () => {
  const key = await generateKey();
  const big = new Uint8Array(2 * 1024 * 1024);
  for (let i = 0; i < big.length; i++) big[i] = (i * 31 + 7) & 0xff;
  const envelope = await encrypt(key, big);
  const out = await decrypt(key, envelope);
  assert.deepEqual(out, big);
});

test("round-trips empty input", async () => {
  const key = await generateKey();
  const envelope = await encrypt(key, new Uint8Array(0));
  const out = await decrypt(key, envelope);
  assert.equal(out.length, 0);
});
