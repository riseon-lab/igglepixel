// Run with:  npm test   (node --test)
//
// Verifies the server vault round-trips ciphertext + metadata and is hardened
// against path traversal. Uses a throwaway temp dir so it never touches .vault.

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

const TMP = path.join(os.tmpdir(), `citivia-vault-test-${process.pid}`);
process.env.CITIVIA_DATA_DIR = TMP;

// Import AFTER setting the env var (the storage layer reads it per call).
const vault = await import("./storage.ts");

before(async () => {
  await fs.mkdir(TMP, { recursive: true });
});
after(async () => {
  await fs.rm(TMP, { recursive: true, force: true });
});

test("put → list → getMeta → getCiphertext round-trip", async () => {
  const cipher = new Uint8Array([0x43, 0x49, 0x54, 0x56, 1, 0, 9, 9, 9]);
  const saved = await vault.put(cipher, {
    name: "photo.png",
    kind: "upload",
    mime: "image/png",
    width: 800,
    height: 600,
    size: 12345,
  });
  assert.match(saved.id, /^[A-Za-z0-9-]+$/);
  assert.ok(saved.createdAt);

  const all = await vault.list();
  assert.ok(all.some((m) => m.id === saved.id));

  const meta = await vault.getMeta(saved.id);
  assert.equal(meta?.name, "photo.png");
  assert.equal(meta?.width, 800);

  const back = await vault.getCiphertext(saved.id);
  assert.deepEqual(new Uint8Array(back!), cipher);
});

test("list ignores non-asset .json files (e.g. account.json)", async () => {
  const isolated = path.join(TMP, "reserved-files");
  process.env.CITIVIA_DATA_DIR = isolated;
  try {
    await fs.mkdir(path.join(isolated, "vault"), { recursive: true });
    // Simulate the account record / other reserved files living in the same dir.
    await fs.writeFile(
      path.join(isolated, "vault", "account.json"),
      JSON.stringify({ username: "admin", password: "scrypt:secret" }),
    );
    await fs.writeFile(path.join(isolated, "vault", "kdf.salt"), "c2FsdA==");
    const saved = await vault.put(new Uint8Array([1]), {
      name: "real.png",
      kind: "upload",
      mime: "image/png",
      width: 1,
      height: 1,
      size: 1,
    });
    const all = await vault.list();
    assert.deepEqual(
      all.map((m) => m.id),
      [saved.id],
      "only real assets are listed; account.json must never leak as an asset",
    );
    assert.ok(
      !JSON.stringify(all).includes("scrypt"),
      "the password hash must never appear in a vault listing",
    );
  } finally {
    process.env.CITIVIA_DATA_DIR = TMP;
  }
});

test("remove deletes both blob and metadata", async () => {
  const saved = await vault.put(new Uint8Array([1, 2, 3]), {
    name: "x",
    kind: "upload",
    mime: "image/png",
    width: 1,
    height: 1,
    size: 3,
  });
  assert.equal(await vault.remove(saved.id), true);
  assert.equal(await vault.getMeta(saved.id), null);
  assert.equal(await vault.getCiphertext(saved.id), null);
});

test("rejects path-traversal ids", async () => {
  assert.equal(vault.isSafeId("../../etc/passwd"), false);
  assert.equal(vault.isSafeId("a/b"), false);
  assert.equal(vault.isSafeId("good-id_123"), true);
  assert.equal(await vault.getCiphertext("../../etc/passwd"), null);
  assert.equal(await vault.remove("../../secret"), false);
});

test("getOrCreateSalt is stable across calls", async () => {
  const a = await vault.getOrCreateSalt();
  const b = await vault.getOrCreateSalt();
  assert.equal(a, b);
  assert.ok(a.length >= 16);
});

test("getOrCreateSalt is race-safe: concurrent first calls agree", async () => {
  // Fresh dir so there's no pre-existing salt — forces the create path to race.
  const raceDir = path.join(TMP, "race");
  process.env.CITIVIA_DATA_DIR = raceDir;
  try {
    const salts = await Promise.all(
      Array.from({ length: 20 }, () => vault.getOrCreateSalt()),
    );
    assert.equal(new Set(salts).size, 1, "all concurrent calls must see one salt");
  } finally {
    process.env.CITIVIA_DATA_DIR = TMP;
  }
});
