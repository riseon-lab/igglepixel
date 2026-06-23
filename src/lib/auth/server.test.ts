import { after, test } from "node:test";
import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

const TMP = path.join(os.tmpdir(), `citivia-auth-test-${process.pid}`);
process.env.CITIVIA_DATA_DIR = TMP;

const auth = await import("./store.ts");

after(async () => {
  await fs.rm(TMP, { recursive: true, force: true });
});

test("createAccount is race-safe: only one concurrent setup wins", async () => {
  const results = await Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      auth.createAccount(`admin${i}`, "password123"),
    ),
  );
  assert.equal(results.filter((r) => "token" in r).length, 1);
  assert.equal(results.filter((r) => "error" in r).length, 11);
  assert.ok(await auth.getAccount());
});
