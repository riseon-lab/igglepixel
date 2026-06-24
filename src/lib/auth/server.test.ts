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

test("resetAccount removes the account so setup can run again", async () => {
  await auth.resetAccount();
  assert.equal(await auth.getAccount(), null);

  const result = await auth.createAccount("admin", "password123");
  assert.ok("token" in result);
});

test("login is password-only: a correct password wins without the username", async () => {
  await auth.resetAccount();
  await auth.createAccount("studio-admin", "password123");

  const noName = await auth.login("", "password123");
  assert.ok("token" in noName, "correct password alone should log in");

  const wrongName = await auth.login("someone-else", "password123");
  assert.ok("error" in wrongName, "a supplied-but-wrong username is still rejected");

  const wrongPw = await auth.login("", "nope");
  assert.ok("error" in wrongPw, "a wrong password is rejected");
});

test("accountStatus distinguishes missing, ok, and corrupt", async () => {
  await auth.resetAccount();
  assert.equal(await auth.accountStatus(), "missing");

  await auth.createAccount("admin", "password123");
  assert.equal(await auth.accountStatus(), "ok");

  // Simulate a torn write (pod SIGKILLed mid-write before atomic rename existed):
  // the file exists but isn't valid/usable JSON.
  await fs.writeFile(path.join(TMP, "vault", "account.json"), "{ truncated");
  assert.equal(await auth.accountStatus(), "corrupt");
});
