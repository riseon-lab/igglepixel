import { test } from "node:test";
import assert from "node:assert/strict";
import { hashPassword, verifyPassword } from "./password.ts";

test("verifies the correct password", () => {
  const stored = hashPassword("correct horse battery staple");
  assert.equal(verifyPassword("correct horse battery staple", stored), true);
});

test("rejects the wrong password", () => {
  const stored = hashPassword("s3cret-passw0rd");
  assert.equal(verifyPassword("s3cret-passw0rdX", stored), false);
  assert.equal(verifyPassword("", stored), false);
});

test("salts each hash (same password ⇒ different stored value)", () => {
  const a = hashPassword("same-password");
  const b = hashPassword("same-password");
  assert.notEqual(a, b);
  assert.equal(verifyPassword("same-password", a), true);
  assert.equal(verifyPassword("same-password", b), true);
});

test("rejects malformed stored values", () => {
  assert.equal(verifyPassword("x", "not-a-valid-hash"), false);
  assert.equal(verifyPassword("x", "scrypt:onlyonepart"), false);
  assert.equal(verifyPassword("x", ""), false);
});
