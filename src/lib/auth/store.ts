// Pure account/session storage. Kept free of Next imports so unit tests can
// exercise auth behavior without loading Next's route runtime.

import { randomBytes, timingSafeEqual } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { vaultDir } from "../vault/storage.ts";
import { hashPassword, verifyPassword } from "./password.ts";

export interface AccountRecord {
  username: string;
  /** "scrypt:<saltHex>:<hashHex>" */
  password: string;
  sessionToken: string | null;
  sessionStartedAt: string | null;
}

function accountFile(): string {
  return path.join(/* turbopackIgnore: true */ vaultDir(), "account.json");
}

async function readAccount(): Promise<AccountRecord | null> {
  try {
    return JSON.parse(await fs.readFile(accountFile(), "utf8"));
  } catch {
    return null;
  }
}

async function writeAccount(rec: AccountRecord, flag = "w"): Promise<void> {
  await fs.mkdir(vaultDir(), { recursive: true });
  await fs.writeFile(accountFile(), JSON.stringify(rec), { flag });
}

function newToken(): string {
  return randomBytes(32).toString("hex");
}

export function tokensMatch(a: string, b: string): boolean {
  const ba = Buffer.from(a);
  const bb = Buffer.from(b);
  return ba.length === bb.length && timingSafeEqual(ba, bb);
}

export async function getAccount(): Promise<AccountRecord | null> {
  return readAccount();
}

export async function resetAccount(): Promise<void> {
  try {
    await fs.unlink(accountFile());
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
  }
}

export async function createAccount(
  username: string,
  password: string,
): Promise<{ token: string } | { error: string }> {
  const token = newToken();
  try {
    await writeAccount(
      {
        username,
        password: hashPassword(password),
        sessionToken: token,
        sessionStartedAt: new Date().toISOString(),
      },
      "wx",
    );
    return { token };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "EEXIST") {
      return { error: "An account already exists." };
    }
    throw err;
  }
}

export async function login(
  username: string,
  password: string,
): Promise<{ token: string } | { error: string }> {
  const acc = await readAccount();
  if (!acc) return { error: "No account exists yet. Complete setup first." };
  if (acc.username !== username || !verifyPassword(password, acc.password)) {
    return { error: "Incorrect username or password." };
  }
  const token = newToken(); // rotating the token invalidates any other session
  await writeAccount({
    ...acc,
    sessionToken: token,
    sessionStartedAt: new Date().toISOString(),
  });
  return { token };
}

export async function logout(): Promise<void> {
  const acc = await readAccount();
  if (acc) await writeAccount({ ...acc, sessionToken: null, sessionStartedAt: null });
}
