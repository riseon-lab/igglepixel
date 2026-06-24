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

export type AccountStatus = "missing" | "corrupt" | "ok";

type AccountState =
  | { status: "missing" }
  | { status: "corrupt" }
  | { status: "ok"; account: AccountRecord };

// Distinguishes a genuinely absent account (→ run setup) from a present-but-
// unreadable one (truncated/garbled — e.g. a pod SIGKILLed mid-write before
// writes were made atomic). The corrupt case must NOT look like "no account", or
// the UI sends you to setup and createAccount then fails with "already exists"
// forever — a permanent dead-end.
async function readAccountState(): Promise<AccountState> {
  let raw: string;
  try {
    raw = await fs.readFile(accountFile(), "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return { status: "missing" };
    return { status: "corrupt" }; // exists but unreadable (perms, I/O error)
  }
  try {
    const account = JSON.parse(raw) as AccountRecord;
    if (!account || typeof account.password !== "string" || !account.password) {
      return { status: "corrupt" }; // parsed, but not a usable record
    }
    return { status: "ok", account };
  } catch {
    return { status: "corrupt" }; // present but not valid JSON
  }
}

async function readAccount(): Promise<AccountRecord | null> {
  const state = await readAccountState();
  return state.status === "ok" ? state.account : null;
}

/** "missing" (no account yet) | "corrupt" (present but unreadable) | "ok". */
export async function accountStatus(): Promise<AccountStatus> {
  return (await readAccountState()).status;
}

async function writeAccount(rec: AccountRecord, flag = "w"): Promise<void> {
  const dir = vaultDir();
  await fs.mkdir(dir, { recursive: true });
  const file = accountFile();
  const json = JSON.stringify(rec);

  if (flag === "wx") {
    // First-time setup: O_EXCL fails atomically if an account already exists,
    // which is what makes concurrent setup race-safe. A single small write of a
    // brand-new file doesn't need the temp dance.
    await fs.writeFile(file, json, { flag: "wx" });
    return;
  }

  // Every login rotates the session token, so this overwrite runs often. Write a
  // temp file, fsync it, then rename over the target: a crash or RunPod SIGKILL
  // can interrupt the temp write, but the live account.json is only ever replaced
  // by an atomic rename — so it can never be left half-written and corrupt.
  const tmp = path.join(dir, `account.${randomBytes(6).toString("hex")}.tmp`);
  const handle = await fs.open(tmp, "w");
  try {
    await handle.writeFile(json);
    await handle.sync();
  } finally {
    await handle.close();
  }
  try {
    await fs.rename(tmp, file);
  } catch (err) {
    await fs.rm(tmp, { force: true }).catch(() => {});
    throw err;
  }
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
): Promise<{ token: string; username: string } | { error: string }> {
  const acc = await readAccount();
  if (!acc) return { error: "No account exists yet. Complete setup first." };
  // Single-account deployment: the password is the only secret. A username, if
  // supplied, must still match — but it's optional, so a correct password is
  // never rejected just because the (non-secret) username was misremembered.
  if (username && acc.username !== username)
    return { error: "Incorrect username or password." };
  if (!verifyPassword(password, acc.password))
    return { error: "Incorrect username or password." };
  const token = newToken(); // rotating the token invalidates any other session
  await writeAccount({
    ...acc,
    sessionToken: token,
    sessionStartedAt: new Date().toISOString(),
  });
  return { token, username: acc.username };
}

export async function logout(): Promise<void> {
  const acc = await readAccount();
  if (acc) await writeAccount({ ...acc, sessionToken: null, sessionStartedAt: null });
}
