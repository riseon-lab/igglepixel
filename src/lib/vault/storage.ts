// Server-side encrypted vault storage.
//
// The server only ever sees ciphertext: each asset is written as `<id>.bin`
// (the AES-256-GCM envelope produced client-side) plus an `<id>.json` sidecar of
// non-sensitive metadata. Plaintext image bytes never reach this layer.
//
// Storage root resolution (plan.md: "Everything should save to /workspace only"):
//   - $CITIVIA_DATA_DIR/vault   when CITIVIA_DATA_DIR is set (set it to /workspace on RunPod)
//   - <cwd>/.vault              locally (gitignored)

import { randomBytes, randomUUID } from "node:crypto";
import { existsSync, promises as fs } from "node:fs";
import path from "node:path";
import type { AssetMeta, NewAssetMeta } from "./types";

const ASSET_KINDS = new Set(["upload", "reference", "generated"]);

export function vaultDir(): string {
  const base = process.env.CITIVIA_DATA_DIR;
  if (base) return `${base.replace(/\/+$/, "")}/vault`;
  // Mirror the runner, which defaults its data root to /workspace: whenever the
  // persistent volume exists, always resolve there — regardless of NODE_ENV or
  // the current working directory. This is what keeps the account/session from
  // landing in a moving/ephemeral path across restarts (the bug where models
  // persisted on /workspace but the login did not).
  try {
    if (existsSync("/workspace")) return "/workspace/vault";
  } catch {
    /* fall through */
  }
  return ".vault";
}

async function ensureDir(): Promise<string> {
  const dir = vaultDir();
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

// Asset sidecars use a distinct ".meta.json" suffix so they can never collide
// with reserved files in the same directory (account.json, kdf.salt, …). This
// is what list() matches on, so non-asset files are never returned as assets.
const META_SUFFIX = ".meta.json";

/** Reject anything that isn't a plain id, so a path can never escape the vault dir. */
export function isSafeId(id: string): boolean {
  return /^[A-Za-z0-9_-]{1,128}$/.test(id);
}

function isAssetMeta(value: unknown): value is AssetMeta {
  if (!value || typeof value !== "object") return false;
  const meta = value as Record<string, unknown>;
  return (
    typeof meta.id === "string" &&
    isSafeId(meta.id) &&
    typeof meta.name === "string" &&
    typeof meta.mime === "string" &&
    typeof meta.createdAt === "string" &&
    typeof meta.kind === "string" &&
    ASSET_KINDS.has(meta.kind) &&
    typeof meta.width === "number" &&
    Number.isFinite(meta.width) &&
    typeof meta.height === "number" &&
    Number.isFinite(meta.height) &&
    typeof meta.size === "number" &&
    Number.isFinite(meta.size)
  );
}

export async function list(): Promise<AssetMeta[]> {
  const dir = await ensureDir();
  const files = await fs.readdir(dir);
  const metas: AssetMeta[] = [];
  for (const f of files) {
    if (!f.endsWith(META_SUFFIX)) continue;
    try {
      const meta = JSON.parse(
        await fs.readFile(path.join(/* turbopackIgnore: true */ dir, f), "utf8"),
      );
      if (isAssetMeta(meta)) metas.push(meta);
    } catch {
      /* skip unreadable/partial sidecars */
    }
  }
  return metas.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function put(
  ciphertext: Uint8Array,
  meta: NewAssetMeta,
): Promise<AssetMeta> {
  const dir = await ensureDir();
  const id = randomUUID();
  const record: AssetMeta = { id, createdAt: new Date().toISOString(), ...meta };
  await fs.writeFile(path.join(/* turbopackIgnore: true */ dir, `${id}.bin`), ciphertext);
  await fs.writeFile(
    path.join(/* turbopackIgnore: true */ dir, `${id}${META_SUFFIX}`),
    JSON.stringify(record),
  );
  return record;
}

export async function getCiphertext(id: string): Promise<Uint8Array | null> {
  if (!isSafeId(id)) return null;
  try {
    return await fs.readFile(
      path.join(/* turbopackIgnore: true */ vaultDir(), `${id}.bin`),
    );
  } catch {
    return null;
  }
}

export async function getMeta(id: string): Promise<AssetMeta | null> {
  if (!isSafeId(id)) return null;
  try {
    return JSON.parse(
      await fs.readFile(
        path.join(/* turbopackIgnore: true */ vaultDir(), `${id}${META_SUFFIX}`),
        "utf8",
      ),
    );
  } catch {
    return null;
  }
}

export async function remove(id: string): Promise<boolean> {
  if (!isSafeId(id)) return false;
  const dir = vaultDir();
  let removed = false;
  for (const ext of [".bin", META_SUFFIX]) {
    try {
      await fs.unlink(path.join(/* turbopackIgnore: true */ dir, `${id}${ext}`));
      removed = true;
    } catch {
      /* already gone */
    }
  }
  return removed;
}

/**
 * The PBKDF2 salt for the account's encryption key. Not secret — a salt only
 * needs to be unique/stable — so storing it server-side lets every device derive
 * the same key from the password and decrypt previously stored assets.
 */
export async function getOrCreateSalt(): Promise<string> {
  const dir = await ensureDir();
  const file = path.join(/* turbopackIgnore: true */ dir, "kdf.salt");
  try {
    const existing = (await fs.readFile(file, "utf8")).trim();
    if (existing) return existing;
  } catch {
    /* fall through to create */
  }
  const salt = randomBytes(16).toString("base64");
  try {
    // Atomic create: "wx" fails if the file already exists, so two concurrent
    // first requests can't each write a different salt and clobber each other.
    await fs.writeFile(file, salt, { flag: "wx" });
    return salt;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "EEXIST") {
      // Another request won the race — use the salt it persisted.
      return (await fs.readFile(file, "utf8")).trim();
    }
    throw err;
  }
}
