import { createWriteStream, promises as fs } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import { readApiKeys } from "@/lib/settings/keys";
import type { Lora, LoraSource } from "@/lib/types";

const LORA_EXTENSIONS = new Set([".bin", ".ckpt", ".pt", ".safetensors"]);
const META_SUFFIX = ".meta.json";

export function loraDir(): string {
  const base = process.env.CITIVIA_DATA_DIR;
  if (base) return `${base.replace(/\/+$/, "")}/loras`;
  if (process.env.NODE_ENV === "production") return "/workspace/loras";
  return ".vault/loras";
}

function isSafeFileName(value: string): boolean {
  return (
    /^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$/.test(value) &&
    LORA_EXTENSIONS.has(path.extname(value).toLowerCase())
  );
}

function toSafeFileName(value: string): string {
  const ext = path.extname(value).toLowerCase();
  const base = path
    .basename(value, ext)
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 120);
  const safeExt = LORA_EXTENSIONS.has(ext) ? ext : ".safetensors";
  return `${base || "lora"}${safeExt}`;
}

async function ensureDir(): Promise<string> {
  const dir = loraDir();
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

async function uniqueFileName(dir: string, wanted: string): Promise<string> {
  const parsed = path.parse(toSafeFileName(wanted));
  for (let i = 0; i < 1000; i += 1) {
    const name = i === 0 ? `${parsed.name}${parsed.ext}` : `${parsed.name}-${i}${parsed.ext}`;
    try {
      await fs.access(path.join(/* turbopackIgnore: true */ dir, name));
    } catch {
      return name;
    }
  }
  throw new Error("Could not allocate a LoRA filename.");
}

function sourceFromUrl(url: string): LoraSource {
  if (url.includes("huggingface.co")) return "huggingface";
  if (url.includes("civitai.com")) return "civitai";
  return "upload";
}

function displayName(fileName: string): string {
  return path
    .basename(fileName, path.extname(fileName))
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function readMeta(fileName: string, sizeBytes: number): Promise<Lora> {
  const dir = await ensureDir();
  try {
    const raw = await fs.readFile(
      path.join(/* turbopackIgnore: true */ dir, `${fileName}${META_SUFFIX}`),
      "utf8",
    );
    const parsed = JSON.parse(raw) as Partial<Lora>;
    if (parsed.id === fileName && parsed.path === fileName) {
      return {
        id: fileName,
        name: parsed.name || displayName(fileName),
        source: parsed.source || "upload",
        baseModel: parsed.baseModel || "Qwen",
        sizeBytes,
        triggerWords: Array.isArray(parsed.triggerWords) ? parsed.triggerWords : [],
        installedAt: parsed.installedAt || new Date().toISOString(),
        path: fileName,
      };
    }
  } catch {
    /* derive metadata from the file */
  }
  return {
    id: fileName,
    name: displayName(fileName),
    source: "upload",
    baseModel: "Qwen",
    sizeBytes,
    triggerWords: [],
    installedAt: new Date().toISOString(),
    path: fileName,
  };
}

async function writeMeta(lora: Lora): Promise<void> {
  const dir = await ensureDir();
  await fs.writeFile(
    path.join(/* turbopackIgnore: true */ dir, `${lora.path}${META_SUFFIX}`),
    JSON.stringify(lora),
  );
}

export async function listLoras(): Promise<Lora[]> {
  const dir = await ensureDir();
  const files = await fs.readdir(dir);
  const loras = await Promise.all(
    files
      .filter((fileName) => isSafeFileName(fileName))
      .map(async (fileName) => {
        const stat = await fs.stat(path.join(/* turbopackIgnore: true */ dir, fileName));
        return readMeta(fileName, stat.size);
      }),
  );
  return loras.sort((a, b) => b.installedAt.localeCompare(a.installedAt));
}

export async function saveUploadedLora(file: File): Promise<Lora> {
  const dir = await ensureDir();
  const fileName = await uniqueFileName(dir, file.name);
  const bytes = new Uint8Array(await file.arrayBuffer());
  await fs.writeFile(path.join(/* turbopackIgnore: true */ dir, fileName), bytes, {
    mode: 0o600,
  });
  const lora: Lora = {
    id: fileName,
    name: displayName(fileName),
    source: "upload",
    baseModel: "Qwen",
    sizeBytes: bytes.byteLength,
    triggerWords: [],
    installedAt: new Date().toISOString(),
    path: fileName,
  };
  await writeMeta(lora);
  return lora;
}

function contentDispositionName(value: string | null): string | null {
  if (!value) return null;
  const utf = /filename\*=UTF-8''([^;]+)/i.exec(value);
  if (utf?.[1]) return decodeURIComponent(utf[1]);
  const ascii = /filename="?([^";]+)"?/i.exec(value);
  return ascii?.[1] ?? null;
}

async function resolveCivitaiDownload(url: URL, token?: string): Promise<URL> {
  if (url.pathname.startsWith("/api/download/models/")) return url;
  const modelId = /^\/models\/(\d+)/.exec(url.pathname)?.[1];
  if (!modelId) return url;

  const api = new URL(`/api/v1/models/${modelId}`, url.origin);
  const headers: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(api, { headers, cache: "no-store" });
  if (!res.ok) throw new Error(`Civitai lookup failed (${res.status}).`);
  const data = (await res.json()) as {
    modelVersions?: Array<{
      files?: Array<{ name?: string; downloadUrl?: string; type?: string }>;
    }>;
  };
  const file = data.modelVersions
    ?.flatMap((version) => version.files ?? [])
    .find(
      (candidate) =>
        candidate.downloadUrl &&
        (!candidate.name || path.extname(candidate.name).toLowerCase() === ".safetensors") &&
        (!candidate.type || candidate.type.toLowerCase() === "model"),
    );
  if (!file?.downloadUrl) throw new Error("No downloadable LoRA file found.");
  return new URL(file.downloadUrl);
}

function resolveHuggingFaceDownload(url: URL): URL {
  if (url.pathname.includes("/blob/")) {
    url.pathname = url.pathname.replace("/blob/", "/resolve/");
  }
  return url;
}

async function resolveDownload(urlValue: string): Promise<{
  url: URL;
  headers: HeadersInit;
  source: LoraSource;
}> {
  const keys = await readApiKeys();
  const url = new URL(urlValue);
  if (url.protocol !== "https:") {
    throw new Error("LoRA installs require an HTTPS URL.");
  }
  const source = sourceFromUrl(url.hostname);
  const headers: HeadersInit = {};

  if (source === "huggingface") {
    if (keys.huggingface) headers.Authorization = `Bearer ${keys.huggingface}`;
    return { url: resolveHuggingFaceDownload(url), headers, source };
  }
  if (source === "civitai") {
    if (keys.civitai) headers.Authorization = `Bearer ${keys.civitai}`;
    const resolved = await resolveCivitaiDownload(url, keys.civitai);
    if (keys.civitai && resolved.hostname.includes("civitai.com")) {
      resolved.searchParams.set("token", keys.civitai);
    }
    return { url: resolved, headers, source };
  }
  throw new Error("Use a Civitai or Hugging Face URL, or upload a LoRA file.");
}

export async function downloadLora(urlValue: string): Promise<Lora> {
  const dir = await ensureDir();
  const { url, headers, source } = await resolveDownload(urlValue);
  const res = await fetch(url, { headers, redirect: "follow" });
  if (!res.ok) throw new Error(`LoRA download failed (${res.status}).`);
  if (!res.body) throw new Error("LoRA download response was empty.");

  const suggested =
    contentDispositionName(res.headers.get("content-disposition")) ||
    decodeURIComponent(path.basename(new URL(res.url).pathname)) ||
    "lora.safetensors";
  const fileName = await uniqueFileName(dir, suggested);
  const tmp = path.join(/* turbopackIgnore: true */ dir, `${fileName}.part`);
  const out = path.join(/* turbopackIgnore: true */ dir, fileName);

  await pipeline(
    Readable.fromWeb(res.body as unknown as NodeReadableStream<Uint8Array>),
    createWriteStream(tmp, { mode: 0o600 }),
  );
  await fs.rename(tmp, out);
  const stat = await fs.stat(out);
  const lora: Lora = {
    id: fileName,
    name: displayName(fileName),
    source,
    baseModel: "Qwen",
    sizeBytes: stat.size,
    triggerWords: [],
    installedAt: new Date().toISOString(),
    path: fileName,
  };
  await writeMeta(lora);
  return lora;
}

export async function removeLora(id: string): Promise<boolean> {
  if (!isSafeFileName(id)) return false;
  const dir = await ensureDir();
  let removed = false;
  for (const fileName of [id, `${id}${META_SUFFIX}`]) {
    try {
      await fs.unlink(path.join(/* turbopackIgnore: true */ dir, fileName));
      removed = true;
    } catch {
      /* already gone */
    }
  }
  return removed;
}
