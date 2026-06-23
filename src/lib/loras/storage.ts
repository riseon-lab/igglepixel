import { createWriteStream, promises as fs } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import { readApiKeys } from "@/lib/settings/keys";
import type {
  Lora,
  LoraCandidate,
  LoraResolution,
  LoraSource,
} from "@/lib/types";

const LORA_EXTENSIONS = new Set([".bin", ".ckpt", ".pt", ".safetensors"]);
// Civitai rejects some requests that lack a browser-like User-Agent.
const BROWSER_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";
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

function sourceFromUrl(hostname: string): LoraSource {
  const host = hostname.toLowerCase();
  if (host.includes("huggingface.co") || host.endsWith("hf.co")) {
    return "huggingface";
  }
  // Match Civitai on any TLD (civitai.com, civitai.red, civitai.ai, …).
  if (host.includes("civitai")) return "civitai";
  return "upload";
}

function civitaiApiOrigin(url: URL): string {
  // Use the page's own origin so .com / .red / .ai each hit their own API.
  return /civitai/i.test(url.hostname) ? url.origin : "https://civitai.com";
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
  const headers: Record<string, string> = { "user-agent": BROWSER_UA };
  if (token) headers.Authorization = `Bearer ${token}`;
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

function isLoraFile(name?: string | null): boolean {
  return !!name && LORA_EXTENSIONS.has(path.extname(name).toLowerCase());
}

// ---------------------------------------------------------------------------
// Resolve a Civitai / Hugging Face URL into the list of installable files so the
// UI can show a picker (instead of silently guessing one file).
// ---------------------------------------------------------------------------

interface CivitaiFile {
  id?: number;
  name?: string;
  sizeKB?: number;
  type?: string;
  primary?: boolean;
  downloadUrl?: string;
}
interface CivitaiVersion {
  id?: number;
  name?: string;
  baseModel?: string;
  files?: CivitaiFile[];
  model?: { name?: string };
}

async function civitaiApi(
  origin: string,
  apiPath: string,
  token?: string,
): Promise<unknown> {
  const headers: Record<string, string> = { "user-agent": BROWSER_UA };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${origin}${apiPath}`, {
    headers,
    cache: "no-store",
  });
  if (res.status === 401 || res.status === 403) {
    throw new Error("Civitai requires an API key for this model — add one in Settings.");
  }
  if (!res.ok) throw new Error(`Civitai lookup failed (${res.status}).`);
  return res.json();
}

function civitaiCandidates(version: CivitaiVersion): LoraCandidate[] {
  return (version.files ?? [])
    .filter(
      (f) =>
        f.downloadUrl &&
        (isLoraFile(f.name) || (f.type ?? "").toLowerCase() === "model"),
    )
    .map((f) => ({
      id: `civitai-${version.id}-${f.id ?? f.name}`,
      fileName: toSafeFileName(f.name ?? `${version.name ?? "lora"}.safetensors`),
      label: `${version.name ?? "version"} · ${f.name ?? "model.safetensors"}`,
      downloadUrl: f.downloadUrl!,
      sizeBytes: f.sizeKB ? Math.round(f.sizeKB * 1024) : undefined,
      versionName: version.name,
      baseModel: version.baseModel,
      recommended: !!f.primary,
    }));
}

async function resolveCivitai(url: URL, token?: string): Promise<LoraResolution> {
  const origin = civitaiApiOrigin(url);
  const versionDownload = /^\/api\/download\/models\/(\d+)/.exec(url.pathname);
  if (versionDownload) {
    const v = (await civitaiApi(
      origin,
      `/api/v1/model-versions/${versionDownload[1]}`,
      token,
    )) as CivitaiVersion;
    return {
      source: "civitai",
      modelName: v.model?.name ?? v.name,
      candidates: civitaiCandidates(v),
    };
  }

  const modelPage = /^\/models\/(\d+)/.exec(url.pathname);
  if (modelPage) {
    const data = (await civitaiApi(origin, `/api/v1/models/${modelPage[1]}`, token)) as {
      name?: string;
      modelVersions?: CivitaiVersion[];
    };
    const wanted = url.searchParams.get("modelVersionId");
    const versions = (data.modelVersions ?? []).filter(
      (v) => !wanted || String(v.id) === wanted,
    );
    return {
      source: "civitai",
      modelName: data.name,
      candidates: versions.flatMap(civitaiCandidates),
    };
  }

  throw new Error(
    "Unrecognised Civitai URL. Use a model page (civitai.com/models/…) or a download link.",
  );
}

async function resolveHuggingFace(url: URL, token?: string): Promise<LoraResolution> {
  const parts = url.pathname.split("/").filter(Boolean);

  // Direct file link: /<owner>/<repo>/(blob|resolve)/<rev>/<path...>
  const fileMatch = /\/(?:blob|resolve)\/([^/]+)\/(.+)$/.exec(url.pathname);
  if (fileMatch && isLoraFile(fileMatch[2])) {
    const dl = new URL(url.toString());
    dl.pathname = dl.pathname.replace("/blob/", "/resolve/");
    const name = decodeURIComponent(fileMatch[2].split("/").pop()!);
    return {
      source: "huggingface",
      modelName: parts.length >= 2 ? `${parts[0]}/${parts[1]}` : undefined,
      candidates: [
        {
          id: `hf-${name}`,
          fileName: toSafeFileName(name),
          label: fileMatch[2],
          downloadUrl: dl.toString(),
          recommended: true,
        },
      ],
    };
  }

  // Repo URL (optionally /tree/<rev>) → list the repo's LoRA files.
  if (parts.length >= 2) {
    const [owner, repo] = parts;
    const rev = /\/tree\/([^/]+)/.exec(url.pathname)?.[1] ?? "main";
    const api =
      `https://huggingface.co/api/models/${owner}/${repo}` +
      (rev !== "main" ? `/revision/${encodeURIComponent(rev)}` : "");
    const res = await fetch(api, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      cache: "no-store",
    });
    if (res.status === 401 || res.status === 403) {
      throw new Error(
        "This Hugging Face repo is gated — add an access token in Settings.",
      );
    }
    if (!res.ok) throw new Error(`Hugging Face lookup failed (${res.status}).`);
    const data = (await res.json()) as { siblings?: Array<{ rfilename?: string }> };
    const files = (data.siblings ?? [])
      .map((s) => s.rfilename)
      .filter((n): n is string => isLoraFile(n));
    return {
      source: "huggingface",
      modelName: `${owner}/${repo}`,
      candidates: files.map((name) => ({
        id: `hf-${name}`,
        fileName: toSafeFileName(name.split("/").pop()!),
        label: name,
        downloadUrl: `https://huggingface.co/${owner}/${repo}/resolve/${rev}/${name}`,
        recommended: files.length === 1,
      })),
    };
  }

  throw new Error("Unrecognised Hugging Face URL.");
}

export async function resolveLoraCandidates(
  urlValue: string,
): Promise<LoraResolution> {
  const keys = await readApiKeys();
  let url: URL;
  try {
    url = new URL(urlValue);
  } catch {
    throw new Error("That doesn't look like a valid URL.");
  }
  if (url.protocol !== "https:") {
    throw new Error("LoRA installs require an HTTPS URL.");
  }
  const source = sourceFromUrl(url.hostname);
  if (source === "civitai") return resolveCivitai(url, keys.civitai);
  if (source === "huggingface") return resolveHuggingFace(url, keys.huggingface);
  throw new Error("Use a Civitai or Hugging Face URL, or upload a LoRA file.");
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
    if (keys.civitai && /civitai/i.test(resolved.hostname)) {
      resolved.searchParams.set("token", keys.civitai);
    }
    return { url: resolved, headers, source };
  }
  throw new Error("Use a Civitai or Hugging Face URL, or upload a LoRA file.");
}

export async function downloadLora(urlValue: string): Promise<Lora> {
  const dir = await ensureDir();
  const { url, headers, source } = await resolveDownload(urlValue);
  const res = await fetch(url, {
    headers: { "user-agent": BROWSER_UA, ...(headers as Record<string, string>) },
    redirect: "follow",
  });
  if (res.status === 401 || res.status === 403) {
    const where = source === "civitai" ? "Civitai" : "Hugging Face";
    throw new Error(
      `${where} rejected the download (auth required) — add your ${where} API key in Settings.`,
    );
  }
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
