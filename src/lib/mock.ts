// Sample data for dashboards, default queues, and local empty states. Core
// generation, auth, vault, and LoRA paths are backed by server APIs.

import type {
  Asset,
  DownloadItem,
  Lora,
  ModelInfo,
  QueueJob,
  ResolutionPreset,
} from "./types";

export const DEFAULT_NEGATIVE_PROMPT =
  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, " +
  "fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, " +
  "watermark, username, blurry";

export const RESOLUTION_PRESETS: ResolutionPreset[] = [
  { label: "Square", width: 512, height: 512 },
  { label: "Square HD", width: 1024, height: 1024 },
  { label: "Square XL", width: 1380, height: 1380 },
  { label: "Square 2K", width: 2048, height: 2048 },
  { label: "Portrait", width: 1344, height: 1792 },
  { label: "Portrait Tall", width: 1024, height: 1360 },
  { label: "Story", width: 1080, height: 1920 },
  { label: "Landscape", width: 1920, height: 1080 },
];

export const MODELS: ModelInfo[] = [
  {
    id: "qwen-2512",
    name: "Qwen 2512",
    tagline: "Base Generation",
    description:
      "High-fidelity text-to-image generation. Best for original artwork, concepts and renders.",
    kind: "generation",
    status: "running",
    vramGb: 38,
  },
  {
    id: "qwen-edit-2511",
    name: "Qwen Edit 2511",
    tagline: "Image Editing",
    description:
      "Reference-guided editing. Supply an image and a prompt to transform, restyle or extend it.",
    kind: "editing",
    status: "stopped",
    vramGb: 41,
  },
];

const now = Date.UTC(2026, 5, 23, 12, 0, 0); // fixed clock for deterministic mock data
const minutesAgo = (m: number) => new Date(now - m * 60_000).toISOString();
const daysAgo = (d: number) => new Date(now - d * 86_400_000).toISOString();

export const ASSETS: Asset[] = [
  { id: "a1", name: "portrait-study-01.png", kind: "generated", width: 1344, height: 1792, createdAt: minutesAgo(8), sizeBytes: 2_410_000, hue: 268, encrypted: true },
  { id: "a2", name: "city-reference.jpg", kind: "reference", width: 1920, height: 1080, createdAt: minutesAgo(42), sizeBytes: 1_180_000, hue: 210, encrypted: true },
  { id: "a3", name: "product-mock.png", kind: "upload", width: 1024, height: 1024, createdAt: daysAgo(1), sizeBytes: 980_000, hue: 150, encrypted: true },
  { id: "a4", name: "landscape-final.png", kind: "generated", width: 1920, height: 1080, createdAt: daysAgo(1), sizeBytes: 3_120_000, hue: 32, encrypted: true },
  { id: "a5", name: "character-sheet.png", kind: "generated", width: 1380, height: 1380, createdAt: daysAgo(2), sizeBytes: 2_760_000, hue: 320, encrypted: true },
  { id: "a6", name: "moodboard.jpg", kind: "reference", width: 1024, height: 1360, createdAt: daysAgo(3), sizeBytes: 1_540_000, hue: 190, encrypted: true },
];

export const LORAS: Lora[] = [
  { id: "filmic-portrait-v3.safetensors", name: "Filmic Portrait v3", source: "civitai", baseModel: "Qwen 2512", sizeBytes: 223_000_000, triggerWords: ["filmic", "cinematic portrait"], installedAt: daysAgo(2), path: "filmic-portrait-v3.safetensors" },
  { id: "studio-product-light.safetensors", name: "Studio Product Light", source: "huggingface", baseModel: "Qwen 2512", sizeBytes: 198_000_000, triggerWords: ["studio lighting", "product shot"], installedAt: daysAgo(5), path: "studio-product-light.safetensors" },
  { id: "ink-watercolour.safetensors", name: "Ink & Watercolour", source: "civitai", baseModel: "Qwen 2512", sizeBytes: 167_000_000, triggerWords: ["ink wash", "watercolour"], installedAt: daysAgo(9), path: "ink-watercolour.safetensors" },
  { id: "architectural-detail.safetensors", name: "Architectural Detail", source: "upload", baseModel: "Qwen Edit 2511", sizeBytes: 240_000_000, triggerWords: ["archviz"], installedAt: daysAgo(12), path: "architectural-detail.safetensors" },
];

export const DOWNLOADS: DownloadItem[] = [
  { id: "d1", name: "Filmic Portrait v4", kind: "lora", source: "civitai", status: "downloading", progress: 64, sizeBytes: 231_000_000, startedAt: minutesAgo(3) },
  { id: "d2", name: "Qwen Edit 2511 weights", kind: "model", source: "huggingface", status: "downloading", progress: 21, sizeBytes: 17_400_000_000, startedAt: minutesAgo(11) },
  { id: "d3", name: "Neon Signage LoRA", kind: "lora", source: "civitai", status: "queued", progress: 0, sizeBytes: 184_000_000, startedAt: minutesAgo(1) },
  { id: "d4", name: "Studio Product Light", kind: "lora", source: "huggingface", status: "completed", progress: 100, sizeBytes: 198_000_000, startedAt: daysAgo(5) },
  { id: "d5", name: "Anime Lineart", kind: "lora", source: "civitai", status: "failed", progress: 38, sizeBytes: 176_000_000, startedAt: daysAgo(1) },
];

export const QUEUE: QueueJob[] = [
  { id: "q1", model: "qwen-2512", prompt: "A serene mountain lake at golden hour, mist rising, ultra detailed", width: 1920, height: 1080, steps: 28, cfg: 4, seed: 84213, status: "running", progress: 52, hue: 32, createdAt: minutesAgo(1) },
  { id: "q2", model: "qwen-2512", prompt: "Studio portrait of a woman, soft rim light, 85mm", width: 1344, height: 1792, steps: 24, cfg: 4.5, seed: 11902, status: "pending", progress: 0, hue: 280, createdAt: minutesAgo(1) },
  { id: "q3", model: "qwen-edit-2511", prompt: "Replace background with a sunlit forest", width: 1024, height: 1024, steps: 20, cfg: 4, seed: 55001, status: "pending", progress: 0, hue: 140, createdAt: minutesAgo(2) },
  { id: "q4", model: "qwen-2512", prompt: "Isometric cyberpunk alleyway, neon, rain", width: 1380, height: 1380, steps: 30, cfg: 5, seed: 77310, status: "completed", progress: 100, hue: 300, createdAt: minutesAgo(9) },
  { id: "q5", model: "qwen-2512", prompt: "Minimalist product render of a ceramic mug on stone", width: 1024, height: 1024, steps: 22, cfg: 3.5, seed: 20445, status: "completed", progress: 100, hue: 90, createdAt: minutesAgo(14) },
];

/** GPU/host telemetry shown on the Running page (mock). */
export const RESOURCE_USAGE = {
  gpuName: "RTX PRO 6000",
  vramUsedGb: 38,
  vramTotalGb: 96,
  gpuUtil: 71,
  ramUsedGb: 22,
  ramTotalGb: 128,
  temperatureC: 64,
};
