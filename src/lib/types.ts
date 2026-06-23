// Shared domain types for Citivia Studio.

export type ModelId = "qwen-2512" | "qwen-edit-2511";

export type ModelStatus = "running" | "stopped" | "starting" | "stopping";

export interface ModelInfo {
  id: ModelId;
  name: string;
  tagline: string;
  description: string;
  kind: "generation" | "editing";
  status: ModelStatus;
  /** GPU memory footprint in GB when loaded. */
  vramGb: number;
}

export interface ResolutionPreset {
  label: string;
  width: number;
  height: number;
}

export type AssetKind = "upload" | "reference" | "generated";

export interface Asset {
  id: string;
  name: string;
  kind: AssetKind;
  width: number;
  height: number;
  /** ISO date string. */
  createdAt: string;
  sizeBytes: number;
  /** Placeholder gradient seed so previews look distinct without real files. */
  hue: number;
  encrypted: boolean;
}

export type LoraSource = "civitai" | "huggingface" | "upload";

export interface Lora {
  id: string;
  name: string;
  source: LoraSource;
  baseModel: string;
  sizeBytes: number;
  triggerWords: string[];
  installedAt: string;
  /** Path relative to the runner LORA_DIR. */
  path: string;
}

export type DownloadKind = "model" | "lora";
export type DownloadStatus =
  | "queued"
  | "downloading"
  | "completed"
  | "failed"
  | "paused";

export interface DownloadItem {
  id: string;
  name: string;
  kind: DownloadKind;
  source: LoraSource;
  status: DownloadStatus;
  /** 0–100 */
  progress: number;
  sizeBytes: number;
  startedAt: string;
}

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface QueueJob {
  id: string;
  model: ModelId;
  prompt: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  seed: number;
  status: JobStatus;
  /** 0–100, meaningful while running. */
  progress: number;
  hue: number;
  createdAt: string;
  imageDataUrl?: string;
  outputPath?: string;
  error?: string;
  loras?: string[];
}
