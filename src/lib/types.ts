// Shared domain types for Citivia Studio.

export type ModelId = "qwen-2512" | "qwen-edit-2511";

export interface ModelInfo {
  id: ModelId;
  name: string;
  tagline: string;
  description: string;
  kind: "generation" | "editing";
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

export interface LoraSelection {
  path: string;
  strength: number;
  enabled?: boolean;
}

/** A single downloadable file resolved from a Civitai/HF URL (for the picker). */
export interface LoraCandidate {
  id: string;
  fileName: string;
  label: string;
  downloadUrl: string;
  sizeBytes?: number;
  versionName?: string;
  baseModel?: string;
  /** Pre-selected in the picker (Civitai "primary" file / single-file repos). */
  recommended?: boolean;
}

export interface LoraResolution {
  source: LoraSource;
  modelName?: string;
  candidates: LoraCandidate[];
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
  createdAt: string;
  imageDataUrl?: string;
  outputPath?: string;
  /** Id of the encrypted vault asset this image was saved as, if any. */
  vaultId?: string;
  error?: string;
  loras?: LoraSelection[];
}
