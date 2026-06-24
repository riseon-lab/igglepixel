import type { LoraSelection, ModelId } from "@/lib/types";

const DEFAULT_URLS: Record<ModelId, string> = {
  "qwen-2512": "http://127.0.0.1:8011",
  "qwen-edit-2511": "http://127.0.0.1:8012",
};

const ENV_URLS: Record<ModelId, string | undefined> = {
  "qwen-2512": process.env.QWEN_2512_RUNNER_URL,
  "qwen-edit-2511": process.env.QWEN_EDIT_2511_RUNNER_URL,
};

export interface RunnerGenerateBody {
  model: ModelId;
  prompt: string;
  negativePrompt?: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  seed: number;
  imageBase64?: string;
  /** Edit only: when true the output matches the reference image's size. */
  matchRef?: boolean;
  loras?: LoraSelection[];
}

export interface RunnerGenerateResult {
  path?: string | null;
  mime: string;
  width: number;
  height: number;
  seed: number;
  image_base64: string;
}

export interface RunnerGenerationProgress {
  active: boolean;
  step: number;
  steps: number;
  progress: number;
  seed?: number | null;
  preview_mime?: string;
  preview_base64?: string | null;
  error?: string | null;
}

/** Shape returned by the Python runner's GET /health. */
export interface RunnerHealth {
  ok: boolean;
  mode?: string;
  model?: string;
  loaded: boolean;
  loading?: boolean;
  load_error?: string | null;
  torch?: string;
  cuda?: boolean;
  device?: string | null;
  vram_used_gb?: number | null;
  vram_total_gb?: number | null;
  logs?: string[];
  generation?: RunnerGenerationProgress;
}

export function runnerUrl(model: ModelId): string {
  return ENV_URLS[model] || DEFAULT_URLS[model];
}

export async function runnerHealth(model: ModelId): Promise<RunnerHealth> {
  const res = await fetch(`${runnerUrl(model)}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Runner health failed (${res.status}).`);
  return res.json();
}

/** Start loading the model (non-blocking on the runner — poll health for progress). */
export async function runnerLoad(model: ModelId): Promise<RunnerHealth> {
  const res = await fetch(`${runnerUrl(model)}/load`, { method: "POST" });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Runner load failed (${res.status}).`);
  }
  return res.json();
}

/** Unload the model and free its VRAM. */
export async function runnerUnload(model: ModelId): Promise<RunnerHealth> {
  const res = await fetch(`${runnerUrl(model)}/unload`, { method: "POST" });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Runner unload failed (${res.status}).`);
  }
  return res.json();
}

/** Ask the runner to interrupt the in-flight generation. */
export async function runnerCancel(model: ModelId): Promise<void> {
  await fetch(`${runnerUrl(model)}/cancel`, { method: "POST" }).catch(() => {});
}

/** Unload and delete the model's cached weights from disk. */
export async function runnerDeleteWeights(model: ModelId): Promise<RunnerHealth> {
  const res = await fetch(`${runnerUrl(model)}/delete-weights`, { method: "POST" });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || body.ok === false) {
    throw new Error(body.error || `Could not delete weights (${res.status}).`);
  }
  return body;
}

export async function runGeneration(
  body: RunnerGenerateBody,
): Promise<RunnerGenerateResult> {
  const res = await fetch(`${runnerUrl(body.model)}/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      prompt: body.prompt,
      negative_prompt: body.negativePrompt,
      width: body.width,
      height: body.height,
      steps: body.steps,
      cfg: body.cfg,
      seed: body.seed,
      image_base64: body.imageBase64,
      match_ref: body.matchRef,
      loras: body.loras,
    }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Runner generation failed (${res.status}).`);
  }
  return res.json();
}
