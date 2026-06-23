import type { ModelId } from "@/lib/types";

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
  loras?: string[];
}

export interface RunnerGenerateResult {
  path: string;
  mime: string;
  width: number;
  height: number;
  seed: number;
  image_base64: string;
}

export function runnerUrl(model: ModelId): string {
  return ENV_URLS[model] || DEFAULT_URLS[model];
}

export async function runnerHealth(model: ModelId): Promise<unknown> {
  const res = await fetch(`${runnerUrl(model)}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Runner health failed (${res.status}).`);
  return res.json();
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
      loras: body.loras,
    }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Runner generation failed (${res.status}).`);
  }
  return res.json();
}
