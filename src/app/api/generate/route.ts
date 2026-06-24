import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { runGeneration } from "@/lib/runners/client";
import type { LoraSelection } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function parseLoras(value: unknown): LoraSelection[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.flatMap((item) => {
    if (typeof item === "string") return [{ path: item, strength: 1 }];
    if (!item || typeof item !== "object") return [];
    if ((item as { enabled?: unknown }).enabled === false) return [];
    const path = (item as { path?: unknown }).path;
    if (typeof path !== "string") return [];
    const raw = Number((item as { strength?: unknown }).strength);
    return [{ path, strength: Math.max(0.1, Number.isFinite(raw) ? raw : 1) }];
  });
}

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON." }, { status: 400 });
  }

  const model = body.model;
  if (model !== "qwen-2512" && model !== "qwen-edit-2511") {
    return Response.json({ error: "Unknown model." }, { status: 400 });
  }

  const prompt = String(body.prompt ?? "").trim();
  if (!prompt) return Response.json({ error: "Prompt is required." }, { status: 400 });
  if (model === "qwen-edit-2511" && typeof body.imageBase64 !== "string") {
    return Response.json({ error: "Reference image is required." }, { status: 400 });
  }

  try {
    const result = await runGeneration({
      model,
      prompt,
      negativePrompt: String(body.negativePrompt ?? " "),
      width: Number(body.width) || 1024,
      height: Number(body.height) || 1024,
      steps: Number(body.steps) || 30,
      cfg: Number(body.cfg) || 4,
      seed: Number(body.seed) || Date.now(),
      imageBase64:
        typeof body.imageBase64 === "string" ? body.imageBase64 : undefined,
      loras: parseLoras(body.loras),
    });
    return Response.json(result);
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : "Runner failed." },
      { status: 502 },
    );
  }
}
