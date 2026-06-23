import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { runGeneration } from "@/lib/runners/client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
      loras: Array.isArray(body.loras)
        ? body.loras.filter((x): x is string => typeof x === "string")
        : undefined,
    });
    return Response.json(result);
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : "Runner failed." },
      { status: 502 },
    );
  }
}
