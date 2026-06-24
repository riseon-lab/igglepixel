import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { runGeneration, runnerHealth } from "@/lib/runners/client";
import type { LoraSelection, ModelId } from "@/lib/types";

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

  const args = {
    model: model as ModelId,
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
  };

  // Generation runs for minutes (model load + sampling) and returns a single JSON
  // blob only at the very end. The browser<->Next hop goes through RunPod's proxy,
  // which kills any request that sends no bytes for ~100s. So instead of awaiting
  // the whole thing, we stream NDJSON: forward the runner's live progress/preview
  // frames (polled over loopback, which has no proxy/timeout) as keepalive, then
  // emit the final result. One JSON object per line.
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false;
      const send = (obj: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
        } catch {
          closed = true;
        }
      };

      // Flush an immediate frame so the proxy sees bytes before the first poll.
      send({ type: "progress", progress: 0, step: 0, steps: args.steps });

      let finished = false;
      const generation = runGeneration(args).then(
        (result) => ({ ok: true as const, result }),
        (err) => ({
          ok: false as const,
          error: err instanceof Error ? err.message : "Runner failed.",
        }),
      );

      // Poll the runner's /health (loopback) and forward progress as keepalive.
      const poll = (async () => {
        while (!finished) {
          await new Promise((r) => setTimeout(r, 1200));
          if (finished) break;
          try {
            const g = (await runnerHealth(args.model)).generation;
            if (g) {
              send({
                type: "progress",
                progress: g.progress,
                step: g.step,
                steps: g.steps,
                preview_mime: g.preview_mime,
                preview_base64: g.preview_base64,
              });
            } else {
              send({ type: "ping" });
            }
          } catch {
            send({ type: "ping" });
          }
        }
      })();

      const outcome = await generation;
      finished = true;
      await poll;
      if (outcome.ok) send({ type: "result", ...outcome.result });
      else send({ type: "error", error: outcome.error });
      closed = true;
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-store, no-transform",
      // Discourage any intermediary from buffering the stream.
      "x-accel-buffering": "no",
    },
  });
}
