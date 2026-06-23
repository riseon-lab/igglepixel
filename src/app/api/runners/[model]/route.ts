import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import {
  runnerDeleteWeights,
  runnerHealth,
  runnerLoad,
  runnerUnload,
} from "@/lib/runners/client";
import type { ModelId } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MODELS: ModelId[] = ["qwen-2512", "qwen-edit-2511"];

function isModel(value: string): value is ModelId {
  return (MODELS as string[]).includes(value);
}

// GET  → current health for one model
// POST { action: "start" | "stop" } → load / unload the model on its runner
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ model: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { model } = await params;
  if (!isModel(model)) return new Response("Unknown model", { status: 404 });
  try {
    return Response.json(await runnerHealth(model));
  } catch (err) {
    return Response.json(
      { ok: false, error: err instanceof Error ? err.message : "unreachable" },
      { status: 502 },
    );
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ model: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { model } = await params;
  if (!isModel(model)) return new Response("Unknown model", { status: 404 });

  let action: unknown;
  try {
    action = (await req.json()).action;
  } catch {
    return new Response("Invalid request body.", { status: 400 });
  }
  if (action !== "start" && action !== "stop") {
    return new Response("action must be 'start' or 'stop'.", { status: 400 });
  }

  try {
    const health =
      action === "start" ? await runnerLoad(model) : await runnerUnload(model);
    return Response.json(health);
  } catch (err) {
    return Response.json(
      {
        ok: false,
        error:
          err instanceof Error
            ? err.message
            : `Could not reach the ${model} runner.`,
      },
      { status: 502 },
    );
  }
}

// DELETE → unload the model and remove its cached weights from disk.
export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ model: string }> },
) {
  if (!(await requireSession(req))) return unauthorized();
  const { model } = await params;
  if (!isModel(model)) return new Response("Unknown model", { status: 404 });

  try {
    return Response.json(await runnerDeleteWeights(model));
  } catch (err) {
    return Response.json(
      {
        ok: false,
        error:
          err instanceof Error ? err.message : `Could not reach the ${model} runner.`,
      },
      { status: 502 },
    );
  }
}
