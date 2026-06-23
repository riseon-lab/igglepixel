import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { runnerHealth } from "@/lib/runners/client";
import type { ModelId } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MODELS: ModelId[] = ["qwen-2512", "qwen-edit-2511"];

export async function GET(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  const checks = await Promise.all(
    MODELS.map(async (model) => {
      try {
        return { model, ok: true, detail: await runnerHealth(model) };
      } catch (err) {
        return {
          model,
          ok: false,
          error: err instanceof Error ? err.message : "unreachable",
        };
      }
    }),
  );
  return Response.json(checks);
}
