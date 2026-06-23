import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { getApiKeyStatus, saveApiKeys } from "@/lib/settings/keys";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  return Response.json(await getApiKeyStatus());
}

export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();
  const body = await req.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return new Response("Invalid request.", { status: 400 });
  }
  await saveApiKeys({
    civitai: typeof body.civitai === "string" ? body.civitai : undefined,
    huggingface:
      typeof body.huggingface === "string" ? body.huggingface : undefined,
  });
  return Response.json(await getApiKeyStatus());
}
