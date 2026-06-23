import type { NextRequest } from "next/server";
import { requireSession, unauthorized } from "@/lib/auth/server";
import { resolveLoraCandidates } from "@/lib/loras/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Resolves a Civitai/HF URL into its installable files WITHOUT downloading, so
// the UI can present a picker.
export async function POST(req: NextRequest) {
  if (!(await requireSession(req))) return unauthorized();

  const body = (await req.json().catch(() => null)) as { url?: unknown } | null;
  if (!body || typeof body.url !== "string" || !body.url.trim()) {
    return Response.json({ error: "A URL is required." }, { status: 400 });
  }

  try {
    return Response.json(await resolveLoraCandidates(body.url.trim()));
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : "Could not resolve that URL." },
      { status: 400 },
    );
  }
}
